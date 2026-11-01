"""Method33: complete standalone traffic-defense core.

This file contains the entire executable Method33 algorithm.  The
implementation has two local stages:

1. a deterministic base stage implementing input validation, the random
   weakener, code-dependent Trigger positions, budget accounting, fixed-window
   rendering, metadata, and optional complete event provenance;
2. a Method33 stage that applies per-flow, zero-sum internal jitter to the two
   Trigger packets while preserving Trigger bytes, delay, and dummy count
   exactly.

Keeping both stages in one module makes ``method33.py`` a deployable algorithm
core rather than a wrapper around another method. ``Method33Config`` is the
single public runtime-policy type.

The returned application keeps the stable seven-element protocol:

    (lengths, gaps, directions, mask, stats, metadata, audit_rows)
"""

from __future__ import annotations

# =============================================================================
# 独立性说明
# =============================================================================
# 本文件不是“调用其他算法后做一次后处理”的薄包装。下面首先完整定义预算、弱化
# 器配置、运行配置、输入校验、随机弱化、Trigger插入、成本账本、固定窗口渲染
# 和事件trace；随后再定义Method33专属的5%零和Trigger内部波动。
#
# 为了保证既有实验数值连续，确定性基础阶段逐语义继承已验证实现，没有重新设计
# 随机数顺序或浮点账本；Trigger改为随码字轮换的位置组合。它在本文件中的私有入口是
# apply_method33_base；正式防御入口是apply_method33。

# =============================================================================
# 中文阅读导图：本文件是“纯算法核心”
# =============================================================================
# 本文件刻意不读取配置文件、不访问 PCAP、不训练模型、不输出日志，也不导入
# 其他算法脚本。调用者必须先准备好五个输入数组和已经验证的 Method33Config，
# 然后调用公开入口apply_method33。这样的设计让论文方法与实验工程解耦：算法
# 核心可以被命令行、Notebook、单元测试或未来的在线网关共同复用。
#
# apply_method33_base 的输入约定：
#   lengths / gaps / directions / mask：形状相同的 [N, L] 固定窗口数组；
#   labels：形状 [N]，只有 trigger 层会读取；
#   config：不可变运行策略，包含弱化器、码本、映射和三层预算；
#   include_event_trace：默认False；仅inspection设True，把完整事件来源附加到
#   audit_rows，不改变防御数组、随机路径、预算或稳定七元组的任何外层位置。
#   seed：批次随机种子；同一输入顺序、配置和 seed 会得到同一结果。
#
# 确定性基础阶段 apply_method33_base 的执行顺序：
#   1. 校验输入，解析 old/next/显式 mapping；
#   2. 为每条流建立独立 RNG，并提前抽取与标签无关的弱化 dummy 计划；
#   3. 先为计划中的 dummy 预留预算，再在线逐个处理真实事件；
#   4. 仅对上行真实包做随机 padding/delay；
#   5. 按code选择两个真实分组锚点，在对应位置分别插入Trigger；
#   6. 在抽中的上行序号后插入配置数量的弱化 dummy；
#   7. 检查逐流预算，生成 metadata/audit；
#   8. 将完整事件序列渲染为分类器看到的前 L 个事件；
#   9. apply_method33定位两个Trigger事件，对长度和gap做5%零和内部波动，
#      同步更新完整trace和前L输出矩阵，并再次验证Trigger分账与硬预算。
#
# “完整事件序列”与“前 L 观察窗口”必须区分：插 dummy 不会删除网络中的真实
# 包，但会把末尾若干真实包推到分类器窗口之外。displaced_real_packets 统计的
# 是这种观察窗口位移，而不是网络层丢包。

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
from experiment_support.metrics import timing_metrics, store_timing


@dataclass(frozen=True)
class Budget:
    """Per-flow limits ordered as bytes, added delay, and dummy count."""

    bytes: float
    delay_seconds: float
    dummy_packets: int

    # JSON/metadata 常用 [bytes, delay, dummy] 顺序；集中转换可避免不同调用点
    # 把三个量排错。显式 float/int 也把 NumPy 标量隔离在公共接口之外。
    def as_list(self) -> list[float | int]:
        return [float(self.bytes), float(self.delay_seconds), int(self.dummy_packets)]


@dataclass(frozen=True)
class WeakenerConfig:
    """All stochastic parameters used by the class-independent weakener."""

    # length_targets/gap_targets 用于修改真实上行包；dummy_*_targets 用于抽
    # 新增 dummy 的属性。每组 probabilities 与对应 targets 一一配对。
    length_targets: tuple[float, ...]
    length_probabilities: tuple[float, ...]
    gap_targets: tuple[float, ...]
    gap_probabilities: tuple[float, ...]
    max_padding_per_packet: float
    max_delay_per_packet: float
    dummy_length_targets: tuple[float, ...]
    dummy_length_probabilities: tuple[float, ...]
    dummy_gap_targets: tuple[float, ...]
    dummy_gap_probabilities: tuple[float, ...]
    # 插入位置按“第几个上行真实包”计数，而不是按原数组列号计数，因此下行
    # 包不会消耗 slot。padding_probability 与 delay_probability 是两次独立抽样。
    dummy_insertion_outbound_slots: tuple[int, ...]
    padding_probability: float
    delay_probability: float


@dataclass(frozen=True)
class Method33Config:
    """Complete runtime policy after file I/O and validation.

    Keeping the policy immutable prevents a long experiment from changing its
    budget or mapping halfway through.  ``method33_tools.load_config`` is the
    normal constructor; direct construction is mainly useful for tests.
    """

    # 三层预算关系：weakener_budget 管随机弱化层，trigger_budget 管码字层，
    # hard_budget 管两者总和。tools.validate_config 会验证分账不超过总预算。
    version: int
    method_key: str
    weakener_candidate_id: str
    trigger_codebook_id: str
    weakener_dummy_count: int
    trigger_packet_count: int
    hard_budget: Budget
    weakener_budget: Budget
    trigger_budget: Budget
    weakener: WeakenerConfig
    # trigger_codes 形状概念上为 [class][trigger_packet][length, gap]。
    # old_mapping/next_mapping 将可信设备标签映射到码字索引；next 必须无固定点。
    trigger_codes: tuple[tuple[tuple[float, float], ...], ...]
    old_mapping: tuple[int, ...]
    next_mapping: tuple[int, ...]
    max_frame_length: float = 1514.0
    min_dummy_frame_length: float = 60.0
    require_all_dummies_in_window: bool = False
    # One ordered pair per CODE, counted after original real packets (both directions).
    # Empty means the documented deterministic default; mapping selects the pair.
    trigger_positions: tuple[tuple[int, int], ...] = ()

    @property
    def class_count(self) -> int:
        # 类别数直接由码本数推出，避免再维护一个可能与 trigger_codes 冲突
        # 的独立常量。配置验证会进一步要求它等于 dataset.devices 数量。
        return len(self.trigger_codes)


def default_trigger_positions(class_count: int) -> tuple[tuple[int, int], ...]:
    """Balanced, preregistered pairs over real-packet anchors 1/4/8/12/16.

    Cycle-distance two edges precede distance one edges. No labels, data,
    random seeds or classifier scores are used to choose these positions.
    """
    slots = (1, 4, 8, 12, 16)
    pairs = tuple(tuple(sorted((slots[i], slots[(i + step) % 5])))
                  for step in (2, 1) for i in range(5))
    if not 1 <= class_count <= len(pairs):
        raise ValueError("default positions support up to 10 codes; supply trigger_positions")
    return pairs[:class_count]


def resolve_trigger_positions(config: Method33Config) -> tuple[tuple[int, int], ...]:
    """Validate code-indexed positions; the two packets remain in code order."""
    pairs = config.trigger_positions or default_trigger_positions(config.class_count)
    if len(pairs) != config.class_count:
        raise ValueError("trigger_positions must contain exactly one pair per code")
    result = []
    for pair in pairs:
        if len(pair) != 2 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in pair):
            raise ValueError("each trigger_positions pair must contain two integers")
        first, second = map(int, pair)
        if not 1 <= first < second:
            raise ValueError("Trigger positions must be positive and strictly increasing")
        result.append((first, second))
    if len(set(result)) != len(result):
        raise ValueError("different trigger codes must have distinct position pairs")
    return tuple(result)


@dataclass(frozen=True)
class _Event:
    """Internal full-stream event used before fixed-window rendering."""

    # source_index 只对真实事件有值，用于证明源包未被删除、复制或重排；
    # role 进一步区分 real / weakener_dummy / trigger_dummy，便于窗口审计。
    length: float
    gap: float
    direction: int
    is_dummy: bool
    source_index: int | None
    role: str


@dataclass(frozen=True)
class _DummyPlanItem:
    """One class-independent dummy action sampled for a flow."""

    # 一个计划项完全由配置和流级 RNG 产生，不含 label，也不读取任何未来包。
    outbound_slot: int
    length: float
    gap: float


# 七元组是本项目防御方法的公共返回协议：前四项是受保护流量矩阵，第五项
# 是逐流成本数组，第六项是批次 metadata，第七项是每条流的详细 audit。
# 使用 TypeAlias 只增强可读性，不会在运行时复制或转换任何数据。
Application: TypeAlias = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
    list[dict[str, Any]],
]


# =============================================================================
# 成本统计模式
# =============================================================================
# These names are a compatibility contract with ``defense_baselines.common``.
# Defining the small schema locally avoids making our core method depend on the
# baseline implementation while allowing the same evaluators to consume both.
STAT_KEYS = (
    "real_added_bytes",
    "real_added_delay",
    "dummy_count",
    "dummy_bytes",
    "dummy_delay",
    "retained_dummy_count",
    "displaced_real_packets",
    "outbound_real_added_bytes",
    "outbound_real_added_delay",
    "inbound_real_added_bytes",
    "inbound_real_added_delay",
    "outbound_dummy_count",
    "outbound_dummy_bytes",
    "outbound_dummy_delay",
    "inbound_dummy_count",
    "inbound_dummy_bytes",
    "inbound_dummy_delay",
    # Explicit name retained in the shared application-statistics contract.
    "dummy_gap_seconds",
)


def _empty_stats(flow_count: int) -> dict[str, np.ndarray]:
    """Allocate every cost channel once so action code cannot omit a field."""

    # 每个字段预先分配长度 N 的 float64 零数组。集中分配可以确保任一执行
    # 分支都返回完整 schema，并允许下游统一比较 Method33 与 baselines。
    return {
        key: np.zeros(int(flow_count), dtype=np.float64) for key in STAT_KEYS
    }


def _coerce_inputs(
    lengths: np.ndarray,
    gaps: np.ndarray,
    directions: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    config: Method33Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform the minimal safety checks required by the runtime loop.

    Full human-friendly configuration and dataset validation belongs in
    ``method33_tools``.  The core still rejects malformed arrays because silent
    broadcasting or a non-prefix mask would invalidate source-order auditing.
    """

    # 核心层只做保证安全运行所必需的检查；更友好的数据集和配置错误信息在
    # method33_tools 中提供。np.asarray 不强制复制，避免无谓增加大批量内存。
    lengths, gaps, directions, mask = map(
        np.asarray, (lengths, gaps, directions, mask)
    )
    labels = np.asarray(labels, dtype=int)
    if lengths.ndim != 2 or not (
        lengths.shape == gaps.shape == directions.shape == mask.shape
    ):
        raise ValueError("lengths/gaps/directions/mask must share a 2-D shape")
    if labels.shape != (len(lengths),):
        raise ValueError("labels must be one-dimensional and match flow count")

    # valid 只选择真实事件位置。mask 必须原本就是 0/1；若值为 2，虽然
    # astype(bool) 也会变 True，但与转换回原 dtype 后不相等，因而会被拒绝。
    valid = mask.astype(bool)
    if np.any(mask != valid.astype(mask.dtype)):
        raise ValueError("mask must contain only zero and one")
    if np.any(~np.isin(directions[valid], (-1, 1))):
        raise ValueError("valid directions must be -1 or +1")
    if not np.all(np.isfinite(lengths[valid])) or not np.all(
        np.isfinite(gaps[valid])
    ):
        raise ValueError("valid events contain non-finite values")
    if np.any(lengths[valid] <= 0.0) or np.any(gaps[valid] < 0.0):
        raise ValueError("valid lengths must be positive and gaps nonnegative")
    if np.any(lengths[valid] > config.max_frame_length + 1e-9):
        raise ValueError("a source frame exceeds max_frame_length")
    # 每行有效位置必须严格是 [0, count) 的左对齐前缀。固定前缀保证按
    # np.flatnonzero 遍历时，source_index 与真实事件顺序具有唯一解释。
    # A prefix is precisely a boolean row with no False -> True transition.
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("each mask row must be a contiguous prefix")
    if np.any(labels < 0) or np.any(labels >= config.class_count):
        raise ValueError("labels exceed the configured trigger-code classes")
    return lengths, gaps, directions, mask, labels


def _resolve_mapping(
    config: Method33Config, mapping: str | Sequence[int]
) -> tuple[tuple[int, ...], str]:
    """Resolve a named phase or an explicit pre-registered permutation."""

    # 命名映射用于普通 old/next 实验；显式排列用于预先登记的其他错排。
    # 显式值只验证“是排列”，不在核心层强制无固定点，便于诊断控制实验。
    if isinstance(mapping, str):
        normalized = mapping.strip().lower()
        if normalized == "old":
            return config.old_mapping, "old"
        if normalized in {"next", "new", "switched"}:
            return config.next_mapping, "next"
        raise ValueError("mapping must be 'old', 'next', or an explicit permutation")

    explicit = tuple(int(value) for value in mapping)
    expected = tuple(range(config.class_count))
    if tuple(sorted(explicit)) != expected:
        raise ValueError(f"explicit mapping must be a permutation of {expected}")
    return explicit, "explicit"


def _sample_dummy_plan(
    rng: np.random.Generator,
    weakener: WeakenerConfig,
    dummy_count: int,
) -> tuple[_DummyPlanItem, ...]:
    """Sample the entire class-independent dummy plan without reading a label."""

    # 整份计划在逐包循环前一次抽完，但只读取候选 slot/长度/gap 分布；因此
    # 它既不依赖设备标签，也不窥视后续真实包的长度或间隔。
    count = int(dummy_count)
    if count == 0:
        return ()
    slots = np.asarray(weakener.dummy_insertion_outbound_slots, dtype=int)
    # replace=False 防止两个 dummy 选中同一候选 slot；排序后可用单调指针
    # emitted_weakener_dummies 在线消费计划，无需每包搜索整个列表。
    selected_slots = np.sort(rng.choice(slots, size=count, replace=False))
    selected_lengths = rng.choice(
        np.asarray(weakener.dummy_length_targets, dtype=np.float64),
        size=count,
        p=np.asarray(weakener.dummy_length_probabilities, dtype=np.float64),
    )
    selected_gaps = rng.choice(
        np.asarray(weakener.dummy_gap_targets, dtype=np.float64),
        size=count,
        p=np.asarray(weakener.dummy_gap_probabilities, dtype=np.float64),
    )
    return tuple(
        _DummyPlanItem(int(slot), float(length), float(gap))
        for slot, length, gap in zip(
            selected_slots, selected_lengths, selected_gaps, strict=True
        )
    )


def _record_dummy_cost(
    stats: dict[str, np.ndarray], sample: int, length: float, gap: float
) -> None:
    """Update the shared cost ledger for one outbound dummy event."""

    # 当前方法只插入上行 dummy，所以总 dummy 与 outbound dummy 同步增加；
    # inbound_* 字段仍保留为 0，以维持跨防御方法的统一统计模式。
    stats["dummy_count"][sample] += 1.0
    stats["dummy_bytes"][sample] += length
    stats["dummy_delay"][sample] += gap
    stats["dummy_gap_seconds"][sample] += gap
    stats["outbound_dummy_count"][sample] += 1.0
    stats["outbound_dummy_bytes"][sample] += length
    stats["outbound_dummy_delay"][sample] += gap


def _ordered_unique(values: Sequence[int]) -> list[int]:
    """Collapse adjacent segment IDs for source-order verification."""

    # 该辅助函数兼容未来可能把一个源事件分成相邻多个 segment 的渲染器：
    # 相邻重复 source_index 合并，但非相邻重复仍会暴露为复制/重排错误。
    result: list[int] = []
    for value in values:
        if not result or result[-1] != int(value):
            result.append(int(value))
    return result


def _render_fixed_window(
    source_lengths: np.ndarray,
    source_gaps: np.ndarray,
    source_directions: np.ndarray,
    source_mask: np.ndarray,
    events_by_flow: list[list[_Event]],
    stats: dict[str, np.ndarray],
    metadata: dict[str, Any],
    method_audits: list[dict[str, Any]],
    include_event_trace: bool = False,
) -> Application:
    """Render full network events into the classifier's leading-L window.

    The complete event list retains every source event.  Only the returned
    classifier arrays are truncated to their original width.  This distinction
    is essential: an event displaced from the first-L observation is not a
    packet deleted from the network flow.
    """

    # 渲染阶段不再作任何随机决策，只把已经生成的完整事件列表投影到 [N,L]。
    # 输出长度/gap 用 float64，以容纳精确的预算裁剪和 trigger 小数间隔。
    flow_count, observed_events = source_lengths.shape
    out_lengths = np.zeros_like(source_lengths, dtype=np.float64)
    out_gaps = np.zeros_like(source_gaps, dtype=np.float64)
    out_directions = np.zeros_like(source_directions)
    out_mask = np.zeros_like(source_mask)
    audit_rows: list[dict[str, Any]] = []

    for sample, events in enumerate(events_by_flow):
        store_timing(stats,sample,timing_metrics(
            source_gaps[sample,np.asarray(source_mask[sample],dtype=bool)],
            np.cumsum([event.gap for event in events]) - (float(source_gaps[sample,0]) if np.any(source_mask[sample]) else 0.0),
            [not event.is_dummy for event in events]),flow_count)
        # 先在“完整网络事件”层审计真实源事件。这里故意在截断前检查，否则
        # 被推到观察窗口外的源包会被误判成算法删除了网络包。
        source_count = int(np.sum(source_mask[sample]))
        full_source_ids = [
            int(event.source_index)
            for event in events
            if not event.is_dummy and event.source_index is not None
        ]
        if _ordered_unique(full_source_ids) != list(range(source_count)):
            raise AssertionError(
                "Method33 base stage deleted, duplicated, or reordered source events"
            )

        # 分类器威胁模型固定观察前 L 个事件；新增 dummy 可能占据其中位置，
        # 后面的真实包仍存在于 events，只是不出现在 returned arrays 中。
        retained = events[:observed_events]
        retained_source_ids = {
            int(event.source_index)
            for event in retained
            if not event.is_dummy and event.source_index is not None
        }
        for position, event in enumerate(retained):
            out_lengths[sample, position] = event.length
            out_gaps[sample, position] = event.gap
            out_directions[sample, position] = event.direction
            out_mask[sample, position] = 1
        if retained:
            out_gaps[sample,0] = 0.0

        # complete_* 与 retained_* 成对报告，便于审稿或调试时区分“确实发出”
        # 和“仍处于分类器窗口内”这两种语义。
        retained_roles = [event.role for event in retained]
        complete_roles = [event.role for event in events]
        retained_dummy_count = int(sum(event.is_dummy for event in retained))
        displaced = source_count - len(retained_source_ids)
        stats["retained_dummy_count"][sample] = retained_dummy_count
        stats["displaced_real_packets"][sample] = displaced

        # 事件级trace只服务于人工inspection，默认关闭，避免候选选择和正式评测
        # 为每个流长期保存数十个dict。启用时记录完整网络事件（不只前L项），
        # HTML即可同时解释dummy、被挤出观察窗口的真实包及其源PCAP位置。
        # dummy没有对应的PCAP frame/IP/epoch timestamp，因此source_index必须是
        # None；展示层会明确写“synthetic”，绝不能从相邻包信息中伪造来源字段。
        trace_payload: dict[str, Any] = {}
        if include_event_trace:
            source_positions = np.flatnonzero(source_mask[sample])
            complete_event_trace: list[dict[str, Any]] = []
            for event_position, event in enumerate(events):
                if not event.is_dummy and event.source_index is not None:
                    source_index = int(event.source_index)
                    source_position = int(source_positions[source_index])
                    source_length = float(source_lengths[sample, source_position])
                    source_gap = float(source_gaps[sample, source_position])
                    source_direction = int(source_directions[sample, source_position])
                else:
                    source_index = None
                    source_length = source_gap = source_direction = None
                complete_event_trace.append(
                    {
                        "event_position": int(event_position + 1),
                        "in_observation_window": bool(event_position < observed_events),
                        "role": event.role,
                        "source_index": source_index,
                        # 三个source_*值是把application与原始CSV逐事件精确联结的
                        # 数值指纹；它们不是扰动后值，dummy也绝不能拥有这些字段。
                        "source_length_bytes": source_length,
                        "source_gap_seconds": source_gap,
                        "source_direction": source_direction,
                        "length_bytes": float(event.length),
                        "gap_seconds": float(event.gap),
                        "direction": int(event.direction),
                    }
                )
            trace_payload["complete_event_trace"] = complete_event_trace

        audit_rows.append(
            {
                "flow_index": sample,
                "source_real_packets": source_count,
                "complete_emitted_events": len(events),
                "complete_emitted_real_events": len(full_source_ids),
                "complete_emitted_dummy_events": int(
                    sum(event.is_dummy for event in events)
                ),
                "retained_events": len(retained),
                "retained_dummy_events": retained_dummy_count,
                "source_packets_fully_retained": len(retained_source_ids),
                "source_packets_not_fully_in_window": displaced,
                "source_event_order_preserved": True,
                "complete_weakener_dummy_events": complete_roles.count(
                    "weakener_dummy"
                ),
                "complete_trigger_dummy_events": complete_roles.count(
                    "trigger_dummy"
                ),
                "retained_weakener_dummy_events": retained_roles.count(
                    "weakener_dummy"
                ),
                "retained_trigger_dummy_events": retained_roles.count(
                    "trigger_dummy"
                ),
                **trace_payload,
                **method_audits[sample],
            }
        )

    # 返回顺序是稳定兼容契约。调用者不应重新解释索引含义；更清楚的统计
    # 和保存操作由 method33_tools 封装。
    return (
        out_lengths,
        out_gaps,
        out_directions,
        out_mask,
        stats,
        metadata,
        audit_rows,
    )


def apply_method33_base(
    lengths: np.ndarray,
    gaps: np.ndarray,
    directions: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    config: Method33Config,
    seed: int,
    *,
    mapping: str | Sequence[int] = "old",
    include_weakener: bool = True,
    include_trigger: bool = True,
    include_event_trace: bool = False,
) -> Application:
    """Build Method33's deterministic pre-jitter traffic for one batch.

    This is the complete local base stage. Normal users should call
    :func:`apply_method33`; the tools layer uses the public
    :func:`apply_method33_base` name only for paired policy selection and
    deterministic controls.

    Parameters
    ----------
    mapping:
        ``"old"`` for attacker retraining, ``"next"`` for the configured
        deployment rotation, or an explicit permutation for a pre-registered
        derangement evaluation.
    include_weakener / include_trigger:
        Component switches intended for diagnostics.  Disabling the trigger
        also guarantees that changing ``labels`` cannot change emitted traffic.
    include_event_trace:
        When true, add the complete per-event provenance trace to each audit
        row for the inspection HTML.  It is false by default so model selection
        and formal evaluation retain their compact audit representation.
    """

    # -------------------------------------------------------------------------
    # 阶段 A：统一输入、选择映射、把离散分布准备成 NumPy 数组
    # -------------------------------------------------------------------------
    lengths, gaps, directions, mask, labels = _coerce_inputs(
        lengths, gaps, directions, mask, labels, config
    )
    active_mapping, mapping_name = _resolve_mapping(config, mapping)
    position_codes = resolve_trigger_positions(config)
    weakener = config.weakener
    length_targets = np.asarray(weakener.length_targets, dtype=np.float64)
    length_probabilities = np.asarray(
        weakener.length_probabilities, dtype=np.float64
    )
    gap_targets = np.asarray(weakener.gap_targets, dtype=np.float64)
    gap_probabilities = np.asarray(weakener.gap_probabilities, dtype=np.float64)

    # stats 按“流”累计成本；events_by_flow 暂存完整事件序列；method_audits
    # 记录确定性基础阶段特有的不变量，最终会与通用窗口审计合并。
    stats = _empty_stats(len(lengths))
    events_by_flow: list[list[_Event]] = []
    method_audits: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 阶段 B：逐流执行。流之间没有共享可变预算或共享随机流。
    # -------------------------------------------------------------------------
    for sample in range(len(lengths)):
        # SeedSequence isolates every flow.  Changing flow 0's length or number
        # of random draws cannot shift the random stream used by flow 1.
        # SeedSequence([批次 seed, 局部流序号]) 为每条流派生稳定子序列。
        # 某一流多做一次随机抽样不会让后续流的随机性整体错位。
        flow_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(sample)])
        )
        dummy_plan = (
            _sample_dummy_plan(
                flow_rng, weakener, config.weakener_dummy_count
            )
            if include_weakener
            else ()
        )

        # Reserve the planned weakener dummies before any real-packet action.
        # Without reservation, early padding/delay could spend all weakener
        # budget and silently turn the intended 1+2 design into 0+2.
        # 预算必须先预留再花费：Reduce30 的弱化层规定一定发出 1 个 dummy。
        # 若先把 231 B/0.06125 s 全用于真实包，就可能没有钱发计划 dummy，
        # 悄悄破坏“1 个弱化 dummy + 2 个 trigger dummy”的固定结构。
        planned_dummy_bytes = float(sum(item.length for item in dummy_plan))
        planned_dummy_delay = float(sum(item.gap for item in dummy_plan))
        weakener_bytes_remaining = (
            config.weakener_budget.bytes - planned_dummy_bytes
        )
        weakener_delay_remaining = (
            config.weakener_budget.delay_seconds - planned_dummy_delay
        )
        if weakener_bytes_remaining < -1e-9 or weakener_delay_remaining < -1e-12:
            raise AssertionError("sampled weakener dummy plan exceeds its partition")

        # 只有下面这一步读取 labels[sample]。关闭 trigger 后条件表达式走
        # else 分支，标签甚至不会被索引，这为弱化器标签无关性提供结构保证。
        trigger_code_index = (
            int(active_mapping[int(labels[sample])]) if include_trigger else -1
        )
        trigger_code = (
            config.trigger_codes[trigger_code_index] if include_trigger else ()
        )

        trigger_slots = position_codes[trigger_code_index] if include_trigger else ()
        realized_trigger_slots: list[int] = []
        emitted_trigger_packets = 0

        # outbound_ordinal 只统计真实上行包；dummy 自身不推进 slot，避免新增
        # dummy 递归改变后续 dummy 的锚点。其余变量用于成本与因果审计。
        events: list[_Event] = []
        outbound_ordinal = 0
        emitted_weakener_dummies = 0
        trigger_inserted = False
        fallback_used = False
        real_padding_bytes = 0.0
        real_added_delay = 0.0
        padding_attempts = delay_attempts = 0
        padding_actions = delay_actions = 0

        # ---------------------------------------------------------------------
        # 阶段 C：按源事件顺序在线处理单条流
        # ---------------------------------------------------------------------
        for source_index, position in enumerate(np.flatnonzero(mask[sample])):
            source_length = float(lengths[sample, position])
            source_gap = float(gaps[sample, position])
            direction = int(directions[sample, position])
            changed_length = source_length
            changed_gap = source_gap

            # 弱化器只处理设备上行真实包。padding 与 delay 各自抛一次硬币，
            # 两种动作可以同时发生，也可以都不发生；任何动作都不能缩短包或 gap。
            if include_weakener and direction == 1:
                want_padding = bool(
                    flow_rng.random() < weakener.padding_probability
                )
                want_delay = bool(flow_rng.random() < weakener.delay_probability)
                padding_attempts += int(want_padding)
                delay_attempts += int(want_delay)

                if want_padding and weakener_bytes_remaining > 1e-9:
                    # 抽到的是“希望达到的目标长度”，实际 padding 是以下四者
                    # 最小值：目标缺口、单包上限、剩余分账、MTU/帧长余量。
                    target = float(
                        flow_rng.choice(length_targets, p=length_probabilities)
                    )
                    padding = float(
                        min(
                            max(0.0, target - source_length),
                            weakener.max_padding_per_packet,
                            weakener_bytes_remaining,
                            config.max_frame_length - source_length,
                        )
                    )
                    if padding > 1e-9:
                        changed_length += padding
                        weakener_bytes_remaining -= padding
                        real_padding_bytes += padding
                        padding_actions += 1
                        stats["real_added_bytes"][sample] += padding
                        stats["outbound_real_added_bytes"][sample] += padding

                if want_delay and weakener_delay_remaining > 1e-12:
                    # delay 与 padding 对称：只向目标 gap 增加，并受单包上限
                    # 与剩余时延分账双重裁剪，所以不可能靠一次动作超预算。
                    target = float(flow_rng.choice(gap_targets, p=gap_probabilities))
                    added_delay = float(
                        min(
                            max(0.0, target - source_gap),
                            weakener.max_delay_per_packet,
                            weakener_delay_remaining,
                        )
                    )
                    if added_delay > 1e-12:
                        changed_gap += added_delay
                        weakener_delay_remaining -= added_delay
                        real_added_delay += added_delay
                        delay_actions += 1
                        stats["real_added_delay"][sample] += added_delay
                        stats["outbound_real_added_delay"][sample] += added_delay

            # Every source event is appended exactly once, before any dummy that
            # is anchored to it.  ``source_index`` supports an order audit later.
            # 无论是否发生弱化，每个源事件都在此处恰好 append 一次。dummy
            # 只能在真实事件之后加入，因此真实事件相对次序不会被打乱。
            events.append(
                _Event(
                    changed_length,
                    changed_gap,
                    direction,
                    False,
                    int(source_index),
                    "real",
                )
            )

            # Code positions refer to ORIGINAL real packets, independent of
            # weakener dummies, directions and per-flow zero-sum jitter.
            # At a shared anchor Trigger precedes a weakener dummy.
            while (include_trigger and emitted_trigger_packets < len(trigger_code)
                   and trigger_slots[emitted_trigger_packets] == source_index + 1):
                trigger_length, trigger_gap = trigger_code[emitted_trigger_packets]
                events.append(_Event(float(trigger_length), float(trigger_gap),
                                     1, True, None, "trigger_dummy"))
                _record_dummy_cost(stats, sample, float(trigger_length), float(trigger_gap))
                realized_trigger_slots.append(int(source_index) + 1)
                emitted_trigger_packets += 1

            if direction == 1:
                outbound_ordinal += 1

                # while 为将来 W_n>1 或规则扩展保留正确性；当前 Reduce30 W_n=1。
                # 计划 slot 已排序，故只需检查下一个未发出的计划项。
                while (
                    emitted_weakener_dummies < len(dummy_plan)
                    and dummy_plan[emitted_weakener_dummies].outbound_slot
                    == outbound_ordinal
                ):
                    item = dummy_plan[emitted_weakener_dummies]
                    events.append(
                        _Event(
                            item.length,
                            item.gap,
                            1,
                            True,
                            None,
                            "weakener_dummy",
                        )
                    )
                    _record_dummy_cost(stats, sample, item.length, item.gap)
                    emitted_weakener_dummies += 1

        # An unavailable anchor can only be detected at end-of-flow. Append
        # outstanding Trigger packets in code order; never inspect future values.
        trigger_fallback = bool(include_trigger and emitted_trigger_packets < len(trigger_code))
        while emitted_trigger_packets < len(trigger_code):
            trigger_length, trigger_gap = trigger_code[emitted_trigger_packets]
            events.append(_Event(float(trigger_length), float(trigger_gap),
                                 1, True, None, "trigger_dummy"))
            _record_dummy_cost(stats, sample, float(trigger_length), float(trigger_gap))
            realized_trigger_slots.append(int(np.count_nonzero(mask[sample])))
            emitted_trigger_packets += 1
        trigger_inserted = bool(include_trigger and emitted_trigger_packets == len(trigger_code))

        # Only after the flow ends can an online algorithm know that a planned
        # outbound slot never occurred.  Appending the remaining plan is causal;
        # the audit exposes this fallback so unusual short flows are visible.
        # 短流兜底只在流结束后才能判断。它不读取未来包值，因为此时未来已经
        # 不存在；audit 会明确标记 fallback，避免异常数据被静默掩盖。
        while emitted_weakener_dummies < len(dummy_plan):
            fallback_used = True
            item = dummy_plan[emitted_weakener_dummies]
            events.append(
                _Event(
                    item.length,
                    item.gap,
                    1,
                    True,
                    None,
                    "weakener_dummy",
                )
            )
            _record_dummy_cost(stats, sample, item.length, item.gap)
            emitted_weakener_dummies += 1

        # ---------------------------------------------------------------------
        # 阶段 D：逐流预算复算与确定性基础阶段专属审计
        # ---------------------------------------------------------------------
        # weakener_total_* 只审弱化分账；total_* 同时包含 trigger。两层都检查，
        # 可防止总预算没超但某一子模块挪用了另一模块预算的实现错误。
        total_bytes = float(
            stats["real_added_bytes"][sample] + stats["dummy_bytes"][sample]
        )
        total_delay = float(
            stats["real_added_delay"][sample] + stats["dummy_delay"][sample]
        )
        total_dummies = int(round(stats["dummy_count"][sample]))
        weakener_total_bytes = real_padding_bytes + planned_dummy_bytes
        weakener_total_delay = real_added_delay + planned_dummy_delay
        budget_passed = bool(
            weakener_total_bytes <= config.weakener_budget.bytes + 1e-7
            and weakener_total_delay
            <= config.weakener_budget.delay_seconds + 1e-10
            and total_bytes <= config.hard_budget.bytes + 1e-7
            and total_delay <= config.hard_budget.delay_seconds + 1e-10
            and total_dummies <= config.hard_budget.dummy_packets
        )
        if not budget_passed:
            raise AssertionError("Method33 base stage exceeded a per-flow hard budget")

        # audit 不参与算法决策，只把关键事实落盘：标签使用边界、所选码字、
        # 随机动作次数、两层成本、dummy 数以及预算是否通过。
        method_audits.append(
            {
                "source_label": int(labels[sample]),
                "mapping": mapping_name,
                "trigger_code_index": trigger_code_index,
                "trigger_position_anchor": "original_real_packet_both_directions",
                "trigger_planned_real_slots": list(trigger_slots),
                "trigger_realized_real_slots": realized_trigger_slots,
                "trigger_short_flow_fallback_used": trigger_fallback,
                "runtime_label_used_only_for_trigger": bool(include_trigger),
                "random_weakener_used_label": False,
                "causal_no_future_packet_values": True,
                "weakener_dummy_planned_slots": [
                    int(item.outbound_slot) for item in dummy_plan
                ],
                "weakener_dummy_emitted": emitted_weakener_dummies,
                "weakener_short_flow_fallback_used": fallback_used,
                "trigger_inserted": trigger_inserted,
                "trigger_packet_count": (
                    config.trigger_packet_count if include_trigger else 0
                ),
                "padding_attempts": padding_attempts,
                "delay_attempts": delay_attempts,
                "padding_actions": padding_actions,
                "delay_actions": delay_actions,
                "weakener_real_padding_bytes": real_padding_bytes,
                "weakener_real_added_delay": real_added_delay,
                "weakener_dummy_bytes": planned_dummy_bytes,
                "weakener_dummy_delay": planned_dummy_delay,
                "weakener_total_added_bytes": weakener_total_bytes,
                "weakener_total_added_delay": weakener_total_delay,
                "trigger_added_bytes": (
                    config.trigger_budget.bytes if include_trigger else 0.0
                ),
                "trigger_added_delay": (
                    config.trigger_budget.delay_seconds if include_trigger else 0.0
                ),
                "total_added_bytes": total_bytes,
                "total_added_delay": total_delay,
                "dummy_packets": total_dummies,
                "all_per_flow_hard_budgets_satisfied": budget_passed,
            }
        )
        events_by_flow.append(events)

    # -------------------------------------------------------------------------
    # 阶段 E：批次级元数据、固定窗口渲染与窗口内 dummy 断言
    # -------------------------------------------------------------------------
    # metadata 描述“这一批怎样运行”，而 method_audits 描述“每条流发生什么”。
    metadata = {
        "method": config.method_key,
        "version": int(config.version),
        "weakener_candidate_id": config.weakener_candidate_id,
        "trigger_codebook_id": config.trigger_codebook_id,
        "mapping": mapping_name,
        "active_mapping": list(active_mapping),
        "trigger_position_anchor": "original_real_packet_both_directions",
        "trigger_positions_by_code": [list(pair) for pair in position_codes],
        "trigger_short_flow_policy": "append remaining Trigger packets at end of flow",
        "trigger_positions_changed_by_jitter": False,
        "include_weakener": bool(include_weakener),
        "include_trigger": bool(include_trigger),
        "random_weakener_uses_label": False,
        "trigger_layer_uses_trusted_device_label": bool(include_trigger),
        "weakener_dummy_count": int(config.weakener_dummy_count),
        "trigger_packet_count": int(config.trigger_packet_count),
        "hard_budget": config.hard_budget.as_list(),
        "budget_partition": {
            "weakener": config.weakener_budget.as_list(),
            "trigger": config.trigger_budget.as_list(),
        },
        "seed": int(seed),
        "fixed_leading_observation_window": int(lengths.shape[1]),
        "first_packet_wait_observable": False,
        "observation_origin": "first_emitted_packet",
        "dummy_visibility_policy": "full trace retained; tail allowed unless strict diagnostic requested",
    }
    if include_event_trace:
        # 默认运行的metadata保持旧schema；只有inspection显式开启trace时才增加
        # 这一标志和逐事件audit，最大限度兼容既有结果比较/序列化脚本。
        metadata["event_trace_in_audit"] = True
    result = _render_fixed_window(
        lengths,
        gaps,
        directions,
        mask,
        events_by_flow,
        stats,
        metadata,
        method_audits,
        include_event_trace,
    )

    # 当前论文威胁模型观察前 L 个事件，因此要求配置中的所有 dummy 都被观察到。
    # 若未来研究完整事件序列，可在配置中关闭此约束，但必须在论文中说明。
    if config.require_all_dummies_in_window:
        expected_weakener = config.weakener_dummy_count if include_weakener else 0
        expected_trigger = config.trigger_packet_count if include_trigger else 0
        for row in result[6]:
            if row["retained_weakener_dummy_events"] != expected_weakener:
                raise AssertionError("a weakener dummy fell outside the observation window")
            if row["retained_trigger_dummy_events"] != expected_trigger:
                raise AssertionError("a trigger dummy fell outside the observation window")
    return result



# 确定性基础阶段沿用历史method_key以保证旧结果可追溯，但Method33最终metadata
# 必须明确改写方法身份。码本仍来自传入的Method33Config，因此同时保留
# base_method和base_trigger_codebook_id，方便核验基础策略与波动策略的关系。
METHOD33_KEY = "method33_position_coded_zero_sum_jittered_rotating_trigger"
METHOD33_VERSION = 2

# SeedSequence的第三、第四个整数只是Method33专用的域分隔标记。弱化器仍使用
# 基础阶段使用SeedSequence([seed, flow_index])；Trigger jitter使用下面的额外域，
# 因而新增两个随机数绝不会让弱化器后续抽样错位。
_METHOD33_JITTER_SEED_DOMAIN = (32, 5_000_000)


@dataclass(frozen=True)
class TriggerJitterConfig:
    """Immutable runtime policy for Method33's Trigger-only randomization.

    ``length_relative_bound`` and ``gap_relative_bound`` are per-coordinate
    upper bounds.  The defaults mean that neither packet length nor packet gap
    may move by more than five percent of its own deterministic base value.

    ``minimum_gap_seconds`` keeps each realized Trigger gap away from an
    infeasible/undesired near-zero value.  It defaults to the 0.004-second
    minimum used to generate the current Reduce30 Trigger codebook.  This value
    lives here rather than in Method33Config because the user requested that
    existing code and configuration remain untouched.
    """

    length_relative_bound: float = 0.05
    gap_relative_bound: float = 0.05
    minimum_gap_seconds: float = 0.004


DEFAULT_TRIGGER_JITTER = TriggerJitterConfig()


@dataclass(frozen=True)
class JitteredTriggerCode:
    """One realized two-packet Trigger plus complete randomization evidence."""

    # packets保持基础码字的嵌套形状：((length1,gap1),(length2,gap2))。
    packets: tuple[tuple[float, float], tuple[float, float]]
    # delta定义为“从第二包转给第一包”的量；负数表示从第一包转给第二包。
    length_transfer_bytes: int
    gap_transfer_seconds: float
    # 两个区间记录本码字在5%和物理边界共同限制下实际可抽到的范围。
    # 若原码字有一个60B包，则长度区间可能不是关于0对称，这是正确的安全裁剪。
    feasible_length_transfer_bytes: tuple[int, int]
    feasible_gap_transfer_seconds: tuple[float, float]
    # uniform_quantile不依赖label、mapping或码字。old/next配对实验对同一flow
    # 使用相同quantile，再映射到各自码字的可行区间，减少无关随机方差。
    length_uniform_quantile: float
    gap_uniform_quantile: float


def _validate_jitter_config(jitter: TriggerJitterConfig) -> None:
    """Reject a jitter policy that cannot represent a bounded perturbation."""

    values = (
        jitter.length_relative_bound,
        jitter.gap_relative_bound,
        jitter.minimum_gap_seconds,
    )
    if any(not np.isfinite(value) for value in values):
        raise ValueError("Method33 jitter parameters must be finite")
    # 小于1保证坐标的相对下界仍为正。当前默认只有0.05；允许显式0便于做
    # “同一Method33代码路径、关闭波动”的控制实验。
    if not 0.0 <= jitter.length_relative_bound < 1.0:
        raise ValueError("length_relative_bound must be in [0, 1)")
    if not 0.0 <= jitter.gap_relative_bound < 1.0:
        raise ValueError("gap_relative_bound must be in [0, 1)")
    if jitter.minimum_gap_seconds < 0.0:
        raise ValueError("minimum_gap_seconds cannot be negative")


def _validate_unit_quantile(value: float, name: str) -> float:
    """Return a finite random quantile in the half-open interval [0, 1)."""

    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1)")
    return result


def _integer_transfer_interval(
    first: float,
    second: float,
    *,
    relative_bound: float,
    minimum_value: float,
    maximum_value: float,
) -> tuple[int, int]:
    """Find every legal integer transfer for a pair of packet lengths.

    A transfer ``delta`` produces ``(first + delta, second - delta)``.  The
    interval simultaneously enforces:

    * at most ``relative_bound`` change for *both* coordinates;
    * the minimum Ethernet-frame length for both packets;
    * the configured maximum frame length for both packets; and
    * integer-byte output.
    """

    # PCAP/frame长度以byte计数，真实部署不能产生76.4B的frame。当前码本本来
    # 都是整数；这里明确拒绝非整数基础值，避免round悄悄改变码字总成本。
    first_integer = int(round(float(first)))
    second_integer = int(round(float(second)))
    if not np.isclose(first, first_integer, rtol=0.0, atol=1e-9) or not np.isclose(
        second, second_integer, rtol=0.0, atol=1e-9
    ):
        raise ValueError("Method33 requires integer-byte Trigger base lengths")

    # 相对边界：first变化+delta，second变化-delta。要让两者都不超过5%，
    # |delta|必须不超过两者各自允许绝对变化量中的较小者。
    relative_limit = min(
        relative_bound * first_integer,
        relative_bound * second_integer,
    )
    lower = max(
        -relative_limit,
        minimum_value - first_integer,   # first + delta >= minimum
        second_integer - maximum_value,  # second - delta <= maximum
    )
    upper = min(
        relative_limit,
        maximum_value - first_integer,   # first + delta <= maximum
        second_integer - minimum_value,  # second - delta >= minimum
    )

    # delta必须是整数。微小1e-12只吸收浮点表达误差，不扩大实际可行集合。
    lower_integer = int(np.ceil(lower - 1e-12))
    upper_integer = int(np.floor(upper + 1e-12))
    if lower_integer > upper_integer:
        raise ValueError("no feasible integer zero-sum Trigger length jitter exists")
    return lower_integer, upper_integer


def _continuous_transfer_interval(
    first: float,
    second: float,
    *,
    relative_bound: float,
    minimum_value: float,
) -> tuple[float, float]:
    """Find the legal continuous zero-sum transfer interval for two gaps."""

    first = float(first)
    second = float(second)
    if not np.isfinite(first) or not np.isfinite(second):
        raise ValueError("Method33 Trigger base gaps must be finite")
    if first < minimum_value or second < minimum_value:
        raise ValueError(
            "a Trigger base gap is below Method33 minimum_gap_seconds"
        )

    relative_limit = min(relative_bound * first, relative_bound * second)
    lower = max(
        -relative_limit,
        minimum_value - first,  # first + delta >= minimum
    )
    upper = min(
        relative_limit,
        second - minimum_value,  # second - delta >= minimum
    )
    if lower > upper + 1e-15:
        raise ValueError("no feasible zero-sum Trigger gap jitter exists")
    # 若只差二进制浮点尾数，把区间收缩成唯一安全点，不制造负宽度。
    if lower > upper:
        midpoint = float((lower + upper) / 2.0)
        return midpoint, midpoint
    return float(lower), float(upper)


def jitter_trigger_code(
    base_code: Sequence[Sequence[float]],
    method_config: Method33Config,
    jitter_config: TriggerJitterConfig = DEFAULT_TRIGGER_JITTER,
    *,
    length_uniform_quantile: float,
    gap_uniform_quantile: float,
) -> JitteredTriggerCode:
    """Realize one exactly budget-preserving two-packet Trigger code.

    The caller supplies the two unit-uniform quantiles explicitly.  Separating
    random-number generation from constrained realization makes the function
    deterministic, independently testable, and usable for paired old/next
    comparisons with common random numbers.
    """

    _validate_jitter_config(jitter_config)
    length_q = _validate_unit_quantile(
        length_uniform_quantile, "length_uniform_quantile"
    )
    gap_q = _validate_unit_quantile(gap_uniform_quantile, "gap_uniform_quantile")

    # Method33当前定义就是“两包Trigger内部零和波动”。显式拒绝其他包数比
    # 静默只修改前两包更安全；类别数K仍完全动态，不受这一条件限制。
    if method_config.trigger_packet_count != 2 or len(base_code) != 2:
        raise ValueError("Method33 currently requires exactly two Trigger packets")
    try:
        first_length, first_gap = map(float, base_code[0])
        second_length, second_gap = map(float, base_code[1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "each Method33 Trigger packet must contain [length, gap]"
        ) from error

    base_total_bytes = first_length + second_length
    base_total_gap = first_gap + second_gap
    if not np.isclose(
        base_total_bytes,
        method_config.trigger_budget.bytes,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("base Trigger code does not equal its byte partition")
    if not np.isclose(
        base_total_gap,
        method_config.trigger_budget.delay_seconds,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("base Trigger code does not equal its delay partition")

    length_lower, length_upper = _integer_transfer_interval(
        first_length,
        second_length,
        relative_bound=jitter_config.length_relative_bound,
        minimum_value=method_config.min_dummy_frame_length,
        maximum_value=method_config.max_frame_length,
    )
    # 使用同一个[0,1) quantile映射到闭整数区间。与rng.integers相比，这种
    # 写法能把实际使用的公共随机数记录进audit，方便old/next配对核验。
    number_of_length_choices = length_upper - length_lower + 1
    length_choice_offset = min(
        int(np.floor(length_q * number_of_length_choices)),
        number_of_length_choices - 1,
    )
    delta_bytes = int(length_lower + length_choice_offset)

    gap_lower, gap_upper = _continuous_transfer_interval(
        first_gap,
        second_gap,
        relative_bound=jitter_config.gap_relative_bound,
        minimum_value=jitter_config.minimum_gap_seconds,
    )
    delta_gap = float(gap_lower + gap_q * (gap_upper - gap_lower))

    realized_first_length = float(int(round(first_length)) + delta_bytes)
    # 第二包用预算减第一包，而不是独立计算second-delta；这样即使未来预算
    # 来自可精确表示范围之外的JSON小数，也把所有byte求和残差固定收在此处。
    realized_second_length = float(
        method_config.trigger_budget.bytes - realized_first_length
    )
    realized_first_gap = float(first_gap + delta_gap)
    realized_second_gap = float(
        method_config.trigger_budget.delay_seconds - realized_first_gap
    )
    # 二进制float执行“总量-第一包”以后，再与第一包相加时可能留下约1e-18秒
    # 的表示尾差。把可观测残差一次性回填给第二包；这不是放宽预算，而是让
    # 浮点实现尽可能贴合零和公式。下面仍会用rtol=0的严格绝对容差复核。
    realized_second_gap += float(
        method_config.trigger_budget.delay_seconds
        - (realized_first_gap + realized_second_gap)
    )
    # “严格预算”首先意味着任何flow都绝不能因为最后1 ULP而位于上限之上。
    # 如果普通浮点加法仍向上舍入，就把第二包向负无穷取下一个可表示数；最多
    # 只减少约1e-18秒，远小于配置审计容差，却把不超预算从容差判断加强为
    # 直接的 <= 判断。循环通常0次，极少数输入1次即可。
    while (
        realized_first_gap + realized_second_gap
        > method_config.trigger_budget.delay_seconds
    ):
        realized_second_gap = float(
            np.nextafter(realized_second_gap, -np.inf)
        )

    packets = (
        (realized_first_length, realized_first_gap),
        (realized_second_length, realized_second_gap),
    )

    # ---------------------------------------------------------------------
    # 本地强断言：绝不能依赖“公式看起来应该守预算”
    # ---------------------------------------------------------------------
    realized_lengths = np.asarray([packet[0] for packet in packets])
    realized_gaps = np.asarray([packet[1] for packet in packets])
    base_lengths = np.asarray([first_length, second_length])
    base_gaps = np.asarray([first_gap, second_gap])
    if not np.isclose(
        float(np.sum(realized_lengths)),
        method_config.trigger_budget.bytes,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Method33 failed to preserve Trigger byte cost")
    if not np.isclose(
        float(np.sum(realized_gaps)),
        method_config.trigger_budget.delay_seconds,
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("Method33 failed to preserve Trigger delay cost")
    if np.any(realized_lengths < method_config.min_dummy_frame_length - 1e-9):
        raise AssertionError("Method33 produced an undersized Trigger frame")
    if np.any(realized_lengths > method_config.max_frame_length + 1e-9):
        raise AssertionError("Method33 produced an oversized Trigger frame")
    if np.any(realized_gaps < jitter_config.minimum_gap_seconds - 1e-15):
        raise AssertionError("Method33 produced a Trigger gap below its minimum")
    if np.any(
        np.abs(realized_lengths - base_lengths)
        > jitter_config.length_relative_bound * base_lengths + 1e-9
    ):
        raise AssertionError("Method33 length jitter exceeded its relative bound")
    if np.any(
        np.abs(realized_gaps - base_gaps)
        > jitter_config.gap_relative_bound * base_gaps + 1e-15
    ):
        raise AssertionError("Method33 gap jitter exceeded its relative bound")

    return JitteredTriggerCode(
        packets=packets,
        length_transfer_bytes=delta_bytes,
        gap_transfer_seconds=delta_gap,
        feasible_length_transfer_bytes=(length_lower, length_upper),
        feasible_gap_transfer_seconds=(gap_lower, gap_upper),
        length_uniform_quantile=length_q,
        gap_uniform_quantile=gap_q,
    )


def _flow_jitter_quantiles(seed: int, flow_index: int) -> tuple[float, float]:
    """Draw two label-independent Method33 random numbers for one flow."""

    if int(seed) < 0:
        raise ValueError("Method33 seed must be non-negative")
    # seed只使用批次seed、flow行号和固定域分隔，不使用label或mapping。
    # 因此改变old→next不会偷偷改变公共随机数；改变某条流的码字也不会让
    # 后续flow随机流错位。
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(flow_index),
                *_METHOD33_JITTER_SEED_DOMAIN,
            ]
        )
    )
    return float(rng.random()), float(rng.random())


def _method33_metadata(
    base_metadata: dict[str, Any],
    base_config: Method33Config,
    jitter_config: TriggerJitterConfig,
    *,
    include_trigger: bool,
    include_event_trace: bool,
) -> dict[str, Any]:
    """Create an explicit Method33 identity from local base-stage metadata."""

    metadata = dict(base_metadata)
    metadata.update(
        {
            "method": METHOD33_KEY,
            "version": METHOD33_VERSION,
            "base_method": base_config.method_key,
            "base_method_version": int(base_config.version),
            "base_trigger_codebook_id": base_config.trigger_codebook_id,
            "trigger_internal_jitter_enabled": bool(include_trigger),
            "trigger_internal_jitter_zero_sum": True,
            "trigger_total_budget_preserved": True,
            "trigger_jitter_length_relative_bound": float(
                jitter_config.length_relative_bound
            ),
            "trigger_jitter_gap_relative_bound": float(
                jitter_config.gap_relative_bound
            ),
            "trigger_jitter_minimum_gap_seconds": float(
                jitter_config.minimum_gap_seconds
            ),
            "trigger_jitter_seed_domain": list(_METHOD33_JITTER_SEED_DOMAIN),
        }
    )
    if not include_event_trace:
        # apply_method33可能为了找到完整Trigger事件而内部开启基础阶段trace；
        # 调用者没要求trace时，最终metadata不能谎称返回结果仍包含它。
        metadata.pop("event_trace_in_audit", None)
    return metadata


def apply_method33(
    lengths: np.ndarray,
    gaps: np.ndarray,
    directions: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    config: Method33Config,
    seed: int,
    *,
    mapping: str | Sequence[int] = "old",
    include_weakener: bool = True,
    include_trigger: bool = True,
    include_event_trace: bool = False,
    jitter_config: TriggerJitterConfig = DEFAULT_TRIGGER_JITTER,
) -> Application:
    """Apply Method33 to a batch of fixed-window flows.

    All arguments except ``jitter_config`` intentionally match
    local :func:`apply_method33_base`. This keeps the deterministic base and
    jittered Method33 paths paired without importing any external algorithm
    core, while preserving the same input/output contract.

    ``mapping`` may be ``"old"``, ``"next"``, or an explicit registered
    permutation exactly as in the deterministic base stage. The per-flow
    jitter quantiles do not depend on mapping, so old/next runs with the same
    batch seed are paired.
    """

    _validate_jitter_config(jitter_config)
    if int(seed) < 0:
        raise ValueError("Method33 seed must be non-negative")
    if include_trigger and config.trigger_packet_count != 2:
        raise ValueError("Method33 currently requires exactly two Trigger packets")

    # Method33需要知道完整网络序列中哪些事件是Trigger dummy。基础阶段的公共
    # 七元组只有在include_event_trace=True时保留role/event_position，因此
    # Trigger开启时内部强制请求trace。若调用者没请求，末尾再安全删除。
    internal_trace_required = bool(include_trigger)
    base_application = apply_method33_base(
        lengths,
        gaps,
        directions,
        mask,
        labels,
        config,
        seed,
        mapping=mapping,
        include_weakener=include_weakener,
        include_trigger=include_trigger,
        include_event_trace=bool(include_event_trace or internal_trace_required),
    )
    (
        protected_lengths,
        protected_gaps,
        protected_directions,
        protected_mask,
        stats,
        base_metadata,
        base_audit_rows,
    ) = base_application

    # protected_*是本地基础阶段新分配的输出矩阵，不是用户输入数组，因此就地只改
    # Trigger位置不会污染原始dataset。direction、mask、stats完全不需要改变。
    audit_rows: list[dict[str, Any]] = []
    for flow_index, base_row in enumerate(base_audit_rows):
        row = dict(base_row)
        if include_trigger:
            trace = row.get("complete_event_trace")
            if not isinstance(trace, list):
                raise AssertionError("Method33 internal base-stage trace is missing")
            trigger_events = [
                event for event in trace if event.get("role") == "trigger_dummy"
            ]
            if len(trigger_events) != 2:
                raise AssertionError(
                    "Method33 expected exactly two Trigger events in each flow"
                )

            trigger_code_index = int(row["trigger_code_index"])
            if not 0 <= trigger_code_index < config.class_count:
                raise AssertionError("Method33 received an invalid Trigger code index")
            base_code = config.trigger_codes[trigger_code_index]

            # 两个quantile由flow身份决定而非label/mapping决定；实现old/next公共
            # 随机数。码字边界不同，所以最终delta可不同，但随机分位点完全配对。
            length_q, gap_q = _flow_jitter_quantiles(seed, flow_index)
            realized = jitter_trigger_code(
                base_code,
                config,
                jitter_config,
                length_uniform_quantile=length_q,
                gap_uniform_quantile=gap_q,
            )

            for packet_index, (event, packet) in enumerate(
                zip(trigger_events, realized.packets, strict=True)
            ):
                base_length, base_gap = map(float, base_code[packet_index])
                # 这两项证明trace确实来自同一个base config，而不是另一个同形
                # application。失败时停止，绝不把错误事件静默标成Method33。
                if not np.isclose(
                    float(event["length_bytes"]),
                    base_length,
                    rtol=0.0,
                    atol=1e-12,
                ) or not np.isclose(
                    float(event["gap_seconds"]),
                    base_gap,
                    rtol=0.0,
                    atol=1e-15,
                ):
                    raise AssertionError(
                        "Method33 Trigger trace disagrees with the base codebook"
                    )

                realized_length, realized_gap = packet
                event["length_bytes"] = float(realized_length)
                event["gap_seconds"] = float(realized_gap)

                # event_position是完整事件流中的1-based位置。只有它还在前L观察
                # 窗口内时，七元组前两张[N,L]矩阵才有对应单元需要同步更新。
                if bool(event["in_observation_window"]):
                    array_position = int(event["event_position"]) - 1
                    if not 0 <= array_position < protected_lengths.shape[1]:
                        raise AssertionError(
                            "Method33 trace window position is outside output arrays"
                        )
                    protected_lengths[flow_index, array_position] = realized_length
                    protected_gaps[flow_index, array_position] = realized_gap

            # 流级audit同时保存base与realized，后续HTML或论文复核无需凭seed猜测。
            base_code_json = [
                [float(packet[0]), float(packet[1])] for packet in base_code
            ]
            realized_code_json = [
                [float(packet[0]), float(packet[1])]
                for packet in realized.packets
            ]
            row.update(
                {
                    "method33_trigger_jitter_enabled": True,
                    "method33_trigger_base_code": base_code_json,
                    "method33_trigger_realized_code": realized_code_json,
                    "method33_length_transfer_bytes": int(
                        realized.length_transfer_bytes
                    ),
                    "method33_gap_transfer_seconds": float(
                        realized.gap_transfer_seconds
                    ),
                    "method33_feasible_length_transfer_bytes": list(
                        realized.feasible_length_transfer_bytes
                    ),
                    "method33_feasible_gap_transfer_seconds": list(
                        realized.feasible_gap_transfer_seconds
                    ),
                    "method33_length_uniform_quantile": float(
                        realized.length_uniform_quantile
                    ),
                    "method33_gap_uniform_quantile": float(
                        realized.gap_uniform_quantile
                    ),
                    "method33_trigger_total_bytes": float(
                        sum(packet[0] for packet in realized.packets)
                    ),
                    "method33_trigger_total_delay": float(
                        sum(packet[1] for packet in realized.packets)
                    ),
                    "method33_trigger_budget_preserved": True,
                }
            )
        else:
            # 组件消融关闭Trigger时，Method33不读取label来产生任何新动作，也
            # 不抽取jitter随机数。此时输出流量应与确定性weakener-only一致。
            row.update(
                {
                    "method33_trigger_jitter_enabled": False,
                    "method33_trigger_base_code": [],
                    "method33_trigger_realized_code": [],
                    "method33_length_transfer_bytes": 0,
                    "method33_gap_transfer_seconds": 0.0,
                    "method33_trigger_total_bytes": 0.0,
                    "method33_trigger_total_delay": 0.0,
                    "method33_trigger_budget_preserved": True,
                }
            )

        if include_trigger:
            store_timing(stats,flow_index,timing_metrics(
                np.asarray(gaps)[flow_index,np.asarray(mask[flow_index],dtype=bool)],
                np.cumsum([e['gap_seconds'] for e in trace]) - (float(np.asarray(gaps)[flow_index,0]) if np.any(mask[flow_index]) else 0.0),
                [e['role']=='real' for e in trace]),len(base_audit_rows))
        if not include_event_trace:
            # 保持基础阶段默认轻量audit语义。内部trace只是一座安全定位Trigger
            # 事件的桥，不能因为实现细节而永久增加候选选择/正式评测的结果体积。
            row.pop("complete_event_trace", None)
        audit_rows.append(row)

    metadata = _method33_metadata(
        base_metadata,
        config,
        jitter_config,
        include_trigger=include_trigger,
        include_event_trace=include_event_trace,
    )

    # 最终批次级复核。Method33没有修改stats，因为零和波动使每条flow的Trigger
    # 总成本严格不变；这里仍逐流核对账本中的trigger分账和硬预算标志。
    for flow_index, row in enumerate(audit_rows):
        expected_trigger_bytes = config.trigger_budget.bytes if include_trigger else 0.0
        expected_trigger_delay = (
            config.trigger_budget.delay_seconds if include_trigger else 0.0
        )
        if not np.isclose(
            row["method33_trigger_total_bytes"],
            expected_trigger_bytes,
            rtol=0.0,
            atol=1e-12,
        ) or not np.isclose(
            row["method33_trigger_total_delay"],
            expected_trigger_delay,
            rtol=0.0,
            atol=1e-15,
        ):
            raise AssertionError("Method33 changed the Trigger partition total")
        if not bool(row["all_per_flow_hard_budgets_satisfied"]):
            raise AssertionError(f"Method33 budget failed for flow {flow_index}")

    return (
        protected_lengths,
        protected_gaps,
        protected_directions,
        protected_mask,
        stats,
        metadata,
        audit_rows,
    )

__all__ = [
    "Application",
    "Budget",
    "DEFAULT_TRIGGER_JITTER",
    "JitteredTriggerCode",
    "METHOD33_KEY",
    "METHOD33_VERSION",
    "Method33Config",
    "STAT_KEYS",
    "TriggerJitterConfig",
    "WeakenerConfig",
    "apply_method33",
    "apply_method33_base",
    "jitter_trigger_code",
]
