"""Standalone Method33 configuration and CLI; legacy JSON/TXT files supported.

Only the local Method33 core and the shared dataset module are imported.
Legacy selection metadata is parsed/validated for compatibility; this module
provides inspect/apply, not a new policy-selection procedure.
"""

# =============================================================================
# 中文阅读导图：本文件是“配置层 + 命令行层”，不是算法本体
# =============================================================================
# 与 method33.py 的分工：
#   method33.py        只接受已经构造好的不可变 Method33Config，执行纯算法；
#   method33_tools.py  负责把磁盘上的 JSON 变成该对象，并在进入算法前后做
#                      数据集对齐、硬预算和七元组 schema 的校验。
# 因此建议先读 method33.apply_method33，再回到本文件理解每个配置字段的含义。
#
# 典型调用链（也是 CLI 的执行顺序）：
#   load_config(path)           读 JSON（允许 UTF-8 BOM）→ ProjectConfig
#     ├─ _budget/_weakener/_evaluation/_selection/_trigger_codes/_trigger_jitter
#     │    把嵌套字典逐字段转成 frozen dataclass；错误信息带字段路径
#     └─ ProjectConfig 保存 source_path，供 manifest 追溯“这份策略从哪来”
#   validate_config(project)    跨字段一致性：预算分账、概率归一、码字等成本、
#                               next 映射无固定点、位置表与类别数匹配、
#                               数据集设备顺序/特征上界一致
#   apply_configured_method()   数据↔配置对齐 → apply_method33 → 立即审计预算
#   validate_application()      七元组 schema + 逐流硬预算复核（失败直接抛异常）
#   main()                      两个子命令：inspect（只读配置）与 apply（真正变换）
#
# 复杂语法提示：
#   * _number_tuple/_integer_tuple 先把 JSON list 转 tuple：加载完成后不可再
#     append，且 float→int 会回比较，避免 1.9 被静默截断成 1；
#   * _integer 显式拒绝 bool：Python 里 bool 是 int 子类，配置写 true 通常
#     意味着字段类型写错了；
#   * main() 用 argparse subparsers 实现 inspect/apply，参数合法性在解析阶段
#     处理，算法函数无需再判断“用户到底想干什么”。
from __future__ import annotations
import argparse
import sys
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
if __package__:
    from data_pipeline import pcap as dataset_pcap
    from .method33 import (Application, Budget, Method33Config, WeakenerConfig,
        TriggerJitterConfig, STAT_KEYS, METHOD33_KEY, METHOD33_VERSION,
        apply_method33, apply_method33_base, jitter_trigger_code,
        default_trigger_positions, resolve_trigger_positions)
else:
    from data_pipeline import pcap as dataset_pcap
    from method33 import (Application, Budget, Method33Config, WeakenerConfig,
        TriggerJitterConfig, STAT_KEYS, METHOD33_KEY, METHOD33_VERSION,
        apply_method33, apply_method33_base, jitter_trigger_code,
        default_trigger_positions, resolve_trigger_positions)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent / "configs/methods/fade_cic_active.json"


def _position_codes(raw):
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ValueError("trigger_positions must be a nonempty array of pairs")
    result = []
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each trigger_positions entry must be a pair")
        result.append(tuple(_integer(x, "trigger_positions") for x in pair))
    return tuple(result)


@dataclass(frozen=True)
class EvaluationConfig:
    """Non-algorithm settings used by splits, quick checks, and CLI defaults."""

    # 这些参数不改变 Method33 流量变换本身，只控制数据划分、快速评测以及
    # CLI 默认随机种子，因此与 Method33Config 分开保存。
    capture_train_ratio: float
    split_seed: int
    default_apply_seed: int
    retraining_mask_seed: int
    model_seed: int
    retraining_ratios: tuple[float, ...]


@dataclass(frozen=True)
class WeakenerCandidate:
    """One named, fully specified weakener considered during offline selection.

    候选ID只负责追溯身份；真正决定流量动作的是 ``weakener`` 中的13个字段。
    两者绑在一个冻结对象中，可防止候选名称和候选参数在循环中错位。
    """

    candidate_id: str
    weakener: WeakenerConfig


@dataclass(frozen=True)
class SelectionConfig:
    """All offline-only hyperparameters for the two-stage policy search.

    这些参数不属于在线防御策略，因此不能放进 ``method33.py``。它们控制
    capture隔离、代理模型重复、候选码本生成和选择目标；最终部署只消费搜索
    产生的 ``Method33Config``，不需要scikit-learn或任何候选列表。
    """

    # 数据协议：每类固定留出的完整capture数；以及攻击者真正见过防御样本时
    # 才参与候选选择的重训练比例。这里故意不含0%，它与完整效果曲线分开。
    validation_captures_per_class: int
    retraining_ratios: tuple[float, ...]
    # 随机弱化器训练重复次数、每个repeat上的独立validation扰动次数。
    repeats: int
    validation_trials: int
    # Trigger代理模式的old-mapping可学习性门槛；mode=geometry时仍被保存但不参与排序。
    same_mapping_floor: float
    trigger_selection_mode: str
    # 每次生成多少套码本、每套码本先生成多大的等成本候选池，以及每包最小gap。
    trigger_codebook_candidates: int
    trigger_pool_size: int
    trigger_min_gap_seconds: float
    trigger_codebook_prefix: str
    # 六类随机过程分别使用显式seed。拆开保存可防止新增一次随机调用后，后面所有
    # 阶段的结果都跟着漂移；候选ID绝不进入这些seed。
    inner_split_seed: int
    weakener_train_seed: int
    weakener_validation_seed: int
    trigger_train_seed: int
    trigger_validation_seed: int
    codebook_seed: int
    # 每个候选都保存13个完整WeakenerConfig字段，不依赖Python中的隐藏base/override。
    weakener_candidates: tuple[WeakenerCandidate, ...]


@dataclass(frozen=True)
class ProjectConfig:
    """Typed view of the complete JSON-in-TXT configuration."""

    # source_path保存实际读取的配置绝对路径，便于日志与产物追溯；子配置
    # 分别描述数据、Method33基础策略、Trigger内部波动、快速效果检查和
    # 离线两阶段策略选择。
    schema_version: int
    dataset: dataset_pcap.DatasetConfig
    method33: Method33Config
    # 基础策略与三个Trigger波动参数分开存储，但两者都是
    # 完整Method33必需的配置。
    trigger_jitter: TriggerJitterConfig
    evaluation: EvaluationConfig
    selection: SelectionConfig
    source_path: Path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require a JSON object and produce a field-specific error."""

    # 统一入口可让错误信息携带字段路径，而不是在后续 value["..."] 时报一个
    # 难以定位的 TypeError。JSON object 在 Python 中通常解析为 dict。
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _number_tuple(value: Any, name: str) -> tuple[float, ...]:
    """Convert a nonempty JSON list of finite numbers to an immutable tuple."""

    # 配置最终使用 tuple，表达“加载后不可随意 append/pop”。先转 float 再用
    # isfinite 拒绝 NaN/Infinity，避免它们污染概率和预算计算。
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty JSON array")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numbers") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Convert a JSON integer array without silently truncating decimals."""

    # 不能直接 int(1.9)，因为那会静默得到 1。先转数值、再转整数并回比较，
    # 确保只有数学意义上的整数能够通过。
    numbers = _number_tuple(value, name)
    result = tuple(int(item) for item in numbers)
    if any(float(integer) != number for integer, number in zip(result, numbers, strict=True)):
        raise ValueError(f"{name} must contain integers")
    return result


def _integer(value: Any, name: str) -> int:
    """Parse one JSON integer without accepting booleans or decimal truncation."""

    # Python 中 bool 是 int 的子类，int(True)==1；配置里的 true 通常意味着
    # 用户写错字段类型，因此必须在普通整数转换前单独拒绝。
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        number = float(value)
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not np.isfinite(number) or float(result) != number:
        raise ValueError(f"{name} must be an integer")
    return result


def _budget(raw: Any, name: str) -> Budget:
    """Parse one named per-flow budget object."""

    # 本函数只负责字段存在性与基础类型转换；非负性、分账之和等跨字段关系
    # 统一留给 validate_config，避免解析函数之间重复逻辑。
    value = _mapping(raw, name)
    try:
        budget = Budget(
            bytes=float(value["bytes"]),
            delay_seconds=float(value["delay_seconds"]),
            dummy_packets=_integer(value["dummy_packets"], f"{name}.dummy_packets"),
        )
    except KeyError as error:
        raise ValueError(f"{name} is missing {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} contains an invalid numeric value") from error
    return budget


def _weakener(raw: Any) -> WeakenerConfig:
    """Parse weakener arrays while preserving their JSON order."""

    # JSON 数组顺序会决定 rng.choice 的“值—概率”对应关系，不能排序或转 set。
    # 所有数组转成 tuple 后按原顺序交给纯算法核心。
    value = _mapping(raw, "method33.weakener")
    try:
        return WeakenerConfig(
            length_targets=_number_tuple(
                value["length_targets"], "weakener.length_targets"
            ),
            length_probabilities=_number_tuple(
                value["length_probabilities"], "weakener.length_probabilities"
            ),
            gap_targets=_number_tuple(value["gap_targets"], "weakener.gap_targets"),
            gap_probabilities=_number_tuple(
                value["gap_probabilities"], "weakener.gap_probabilities"
            ),
            max_padding_per_packet=float(value["max_padding_per_packet"]),
            max_delay_per_packet=float(value["max_delay_per_packet"]),
            dummy_length_targets=_number_tuple(
                value["dummy_length_targets"], "weakener.dummy_length_targets"
            ),
            dummy_length_probabilities=_number_tuple(
                value["dummy_length_probabilities"],
                "weakener.dummy_length_probabilities",
            ),
            dummy_gap_targets=_number_tuple(
                value["dummy_gap_targets"], "weakener.dummy_gap_targets"
            ),
            dummy_gap_probabilities=_number_tuple(
                value["dummy_gap_probabilities"],
                "weakener.dummy_gap_probabilities",
            ),
            dummy_insertion_outbound_slots=_integer_tuple(
                value["dummy_insertion_outbound_slots"],
                "weakener.dummy_insertion_outbound_slots",
            ),
            padding_probability=float(value["padding_probability"]),
            delay_probability=float(value["delay_probability"]),
        )
    except KeyError as error:
        raise ValueError(f"method33.weakener is missing {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("weakener."):
            raise
        raise ValueError("method33.weakener contains an invalid value") from error


def _selection(raw: Any) -> SelectionConfig:
    """Parse the complete offline two-stage selection protocol from JSON.

    候选弱化器使用与活动弱化器相同的字段模式。解析阶段只负责字段存在性、
    基础类型和不可变转换；概率、预算、候选ID唯一性等组合约束随后统一交给
    ``validate_config``，避免“解析成功”被误解成“实验协议一定合法”。
    """

    value = _mapping(raw, "selection")
    try:
        raw_candidates = value["weakener_candidates"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("selection.weakener_candidates must be a nonempty JSON array")
        candidates: list[WeakenerCandidate] = []
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _mapping(raw_candidate, f"selection.weakener_candidates[{index}]")
            try:
                candidate_id = str(candidate["candidate_id"])
                weakener_raw = candidate["weakener"]
            except KeyError as error:
                raise ValueError(
                    f"selection.weakener_candidates[{index}] is missing {error.args[0]!r}"
                ) from error
            candidates.append(
                WeakenerCandidate(candidate_id=candidate_id, weakener=_weakener(weakener_raw))
            )

        return SelectionConfig(
            validation_captures_per_class=_integer(
                value["validation_captures_per_class"],
                "selection.validation_captures_per_class",
            ),
            retraining_ratios=_number_tuple(
                value["retraining_ratios"], "selection.retraining_ratios"
            ),
            repeats=_integer(value["repeats"], "selection.repeats"),
            validation_trials=_integer(
                value["validation_trials"], "selection.validation_trials"
            ),
            same_mapping_floor=float(value["same_mapping_floor"]),
            trigger_selection_mode=str(value["trigger_selection_mode"]).strip().lower(),
            trigger_codebook_candidates=_integer(
                value["trigger_codebook_candidates"],
                "selection.trigger_codebook_candidates",
            ),
            trigger_pool_size=_integer(
                value["trigger_pool_size"], "selection.trigger_pool_size"
            ),
            trigger_min_gap_seconds=float(value["trigger_min_gap_seconds"]),
            trigger_codebook_prefix=str(value["trigger_codebook_prefix"]),
            inner_split_seed=_integer(
                value["inner_split_seed"], "selection.inner_split_seed"
            ),
            weakener_train_seed=_integer(
                value["weakener_train_seed"], "selection.weakener_train_seed"
            ),
            weakener_validation_seed=_integer(
                value["weakener_validation_seed"],
                "selection.weakener_validation_seed",
            ),
            trigger_train_seed=_integer(
                value["trigger_train_seed"], "selection.trigger_train_seed"
            ),
            trigger_validation_seed=_integer(
                value["trigger_validation_seed"],
                "selection.trigger_validation_seed",
            ),
            codebook_seed=_integer(value["codebook_seed"], "selection.codebook_seed"),
            weakener_candidates=tuple(candidates),
        )
    except KeyError as error:
        raise ValueError(f"selection is missing {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("selection.", "method33.weakener", "weakener.")
        ):
            raise
        raise ValueError("selection contains an invalid value") from error


def _trigger_codes(raw: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Parse ``[class][trigger_packet][length,gap]`` from JSON."""

    # 预期嵌套层次：[设备类别][该码字中的包][包长, 包间隔]。这里只验证结构
    # 与数值可转换性；每个码字的包数、总成本和帧长合法性在 validate_config。
    if not isinstance(raw, list) or not raw:
        raise ValueError("method33.trigger_codes must be a nonempty JSON array")
    codes: list[tuple[tuple[float, float], ...]] = []
    for class_index, code in enumerate(raw):
        if not isinstance(code, list) or not code:
            raise ValueError(f"trigger code {class_index} must be a nonempty array")
        packets: list[tuple[float, float]] = []
        for packet_index, pair in enumerate(code):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(
                    f"trigger_codes[{class_index}][{packet_index}] must be [length, gap]"
                )
            try:
                packets.append((float(pair[0]), float(pair[1])))
            except (TypeError, ValueError) as error:
                raise ValueError("trigger packet length/gap must be numeric") from error
        codes.append(tuple(packets))
    return tuple(codes)


def _trigger_jitter(raw: Any) -> TriggerJitterConfig:
    """Parse the three Method33 Trigger-jitter parameters from strict JSON.

    The separate ``trigger_jitter`` block contains only the relative
    length/gap bounds and realized per-packet gap floor. Budgets and codeword
    centers stay in the ``method33`` policy block so there is one source of
    truth for every cost constraint.
    """

    value = _mapping(raw, "trigger_jitter")
    try:
        return TriggerJitterConfig(
            length_relative_bound=float(value["length_relative_bound"]),
            gap_relative_bound=float(value["gap_relative_bound"]),
            minimum_gap_seconds=float(value["minimum_gap_seconds"]),
        )
    except KeyError as error:
        raise ValueError(f"trigger_jitter is missing {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ValueError("trigger_jitter contains an invalid numeric value") from error


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Read strict JSON from a TXT file, construct typed objects, and validate.

    The ``.txt`` extension is intentional for convenient editing, but comments,
    trailing commas, NaN, and other non-JSON syntax are rejected.  UTF-8 with or
    without a BOM is accepted.
    """

    # -------------------------------------------------------------------------
    # 步骤 1：读取严格 JSON
    # -------------------------------------------------------------------------
    # utf-8-sig 同时兼容普通 UTF-8 与带 BOM 的 Windows 编辑器文件。
    # parse_constant 主动拒绝 Python json 默认可能接受的 NaN/Infinity。
    source = Path(path).resolve()
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8-sig"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value!r} is not allowed")
            ),
        )
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON configuration {source}: {error}") from error
    top = _mapping(raw, "configuration root")

    # -------------------------------------------------------------------------
    # 步骤 2：先取得顶层对象
    # -------------------------------------------------------------------------
    # limits 位于 method33 内部，但单独取出能让后续构造代码保持平坦、可读。
    try:
        dataset_raw = _mapping(top["dataset"], "dataset")
        if "method33" in top and "method32" in top:
            raise ValueError("supply one policy block: method33 or legacy method32")
        policy_key = "method33" if "method33" in top else "method32"
        method_raw = _mapping(top[policy_key], policy_key)
        anchor = method_raw.get("trigger_position_anchor", "original_real_packet_both_directions")
        if anchor != "original_real_packet_both_directions":
            raise ValueError("method33 trigger_position_anchor must count original real packets")
        evaluation_raw = _mapping(top["evaluation"], "evaluation")
        selection_raw = _mapping(top["selection"], "selection")
        limits_raw = _mapping(method_raw["limits"], "method33.limits")
    except KeyError as error:
        raise ValueError(f"configuration is missing top-level field {error.args[0]!r}") from error

    # -------------------------------------------------------------------------
    # 步骤 3：构造数据层配置
    # -------------------------------------------------------------------------
    try:
        devices_value = dataset_raw["devices"]
        if not isinstance(devices_value, list):
            raise ValueError("dataset.devices must be a JSON array")
        dataset_config = dataset_pcap.DatasetConfig(
            devices=tuple(str(item) for item in devices_value),
            min_packets=_integer(dataset_raw["min_packets"], "dataset.min_packets"),
            max_packets=_integer(dataset_raw["max_packets"], "dataset.max_packets"),
            tls_port=_integer(dataset_raw["tls_port"], "dataset.tls_port"),
            mac_filename=str(dataset_raw["mac_filename"]),
            pcap_pattern=str(dataset_raw["pcap_pattern"]),
            max_feature_gap_seconds=float(dataset_raw["max_feature_gap_seconds"]),
            max_frame_length=float(dataset_raw["max_frame_length"]),
        )
    except KeyError as error:
        raise ValueError(f"dataset is missing {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ValueError("dataset contains an invalid value") from error

    # -------------------------------------------------------------------------
    # 步骤 4：构造算法层与评测层配置
    # -------------------------------------------------------------------------
    # require_all_dummies_in_window 必须是真正的 JSON boolean，不能用 0/1 或
    # 字符串冒充；其余离散计数也都通过 _integer 防止小数被截断。
    try:
        require_all = limits_raw["require_all_dummies_in_window"]
        if not isinstance(require_all, bool):
            raise ValueError("limits.require_all_dummies_in_window must be true or false")
        method_config = Method33Config(
            version=_integer(method_raw["version"], "method33.version"),
            method_key=METHOD33_KEY,
            weakener_candidate_id=str(method_raw["weakener_candidate_id"]),
            trigger_codebook_id=str(method_raw["trigger_codebook_id"]),
            weakener_dummy_count=_integer(
                method_raw["weakener_dummy_count"], "method33.weakener_dummy_count"
            ),
            trigger_packet_count=_integer(
                method_raw["trigger_packet_count"], "method33.trigger_packet_count"
            ),
            hard_budget=_budget(method_raw["hard_budget"], "method33.hard_budget"),
            weakener_budget=_budget(
                method_raw["weakener_budget"], "method33.weakener_budget"
            ),
            trigger_budget=_budget(
                method_raw["trigger_budget"], "method33.trigger_budget"
            ),
            weakener=_weakener(method_raw["weakener"]),
            trigger_codes=_trigger_codes(method_raw["trigger_codes"]),
            old_mapping=_integer_tuple(method_raw["old_mapping"], "old_mapping"),
            next_mapping=_integer_tuple(method_raw["next_mapping"], "next_mapping"),
            max_frame_length=float(limits_raw["max_frame_length"]),
            min_dummy_frame_length=float(limits_raw["min_dummy_frame_length"]),
            require_all_dummies_in_window=require_all,
            trigger_positions=_position_codes(method_raw.get("trigger_positions")),
        )
        evaluation_config = EvaluationConfig(
            capture_train_ratio=float(evaluation_raw["capture_train_ratio"]),
            split_seed=_integer(evaluation_raw["split_seed"], "evaluation.split_seed"),
            default_apply_seed=_integer(
                evaluation_raw["default_apply_seed"], "evaluation.default_apply_seed"
            ),
            retraining_mask_seed=_integer(
                evaluation_raw["retraining_mask_seed"], "evaluation.retraining_mask_seed"
            ),
            model_seed=_integer(evaluation_raw["model_seed"], "evaluation.model_seed"),
            retraining_ratios=_number_tuple(
                evaluation_raw["retraining_ratios"], "evaluation.retraining_ratios"
            ),
        )
        selection_config = _selection(selection_raw)
        # Method33的零和波动是正式方法的一部分，因此该顶层块是必需的。
        trigger_jitter_config = _trigger_jitter(top["trigger_jitter"])
        project = ProjectConfig(
            schema_version=_integer(top["schema_version"], "schema_version"),
            dataset=dataset_config,
            method33=method_config,
            trigger_jitter=trigger_jitter_config,
            evaluation=evaluation_config,
            selection=selection_config,
            source_path=source,
        )
    except KeyError as error:
        raise ValueError(f"configuration is missing {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            (
                "method33.",
                "trigger_jitter",
                "weakener.",
                "old_mapping",
                "next_mapping",
                "evaluation.",
                "selection.",
                "limits.",
            )
        ):
            raise
        raise ValueError("method33/evaluation contains an invalid value") from error

    # -------------------------------------------------------------------------
    # 步骤 5：执行跨字段验证
    # -------------------------------------------------------------------------
    # 只有成功通过 validate_config 的 ProjectConfig 才会返回给调用者。
    validate_config(project)
    return project


def _validate_distribution(
    values: Sequence[float], probabilities: Sequence[float], name: str
) -> None:
    """Validate one ordered discrete distribution."""

    # 一个离散分布必须同时满足：值非空且有限非负、概率同形且有限非负、
    # 概率和在微小浮点容差内等于 1。排序不是要求，因为顺序本身可有语义。
    values_array = np.asarray(values, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if values_array.ndim != 1 or len(values_array) == 0:
        raise ValueError(f"{name} values must be a nonempty vector")
    if probability_array.shape != values_array.shape:
        raise ValueError(f"{name} values/probabilities have different shapes")
    if np.any(~np.isfinite(values_array)) or np.any(values_array < 0.0):
        raise ValueError(f"{name} contains a negative or non-finite value")
    if np.any(~np.isfinite(probability_array)) or np.any(probability_array < 0.0):
        raise ValueError(f"{name} contains an invalid probability")
    if not np.isclose(float(np.sum(probability_array)), 1.0, atol=1e-10):
        raise ValueError(f"{name} probabilities must sum to one")


def _validate_weakener_for_method(
    weakener: WeakenerConfig,
    config: Method33Config,
    *,
    name: str,
) -> None:
    """Validate one active or candidate weakener against Method33 constraints."""

    # 候选与当前活动弱化器必须经过完全相同的规则。若只验证当前选中项，某个
    # 非法候选可能在耗时搜索运行到一半时才失败，既浪费时间也会留下不完整表。
    _validate_distribution(
        weakener.length_targets,
        weakener.length_probabilities,
        f"{name} length targets",
    )
    _validate_distribution(
        weakener.gap_targets,
        weakener.gap_probabilities,
        f"{name} gap targets",
    )
    _validate_distribution(
        weakener.dummy_length_targets,
        weakener.dummy_length_probabilities,
        f"{name} dummy length targets",
    )
    _validate_distribution(
        weakener.dummy_gap_targets,
        weakener.dummy_gap_probabilities,
        f"{name} dummy gap targets",
    )
    if max(weakener.length_targets) > config.max_frame_length + 1e-9:
        raise ValueError(f"{name} has a length target above max_frame_length")
    if min(weakener.dummy_length_targets) < config.min_dummy_frame_length - 1e-9:
        raise ValueError(f"{name} has a dummy target below min_dummy_frame_length")
    if max(weakener.dummy_length_targets) > config.max_frame_length + 1e-9:
        raise ValueError(f"{name} has a dummy target above max_frame_length")
    if (
        not np.isfinite(weakener.max_padding_per_packet)
        or not np.isfinite(weakener.max_delay_per_packet)
        or weakener.max_padding_per_packet < 0.0
        or weakener.max_delay_per_packet < 0.0
    ):
        raise ValueError(f"{name} per-packet caps must be finite and nonnegative")
    if not 0.0 <= weakener.padding_probability <= 1.0:
        raise ValueError(f"{name} padding_probability must be in [0, 1]")
    if not 0.0 <= weakener.delay_probability <= 1.0:
        raise ValueError(f"{name} delay_probability must be in [0, 1]")
    slots = weakener.dummy_insertion_outbound_slots
    if len(set(slots)) != len(slots) or any(slot < 1 for slot in slots):
        raise ValueError(f"{name} dummy insertion slots must be unique positive integers")
    if len(slots) < config.weakener_dummy_count:
        raise ValueError(f"{name} has too few insertion slots for weakener_dummy_count")
    if (
        config.weakener_dummy_count * max(weakener.dummy_length_targets)
        > config.weakener_budget.bytes + 1e-9
        or config.weakener_dummy_count * max(weakener.dummy_gap_targets)
        > config.weakener_budget.delay_seconds + 1e-12
    ):
        raise ValueError(f"{name} worst dummy plan cannot fit the weakener partition")


def validate_config(project: ProjectConfig) -> None:
    """Validate all cross-field Method33, dataset, and evaluation invariants."""

    # 为便于审计，检查按“版本与数据 → 预算 → 弱化器 → trigger → mapping →
    # 评测参数”的顺序组织。函数没有返回值；不抛异常即表示全部约束成立。
    dataset = project.dataset
    config = project.method33
    evaluation = project.evaluation
    selection = project.selection
    weakener = config.weakener
    pairs = resolve_trigger_positions(config)
    if (config.require_all_dummies_in_window and
        max(max(pair) for pair in pairs) + config.weakener_dummy_count + config.trigger_packet_count > dataset.max_packets):
        raise ValueError("Trigger anchors plus dummy count exceed the observation width; supply earlier positions")

    if project.schema_version != 1 or config.version != 1:
        raise ValueError("only configuration schema/method version 1 is supported")
    if not dataset.devices or len(set(dataset.devices)) != len(dataset.devices):
        raise ValueError("dataset.devices must contain unique device names")
    if any(not name.strip() for name in dataset.devices):
        raise ValueError("dataset.devices cannot contain an empty name")
    if dataset.min_packets < 1 or dataset.max_packets < dataset.min_packets:
        raise ValueError("dataset packet limits are inconsistent")
    if not 1 <= dataset.tls_port <= 65535:
        raise ValueError("dataset.tls_port is outside the valid port range")
    if dataset.max_frame_length <= 0.0 or dataset.max_feature_gap_seconds <= 0.0:
        raise ValueError("dataset scaling limits must be positive")
    if not config.method_key.strip():
        raise ValueError("method33.method_key cannot be empty")
    if config.class_count != len(dataset.devices):
        raise ValueError("trigger code count must equal dataset device count")
    if not np.isclose(config.max_frame_length, dataset.max_frame_length):
        raise ValueError("dataset and Method33 max_frame_length must match")
    if config.min_dummy_frame_length <= 0.0:
        raise ValueError("min_dummy_frame_length must be positive")

    # 三个预算对象逐一检查基础数值域。这里不能只看 hard budget，因为负的
    # 子预算可能在求和时与另一分账相抵，掩盖配置错误。
    for name, budget in (
        ("hard", config.hard_budget),
        ("weakener", config.weakener_budget),
        ("trigger", config.trigger_budget),
    ):
        if not np.isfinite(budget.bytes) or budget.bytes < 0.0:
            raise ValueError(f"{name} byte budget is invalid")
        if not np.isfinite(budget.delay_seconds) or budget.delay_seconds < 0.0:
            raise ValueError(f"{name} delay budget is invalid")
        if budget.dummy_packets < 0:
            raise ValueError(f"{name} dummy budget is invalid")

    # 分账检查既要求 dummy 计数声明一致，也要求 weakener+trigger 不超过 hard。
    # 使用极小容差只是吸收十进制浮点表示误差，不允许真实意义上的超预算。
    if config.weakener_dummy_count != config.weakener_budget.dummy_packets:
        raise ValueError("weakener_dummy_count must equal weakener dummy budget")
    if config.trigger_packet_count != config.trigger_budget.dummy_packets:
        raise ValueError("trigger_packet_count must equal trigger dummy budget")
    if (
        config.weakener_budget.bytes + config.trigger_budget.bytes
        > config.hard_budget.bytes + 1e-9
        or config.weakener_budget.delay_seconds
        + config.trigger_budget.delay_seconds
        > config.hard_budget.delay_seconds + 1e-12
        or config.weakener_budget.dummy_packets
        + config.trigger_budget.dummy_packets
        > config.hard_budget.dummy_packets
    ):
        raise ValueError("weakener and trigger partitions exceed the hard budget")

    # 当前活动弱化器和所有离线候选使用同一验证器，确保选择过程不会遇到一个
    # “能解析但无法在Reduce30分账内执行”的候选。
    _validate_weakener_for_method(weakener, config, name="active weakener")

    # 所有码字必须具有相同包数，并且每个码字的总 byte/delay 都精确等于
    # trigger 分账。这样类别不能通过“总开销”被区分，只能通过码字内部结构。
    for class_index, code in enumerate(config.trigger_codes):
        if len(code) != config.trigger_packet_count:
            raise ValueError(
                f"trigger code {class_index} has {len(code)} packets; "
                f"expected {config.trigger_packet_count}"
            )
        lengths = np.asarray([packet[0] for packet in code], dtype=np.float64   )
        gaps = np.asarray([packet[1] for packet in code], dtype=np.float64)
        if np.any(~np.isfinite(lengths)) or np.any(~np.isfinite(gaps)):
            raise ValueError(f"trigger code {class_index} contains a non-finite value")
        if np.any(lengths < config.min_dummy_frame_length - 1e-9):
            raise ValueError(f"trigger code {class_index} contains an undersized packet")
        if np.any(lengths > config.max_frame_length + 1e-9) or np.any(gaps < 0.0):
            raise ValueError(f"trigger code {class_index} contains an invalid packet")
        if not np.isclose(np.sum(lengths), config.trigger_budget.bytes, atol=1e-9):
            raise ValueError(f"trigger code {class_index} has the wrong total bytes")
        if not np.isclose(
            np.sum(gaps), config.trigger_budget.delay_seconds, atol=1e-12
        ):
            raise ValueError(f"trigger code {class_index} has the wrong total delay")

    # old/next 都必须是 0..K-1 的排列，确保类别和码字一一对应；next 还必须
    # 无固定点，使每个类别在部署切换后都不再使用攻击者训练时看到的原码字。
    expected_mapping = tuple(range(config.class_count))
    for name, mapping in (
        ("old_mapping", config.old_mapping),
        ("next_mapping", config.next_mapping),
    ):
        if tuple(sorted(mapping)) != expected_mapping:
            raise ValueError(f"{name} must be a permutation of {expected_mapping}")
    # “切换”应相对于old映射定义，而不是相对于恒等排列定义。当前old恰好是
    # identity，但使用逐label比较可保证未来更换old时仍不会给任何设备保留旧码字。
    if any(
        old_code == next_code
        for old_code, next_code in zip(
            config.old_mapping, config.next_mapping, strict=True
        )
    ):
        raise ValueError("next_mapping must assign every label a code different from old_mapping")

    # 在任何PCAP提取或模型训练之前验证5%零和波动是否对每一个
    # 冻结码字都可行。用0与小于1的最大float覆盖可行区间两端；
    # jitter_trigger_code还会复核整数frame、最小gap、相对上界和等总预算。
    jitter = project.trigger_jitter
    if config.trigger_packet_count != 2:
        raise ValueError("configured Method33 requires exactly two Trigger packets")
    upper_quantile = float(np.nextafter(1.0, 0.0))
    for code_index, code in enumerate(config.trigger_codes):
        try:
            for quantile in (0.0, upper_quantile):
                jitter_trigger_code(
                    code,
                    config,
                    jitter,
                    length_uniform_quantile=quantile,
                    gap_uniform_quantile=quantile,
                )
        except (ValueError, AssertionError) as error:
            raise ValueError(
                f"method33 jitter is infeasible for trigger code {code_index}: {error}"
            ) from error

    # 评测比例必须有序且不重复。build_stratified_nested_masks 依赖这些比例
    # 形成可解释的嵌套样本轨迹。
    if not 0.0 < evaluation.capture_train_ratio < 1.0:
        raise ValueError("capture_train_ratio must be strictly between zero and one")
    ratios = np.asarray(evaluation.retraining_ratios, dtype=np.float64)
    if np.any(~np.isfinite(ratios)) or np.any((ratios < 0.0) | (ratios > 1.0)):
        raise ValueError("retraining_ratios must be finite values in [0, 1]")
    if len(np.unique(ratios)) != len(ratios) or np.any(np.diff(ratios) < 0.0):
        raise ValueError("retraining_ratios must be unique and sorted")

    # 所有随机源都必须能被NumPy SeedSequence接受。模型seed还会以seed+1、
    # seed+2传给scikit-learn的RandomState接口，所以需为这两个固定偏移预留
    # uint32空间；在训练数分钟后才遇到随机状态错误会浪费整次实验。
    seed_values = {
        "evaluation.split_seed": evaluation.split_seed,
        "evaluation.default_apply_seed": evaluation.default_apply_seed,
        "evaluation.retraining_mask_seed": evaluation.retraining_mask_seed,
        "evaluation.model_seed": evaluation.model_seed,
        "selection.inner_split_seed": selection.inner_split_seed,
        "selection.weakener_train_seed": selection.weakener_train_seed,
        "selection.weakener_validation_seed": selection.weakener_validation_seed,
        "selection.trigger_train_seed": selection.trigger_train_seed,
        "selection.trigger_validation_seed": selection.trigger_validation_seed,
        "selection.codebook_seed": selection.codebook_seed,
    }
    negative_seeds = [name for name, value in seed_values.items() if value < 0]
    if negative_seeds:
        raise ValueError(
            "random seeds must be non-negative: " + ", ".join(negative_seeds)
        )
    maximum_model_seed = int(np.iinfo(np.uint32).max) - 2
    if evaluation.model_seed > maximum_model_seed:
        raise ValueError(
            "evaluation.model_seed must be at most "
            f"{maximum_model_seed} because proxy models also use seed+1/seed+2"
        )

    # 离线选择协议也属于可复现实验配置。所有候选在正式搜索开始前一次性验证，
    # 并要求当前活动候选出现在列表中，使默认配置自身就是一次可重跑选择的输入。
    if selection.validation_captures_per_class < 1:
        raise ValueError(
            "selection.validation_captures_per_class must be a positive integer"
        )
    selection_ratios = np.asarray(selection.retraining_ratios, dtype=np.float64)
    if (
        np.any(~np.isfinite(selection_ratios))
        or np.any((selection_ratios <= 0.0) | (selection_ratios > 1.0))
        or len(np.unique(selection_ratios)) != len(selection_ratios)
        or np.any(np.diff(selection_ratios) < 0.0)
    ):
        raise ValueError(
            "selection.retraining_ratios must be unique sorted values in (0, 1]"
        )
    if selection.repeats < 1 or selection.validation_trials < 1:
        raise ValueError("selection repeats and validation_trials must be positive")
    if not 0.0 <= selection.same_mapping_floor <= 1.0:
        raise ValueError("selection.same_mapping_floor must be in [0, 1]")
    if selection.trigger_selection_mode not in {"geometry", "proxy"}:
        raise ValueError("selection.trigger_selection_mode must be 'geometry' or 'proxy'")
    if config.class_count < 2 or config.trigger_packet_count < 2:
        raise ValueError("Trigger selection requires at least two classes and two packets")
    if config.trigger_budget.bytes <= 0.0 or config.trigger_budget.delay_seconds <= 0.0:
        raise ValueError("Trigger selection requires positive byte and delay budgets")
    if selection.trigger_codebook_candidates < 1 or selection.trigger_pool_size < config.class_count:
        raise ValueError(
            "selection needs at least one codebook and a pool no smaller than class_count"
        )
    if (
        not np.isfinite(selection.trigger_min_gap_seconds)
        or selection.trigger_min_gap_seconds < 0.0
        or config.trigger_packet_count * selection.trigger_min_gap_seconds
        > config.trigger_budget.delay_seconds + 1e-12
    ):
        raise ValueError("selection.trigger_min_gap_seconds is infeasible")
    if not selection.trigger_codebook_prefix.strip():
        raise ValueError("selection.trigger_codebook_prefix cannot be empty")
    rounded_trigger_bytes = int(round(config.trigger_budget.bytes))
    if not np.isclose(config.trigger_budget.bytes, rounded_trigger_bytes, atol=1e-9):
        raise ValueError("trigger codebook generation requires an integer byte budget")
    if (
        config.trigger_packet_count * config.min_dummy_frame_length
        > rounded_trigger_bytes + 1e-9
    ):
        raise ValueError("trigger budget cannot fit the minimum frame length per packet")
    rounded_minimum_frame = int(round(config.min_dummy_frame_length))
    if not np.isclose(
        config.min_dummy_frame_length, rounded_minimum_frame, atol=1e-9
    ):
        raise ValueError("trigger codebook generation requires an integer minimum frame")
    free_trigger_bytes = (
        rounded_trigger_bytes
        - config.trigger_packet_count * rounded_minimum_frame
    )
    feasible_length_tuples = math.comb(
        free_trigger_bytes + config.trigger_packet_count - 1,
        config.trigger_packet_count - 1,
    )
    if selection.trigger_pool_size > feasible_length_tuples:
        raise ValueError(
            "selection.trigger_pool_size exceeds the finite integer length-plan space"
        )

    candidate_ids = [candidate.candidate_id for candidate in selection.weakener_candidates]
    if any(not candidate_id.strip() for candidate_id in candidate_ids):
        raise ValueError("selection weakener candidate IDs cannot be empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("selection weakener candidate IDs must be unique")
    if config.weakener_candidate_id not in set(candidate_ids):
        raise ValueError("active weakener_candidate_id is absent from selection candidates")
    for candidate in selection.weakener_candidates:
        _validate_weakener_for_method(
            candidate.weakener,
            config,
            name=f"weakener candidate {candidate.candidate_id!r}",
        )


def validate_dataset_project_alignment(
    dataset: Mapping[str, np.ndarray], project: ProjectConfig
) -> None:
    """Reject label/device-order mismatches before selection or evaluation.

    ``dataset_pcap.validate_dataset`` proves that labels are internally
    contiguous, but it cannot know which configured device each integer means.
    Trigger mappings are label-indexed, so silently swapping two device names
    would invalidate the experiment even though every array shape remained
    legal.  This cross-object check closes that gap.
    """

    dataset_pcap.validate_dataset(dataset, expected_width=project.dataset.max_packets)
    labels = np.asarray(dataset["labels"], dtype=int)
    expected_labels = np.arange(project.method33.class_count)
    if not np.array_equal(np.unique(labels), expected_labels):
        raise ValueError(
            "dataset labels do not match configured class count: "
            f"found {np.unique(labels).tolist()}, expected {expected_labels.tolist()}"
        )
    if "device_names" not in dataset:
        raise ValueError("dataset must include device_names for config-alignment validation")
    actual_devices = tuple(np.asarray(dataset["device_names"]).astype(str).tolist())
    if actual_devices != tuple(project.dataset.devices):
        raise ValueError(
            "dataset device_names order differs from configuration: "
            f"dataset={list(actual_devices)}, config={list(project.dataset.devices)}"
        )

    manifest_key = "unsw_extraction_manifest_json"
    if manifest_key in dataset:
        encoded = np.asarray(dataset[manifest_key])
        if encoded.shape != ():
            raise ValueError("unsw_extraction_manifest_json must be a scalar JSON string")
        try:
            manifest = json.loads(str(encoded.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("unsw_extraction_manifest_json is invalid") from error
        if not isinstance(manifest, Mapping):
            raise ValueError("UNSW extraction manifest must decode to an object")
        manifest_devices = tuple(str(value) for value in manifest.get("devices", ()))
        if manifest_devices != actual_devices:
            raise ValueError(
                "UNSW extraction manifest devices differ from dataset device_names"
            )
        manifest_class_count = manifest.get("class_count", len(manifest_devices))
        if int(manifest_class_count) != project.method33.class_count:
            raise ValueError(
                "UNSW extraction manifest class_count differs from configuration"
            )


def build_stratified_nested_masks(
    labels: np.ndarray,
    ratios: Sequence[float],
    *,
    seed: int,
) -> dict[float, np.ndarray]:
    """Build shared class-stratified masks nested across retraining ratios.

    A class is shuffled once, then each larger ratio takes a longer prefix of
    that same order.  Candidate identity is intentionally absent from the seed;
    all weakener candidates in one repeat must see the same retraining samples.
    """

    # 关键研究约束：seed 只由 repeat/协议预先决定，绝不混入候选弱化器 ID。
    # 因而同一轮中的所有候选都会在完全相同的重训练流上被比较。
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a nonempty one-dimensional array")
    ratios = tuple(float(value) for value in ratios)
    if any(not 0.0 <= value <= 1.0 for value in ratios):
        raise ValueError("ratios must lie in [0, 1]")
    # 每个类别只打乱一次；比例 r 取该固定顺序的前 round(n_k*r) 条。所以
    # 10% 样本天然包含于 25%，25% 又包含于 50%，减少曲线中的采样噪声。
    class_orders: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        local = np.flatnonzero(labels == label)
        rng = np.random.default_rng(int(seed) + 101 * int(label))
        class_orders[int(label)] = rng.permutation(local)
    result: dict[float, np.ndarray] = {}
    for ratio in ratios:
        selected = np.zeros(len(labels), dtype=bool)
        for order in class_orders.values():
            count = int(np.rint(len(order) * ratio))
            selected[order[:count]] = True
        result[ratio] = selected
    return result


def apply_configured_method(
    dataset: Mapping[str, np.ndarray],
    project: ProjectConfig,
    *,
    seed: int | None = None,
    mapping: str | Sequence[int] = "old",
    include_weakener: bool = True,
    include_trigger: bool = True,
    include_event_trace: bool = False,
) -> Application:
    """Apply Method33 through one validated orchestration gateway.

    All public orchestration paths now apply the complete Method33 method.
    """

    validate_dataset_project_alignment(dataset, project)
    actual_seed = (
        project.evaluation.default_apply_seed if seed is None else int(seed)
    )
    common_arguments = (
        *(np.asarray(dataset[name]) for name in ("lengths", "gaps", "directions", "mask")),
        np.asarray(dataset["labels"]),
        project.method33,
        actual_seed,
    )
    common_keywords = {
        "mapping": mapping,
        "include_weakener": bool(include_weakener),
        "include_trigger": bool(include_trigger),
        "include_event_trace": bool(include_event_trace),
    }
    application = apply_method33(
        *common_arguments,
        **common_keywords,
        jitter_config=project.trigger_jitter,
    )

    # 应用后立即验证Method33硬预算和七元组schema，
    # 避免错误结果继续流入HTML、NPZ或攻击模型。
    audit = validate_application(application, config=project.method33)
    if not audit["passed"]:
        raise AssertionError("Method33 failed the configured hard budget")
    return application


def hard_budget_audit(
    stats: Mapping[str, np.ndarray], config: Method33Config
) -> dict[str, Any]:
    """Check total per-flow bytes, delay, and dummy count against hard limits."""

    # 总新增成本 = 真实包修改成本 + dummy 成本。审计必须逐流判断，不能只看
    # 全体平均值；平均不超预算并不代表没有单条流违规。
    added_bytes = np.asarray(stats["real_added_bytes"], dtype=np.float64) + np.asarray(
        stats["dummy_bytes"], dtype=np.float64
    )
    added_delay = np.asarray(stats["real_added_delay"], dtype=np.float64) + np.asarray(
        stats["dummy_delay"], dtype=np.float64
    )
    dummy_count = np.asarray(stats["dummy_count"], dtype=np.float64)
    # 三个布尔条件按位 OR，任一维度超限都将该流标记为 violating。
    violating = (
        (added_bytes > config.hard_budget.bytes + 1e-7)
        | (added_delay > config.hard_budget.delay_seconds + 1e-10)
        | (dummy_count > config.hard_budget.dummy_packets + 1e-9)
    )
    rows = np.column_stack((added_bytes, added_delay, dummy_count))
    return {
        "passed": bool(not np.any(violating)),
        "flows_checked": int(len(rows)),
        "violating_flows": int(np.sum(violating)),
        "maximum": np.max(rows, axis=0).tolist() if len(rows) else [0.0, 0.0, 0.0],
        "mean": np.mean(rows, axis=0).tolist() if len(rows) else [0.0, 0.0, 0.0],
        "limits": config.hard_budget.as_list(),
    }


def validate_application(
    application: Sequence[Any],
    *,
    config: Method33Config | None = None,
) -> dict[str, Any]:
    """Validate the shared seven-element defense result schema.

    ``config`` is optional so basic shape/cost validation can also be applied to
    a retained baseline.  When supplied, Method33 hard budgets are audited too.
    """

    # 七元组验证可同时用于 Method33 与保留的 baseline：先检查公共 schema；
    # 仅当提供 Method33Config 时，再执行 Method33 的具体硬预算审计。
    if len(application) != 7:
        raise ValueError("a defense application must contain seven elements")
    lengths, gaps, directions, mask, stats, metadata, audit_rows = application
    lengths, gaps, directions, mask = map(
        np.asarray, (lengths, gaps, directions, mask)
    )
    if lengths.ndim != 2 or not (
        lengths.shape == gaps.shape == directions.shape == mask.shape
    ):
        raise ValueError("application arrays do not share a two-dimensional shape")
    valid = mask.astype(bool)
    if np.any(mask != valid.astype(mask.dtype)):
        raise ValueError("application mask is not binary")
    if np.any(~np.isfinite(lengths[valid])) or np.any(~np.isfinite(gaps[valid])):
        raise ValueError("application contains non-finite event values")
    if np.any(lengths[valid] <= 0.0) or np.any(gaps[valid] < 0.0):
        raise ValueError("application contains an invalid length or gap")
    if np.any(~np.isin(directions[valid], (-1, 1))):
        raise ValueError("application contains an invalid direction")
    # 前 17 个 STAT_KEYS 是跨方法公共契约；dummy_gap_seconds 是历史别名，
    # 所以 baseline 缺少该别名时仍可通过公共验证。
    missing_stats = set(STAT_KEYS[:17]) - set(stats)
    if missing_stats:
        raise ValueError(f"application stats are missing: {sorted(missing_stats)}")
    for key, values in stats.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(lengths),) or np.any(~np.isfinite(array)):
            raise ValueError(f"stats[{key!r}] has a wrong shape or non-finite value")
        if np.any(array < -1e-9):
            raise ValueError(f"stats[{key!r}] contains negative overhead")
    if not isinstance(metadata, Mapping):
        raise ValueError("application metadata must be a mapping")
    if not isinstance(audit_rows, list) or len(audit_rows) != len(lengths):
        raise ValueError("application audit row count does not match flow count")
    if any(not row.get("source_event_order_preserved", False) for row in audit_rows):
        raise ValueError("a source-event order audit failed")
    # metadata 与 audit 只检查结构和最关键的源顺序标记，避免工具层假设每个
    # baseline 都有完全相同的私有审计字段。
    return hard_budget_audit(stats, config) if config else {
        "passed": True,
        "flows_checked": int(len(lengths)),
    }


def summarize_application(application: Sequence[Any]) -> dict[str, Any]:
    """Return compact means for either Method33 or a compatible baseline."""

    # 汇总只读取公共成本字段，因此同一个打印/报表函数可用于 Method33 和
    # baselines。这里报告均值；是否有逐流超限仍以 hard_budget_audit 为准。
    stats = application[4]
    added_bytes = np.asarray(stats["real_added_bytes"]) + np.asarray(
        stats["dummy_bytes"]
    )
    added_delay = np.asarray(stats["real_added_delay"]) + np.asarray(
        stats["dummy_delay"]
    )
    return {
        "flows": int(len(added_bytes)),
        **{f"mean_{key}": float(np.mean(stats[key])) if len(added_bytes) else 0.0
           for key in ("real_completion_delay_seconds","wire_completion_delay_seconds","cover_tail_seconds") if key in stats},
        "mean_added_bytes": float(np.mean(added_bytes)) if len(added_bytes) else 0.0,
        "mean_action_budget_delay_seconds": (
            float(np.mean(added_delay)) if len(added_delay) else 0.0
        ),
        "mean_dummy_packets": float(np.mean(stats["dummy_count"])),
        "mean_retained_dummy_packets": float(
            np.mean(stats["retained_dummy_count"])
        ),
        "mean_displaced_real_packets": float(
            np.mean(stats["displaced_real_packets"])
        ),
    }


def save_application(
    path: Path,
    source_dataset: Mapping[str, np.ndarray],
    application: Sequence[Any],
) -> None:
    """Save protected arrays, labels/groups, stats, metadata, and audits to NPZ.

    JSON strings are stored as Unicode arrays, so loading never requires
    ``allow_pickle=True``.  Cost arrays use a ``stats__`` prefix to avoid name
    collisions with dataset fields.
    """

    # NPZ 顶层只放 NumPy 数组。嵌套 metadata/audit 先转成 JSON Unicode 字符串，
    # 从而保持 allow_pickle=False 的安全加载能力。
    lengths, gaps, directions, mask, stats, metadata, audits = application
    payload: dict[str, np.ndarray] = {
        "lengths": np.asarray(lengths),
        "gaps": np.asarray(gaps),
        "directions": np.asarray(directions),
        "mask": np.asarray(mask),
        "labels": np.asarray(source_dataset["labels"]),
        "groups": np.asarray(source_dataset["groups"]),
        "metadata_json": np.asarray(json.dumps(_jsonable(metadata), ensure_ascii=False)),
        "audit_json": np.asarray(json.dumps(_jsonable(audits), ensure_ascii=False)),
    }
    if "device_names" in source_dataset:
        payload["device_names"] = np.asarray(source_dataset["device_names"])
    # stats__ 前缀防止成本字段与原始 dataset 字段同名；labels/groups 沿用
    # source_dataset，因为防御只改变流量观察，不改变样本身份。
    for key, values in stats.items():
        payload[f"stats__{key}"] = np.asarray(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _jsonable(value: Any) -> Any:
    """Recursively convert NumPy/Path/dataclass values for JSON output."""

    # json.dumps 不认识 ndarray、NumPy 标量、Path 和 dataclass；递归转换后
    # 只留下 dict/list/str/int/float/bool/None 等 JSON 原生类型。
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="validate a legacy or native configuration")
    apply_parser = sub.add_parser("apply", help="apply independent Method33 to an NPZ")
    apply_parser.add_argument("--dataset", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--mapping", choices=("old", "next"), default="next")
    apply_parser.add_argument("--seed", type=int)
    apply_parser.add_argument("--trace", action="store_true")
    args = parser.parse_args(argv)
    project = load_config(args.config)
    if args.command == "inspect":
        result = _jsonable(asdict(project))
        result["effective_trigger_positions"] = resolve_trigger_positions(project.method33)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if args.output.resolve() == args.dataset.resolve():
            parser.error("output must differ from the original dataset")
        if args.output.exists():
            parser.error("output already exists; choose a new destination")
        data = dataset_pcap.load_dataset(args.dataset)
        app = apply_configured_method(data, project, seed=args.seed, mapping=args.mapping,
                                      include_event_trace=args.trace)
        save_application(args.output, data, app)
        print(json.dumps(summarize_application(app), ensure_ascii=False))


if __name__ == "__main__":
    main()
