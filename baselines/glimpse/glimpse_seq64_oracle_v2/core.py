"""Glimpse-Seq64-Oracle-v2 的全部算法实现。

本文件不导入项目中任何其他模块。实现保留 Glimpse 的三项核心思想：

1. 每台设备只用自己的流量训练 Isolation Forest 奖励模型；
2. 用 DDPG 的 Actor/Critic 学习逐包变形策略；
3. 保留padding、TCP分段、delay三类动作，并审计完整事件流的开销。

Oracle 适配与原论文的差别也必须说清：策略可以看到一条样本完整的前 32 个
分组，因此不再运行 ARIMA。它是离线适配变体，不是实时实现或原方法的效果
上界。分类特征为32个带方向长度加32个对数间隔，代码不依赖其他方法。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import pickle
from typing import Sequence
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# 论文明确值与 Seq64 适配常量
# ---------------------------------------------------------------------------

EVENT_COUNT = 32
FEATURE_COUNT = 64
MAX_FRAME_LENGTH = 1514.0  # Feature scale and outbound action limit, not an inbound capture limit.
MAX_GAP_SECONDS = 10.0
DELAY_MODEL = "outbound_fifo_timestamps_v1"
FRAGMENTATION_MODEL = "tcp_segmentation_integer_payload_v1"
POLICY_FORMAT = "glimpse_seq64_oracle_v2_target_actor"
MIN_FRAME_LENGTH = 60  # Ethernet帧长口径不含FCS、前导码和帧间隙。

IF_TREE_COUNT = 100
IF_MAX_SAMPLES = 256
DDPG_HIDDEN = (32, 32)
DDPG_LEARNING_RATE = 0.001
DDPG_BATCH_SIZE = 32
DDPG_UPDATE_INTERVAL = 100
MAX_ACTIONS_PER_FLOW = 10
PADDING_MIN_BYTES = 200
PADDING_MAX_BYTES = 1500

ACTION_NAMES = (
    "payload_padding",
    "ip_tcp_fragmentation",
    "packet_delay",
)


def _print_training_progress(label: str, completed: int, total: int, reward: float) -> None:
    """Render a compact dependency-free progress bar for long DDPG fitting."""

    interval = max(1, int(total) // 20)
    if completed != total and completed % interval:
        return
    width = 24
    filled = round(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if completed == total else ""
    print(
        f"\r{label} [{bar}] {completed:>3}/{total} iter  reward={reward:.4f}",
        end=ending,
        flush=True,
    )
ACTION_DIMENSION = 4  # 3 个动作分数 + 1 个连续强度
_EULER_GAMMA_FROM_PAPER = 0.5772156649


@dataclass(frozen=True)
class GlimpseConfig:
    """运行配置。

    下列实验参数在论文中只出现了符号/概念或完全没有给数值，因此这里故意
    不设置默认值，调用者必须明确声明。这样不会把我们的选择伪装成论文参数。

    ``max_delay_seconds`` 和 ``max_fragments`` 是 Seq64 数据包适配所必需的
    边界，论文同样没有给出；它们也必须由实验协议明确提供。
    ``frame_header_bytes``的54字节默认值是无选项Ethernet/IPv4/TCP建模假设，
    不是从输入序列推断的真实头长；逐包header_lengths可以覆盖该假设。
    """

    gamma: float
    rho: float
    reward_threshold: float
    training_iterations: int
    max_delay_seconds: float
    max_fragments: int
    target_update_period: int
    replay_buffer_capacity: int
    exploration_noise_stddev: float
    exploration_noise_damping: float
    seed: int
    frame_header_bytes: int = 54  # 显式建模假设：Ethernet14 + IPv4 20 + TCP20。
    shared_budget: dict | None = None
    budget_action_mode: str = "reject"  # Historical mode remains reproducible.

    def __post_init__(self) -> None:
        if self.budget_action_mode not in ("reject", "project"):
            raise ValueError("budget_action_mode must be reject or project")
        if self.shared_budget is not None:
            from experiment_support.budget import PhysicalBudget
            PhysicalBudget(**self.shared_budget)
        for name in (
            "training_iterations", "max_fragments", "target_update_period",
            "replay_buffer_capacity", "seed", "frame_header_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} 必须为整数")
        for name in (
            "gamma", "rho", "reward_threshold", "max_delay_seconds",
            "exploration_noise_stddev", "exploration_noise_damping",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 必须是有限数值")
        if self.seed < 0 or not 54 <= self.frame_header_bytes <= MAX_FRAME_LENGTH:
            raise ValueError("seed必须非负，frame_header_bytes必须位于[54,1514]")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma 必须位于 [0,1]")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho 必须位于 [0,1]")
        if not 0.0 <= self.reward_threshold <= 1.0:
            raise ValueError("reward_threshold 必须位于 [0,1]")
        if self.training_iterations < 1:
            raise ValueError("training_iterations 必须为正整数")
        if self.max_delay_seconds <= 0.0:
            raise ValueError("max_delay_seconds 必须为正数")
        if self.max_fragments < 2:
            raise ValueError("max_fragments 至少为 2")
        if self.target_update_period < 1:
            raise ValueError("target_update_period 必须为正整数")
        if self.replay_buffer_capacity < DDPG_BATCH_SIZE:
            raise ValueError("replay_buffer_capacity 不能小于batch size 32")
        if self.exploration_noise_stddev < 0.0:
            raise ValueError("exploration_noise_stddev 不能为负")
        if not 0.0 <= self.exploration_noise_damping <= 1.0:
            raise ValueError("exploration_noise_damping 必须位于 [0,1]")


@dataclass
class Packet:
    """供训练和评测使用的数据包对象，而不是纯特征向量。

    ``source_index`` 标识原始真实包；分片后多个片段共享它，从而可以检查算法
    没有凭空删除原始业务数据。header_bytes是每段均须保留的头长（含二层）。
    payload_bytes包含业务载荷与追加的payload padding，不含以太网填充。
    未提供真实payload长度时，明确假定length-header_bytes均为载荷。
    """

    length: float
    gap: float
    direction: int
    source_index: int
    destination_port: int | None = None
    fragment_index: int = 0
    fragment_count: int = 1
    padding_bytes: int = 0
    header_bytes: int = 54
    payload_bytes: int | None = None
    payload_offset: int = 0

    def __post_init__(self) -> None:
        for name in ("length", "header_bytes", "padding_bytes", "payload_offset"):
            value = getattr(self, name)
            if not np.isfinite(value) or value != int(value):
                raise ValueError(f"Packet.{name}必须为整数字节")
            setattr(self, name, int(value))
        if self.direction not in (-1, 1):
            raise ValueError("Packet.direction必须为-1或+1")
        if not 54 <= self.header_bytes <= self.length:
            raise ValueError("需要54 <= header_bytes <= length")
        # Inbound packets are observations, never action targets. Captured VLAN
        # frames can exceed 1514 bytes and must not be truncated or rescaled.
        # Keep the existing outbound action domain until protocol metadata can
        # support a different MTU/header model; do not guess it from frame size.
        if self.direction == 1 and self.length > MAX_FRAME_LENGTH:
            raise ValueError("上行帧长超过1514；需要明确的协议头部/MTU模型，不能静默截断或扩展动作上限")
        if self.payload_bytes is None:
            self.payload_bytes = self.length - self.header_bytes
        if not np.isfinite(self.payload_bytes) or self.payload_bytes != int(self.payload_bytes):
            raise ValueError("payload_bytes必须为整数字节")
        self.payload_bytes = int(self.payload_bytes)
        if not 0 <= self.payload_bytes <= self.length - self.header_bytes:
            raise ValueError("payload_bytes与帧长/头长不一致")
        if self.length > max(MIN_FRAME_LENGTH, self.header_bytes + self.payload_bytes):
            raise ValueError("头部和载荷之外只允许最小以太网帧所需的填充")
        if not 0 <= self.padding_bytes <= self.payload_bytes or self.payload_offset < 0:
            raise ValueError("payload padding或载荷偏移非法")

    @property
    def link_padding_bytes(self) -> int:
        return self.length - self.header_bytes - self.payload_bytes


@dataclass
class Flow:
    """一条可被 Oracle 策略反复修改的流。"""

    packets: list[Packet]
    # 首个可见事件在共同时间轴上的时刻，不作为Seq64特征。首包等待导致
    # 整流起点后移时保留此偏移，后续动作仍在同一个时间轴上执行。
    start_time: float = 0.0
    budget_origin: tuple | None = None  # (complete bytes, event count, physical end)

    def clone(self) -> "Flow":
        return deepcopy(self)

    def original_source_indices(self) -> set[int]:
        return {packet.source_index for packet in self.packets}

    def outbound_source_indices(self) -> list[int]:
        # 分片会产生重复 source_index，因此按原始包去重，避免再次修改生成片段。
        result: list[int] = []
        for packet in self.packets:
            if packet.direction == 1 and packet.source_index not in result:
                result.append(packet.source_index)
        return result

    def locate_source(self, source_index: int) -> int:
        for position, packet in enumerate(self.packets):
            if packet.source_index == source_index:
                return position
        raise RuntimeError("变形过程删除了待处理的原始包")


@dataclass
class FlowCost:
    """逐流开销与动作审计。

    added_delay累计策略主动施加的等待秒数，不等于流时长增量，也不等于
    所有后续包被FIFO阻塞的等待时间之和。
    """

    added_bytes: float = 0.0
    added_delay: float = 0.0
    added_fragments: int = 0
    added_payload_bytes: int = 0
    added_header_bytes: int = 0
    added_link_padding_bytes: int = 0  # 新旧以太网填充差；可因填充被载荷占用而为负。
    budget_rejected_actions: int = 0
    budget_projected_actions: int = 0
    action_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in ACTION_NAMES}
    )

    def accumulate(self, other: "FlowCost") -> None:
        for name in (
            "added_bytes", "added_delay", "added_fragments", "added_payload_bytes",
            "added_header_bytes", "added_link_padding_bytes",
            "budget_rejected_actions",
            "budget_projected_actions",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in ACTION_NAMES:
            self.action_counts[name] += other.action_counts[name]


@dataclass
class _DevicePolicy:
    """一台设备对应一个奖励模型和一个 Actor。"""

    forest: IsolationForest
    target_actor_weights: tuple[np.ndarray, ...]


def _validate_arrays(lengths, gaps, directions, mask):
    """统一检查 [N,32] 输入，避免实验脚本悄悄改变观察窗口。"""

    arrays = tuple(np.asarray(value) for value in (lengths, gaps, directions, mask))
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("lengths/gaps/directions/mask 必须是二维数组")
    if not all(value.shape == arrays[0].shape for value in arrays):
        raise ValueError("四个输入数组形状必须一致")
    if arrays[0].shape[1] != EVENT_COUNT:
        raise ValueError("Glimpse-Seq64-Oracle-v2 固定观察前 32 个分组")
    valid = arrays[3].astype(bool)
    if np.any(~np.isfinite(arrays[0])) or np.any(~np.isfinite(arrays[1])):
        raise ValueError("包长和包间隔不能包含 NaN/Inf")
    # 固定窗口采用前缀掩码：一旦出现padding位，后面不能再有真实包。
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("mask 必须是连续的有效前缀，不能在无效位之后重新出现真实包")
    if np.any(~np.isin(arrays[2][valid], (-1, 1))):
        raise ValueError("有效分组方向必须为 -1 或 +1")
    if np.any(arrays[0][valid] <= 0.0) or np.any(arrays[1][valid] < 0.0):
        raise ValueError("有效包长必须为正，包间隔不能为负")
    if np.any(arrays[0][valid] != np.rint(arrays[0][valid])):
        raise ValueError("帧长必须为整数字节，不能对浮点长度静默取整")
    if np.any(arrays[1][valid[:, 0], 0] != 0.0):
        raise ValueError("流内首包没有前驱，gaps[:,0]必须为0；等待开销应单独记录")
    return arrays[0], arrays[1], arrays[2], valid


def make_features(lengths, gaps, directions, mask) -> np.ndarray:
    """生成与 Method32 完全相同的 64 维分类特征。

    前 32 维：direction * length / 1514。
    后 32 维：log1p(min(gap,10秒) * 1000)。
    无效位置严格为 0。Oracle 在每次动作后都重新提取完整向量。
    """

    lengths, gaps, directions, valid = _validate_arrays(
        lengths, gaps, directions, mask
    )
    signed_lengths = directions.astype(np.float64) * lengths / MAX_FRAME_LENGTH
    log_gaps = np.log1p(np.minimum(gaps, MAX_GAP_SECONDS) * 1000.0)
    signed_lengths = np.where(valid, signed_lengths, 0.0)
    log_gaps = np.where(valid, log_gaps, 0.0)
    return np.concatenate((signed_lengths, log_gaps), axis=1)


def _packet_metadata(lengths, valid, config, header_lengths=None, payload_lengths=None):
    """逐包协议长度可由调用者提供；不从Seq64假装推断真实TCP头部。"""

    headers = (
        np.full(lengths.shape, config.frame_header_bytes, dtype=np.int64)
        if header_lengths is None else np.asarray(header_lengths)
    )
    payloads = None if payload_lengths is None else np.asarray(payload_lengths)
    for name, values in (("header_lengths", headers), ("payload_lengths", payloads)):
        if values is not None and (
            values.shape != lengths.shape or not np.all(np.isfinite(values[valid]))
            or np.any(values[valid] != np.rint(values[valid]))
        ):
            raise ValueError(f"{name}必须是与输入同形状的整数字节数组")
    if np.any(headers[valid] < 54) or np.any(headers[valid] > lengths[valid]):
        raise ValueError("逐包头长必须位于[54,帧长]；请检查真实头部元数据或默认假设")
    if payloads is not None:
        if np.any(payloads[valid] < 0) or np.any(payloads[valid] + headers[valid] > lengths[valid]):
            raise ValueError("逐包载荷长度与帧长/头长不一致")
        if np.any(lengths[valid] > np.maximum(MIN_FRAME_LENGTH, headers[valid] + payloads[valid])):
            raise ValueError("载荷与头部之外只能是最小帧所需的以太网填充")
    return headers, payloads


def _flow_from_row(lengths, gaps, directions, mask, header_lengths=None, payload_lengths=None) -> Flow:
    """把一个固定窗口转换为数据包对象列表。"""

    count = int(np.sum(mask))
    packets = [
        Packet(
            float(lengths[index]),
            float(gaps[index]),
            int(directions[index]),
            int(index),
            header_bytes=54 if header_lengths is None else int(header_lengths[index]),
            payload_bytes=None if payload_lengths is None else int(payload_lengths[index]),
        )
        for index in range(count)
    ]
    return Flow(packets)


def validate_packet_inputs(lengths, gaps, directions, mask, *, config,
                           header_lengths=None, payload_lengths=None):
    """Vectorized packet preflight shared by dry-run, fitting and application.

    Preserve observed inbound lengths, including VLAN frames. The 1514-byte
    feature divisor and outbound transformation limit remain unchanged.
    """
    lengths, gaps, directions, valid = _validate_arrays(lengths, gaps, directions, mask)
    headers, payloads = _packet_metadata(lengths, valid, config, header_lengths, payload_lengths)
    unsupported = valid & (directions == 1) & (lengths > MAX_FRAME_LENGTH)
    if np.any(unsupported):
        row, column = np.argwhere(unsupported)[0]
        raise ValueError(
            f"Glimpse上行帧长超过1514：row={row}, packet={column}, length={lengths[row, column]}；"
            "需要明确的协议头部/MTU模型，不能静默截断或扩展动作上限"
        )
    return lengths, gaps, directions, valid, headers, payloads


def _flow_to_row(flow: Flow):
    """把完整事件流投影回分类器观察的前 32 个事件。"""

    lengths = np.zeros(EVENT_COUNT, dtype=np.float64)
    gaps = np.zeros(EVENT_COUNT, dtype=np.float64)
    directions = np.zeros(EVENT_COUNT, dtype=np.int8)
    mask = np.zeros(EVENT_COUNT, dtype=bool)
    for position, packet in enumerate(flow.packets[:EVENT_COUNT]):
        lengths[position] = packet.length
        gaps[position] = packet.gap if position else 0.0
        directions[position] = packet.direction
        mask[position] = True
    return lengths, gaps, directions, mask


def _flow_features(flow: Flow) -> np.ndarray:
    row = _flow_to_row(flow)
    return make_features(*(value.reshape(1, -1) for value in row))[0]


class IsolationForestRewardModel:
    """论文算法1及公式(5)的单设备奖励模型。"""

    def __init__(self, seed: int):
        self.forest = IsolationForest(
            n_estimators=IF_TREE_COUNT,
            max_samples=IF_MAX_SAMPLES,
            random_state=int(seed),
        )

    def fit(self, own_device_features: np.ndarray) -> None:
        """只能传入一台设备自己的训练流量；类别筛选由上层严格完成。"""

        values = np.asarray(own_device_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
            raise ValueError("奖励模型训练输入必须是 [N,64]")
        if len(values) < 2:
            raise ValueError("每台设备至少需要两条训练流")
        # sklearn 会自动使用 min(256,N)，与算法1一致。
        self.forest.set_params(max_samples=min(IF_MAX_SAMPLES, len(values)))
        self.forest.fit(values)
        self.forest.__dict__.pop("_glimpse_tree_arrays", None)

    @staticmethod
    def _normalizer() -> float:
        # 论文公式(5)按 TN=100 写成 2H(TN-1)-2(TN-1)/TN。
        harmonic = math.log(IF_TREE_COUNT - 1) + _EULER_GAMMA_FROM_PAPER
        return 2.0 * harmonic - 2.0 * (IF_TREE_COUNT - 1) / IF_TREE_COUNT

    def reward(self, feature_vector: np.ndarray) -> float:
        """逐树计算根到叶的边数，再严格代入论文公式(5)。"""

        vector = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
        if vector.shape != (FEATURE_COUNT,):
            raise ValueError("奖励输入必须为 64 维")
        # sklearn的decision_path会为每棵树构造一个稀疏矩阵。算法2每次动作只
        # 评估一个向量，直接沿树数组走到叶节点与其结果完全相同，却快得多。
        compiled = getattr(self.forest, "_glimpse_tree_arrays", None)
        if compiled is None:
            compiled = tuple(
                (
                    tree.tree_.children_left,
                    tree.tree_.children_right,
                    tree.tree_.feature,
                    tree.tree_.threshold,
                )
                for tree in self.forest.estimators_
            )
            self.forest._glimpse_tree_arrays = compiled
        path_lengths = []
        for children_left, children_right, features, thresholds in compiled:
            node = depth = 0
            while children_left[node] != -1:
                node = (
                    children_left[node]
                    if vector[features[node]] <= thresholds[node]
                    else children_right[node]
                )
                depth += 1
            path_lengths.append(depth)
        return float(2.0 ** (-np.mean(path_lengths) / self._normalizer()))


class TrafficManipulator:
    """在Packet对象上执行padding、TCP分段与delay。

    论文没有给出 DDPG 连续输出到离散动作的编码。Seq64-Oracle 固定采用一个
    公开、可审计的适配：Actor 前3维作为动作分数，argmax选择动作；第4维
    映射为动作强度。该编码属于适配实现，而不是论文声称的参数。

    分片动作具体采用TCP载荷分段模型；各段独立计入完整头部和以太网填充。
    """

    def __init__(self, config: GlimpseConfig):
        self.config = config

    @staticmethod
    def _delay_outbound(flow: Flow, position: int, delay: float) -> None:
        """在时间轴上保持上行FIFO，下行沿用观测时间，再重建双向序列。

        该离线模型不预测服务器对延迟/ACK的响应。后续上行包可以排到同一
        发送时刻（未模拟链路序列化耗时），但不能越过队首上行包。
        """

        times = flow.start_time + np.cumsum(
            [packet.gap for packet in flow.packets], dtype=np.float64
        )
        times[position] += delay
        previous_outbound = times[position]
        for index in range(position + 1, len(flow.packets)):
            if flow.packets[index].direction == 1:
                times[index] = max(times[index], previous_outbound)
                previous_outbound = times[index]

        # 下行包可先于被扣留的上行包出现。相同时刻保持原顺序，以保持
        # 上行FIFO并避免无延迟时任意改变包序。
        order = np.argsort(times, kind="stable")
        flow.packets = [flow.packets[int(index)] for index in order]
        times = times[order]
        flow.start_time = float(times[0])
        for index, packet in enumerate(flow.packets):
            packet.gap = float(times[index] - times[index - 1]) if index else 0.0

    @staticmethod
    def _decode(action: np.ndarray, packet: Packet, config: GlimpseConfig):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIMENSION,):
            raise ValueError("v2 Actor动作必须为4维（三种动作分数和强度）")
        if not np.all(np.isfinite(action)):
            raise ValueError("动作不能含NaN/Inf")
        action_index = int(np.argmax(action[:3]))
        strength = float(np.clip((action[3] + 1.0) / 2.0, 0.0, 1.0))
        if action_index == 0:
            value = PADDING_MIN_BYTES + strength * (
                PADDING_MAX_BYTES - PADDING_MIN_BYTES
            )
        elif action_index == 1:
            value = 2 + int(round(strength * (config.max_fragments - 2)))
        else:
            value = strength * config.max_delay_seconds
        return action_index, value

    def apply(self, flow: Flow, source_index: int, action: np.ndarray):
        """修改一个原始上行包，并返回新流、开销和所选动作名。"""

        result = flow.clone()
        if self.config.shared_budget is not None and result.budget_origin is None:
            result.budget_origin = (sum(p.length for p in flow.packets), len(flow.packets),
                                   flow.start_time + sum(p.gap for p in flow.packets))
        position = result.locate_source(source_index)
        packet = result.packets[position]
        if packet.direction != 1:
            raise ValueError("Glimpse 只允许修改 forward/outbound PDU")
        original_sources = result.original_source_indices()
        original_port = packet.destination_port
        original_total_bytes = sum(p.length for p in result.packets)
        action_index, value = self._decode(action, packet, self.config)
        action_name = ACTION_NAMES[action_index]
        cost = FlowCost()
        cost.action_counts[action_name] = 1

        if self.config.shared_budget is not None and self.config.budget_action_mode == "project":
            # Reduce only the chosen action's magnitude, never choose a different
            # action or alter actor scores/rewards. Training and inference share this.
            limits = self.config.shared_budget
            source_bytes, source_count, source_end = result.budget_origin
            remaining_bytes = max(0, int(limits['max_added_bytes'] - (original_total_bytes - source_bytes)))
            remaining_packets = max(0, limits['max_added_packets'] - (len(result.packets) - source_count))
            original_value = value
            if action_index == 0:
                native_padding = min(int(round(value)), int(MAX_FRAME_LENGTH) - packet.header_bytes - packet.payload_bytes)
                # Replacing Ethernet minimum-frame padding consumes no wire bytes.
                value = min(native_padding, remaining_bytes + packet.link_padding_bytes)
                cost.budget_projected_actions = int(value < native_padding)
            elif action_index == 1:
                native_count = min(int(value), packet.payload_bytes)
                value = min(native_count, remaining_packets + 1)
                while value >= 2:
                    chunk, remainder = divmod(packet.payload_bytes, value)
                    wire_bytes = sum(max(MIN_FRAME_LENGTH, packet.header_bytes + chunk + int(i < remainder)) for i in range(value))
                    if wire_bytes - packet.length <= remaining_bytes:
                        break
                    value -= 1
                cost.budget_projected_actions = int(value < native_count)
            else:
                packet_time = result.start_time + sum(p.gap for p in result.packets[:position+1])
                value = min(value, max(0., source_end + limits['max_wire_delay_seconds'] - packet_time))
                cost.budget_projected_actions = int(value < original_value)
            if cost.budget_projected_actions and value < (2 if action_index == 1 else 1e-12):
                cost.budget_rejected_actions = 1

        if action_index == 0:
            requested = int(round(value))
            old_length, old_link_padding = packet.length, packet.link_padding_bytes
            added = min(requested, int(MAX_FRAME_LENGTH) - packet.header_bytes - packet.payload_bytes)
            packet.payload_bytes += added
            packet.padding_bytes += added
            packet.length = max(old_length, packet.header_bytes + packet.payload_bytes)
            cost.added_payload_bytes = added
            cost.added_link_padding_bytes = packet.link_padding_bytes - old_link_padding
            cost.added_bytes = packet.length - old_length

        elif action_index == 1:
            # 每段至少携带1字节载荷；空载荷/1字节包不伪造可分片的业务数据。
            fragment_count = min(int(value), packet.payload_bytes)
            if fragment_count >= 2:
                chunk, remainder = divmod(packet.payload_bytes, fragment_count)
                fragments = []
                offset = 0
                padding_start = packet.payload_bytes - packet.padding_bytes
                for fragment_index in range(fragment_count):
                    payload = chunk + int(fragment_index < remainder)
                    fragment = deepcopy(packet)
                    fragment.payload_bytes = payload
                    fragment.payload_offset = packet.payload_offset + offset
                    fragment.padding_bytes = max(0, offset + payload - max(offset, padding_start))
                    fragment.length = max(MIN_FRAME_LENGTH, packet.header_bytes + payload)
                    fragment.gap = packet.gap if fragment_index == 0 else 0.0
                    fragment.fragment_index = fragment_index
                    fragment.fragment_count = fragment_count
                    fragments.append(fragment)
                    offset += payload
                result.packets[position : position + 1] = fragments
                cost.added_fragments = fragment_count - 1
                cost.added_header_bytes = (fragment_count - 1) * packet.header_bytes
                cost.added_link_padding_bytes = sum(p.link_padding_bytes for p in fragments) - packet.link_padding_bytes
                cost.added_bytes = sum(p.length for p in fragments) - packet.length
                if sum(p.payload_bytes for p in fragments) != packet.payload_bytes:
                    raise AssertionError("TCP分段未保持载荷总字节数")

        else:
            delay = float(value)
            self._delay_outbound(result, position, delay)
            cost.added_delay = delay

        if packet.destination_port != original_port:
            raise AssertionError("TCP建立后不得修改目的端口")
        if result.original_source_indices() != original_sources:
            raise AssertionError("变形删除或伪造了原始业务包")
        if cost.added_bytes != sum(p.length for p in result.packets) - original_total_bytes:
            raise AssertionError("开销必须等于完整事件流的新增帧字节数")
        if cost.added_bytes != cost.added_payload_bytes + cost.added_header_bytes + cost.added_link_padding_bytes:
            raise AssertionError("字节开销分解不一致")
        if self.config.shared_budget is not None:
            # Accept or reject the native action as a whole. No replacement action,
            # changed policy scores, fragment resizing, or loss/reward modification.
            limits = self.config.shared_budget
            source_bytes, source_count, source_end = result.budget_origin
            if (sum(p.length for p in result.packets) - source_bytes > limits["max_added_bytes"]
                or len(result.packets) - source_count > limits["max_added_packets"]
                or result.start_time + sum(p.gap for p in result.packets) - source_end > limits["max_wire_delay_seconds"] + 1e-9):
                if self.config.budget_action_mode == 'project':
                    raise AssertionError('Glimpse action projection exceeded physical budget')
                unchanged = flow.clone()
                unchanged.budget_origin = result.budget_origin
                return unchanged, FlowCost(budget_rejected_actions=1), action_name
        return result, cost, action_name


class DDPGMorphAgent:
    """用 TensorFlow-Agents 实现论文算法2及公式(3)(4)。

    论文没有报告经验池容量、探索噪声及target更新周期，三者都由
    :class:`GlimpseConfig` 显式传入。TF-Agents的collect policy加入OU探索，
    正式推理使用不带噪声的policy。``training=False``只构建推理网络。
    """

    def __init__(self, config: GlimpseConfig, *, training: bool = True):
        self.config = config
        self._training_enabled = bool(training)
        self._build_agent()

    def _build_agent(self) -> None:
        """延迟导入重型依赖，使特征和包操作仍可单独检查。"""

        # TF-Agents 0.19配套TensorFlow 2.15使用Keras 2。
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        try:
            import tensorflow as tf
            from tf_agents.agents.ddpg import (
                actor_network,
                critic_network,
                ddpg_agent,
            )
            from tf_agents.replay_buffers import tf_uniform_replay_buffer
            from tf_agents.specs import tensor_spec
            from tf_agents.trajectories import time_step as ts
            from tf_agents.trajectories import trajectory
        except ImportError as exc:
            raise RuntimeError(
                "DDPG必须使用TensorFlow-Agents；请在Python 3.11环境执行 "
                "pip install -r requirements.txt"
            ) from exc

        tf.keras.utils.set_random_seed(self.config.seed)
        observation_spec = tensor_spec.TensorSpec(
            (FEATURE_COUNT,), tf.float32, name="seq64_features"
        )
        action_spec = tensor_spec.BoundedTensorSpec(
            (ACTION_DIMENSION,),
            tf.float32,
            minimum=-np.ones(ACTION_DIMENSION, dtype=np.float32),
            maximum=np.ones(ACTION_DIMENSION, dtype=np.float32),
            name="three_action_scores_and_strength",
        )
        actor = actor_network.ActorNetwork(
            observation_spec, action_spec, fc_layer_params=DDPG_HIDDEN
        )
        critic = critic_network.CriticNetwork(
            (observation_spec, action_spec), joint_fc_layer_params=DDPG_HIDDEN
        )
        optimizer = lambda: tf.keras.optimizers.Adam(  # noqa: E731
            learning_rate=DDPG_LEARNING_RATE
        )
        agent = ddpg_agent.DdpgAgent(
            ts.time_step_spec(observation_spec),
            action_spec,
            actor_network=actor,
            critic_network=critic,
            actor_optimizer=optimizer(),
            critic_optimizer=optimizer(),
            ou_stddev=self.config.exploration_noise_stddev,
            ou_damping=self.config.exploration_noise_damping,
            # 论文：target=rho*target+(1-rho)*main；TF-Agents的tau恰为1-rho。
            target_update_tau=1.0 - self.config.rho,
            target_update_period=self.config.target_update_period,
            dqda_clipping=None,
            # TF-Agents内部据此计算Critic的均方贝尔曼误差（论文公式3）。
            td_errors_loss_fn=tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            gamma=self.config.gamma,
            reward_scale_factor=1.0,
            train_step_counter=tf.Variable(0, dtype=tf.int64),
        )
        agent.initialize()

        self._tf, self._ts, self._trajectory = tf, ts, trajectory
        self._agent, self._actor = agent, actor
        # 算法2输出theta_targ；训练采集仍通过main actor的collect_policy。
        self._target_actor = agent._target_actor_network
        self._replay_buffer = self._dataset_iterator = None
        if self._training_enabled:
            replay = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=agent.collect_data_spec,
                batch_size=1,
                max_length=self.config.replay_buffer_capacity,
            )
            dataset = replay.as_dataset(
                sample_batch_size=DDPG_BATCH_SIZE, num_steps=2
            )
            self._replay_buffer = replay
            self._dataset_iterator = iter(dataset)

    def _observation(self, state: np.ndarray):
        """把一个64维状态变成TF-Agents要求的batch张量。"""

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(state)):
            raise ValueError("DDPG状态必须是有限的64维向量")
        return self._tf.convert_to_tensor(state.reshape(1, -1), self._tf.float32)

    def _observations(self, states: np.ndarray):
        """把多条64维状态组成一个batch；仅用于等价地加速确定性策略推理。"""

        states = np.asarray(states, dtype=np.float32)
        if (
            states.ndim != 2
            or states.shape[1] != FEATURE_COUNT
            or not np.all(np.isfinite(states))
        ):
            raise ValueError("DDPG批量状态必须是有限的(N, 64)矩阵")
        return self._tf.convert_to_tensor(states, self._tf.float32)

    def start(self, state: np.ndarray):
        """构造一条新流的FIRST time step。"""

        return self._ts.restart(self._observation(state), batch_size=1)

    def collect_action(self, time_step):
        """训练期由带OU噪声的collect policy生成动作。"""

        step = self._agent.collect_policy.action(time_step)
        return np.asarray(step.action.numpy()[0], dtype=np.float32), step

    def policy_action(self, state: np.ndarray) -> np.ndarray:
        """Oracle推理期使用确定性Actor，不加入探索噪声。"""

        return self.policy_actions(np.asarray(state).reshape(1, -1))[0]

    def policy_actions(self, states: np.ndarray) -> np.ndarray:
        """部署目标Actor(theta_targ)，不调用绑定主Actor的agent.policy。"""

        observations = self._observations(states)
        time_step = self._ts.restart(observations, batch_size=len(states))
        actions, _ = self._target_actor(
            observations, step_type=time_step.step_type, training=False
        )
        return np.asarray(actions.numpy(), dtype=np.float32)

    def next_time_step(self, state: np.ndarray, reward: float, terminal: bool):
        """按是否达到奖励阈值构造MID或LAST time step。"""

        reward_tensor = self._tf.convert_to_tensor([reward], self._tf.float32)
        if terminal:
            return self._ts.termination(self._observation(state), reward_tensor)
        return self._ts.transition(
            self._observation(state),
            reward_tensor,
            discount=self._tf.convert_to_tensor([1.0], self._tf.float32),
        )

    def remember(self, current_time_step, action_step, next_time_step) -> None:
        """将算法2的一条(f,m,r,f')交互放入TF均匀经验池。"""

        if self._replay_buffer is None:
            raise RuntimeError("推理模式不能写入经验池")
        experience = self._trajectory.from_transition(
            current_time_step, action_step, next_time_step
        )
        self._replay_buffer.add_batch(experience)

    def update(self) -> float:
        """从经验池采样32条序列，执行一次Actor/Critic更新。"""

        if self._dataset_iterator is None:
            raise RuntimeError("推理模式不能训练")
        experience, _ = next(self._dataset_iterator)
        return float(self._agent.train(experience).extra.critic_loss.numpy())

    def main_actor_weights(self) -> tuple[np.ndarray, ...]:
        """主Actor用于采集经验；本方法仅供训练检查。"""

        return tuple(np.asarray(value) for value in self._actor.get_weights())

    def target_actor_weights(self) -> tuple[np.ndarray, ...]:
        """导出论文算法2指定的最终目标Actor。"""

        return tuple(np.asarray(value) for value in self._target_actor.get_weights())

    def set_target_actor_weights(self, weights: Sequence[np.ndarray]) -> None:
        """恢复到推理实际使用的目标Actor。"""

        self._target_actor.set_weights([np.asarray(value) for value in weights])


class GlimpseSeq64OracleV2:
    """每类一个IF奖励模型和Actor的完整Oracle策略。"""

    def __init__(self, config: GlimpseConfig):
        self.config = config
        self.delay_model = DELAY_MODEL
        self.policy_format = POLICY_FORMAT
        self.fragmentation_model = FRAGMENTATION_MODEL
        self.device_policies: dict[int, _DevicePolicy] = {}
        self.training_log: dict[int, dict[str, object]] = {}
        # 推理Agent是可重建缓存，不写入pickle，策略文件只保存Actor权重。
        self._runtime_agents: dict[int, DDPGMorphAgent] = {}

    @classmethod
    def fit(
        cls,
        lengths,
        gaps,
        directions,
        mask,
        labels,
        *,
        config: GlimpseConfig,
        train_indices: Sequence[int] | None = None,
        header_lengths=None,
        payload_lengths=None,
        progress: bool = False,
    ) -> "GlimpseSeq64OracleV2":
        """离线训练算法1和算法2；只有 ``train_indices`` 会参与拟合。"""

        lengths, gaps, directions, valid, headers, payloads = validate_packet_inputs(
            lengths, gaps, directions, mask, config=config,
            header_lengths=header_lengths, payload_lengths=payload_lengths,
        )
        labels = np.asarray(labels, dtype=int)
        if labels.shape != (len(lengths),):
            raise ValueError("labels 必须与流数量一致")
        indices = (
            np.arange(len(lengths), dtype=int)
            if train_indices is None
            else np.asarray(train_indices, dtype=int)
        )
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(lengths)):
            raise ValueError("train_indices 非法或为空")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("train_indices 不能包含重复下标")

        policy = cls(config)
        all_features = make_features(lengths, gaps, directions, valid)
        all_flows = [
            _flow_from_row(
                lengths[i], gaps[i], directions[i], valid[i], headers[i],
                None if payloads is None else payloads[i],
            )
            for i in range(len(lengths))
        ]

        class_labels = sorted(np.unique(labels[indices]).tolist())
        for class_offset, label in enumerate(class_labels):
            class_indices = indices[labels[indices] == label]
            if len(class_indices) < 2:
                raise ValueError(f"设备 {label} 的训练流不足两条")
            progress_label = (
                f"[Glimpse][device {class_offset + 1}/{len(class_labels)}: {label}]"
            )
            if progress:
                print(
                    f"{progress_label} 开始训练（{len(class_indices)} 条训练流）",
                    flush=True,
                )
            # 算法1：这里只把当前设备自己的特征传入IF。
            reward_model = IsolationForestRewardModel(
                config.seed + 1009 * (class_offset + 1)
            )
            reward_model.fit(all_features[class_indices])

            trainer_config = GlimpseConfig(
                **{**asdict(config), "seed": config.seed + 7919 * (class_offset + 1)}
            )
            trainer = DDPGMorphAgent(trainer_config)
            manipulator = TrafficManipulator(trainer_config)
            interactions = updates = 0
            projected_actions = rejected_actions = 0
            losses: list[float] = []
            final_rewards: list[float] = []
            rng = np.random.default_rng(trainer_config.seed)

            # 算法2：随机取训练流；逐个forward PDU收集(f,m,r,f')经验。
            for iteration in range(config.training_iterations):
                chosen = int(rng.choice(class_indices))
                flow = all_flows[chosen].clone()
                source_indices = flow.outbound_source_indices()[:MAX_ACTIONS_PER_FLOW]
                if len(source_indices) <= 2:
                    warnings.warn(
                        f"设备 {label} 的流只有 {len(source_indices)} 个可修改上行包，"
                        "论文指出短流防御效果有限",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                last_reward = reward_model.reward(_flow_features(flow))
                current_time_step = trainer.start(_flow_features(flow))
                for ordinal, source_index in enumerate(source_indices):
                    action, action_step = trainer.collect_action(current_time_step)
                    flow, action_cost, _ = manipulator.apply(flow, source_index, action)
                    projected_actions += action_cost.budget_projected_actions
                    rejected_actions += action_cost.budget_rejected_actions
                    next_state = _flow_features(flow)
                    last_reward = reward_model.reward(next_state)
                    done = bool(
                        last_reward > config.reward_threshold
                        or ordinal == len(source_indices) - 1
                    )
                    next_time_step = trainer.next_time_step(
                        next_state, last_reward, terminal=done
                    )
                    trainer.remember(current_time_step, action_step, next_time_step)
                    interactions += 1

                    # 论文第V-B节：每100次interaction更新一次主网络。
                    if interactions % DDPG_UPDATE_INTERVAL == 0:
                        losses.append(trainer.update())
                        updates += 1
                    current_time_step = next_time_step
                    if last_reward > config.reward_threshold:
                        break
                final_rewards.append(float(last_reward))
                if progress:
                    _print_training_progress(
                        progress_label,
                        iteration + 1,
                        config.training_iterations,
                        final_rewards[-1],
                    )

            policy.device_policies[int(label)] = _DevicePolicy(
                reward_model.forest,
                trainer.target_actor_weights(),
            )
            policy.training_log[int(label)] = {
                "own_device_training_flows": int(len(class_indices)),
                "interactions": int(interactions),
                "network_updates": int(updates),
                "budget_projected_actions": int(projected_actions),
                "budget_rejected_actions": int(rejected_actions),
                "critic_msbe": losses,
                "final_reward_mean": float(np.mean(final_rewards)),
                "deployed_actor": "target",
            }
        return policy

    def _require_current_policy_format(self) -> None:
        if (
            getattr(self, "policy_format", None) != POLICY_FORMAT
            or getattr(self, "delay_model", None) != DELAY_MODEL
            or getattr(self, "fragmentation_model", None) != FRAGMENTATION_MODEL
        ):
            raise ValueError(
                "策略的动作空间、Actor或流量模型不是当前v2格式；请重新训练"
            )

    def _inference_agent(self, label: int) -> DDPGMorphAgent:
        """按需恢复一台设备的Actor，并在批量推理期间复用。"""

        if int(label) not in self.device_policies:
            raise KeyError(f"没有设备标签 {label} 的策略")
        label = int(label)
        if label not in self._runtime_agents:
            agent = DDPGMorphAgent(self.config, training=False)
            agent.set_target_actor_weights(self.device_policies[label].target_actor_weights)
            self._runtime_agents[label] = agent
        return self._runtime_agents[label]

    def transform_flow(self, flow: Flow, label: int):
        """Oracle推理：每次动作都可重新查看当前完整32事件特征。"""

        self._require_current_policy_format()
        agent = self._inference_agent(label)
        forest = self.device_policies[int(label)].forest
        reward_model = IsolationForestRewardModel(self.config.seed)
        reward_model.forest = forest
        manipulator = TrafficManipulator(self.config)
        result = flow.clone()
        total = FlowCost()

        source_indices = result.outbound_source_indices()[:MAX_ACTIONS_PER_FLOW]
        if len(source_indices) <= 2:
            warnings.warn(
                f"测试流只有 {len(source_indices)} 个可修改上行包，防御空间有限",
                RuntimeWarning,
                stacklevel=2,
            )
        for source_index in source_indices:
            state = _flow_features(result)
            action = agent.policy_action(state)
            result, cost, action_name = manipulator.apply(result, source_index, action)
            total.accumulate(cost)
            if reward_model.reward(_flow_features(result)) > self.config.reward_threshold:
                break
        return result, total

    def __getstate__(self):
        """pickle时丢弃可由权重重建的TensorFlow运行时对象。"""

        state = self.__dict__.copy()
        state["_runtime_agents"] = {}
        return state

    def __setstate__(self, state):
        """加载旧策略后初始化空的推理缓存。"""

        self.__dict__.update(state)
        self._runtime_agents = {}

    def save(self, path: str | Path) -> None:
        """保存完整策略；pickle文件只应从可信来源加载。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "GlimpseSeq64OracleV2":
        with Path(path).open("rb") as stream:
            try:
                value = pickle.load(stream)
            except ModuleNotFoundError as exc:
                raise ValueError("策略文件不属于独立v2包；请用v2重新训练") from exc
        if not isinstance(value, GlimpseSeq64OracleV2):
            raise TypeError("文件不是 GlimpseSeq64OracleV2 策略")
        value._require_current_policy_format()
        return value


def apply_policy(policy, lengths, gaps, directions, mask, labels, *, header_lengths=None, payload_lengths=None):
    """批量应用策略，返回数组、逐流开销和可审计元数据。"""

    if not isinstance(policy, GlimpseSeq64OracleV2):
        raise TypeError("policy 必须是 GlimpseSeq64OracleV2")
    policy._require_current_policy_format()
    lengths, gaps, directions, valid, headers, payloads = validate_packet_inputs(
        lengths, gaps, directions, mask, config=policy.config,
        header_lengths=header_lengths, payload_lengths=payload_lengths,
    )
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (len(lengths),):
        raise ValueError("labels 必须与流数量一致")
    out_lengths = np.zeros_like(lengths, dtype=np.float64)
    out_gaps = np.zeros_like(gaps, dtype=np.float64)
    out_directions = np.zeros_like(directions, dtype=np.int8)
    out_mask = np.zeros_like(valid, dtype=bool)
    costs = []

    # 每一轮把属于同一设备的当前状态交给同一个Actor批量前向。
    # 这不会跨流共享动作，也不改变“每次动作后重新观察完整窗口并检查奖励阈值”的
    # Oracle语义；它只是把大量等价的单样本TensorFlow调用合并成少量batch调用。
    results = [
        _flow_from_row(
            lengths[i], gaps[i], directions[i], valid[i], headers[i],
            None if payloads is None else payloads[i],
        )
        for i in range(len(lengths))
    ]
    totals = [FlowCost() for _ in results]
    source_indices = [
        flow.outbound_source_indices()[:MAX_ACTIONS_PER_FLOW] for flow in results
    ]
    active = np.ones(len(results), dtype=bool)
    reward_models: dict[int, IsolationForestRewardModel] = {}
    manipulators: dict[int, TrafficManipulator] = {}

    for index, indices in enumerate(source_indices):
        if len(indices) <= 2:
            warnings.warn(
                f"测试流只含 {len(indices)} 个可修改上行包，防御空间有限",
                RuntimeWarning,
                stacklevel=2,
            )
        if not indices:
            active[index] = False

    for ordinal in range(MAX_ACTIONS_PER_FLOW):
        candidates = np.flatnonzero(
            active
            & np.fromiter(
                (ordinal < len(indices) for indices in source_indices),
                dtype=bool,
                count=len(source_indices),
            )
        )
        if len(candidates) == 0:
            break

        for label in sorted(np.unique(labels[candidates]).tolist()):
            group = candidates[labels[candidates] == label]
            states = np.stack([_flow_features(results[i]) for i in group])
            actions = policy._inference_agent(int(label)).policy_actions(states)

            if int(label) not in reward_models:
                reward_model = IsolationForestRewardModel(policy.config.seed)
                reward_model.forest = policy.device_policies[int(label)].forest
                reward_models[int(label)] = reward_model
                manipulators[int(label)] = TrafficManipulator(policy.config)
            reward_model = reward_models[int(label)]
            manipulator = manipulators[int(label)]

            for index, action in zip(group.tolist(), actions):
                results[index], cost, action_name = manipulator.apply(
                    results[index], source_indices[index][ordinal], action
                )
                total = totals[index]
                total.accumulate(cost)
                if reward_model.reward(_flow_features(results[index])) > policy.config.reward_threshold:
                    active[index] = False

    for index, (transformed, cost) in enumerate(zip(results, totals)):
        row = _flow_to_row(transformed)
        out_lengths[index], out_gaps[index], out_directions[index], out_mask[index] = row
        physical_end=transformed.start_time+sum(p.gap for p in transformed.packets)
        source_end=float(np.sum(gaps[index,:int(np.sum(valid[index]))]))
        delay=max(0.0,physical_end-source_end)
        costs.append(dict(asdict(cost),real_completion_delay_seconds=delay,
                          wire_completion_delay_seconds=delay,cover_tail_seconds=0.0))
    metadata = {
        "method": "Glimpse-Seq64-Oracle-v2",
        "online": False,
        "oracle_full_32_event_window": True,
        "arima_used": False,
        "feature_representation": "32 signed lengths + 32 log gaps",
        "feature_knowledge_ratio": 1.0,
        "actions": list(ACTION_NAMES),
        "policy_format": POLICY_FORMAT,
        "deployed_actor": "target",
        "actor_source": "paper Algorithm 2 output theta_targ (PDF page 6 / printed 1215)",
        "paper_constants": {
            "if_trees": IF_TREE_COUNT,
            "if_max_samples": IF_MAX_SAMPLES,
            "hidden_layers": list(DDPG_HIDDEN),
            "learning_rate": DDPG_LEARNING_RATE,
            "batch_size": DDPG_BATCH_SIZE,
            "update_interval": DDPG_UPDATE_INTERVAL,
            "max_actions_per_flow": MAX_ACTIONS_PER_FLOW,
            "padding_range_bytes": [PADDING_MIN_BYTES, PADDING_MAX_BYTES],
        },
        "adapter_choices": {
            "ddpg_backend": "TensorFlow-Agents",
            "delay_model": DELAY_MODEL,
            "budget_action_mode": policy.config.budget_action_mode,
            "shared_budget": policy.config.shared_budget,
            "first_event_gap_seconds": 0.0,
            "inbound_timestamps": "fixed_observations; server responses not simulated",
            "added_delay_metric": "sum of requested action holds in seconds",
            "action_encoding": "first 3 scores argmax; fourth coordinate is strength",
            "fragmentation_model": FRAGMENTATION_MODEL,
            "frame_header_bytes": policy.config.frame_header_bytes,
            "header_lengths_source": "provided" if header_lengths is not None else "configured_assumption",
            "payload_lengths_source": "provided" if payload_lengths is not None else "frame_length_minus_header_assumption",
            "minimum_fragment_frame_bytes": MIN_FRAME_LENGTH,
            "frame_bytes_exclude": "FCS, preamble, inter-frame gap",
            "byte_cost_scope": "full transformed event list before first-32 truncation",
            "max_delay_seconds": policy.config.max_delay_seconds,
            "max_fragments": policy.config.max_fragments,
            "target_update_period": policy.config.target_update_period,
            "replay_buffer_capacity": policy.config.replay_buffer_capacity,
            "exploration_noise_stddev": policy.config.exploration_noise_stddev,
            "exploration_noise_damping": policy.config.exploration_noise_damping,
        },
    }
    return (
        out_lengths,
        out_gaps,
        out_directions,
        out_mask,
        costs,
        metadata,
    )
