"""Shared trace schema, constraints, validation, and overhead accounting."""

# =============================================================================
# 中文阅读导图：基线共用的“小宪法”
# =============================================================================
# 这里定义的不是算法，而是所有基线必须遵守的常量与校验：
#   OBSERVED_EVENTS=32        分类器观察宽度，也是 arrays 的列数
#   MAX_FRAME_LENGTH/MAX_PADDING/MAX_DELAY/DUMMY_*  单动作物理上限
#   STAT_KEYS                 逐流成本字段（与 method33.STAT_KEYS 同构）
#   EmittedEvent              网络执行层看到的单个事件：长度/gap/方向/是否dummy/
#                             源事件下标/padding 字节/附加延迟
#
# 校验顺序与 pcap.validate_dataset 一致：先形状，再方向/有限性/非负，
# 最后强制 mask 是连续左对齐前缀（否则无法安全地做序列变换）。
# training_view 只索引训练行并重新校验，从结构上防止基线读到 held-out 数据。
#
# 为什么单独一个文件：IoTGAN/iPET/Glimpse/traditional 都要用同一组约束，
# 放在各自实现里迟早会产生“不同基线偷偷用不同 MTU”的不公平比较。
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OBSERVED_EVENTS = 32
SEED = 42
MAX_FRAME_LENGTH = 1514.0
MAX_PADDING = 400.0
MAX_DELAY = 0.20
MAX_DUMMY_GAP = 0.10
DUMMY_MIN_FRAME_LENGTH = 60
DUMMY_MAX_FRAME_LENGTH = 1514
DUMMY_BUDGET_PER_FLOW = 4.0

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
)


@dataclass(frozen=True)
class EmittedEvent:
    """One event handed by a defense session to the network execution layer."""

    length: float
    gap: float
    direction: int
    is_dummy: bool
    source_event_index: int | None
    padding_bytes: float = 0.0
    additional_delay: float = 0.0

    def serializable_tuple(self):
        return (
            self.length,
            self.gap,
            self.direction,
            self.is_dummy,
            self.source_event_index,
            self.padding_bytes,
            self.additional_delay,
        )


def validate_event(length, gap, direction):
    length = float(length)
    gap = float(gap)
    direction = int(direction)
    if not np.isfinite(length) or not np.isfinite(gap):
        raise ValueError("包长和 gap 必须是有限值")
    if length < 0.0 or gap < 0.0:
        raise ValueError("包长和 gap 不能为负")
    if direction not in (-1, 1):
        raise ValueError("真实事件方向必须为 -1 或 +1")
    return length, gap, direction


def validate_array_shapes(lengths, gaps, directions, mask):
    arrays = tuple(np.asarray(value) for value in (lengths, gaps, directions, mask))
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape == arrays[3].shape):
        raise ValueError("lengths/gaps/directions/mask 的形状必须一致")
    if arrays[0].ndim != 2:
        raise ValueError("包序列数组必须是二维数组")
    return arrays


def validate_observation_arrays(lengths, gaps, directions, mask):
    arrays = validate_array_shapes(lengths, gaps, directions, mask)
    if arrays[0].shape[1] != OBSERVED_EVENTS:
        raise ValueError(f"输入必须是 [流数量, {OBSERVED_EVENTS}] 的前缀事件数组")
    return arrays


def validate_dataset(lengths, gaps, directions, mask):
    arrays = validate_array_shapes(lengths, gaps, directions, mask)
    valid = arrays[3].astype(bool)
    if not np.all(np.isin(arrays[2][valid], (-1, 1))):
        raise ValueError("有效包方向必须为 -1 或 +1")
    if not np.all(np.isfinite(arrays[0][valid])) or not np.all(
        np.isfinite(arrays[1][valid])
    ):
        raise ValueError("有效事件中存在非有限包长或 gap")
    if np.any(arrays[0][valid] < 0.0) or np.any(arrays[1][valid] < 0.0):
        raise ValueError("有效事件中存在负包长或 gap")
    for sample_mask in arrays[3]:
        count = int(np.sum(sample_mask))
        if not np.array_equal(np.flatnonzero(sample_mask), np.arange(count)):
            raise ValueError("mask 不是连续前缀，不能安全执行序列变换")
    return arrays


def training_view(lengths, gaps, directions, mask, train_indices, groups=None):
    """Select and validate training rows without reading held-out values."""
    arrays = validate_observation_arrays(lengths, gaps, directions, mask)
    indices = np.asarray(train_indices, dtype=int).reshape(-1)
    if not len(indices):
        raise ValueError("train_indices 不能为空")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("train_indices 不能包含重复行")
    if indices.min() < 0 or indices.max() >= len(arrays[0]):
        raise IndexError("train_indices 越界")
    selected = tuple(value[indices] for value in arrays)
    validate_dataset(*selected)
    training_groups = []
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(arrays[0]):
            raise ValueError("groups 数量与流数量不一致")
        training_groups = sorted(str(value) for value in np.unique(groups[indices]))
    return selected, indices, training_groups


def split_by_capture(labels, groups, train_ratio=0.75, seed=SEED):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.shape != groups.shape or labels.ndim != 1:
        raise ValueError("labels 与 groups 必须是等长一维数组")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须位于 (0, 1)")
    group_labels = {}
    for label, group in zip(labels, groups, strict=True):
        previous = group_labels.setdefault(str(group), label)
        if previous != label:
            raise ValueError("同一个 capture group 不能同时属于多个类别")
    rng = np.random.default_rng(seed)
    train = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        label_groups = np.unique(groups[labels == label]).copy()
        if len(label_groups) < 2:
            raise ValueError(f"类别 {label!r} 少于两个 capture，无法隔离测试集")
        rng.shuffle(label_groups)
        count = max(1, min(len(label_groups) - 1, int(len(label_groups) * train_ratio)))
        train |= (labels == label) & np.isin(groups, label_groups[:count])
    test = ~train
    if set(map(str, groups[train])) & set(map(str, groups[test])):
        raise AssertionError("训练和测试 capture group 发生重叠")
    return train, test


def empty_stats(sample_count):
    return {key: np.zeros(sample_count, dtype=np.float64) for key in STAT_KEYS}


def summarize_overhead(stats):
    """Return per-flow average overhead in the shared result schema."""
    result = {
        "real_packet_padding_bytes": float(np.mean(stats["real_added_bytes"])),
        "real_packet_added_delay_seconds": float(np.mean(stats["real_added_delay"])),
        "dummy_packets": float(np.mean(stats["dummy_count"])),
        "dummy_bytes": float(np.mean(stats["dummy_bytes"])),
        "dummy_added_delay_seconds": float(np.mean(stats["dummy_delay"])),
        "total_added_bytes": float(
            np.mean(stats["real_added_bytes"] + stats["dummy_bytes"])
        ),
        "total_added_delay_seconds": float(
            np.mean(stats["real_added_delay"] + stats["dummy_delay"])
        ),
        "dummy_packets_retained_in_first_32": float(
            np.mean(stats["retained_dummy_count"])
        ),
        "real_packets_displaced_from_first_32": float(
            np.mean(stats["displaced_real_packets"])
        ),
    }
    for prefix in ("outbound", "inbound"):
        result[f"{prefix}_real_padding_bytes"] = float(
            np.mean(stats[f"{prefix}_real_added_bytes"])
        )
        result[f"{prefix}_real_added_delay_seconds"] = float(
            np.mean(stats[f"{prefix}_real_added_delay"])
        )
        result[f"{prefix}_dummy_packets"] = float(
            np.mean(stats[f"{prefix}_dummy_count"])
        )
        result[f"{prefix}_dummy_bytes"] = float(
            np.mean(stats[f"{prefix}_dummy_bytes"])
        )
        result[f"{prefix}_dummy_added_delay_seconds"] = float(
            np.mean(stats[f"{prefix}_dummy_delay"])
        )
    return result


# Compatibility names while the baseline implementations are migrated.
_empty_stats = empty_stats
