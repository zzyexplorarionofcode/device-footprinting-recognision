"""Traditional traffic-transformation baselines on the project's trace format.

This module provides five model-independent baselines through one interface:

``Random``
    Resample an outbound target rate for every inter-event interval.
``Random+``
    Draw a persistent mean rate per flow, or per explicitly supplied deployment
    unit, then jitter the rate around that mean.  Classification labels are
    never used as rate identities.
``Constant Rate``
    Use one fixed outbound slot interval.
``Classic Traffic Morphing``
    Fit and sample a packet-size transition matrix.  The implementation is a
    padding-only, common-target adaptation for this data representation.
``Random Segmentation``
    Approximate endpoint TCP segmentation by splitting an outbound frame's
    estimated payload into randomly sized segments and accounting for repeated
    Ethernet/IPv4/TCP headers.

These are *same-data, same-feature reimplementations*, not official code or
parameter-exact reproductions of the papers/systems that introduced the method
families.  In particular, rate shaping is a finite-trace slot approximation,
dummy events represent protocol-valid TLS cover records rather than raw invalid
TCP packets, and segmentation uses a configurable fixed header estimate.

Except for Random Segmentation, every real event remains present in the full
emitted sequence, in its original order, and its length/gap never decreases.
Random Segmentation preserves application payload and source-event order, but
necessarily replaces one source frame by two or more smaller frames.  All
methods are outbound-only: inbound real events are never padded, delayed, or
segmented.  Outbound dummy events can occur inside otherwise idle intervals.

Every apply function returns the same seven-tuple::

    (lengths, gaps, directions, mask, stats, metadata, audit_rows)

The first five elements are compatible with ``04_train_multiagents_dummy_uap``;
the full shape mirrors the application tuples used by 09/10.  Arrays retain the
input observation width (normally 32).  The audit describes the complete
network event sequence, while the returned arrays contain only its leading
observation window; a real event outside that window is not deleted on network.
"""

# =============================================================================
# 中文阅读导图：5 个“传统”基线，共享同一套七元组协议
# =============================================================================
# 与本包主线的关系：FADE 是“弱化 + 诱导码 + 轮换”，这里是它的对照组。
# 所有方法都只改上行真实包（Random Segmentation 例外，它把上行帧拆成多段），
# 下行真实包永不 padding/delay/分片。
#
# 入口与分支：
#   apply_baseline(name, ...)        分发器：按 METHOD_ALIASES 归一化名称
#   apply_random / apply_random_plus 随机速率；Random+ 为每条流抽一个持续均值再抖动
#   apply_constant_rate              固定上行时隙（20 pps → 50 ms 一个 slot）
#   _apply_rate_shaper               时隙速率整形的公共实现（把事件重排到 slot 网格）
#   fit_classic_traffic_morphing     仅用 outer-train 拟合“包长转移矩阵”：
#                                    对每类取分位数包络 → 目标长度 = max(源, 目标类)
#                                    并受 max_padding 与 MTU 限制 → 按 64B 分箱计数归一
#   apply_classic_traffic_morphing   按转移矩阵抽样目标长度，只做 padding
#   apply_random_segmentation        按 MSS/分段数把上行帧拆成多段，重复计算
#                                    Ethernet/IPv4/TCP 头开销，保留应用层载荷
#   summarize_overhead               把逐流 stats 汇总成 mean/p95/max
#   self_test                        合成数据自检（tests/ 之外的第二道保险）
#
# 复杂语法/工程点：
#   * _TraceEvent/_empty_stats 与 method33 的成本 schema 对齐，故可用同一个
#     evaluator 比较；
#   * 速率整形把“连续时间”近似成离散 slot，这是有限 trace 的工程近似，
#     不是论文原始系统的精确实时调度；
#   * 拟合函数强制 training_indices，测试集统计量永远不会进入策略。
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from experiment_support.metrics import timing_metrics, store_timing

from . import common as uap04

MAX_FRAME_LENGTH = float(uap04.MAX_FRAME_LENGTH)
MAX_PADDING = float(uap04.MAX_PADDING)
MAX_DELAY = float(uap04.MAX_DELAY)
MIN_DUMMY_FRAME_LENGTH = int(uap04.DUMMY_MIN_FRAME_LENGTH)
MAX_DUMMY_PER_FLOW = 4

DEFAULT_TARGET_FRAME_LENGTH = 512.0
DEFAULT_DUMMY_FRAME_LENGTH = 512.0
DEFAULT_CONSTANT_RATE_PPS = 20.0
DEFAULT_RANDOM_RATE_MIN_PPS = 5.0
DEFAULT_RANDOM_RATE_MAX_PPS = 50.0

# Extra per-flow arrays supplement 04's STAT_KEYS.  Existing consumers can keep
# reading the original keys; new evaluators can distinguish scheduled idle-gap
# time from actual duration overhead and inspect segmentation costs.
EXTRA_STAT_KEYS = (
    "dummy_gap_seconds",
    "rate_slots_filled",
    "segmented_source_packets",
    "emitted_real_segments",
    "additional_real_segments",
    "segmentation_header_bytes",
    "segmentation_added_delay",
    "source_payload_bytes_before",
    "source_payload_bytes_after",
)

METHOD_ALIASES = {
    "random": "random",
    "random_rate": "random",
    "random-rate": "random",
    "random+": "random_plus",
    "random_plus": "random_plus",
    "random-rate+": "random_plus",
    "random_rate+": "random_plus",
    "constant": "constant_rate",
    "constant_rate": "constant_rate",
    "constant-rate": "constant_rate",
    "classic_traffic_morphing": "classic_traffic_morphing",
    "classic-traffic-morphing": "classic_traffic_morphing",
    "traffic_morphing": "classic_traffic_morphing",
    "traffic-morphing": "classic_traffic_morphing",
    "random_segmentation": "random_segmentation",
    "random-segmentation": "random_segmentation",
}


@dataclass(frozen=True)
class _TraceEvent:
    length: float
    gap: float
    direction: int
    is_dummy: bool
    source_index: int | None


@dataclass
class ClassicTrafficMorphingPolicy:
    """A fitted padding-only probabilistic packet-size mapping.

    ``transition_matrices[label][source_bin, target_bin]`` is row stochastic.
    The common target is derived only from the fitting subset.  Applying the
    policy is stochastic and therefore still requires an explicit seed.
    """

    size_levels: np.ndarray
    class_labels: tuple[int, ...]
    transition_matrices: dict[int, np.ndarray]
    target_mode: str
    target_label: int | None
    bin_width: int
    quantile_count: int
    max_padding: float
    outbound_packet_counts: dict[int, int]

    def matrix_for(self, label: int) -> np.ndarray:
        try:
            return self.transition_matrices[int(label)]
        except KeyError as error:
            raise ValueError(
                f"Classic Traffic Morphing policy has no source label {label}; "
                f"known labels are {list(self.class_labels)}"
            ) from error


def _empty_stats(sample_count: int) -> dict[str, np.ndarray]:
    stats = uap04._empty_stats(sample_count)
    stats.update(
        {
            key: np.zeros(sample_count, dtype=np.float64)
            for key in EXTRA_STAT_KEYS
        }
    )
    return stats


def _validate_inputs(lengths, gaps, directions, mask, labels=None):
    lengths = np.asarray(lengths)
    gaps = np.asarray(gaps)
    directions = np.asarray(directions)
    mask = np.asarray(mask)
    if lengths.ndim != 2:
        raise ValueError("lengths/gaps/directions/mask must be two-dimensional")
    uap04.validate_dataset(lengths, gaps, directions, mask)
    valid = mask.astype(bool)
    if not np.all(np.isfinite(lengths[valid])) or not np.all(np.isfinite(gaps[valid])):
        raise ValueError("valid packet lengths and gaps must be finite")
    if np.any(lengths[valid] < 0.0) or np.any(lengths[valid] > MAX_FRAME_LENGTH):
        raise ValueError(f"valid frame lengths must be within [0, {MAX_FRAME_LENGTH}]")
    if np.any(gaps[valid] < 0.0):
        raise ValueError("valid packet gaps must be non-negative")
    if labels is not None:
        label_array = np.asarray(labels)
        if label_array.ndim != 1 or len(label_array) != lengths.shape[0]:
            raise ValueError("labels must be a one-dimensional value per flow")
    return lengths, gaps, directions, mask


def _bounded_frame_length(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not MIN_DUMMY_FRAME_LENGTH <= value <= MAX_FRAME_LENGTH:
        raise ValueError(
            f"{name} must be within [{MIN_DUMMY_FRAME_LENGTH}, {MAX_FRAME_LENGTH}]"
        )
    return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return value


def _base_metadata(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "implementation_status": (
            "same-data trace-level reimplementation; not an official or "
            "parameter-exact reproduction"
        ),
        "scope": "outbound_only",
        "real_source_order_preserved": True,
        "network_events_vs_observation_window": (
            "audit counts the complete emitted sequence; returned arrays retain only "
            "the original-width leading window"
        ),
        "dummy_semantics": (
            "an outbound protocol-valid TLS cover-record abstraction, not a raw "
            "invalid TCP packet"
        ),
        "statistics_semantics": {
            "real_added_bytes": "additional network bytes per flow",
            "real_added_delay": "incremental flow-duration delay per flow",
            "dummy_bytes": "all emitted cover bytes, including events outside the window",
            "dummy_delay": "incremental flow-duration delay caused by dummy slots",
            "dummy_gap_seconds": "sum of gaps attached to dummy events",
            "displaced_real_packets": (
                "source packets not fully represented in the leading observation window"
            ),
        },
    }


def _ordered_unique(values: list[int]) -> list[int]:
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def _render_application(
    lengths,
    gaps,
    directions,
    mask,
    events_by_flow: list[list[_TraceEvent]],
    stats: dict[str, np.ndarray],
    metadata: dict[str, Any],
    method_audits: list[dict[str, Any]],
):
    """Render full event streams to the fixed observation window and audit order."""
    sample_count, observed_packets = lengths.shape
    out_lengths = np.zeros_like(lengths, dtype=np.float64)
    out_gaps = np.zeros_like(gaps, dtype=np.float64)
    out_directions = np.zeros_like(directions)
    out_mask = np.zeros_like(mask)
    audit_rows = []

    for sample, events in enumerate(events_by_flow):
        source_count = int(np.sum(mask[sample]))
        store_timing(stats,sample,timing_metrics(gaps[sample,:source_count],
            np.cumsum([e.gap for e in events]) - (float(gaps[sample,0]) if source_count else 0.0),[not e.is_dummy for e in events]),sample_count)
        full_source_ids = [
            int(event.source_index)
            for event in events
            if not event.is_dummy and event.source_index is not None
        ]
        if _ordered_unique(full_source_ids) != list(range(source_count)):
            raise AssertionError("a baseline deleted, duplicated, or reordered source payload")

        retained = events[:observed_packets]
        total_segments = np.bincount(full_source_ids, minlength=source_count)
        retained_ids = [
            int(event.source_index)
            for event in retained
            if not event.is_dummy and event.source_index is not None
        ]
        retained_segments = np.bincount(retained_ids, minlength=source_count)
        fully_retained = int(np.sum(retained_segments == total_segments)) if source_count else 0
        partially_retained = (
            int(np.sum((retained_segments > 0) & (retained_segments < total_segments)))
            if source_count
            else 0
        )

        retained_dummy = 0
        for output_position, event in enumerate(retained):
            out_lengths[sample, output_position] = event.length
            out_gaps[sample, output_position] = event.gap
            out_directions[sample, output_position] = event.direction
            out_mask[sample, output_position] = 1
            retained_dummy += int(event.is_dummy)
        stats["retained_dummy_count"][sample] = retained_dummy
        if retained:out_gaps[sample,0]=0.0
        stats["displaced_real_packets"][sample] = source_count - fully_retained

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
                "retained_dummy_events": retained_dummy,
                "source_packets_fully_retained": fully_retained,
                "source_packets_partially_retained": partially_retained,
                "source_packets_not_fully_in_window": source_count - fully_retained,
                "source_event_order_preserved": True,
                **method_audits[sample],
            }
        )

    return (
        out_lengths,
        out_gaps,
        out_directions,
        out_mask,
        stats,
        metadata,
        audit_rows,
    )


def _padded_length(
    source_length: float,
    target_frame_length: float | None,
    max_padding: float,
) -> float:
    if target_frame_length is None:
        return float(source_length)
    return float(
        min(
            MAX_FRAME_LENGTH,
            source_length + max_padding,
            max(source_length, target_frame_length),
        )
    )


def _apply_rate_shaper(
    method: str,
    lengths,
    gaps,
    directions,
    mask,
    interval_sampler: Callable[[int], float],
    *,
    target_frame_length: float | None,
    dummy_frame_length: float,
    max_padding: float,
    max_dummy_per_flow: int,
    metadata_parameters: dict[str, Any],
):
    lengths, gaps, directions, mask = _validate_inputs(
        lengths, gaps, directions, mask
    )
    if target_frame_length is not None:
        target_frame_length = _bounded_frame_length(
            target_frame_length, "target_frame_length"
        )
    dummy_frame_length = _bounded_frame_length(
        dummy_frame_length, "dummy_frame_length"
    )
    max_padding = float(max_padding)
    if not 0.0 <= max_padding <= MAX_FRAME_LENGTH:
        raise ValueError("max_padding must be between 0 and MAX_FRAME_LENGTH")
    max_dummy_per_flow = int(max_dummy_per_flow)
    if max_dummy_per_flow < 0:
        raise ValueError("max_dummy_per_flow must be non-negative")

    stats = _empty_stats(len(lengths))
    events_by_flow: list[list[_TraceEvent]] = []
    method_audits = []

    for sample in range(len(lengths)):
        valid_positions = np.flatnonzero(mask[sample])
        events: list[_TraceEvent] = []
        emitted_dummy = 0
        intervals = []
        padding_bytes = 0.0
        incremental_delay = 0.0
        raw_dummy_gaps = 0.0

        for source_index, packet_position in enumerate(valid_positions):
            source_gap = float(gaps[sample, packet_position])
            direction = int(directions[sample, packet_position])
            interval = _positive(interval_sampler(sample), "sampled slot interval")
            intervals.append(interval)
            real_gap = source_gap

            # The source gap decides how many nominal cover slots are requested,
            # but the original real-event gap is retained verbatim.  Consequently
            # every cover gap is added time rather than time carved out of an
            # existing gap.  This intentionally preserves the project's strict
            # "only add delay" action domain, at the cost of being a monotone
            # trace surrogate rather than a wall-clock-perfect shaper.
            if source_index > 0 and emitted_dummy < max_dummy_per_flow:
                possible = int(np.floor(max(0.0, source_gap - 1e-12) / interval))
                insert_count = min(possible, max_dummy_per_flow - emitted_dummy)
                for _ in range(insert_count):
                    events.append(
                        _TraceEvent(dummy_frame_length, interval, 1, True, None)
                    )
                    emitted_dummy += 1
                    raw_dummy_gaps += interval
                    stats["dummy_count"][sample] += 1.0
                    stats["dummy_bytes"][sample] += dummy_frame_length
                    stats["dummy_delay"][sample] += interval
                    stats["outbound_dummy_count"][sample] += 1.0
                    stats["outbound_dummy_bytes"][sample] += dummy_frame_length
                    stats["outbound_dummy_delay"][sample] += interval
                    stats["rate_slots_filled"][sample] += 1.0

            source_length = float(lengths[sample, packet_position])
            changed_length = source_length
            changed_gap = real_gap
            if direction == 1:
                changed_length = _padded_length(
                    source_length, target_frame_length, max_padding
                )
                # An outbound real event waits for the next slot.  For the first
                # event, time origin remains unchanged as in 09/10.
                if source_index > 0 and real_gap < interval:
                    changed_gap = interval
                byte_delta = changed_length - source_length
                delay_delta = changed_gap - real_gap
                if byte_delta < -1e-9 or delay_delta < -1e-9:
                    raise AssertionError("rate shaper decreased an outbound real event")
                padding_bytes += byte_delta
                incremental_delay += delay_delta
                stats["real_added_bytes"][sample] += byte_delta
                stats["real_added_delay"][sample] += delay_delta
                stats["outbound_real_added_bytes"][sample] += byte_delta
                stats["outbound_real_added_delay"][sample] += delay_delta

            events.append(
                _TraceEvent(
                    changed_length,
                    changed_gap,
                    direction,
                    False,
                    source_index,
                )
            )

        stats["dummy_gap_seconds"][sample] = raw_dummy_gaps
        events_by_flow.append(events)
        method_audits.append(
            {
                "sampled_slot_interval_mean_seconds": (
                    float(np.mean(intervals)) if intervals else 0.0
                ),
                "sampled_slot_interval_min_seconds": (
                    float(np.min(intervals)) if intervals else 0.0
                ),
                "sampled_slot_interval_max_seconds": (
                    float(np.max(intervals)) if intervals else 0.0
                ),
                "real_padding_bytes": padding_bytes,
                "real_packet_added_delay_seconds": incremental_delay,
                "scheduled_dummy_added_delay_seconds": raw_dummy_gaps,
                "dummy_packets": emitted_dummy,
            }
        )

    metadata = {
        **_base_metadata(method),
        "rate_model": (
            "strictly monotone finite-trace outbound slot approximation; a long "
            "source gap requests cover slots but its real-event gap is not shortened, "
            "short-gap outbound real events wait for the next slot, and no trailing "
            "dummy is emitted after flow end"
        ),
        "inbound_real_events": (
            "length and original gap are unchanged; preceding outbound cover slots "
            "can only shift it later in absolute time"
        ),
        "parameters": {
            "target_frame_length": target_frame_length,
            "dummy_frame_length": dummy_frame_length,
            "max_padding_per_outbound_real_packet": max_padding,
            "max_dummy_per_flow": max_dummy_per_flow,
            **metadata_parameters,
        },
    }
    return _render_application(
        lengths,
        gaps,
        directions,
        mask,
        events_by_flow,
        stats,
        metadata,
        method_audits,
    )


def apply_constant_rate(
    lengths,
    gaps,
    directions,
    mask,
    *,
    rate_packets_per_second: float = DEFAULT_CONSTANT_RATE_PPS,
    target_frame_length: float | None = DEFAULT_TARGET_FRAME_LENGTH,
    dummy_frame_length: float = DEFAULT_DUMMY_FRAME_LENGTH,
    max_padding: float = MAX_PADDING,
    max_dummy_per_flow: int = MAX_DUMMY_PER_FLOW,
):
    """Apply a deterministic fixed-rate outbound slot baseline."""
    rate = _positive(rate_packets_per_second, "rate_packets_per_second")
    interval = 1.0 / rate
    return _apply_rate_shaper(
        "Constant Rate",
        lengths,
        gaps,
        directions,
        mask,
        lambda _sample: interval,
        target_frame_length=target_frame_length,
        dummy_frame_length=dummy_frame_length,
        max_padding=max_padding,
        max_dummy_per_flow=max_dummy_per_flow,
        metadata_parameters={
            "rate_packets_per_second": rate,
            "slot_interval_seconds": interval,
        },
    )


def _validated_rate_range(min_rate_pps: float, max_rate_pps: float):
    minimum = _positive(min_rate_pps, "min_rate_pps")
    maximum = _positive(max_rate_pps, "max_rate_pps")
    if minimum > maximum:
        raise ValueError("min_rate_pps cannot exceed max_rate_pps")
    return minimum, maximum


def apply_random(
    lengths,
    gaps,
    directions,
    mask,
    *,
    seed: int,
    min_rate_pps: float = DEFAULT_RANDOM_RATE_MIN_PPS,
    max_rate_pps: float = DEFAULT_RANDOM_RATE_MAX_PPS,
    target_frame_length: float | None = DEFAULT_TARGET_FRAME_LENGTH,
    dummy_frame_length: float = DEFAULT_DUMMY_FRAME_LENGTH,
    max_padding: float = MAX_PADDING,
    max_dummy_per_flow: int = MAX_DUMMY_PER_FLOW,
):
    """Apply Random Rate: independently resample the slot rate per interval."""
    minimum, maximum = _validated_rate_range(min_rate_pps, max_rate_pps)
    rng = np.random.default_rng(seed)

    def sample_interval(_sample: int) -> float:
        return 1.0 / float(rng.uniform(minimum, maximum))

    return _apply_rate_shaper(
        "Random",
        lengths,
        gaps,
        directions,
        mask,
        sample_interval,
        target_frame_length=target_frame_length,
        dummy_frame_length=dummy_frame_length,
        max_padding=max_padding,
        max_dummy_per_flow=max_dummy_per_flow,
        metadata_parameters={
            "seed": int(seed),
            "rate_distribution": "Uniform(min_rate_pps, max_rate_pps) per interval",
            "min_rate_pps": minimum,
            "max_rate_pps": maximum,
        },
    )


def apply_random_plus(
    lengths,
    gaps,
    directions,
    mask,
    *,
    seed: int,
    mean_unit_ids=None,
    min_mean_rate_pps: float = DEFAULT_RANDOM_RATE_MIN_PPS,
    max_mean_rate_pps: float = DEFAULT_RANDOM_RATE_MAX_PPS,
    log_rate_jitter: float = 0.25,
    target_frame_length: float | None = DEFAULT_TARGET_FRAME_LENGTH,
    dummy_frame_length: float = DEFAULT_DUMMY_FRAME_LENGTH,
    max_padding: float = MAX_PADDING,
    max_dummy_per_flow: int = MAX_DUMMY_PER_FLOW,
):
    """Apply Random Rate+ with a persistent unit- or flow-specific mean rate.

    By default each flow receives its own random mean.  A deployment may pass
    explicit ``mean_unit_ids`` (for example, pseudonymous device-instance IDs)
    to share a mean across that unit's flows.  Classification labels are not an
    implicit fallback because doing so would make the defense label-aware and
    could create a new class-specific rate fingerprint.
    """
    lengths, gaps, directions, mask = _validate_inputs(
        lengths, gaps, directions, mask
    )
    minimum, maximum = _validated_rate_range(
        min_mean_rate_pps, max_mean_rate_pps
    )
    jitter = float(log_rate_jitter)
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("log_rate_jitter must be finite and non-negative")
    rng = np.random.default_rng(seed)

    if mean_unit_ids is None:
        unit_for_sample = np.arange(len(lengths), dtype=np.int64)
        mean_scope = "flow"
    else:
        unit_for_sample = np.asarray(mean_unit_ids)
        if unit_for_sample.ndim != 1 or len(unit_for_sample) != len(lengths):
            raise ValueError("mean_unit_ids must be a one-dimensional value per flow")
        mean_scope = "caller_supplied_deployment_unit"
    unique_units = np.unique(unit_for_sample)
    sampled_means = rng.uniform(minimum, maximum, size=len(unique_units))
    mean_by_unit = {
        unit.item() if isinstance(unit, np.generic) else unit: float(mean)
        for unit, mean in zip(unique_units, sampled_means, strict=True)
    }

    def sample_interval(sample: int) -> float:
        unit = unit_for_sample[sample]
        if isinstance(unit, np.generic):
            unit = unit.item()
        mean = mean_by_unit[unit]
        rate = mean * float(np.exp(rng.normal(0.0, jitter)))
        return 1.0 / float(np.clip(rate, minimum, maximum))

    return _apply_rate_shaper(
        "Random+",
        lengths,
        gaps,
        directions,
        mask,
        sample_interval,
        target_frame_length=target_frame_length,
        dummy_frame_length=dummy_frame_length,
        max_padding=max_padding,
        max_dummy_per_flow=max_dummy_per_flow,
        metadata_parameters={
            "seed": int(seed),
            "persistent_mean_scope": mean_scope,
            "classification_labels_used_for_rate_identity": False,
            "mean_rate_distribution": (
                "Uniform(min_mean_rate_pps, max_mean_rate_pps)"
            ),
            "per_interval_rate_distribution": (
                "clipped mean_rate * exp(Normal(0, log_rate_jitter))"
            ),
            "min_mean_rate_pps": minimum,
            "max_mean_rate_pps": maximum,
            "log_rate_jitter": jitter,
            "persistent_mean_unit_count": int(len(unique_units)),
        },
    )


def _size_levels(bin_width: int) -> np.ndarray:
    bin_width = int(bin_width)
    if bin_width < 1:
        raise ValueError("bin_width must be a positive integer")
    levels = np.arange(0.0, MAX_FRAME_LENGTH, float(bin_width))
    return np.unique(np.append(levels, MAX_FRAME_LENGTH)).astype(np.float64)


def _level_indices(values, levels):
    return np.minimum(np.searchsorted(levels, values, side="left"), len(levels) - 1)


def fit_classic_traffic_morphing(
    lengths,
    directions,
    mask,
    labels,
    *,
    training_indices=None,
    target_label: int | None = None,
    bin_width: int = 64,
    quantile_count: int = 512,
    max_padding: float = MAX_PADDING,
) -> ClassicTrafficMorphingPolicy:
    """Fit a padding-only packet-size transition policy from training flows.

    With ``target_label=None``, the target quantile at each probability is the
    maximum class quantile, producing a common empirical distribution that is
    monotonically reachable before the per-packet padding cap is applied.  With
    an explicit target label, all classes project toward that class; unreachable
    smaller target sizes remain unchanged.
    """
    lengths = np.asarray(lengths)
    directions = np.asarray(directions)
    mask = np.asarray(mask)
    labels = np.asarray(labels)
    if not (lengths.shape == directions.shape == mask.shape) or lengths.ndim != 2:
        raise ValueError("lengths/directions/mask must be equal two-dimensional arrays")
    if labels.ndim != 1 or len(labels) != len(lengths):
        raise ValueError("labels must be a one-dimensional value per flow")
    valid = mask.astype(bool)
    if not np.all(np.isin(directions[valid], (-1, 1))):
        raise ValueError("valid packet directions must be -1 or +1")
    if training_indices is None:
        training_indices = np.arange(len(lengths), dtype=np.int64)
    else:
        training_indices = np.asarray(training_indices, dtype=np.int64)
    if training_indices.ndim != 1 or len(training_indices) == 0:
        raise ValueError("training_indices must select at least one flow")
    if np.any(training_indices < 0) or np.any(training_indices >= len(lengths)):
        raise ValueError("training_indices contains an out-of-range flow index")
    quantile_count = int(quantile_count)
    if quantile_count < 2:
        raise ValueError("quantile_count must be at least 2")
    max_padding = float(max_padding)
    if not 0.0 <= max_padding <= MAX_FRAME_LENGTH:
        raise ValueError("max_padding must be between 0 and MAX_FRAME_LENGTH")

    class_labels = tuple(int(value) for value in np.unique(labels[training_indices]))
    if len(class_labels) < 2:
        raise ValueError("Classic Traffic Morphing requires at least two training labels")
    samples_by_class = {}
    for label in class_labels:
        rows = training_indices[labels[training_indices] == label]
        outbound = mask[rows].astype(bool) & (directions[rows] == 1)
        samples = lengths[rows][outbound].astype(np.float64)
        if len(samples) == 0:
            raise ValueError(f"training label {label} has no outbound packets")
        samples_by_class[label] = samples

    if target_label is not None:
        target_label = int(target_label)
        if target_label not in samples_by_class:
            raise ValueError("target_label is absent from the fitting subset")
    probabilities = (np.arange(quantile_count, dtype=np.float64) + 0.5) / quantile_count
    class_quantiles = {
        label: np.quantile(samples, probabilities, method="higher")
        for label, samples in samples_by_class.items()
    }
    if target_label is None:
        target_quantiles = np.maximum.reduce(list(class_quantiles.values()))
        target_mode = "common_upper_quantile_envelope"
    else:
        target_quantiles = class_quantiles[target_label]
        target_mode = "explicit_target_label"

    levels = _size_levels(bin_width)
    matrices = {}
    for label in class_labels:
        source_quantiles = class_quantiles[label]
        projected_targets = np.maximum(source_quantiles, target_quantiles)
        projected_targets = np.minimum(
            projected_targets, source_quantiles + max_padding
        )
        projected_targets = np.minimum(projected_targets, MAX_FRAME_LENGTH)
        source_bins = _level_indices(source_quantiles, levels)
        target_bins = _level_indices(projected_targets, levels)
        target_bins = np.maximum(source_bins, target_bins)
        matrix = np.zeros((len(levels), len(levels)), dtype=np.float64)
        np.add.at(matrix, (source_bins, target_bins), 1.0)
        for source_bin in range(len(levels)):
            row_sum = float(np.sum(matrix[source_bin]))
            if row_sum == 0.0:
                matrix[source_bin, source_bin] = 1.0
            else:
                matrix[source_bin] /= row_sum
        matrices[label] = matrix

    return ClassicTrafficMorphingPolicy(
        size_levels=levels,
        class_labels=class_labels,
        transition_matrices=matrices,
        target_mode=target_mode,
        target_label=target_label,
        bin_width=int(bin_width),
        quantile_count=quantile_count,
        max_padding=max_padding,
        outbound_packet_counts={
            label: int(len(samples)) for label, samples in samples_by_class.items()
        },
    )


def apply_classic_traffic_morphing(
    lengths,
    gaps,
    directions,
    mask,
    *,
    labels,
    policy: ClassicTrafficMorphingPolicy,
    seed: int,
):
    """Sample a fitted padding-only Classic Traffic Morphing policy."""
    lengths, gaps, directions, mask = _validate_inputs(
        lengths, gaps, directions, mask, labels
    )
    labels = np.asarray(labels)
    if not isinstance(policy, ClassicTrafficMorphingPolicy):
        raise TypeError("policy must be a ClassicTrafficMorphingPolicy")
    rng = np.random.default_rng(seed)
    stats = _empty_stats(len(lengths))
    events_by_flow = []
    method_audits = []

    for sample in range(len(lengths)):
        valid_positions = np.flatnonzero(mask[sample])
        label = int(labels[sample])
        matrix = policy.matrix_for(label)
        events = []
        changed_packets = 0
        added_bytes = 0.0
        for source_index, packet_position in enumerate(valid_positions):
            source_length = float(lengths[sample, packet_position])
            direction = int(directions[sample, packet_position])
            changed_length = source_length
            if direction == 1:
                source_bin = int(_level_indices([source_length], policy.size_levels)[0])
                target_bin = int(
                    rng.choice(len(policy.size_levels), p=matrix[source_bin])
                )
                target = float(policy.size_levels[target_bin])
                changed_length = min(
                    MAX_FRAME_LENGTH,
                    source_length + policy.max_padding,
                    max(source_length, target),
                )
                delta = changed_length - source_length
                if delta < -1e-9:
                    raise AssertionError("Classic Traffic Morphing decreased a packet")
                changed_packets += int(delta > 0.0)
                added_bytes += delta
                stats["real_added_bytes"][sample] += delta
                stats["outbound_real_added_bytes"][sample] += delta
            events.append(
                _TraceEvent(
                    changed_length,
                    float(gaps[sample, packet_position]),
                    direction,
                    False,
                    source_index,
                )
            )
        events_by_flow.append(events)
        method_audits.append(
            {
                "source_label": label,
                "outbound_packets_with_positive_padding": changed_packets,
                "real_padding_bytes": added_bytes,
            }
        )

    metadata = {
        **_base_metadata("Classic Traffic Morphing"),
        "traffic_morphing_model": (
            "empirical quantile coupling converted to a row-stochastic packet-size "
            "transition matrix; outbound packet sizes only; no timing morphing"
        ),
        "source_label_required_at_application": True,
        "parameters": {
            "seed": int(seed),
            "target_mode": policy.target_mode,
            "target_label": policy.target_label,
            "bin_width_bytes": policy.bin_width,
            "quantile_count": policy.quantile_count,
            "max_padding_per_outbound_packet": policy.max_padding,
            "fitted_class_labels": list(policy.class_labels),
            "training_outbound_packet_counts": policy.outbound_packet_counts,
        },
    }
    return _render_application(
        lengths,
        gaps,
        directions,
        mask,
        events_by_flow,
        stats,
        metadata,
        method_audits,
    )


def _random_integer_composition(
    total: int, parts: int, minimum: int, rng: np.random.Generator
) -> np.ndarray:
    remaining = total - parts * minimum
    if remaining < 0:
        raise ValueError("total cannot satisfy the minimum part size")
    if remaining == 0:
        return np.full(parts, minimum, dtype=np.int64)
    weights = rng.dirichlet(np.ones(parts, dtype=np.float64))
    return np.full(parts, minimum, dtype=np.int64) + rng.multinomial(
        remaining, weights
    )


def apply_random_segmentation(
    lengths,
    gaps,
    directions,
    mask,
    *,
    seed: int,
    segmentation_probability: float = 1.0,
    min_segment_payload_bytes: int = 32,
    max_segments_per_packet: int = 4,
    frame_header_bytes: int = 54,
    max_inter_segment_gap_seconds: float = 0.01,
):
    """Approximate random outbound TCP segmentation while conserving payload.

    ``frame_header_bytes`` defaults to Ethernet(14)+IPv4(20)+TCP(20).  The PCAP
    does not retain application write boundaries or exact header/options for all
    packets, so this is a trace-level approximation that needs endpoint socket
    control in a real system.  It is deliberately not represented as dummy
    insertion: emitted segments are real payload-bearing events.
    """
    lengths, gaps, directions, mask = _validate_inputs(
        lengths, gaps, directions, mask
    )
    probability = _probability(
        segmentation_probability, "segmentation_probability"
    )
    min_payload = int(min_segment_payload_bytes)
    max_segments = int(max_segments_per_packet)
    header = int(frame_header_bytes)
    max_segment_gap = float(max_inter_segment_gap_seconds)
    if min_payload < 1:
        raise ValueError("min_segment_payload_bytes must be positive")
    if max_segments < 2:
        raise ValueError("max_segments_per_packet must be at least 2")
    if header < 0 or header >= MAX_FRAME_LENGTH:
        raise ValueError("frame_header_bytes must be within the frame length range")
    if not np.isfinite(max_segment_gap) or not 0.0 <= max_segment_gap <= MAX_DELAY:
        raise ValueError(
            f"max_inter_segment_gap_seconds must be within [0, {MAX_DELAY}]"
        )
    valid_lengths = lengths[mask.astype(bool)]
    if not np.allclose(valid_lengths, np.rint(valid_lengths), atol=1e-9, rtol=0.0):
        raise ValueError("Random Segmentation requires integer-valued captured lengths")

    rng = np.random.default_rng(seed)
    stats = _empty_stats(len(lengths))
    events_by_flow = []
    method_audits = []

    for sample in range(len(lengths)):
        valid_positions = np.flatnonzero(mask[sample])
        events = []
        payload_before = 0
        payload_after = 0
        segmented_packets = 0
        added_segments = 0
        header_overhead = 0.0
        added_delay = 0.0

        for source_index, packet_position in enumerate(valid_positions):
            frame_length = int(round(float(lengths[sample, packet_position])))
            direction = int(directions[sample, packet_position])
            payload = max(0, frame_length - header)
            should_segment = direction == 1 and rng.random() < probability
            feasible_segments = min(max_segments, payload // min_payload)

            if should_segment and feasible_segments >= 2:
                segment_count = int(rng.integers(2, feasible_segments + 1))
                chunks = _random_integer_composition(
                    payload, segment_count, min_payload, rng
                )
                if int(np.sum(chunks)) != payload:
                    raise AssertionError("Random Segmentation did not conserve payload")
                inter_segment_gaps = rng.uniform(
                    0.0, max_segment_gap, size=segment_count - 1
                )
                for segment_index, chunk in enumerate(chunks):
                    event_gap = (
                        float(gaps[sample, packet_position])
                        if segment_index == 0
                        else float(inter_segment_gaps[segment_index - 1])
                    )
                    events.append(
                        _TraceEvent(
                            float(header + int(chunk)),
                            event_gap,
                            1,
                            False,
                            source_index,
                        )
                    )
                extra_headers = float((segment_count - 1) * header)
                segment_delay = float(np.sum(inter_segment_gaps))
                segmented_packets += 1
                added_segments += segment_count - 1
                header_overhead += extra_headers
                added_delay += segment_delay
                payload_before += payload
                payload_after += int(np.sum(chunks))
                stats["real_added_bytes"][sample] += extra_headers
                stats["real_added_delay"][sample] += segment_delay
                stats["outbound_real_added_bytes"][sample] += extra_headers
                stats["outbound_real_added_delay"][sample] += segment_delay
            else:
                events.append(
                    _TraceEvent(
                        float(frame_length),
                        float(gaps[sample, packet_position]),
                        direction,
                        False,
                        source_index,
                    )
                )
                if direction == 1:
                    payload_before += payload
                    payload_after += payload

        emitted_real_segments = sum(not event.is_dummy for event in events)
        stats["segmented_source_packets"][sample] = segmented_packets
        stats["emitted_real_segments"][sample] = emitted_real_segments
        stats["additional_real_segments"][sample] = added_segments
        stats["segmentation_header_bytes"][sample] = header_overhead
        stats["segmentation_added_delay"][sample] = added_delay
        stats["source_payload_bytes_before"][sample] = payload_before
        stats["source_payload_bytes_after"][sample] = payload_after
        events_by_flow.append(events)
        method_audits.append(
            {
                "segmented_outbound_source_packets": segmented_packets,
                "additional_real_segments": added_segments,
                "repeated_header_overhead_bytes": header_overhead,
                "inter_segment_added_delay_seconds": added_delay,
                "outbound_payload_bytes_before": payload_before,
                "outbound_payload_bytes_after": payload_after,
                "outbound_payload_conserved": payload_before == payload_after,
            }
        )

    metadata = {
        **_base_metadata("Random Segmentation"),
        "segmentation_semantics": (
            "outbound source frames are replaced by ordered real payload-bearing "
            "segments; this is not cover insertion and is the sole baseline here "
            "that may reduce an individual emitted frame length"
        ),
        "deployment_requirement": (
            "endpoint/application or socket control; PCAP frame lengths alone cannot "
            "recover exact TCP payload boundaries, options, retransmissions, or TLS "
            "record-to-segment mapping"
        ),
        "parameters": {
            "seed": int(seed),
            "segmentation_probability": probability,
            "min_segment_payload_bytes": min_payload,
            "max_segments_per_packet": max_segments,
            "frame_header_bytes": header,
            "max_inter_segment_gap_seconds": max_segment_gap,
        },
    }
    return _render_application(
        lengths,
        gaps,
        directions,
        mask,
        events_by_flow,
        stats,
        metadata,
        method_audits,
    )


def apply_baseline(
    method: str,
    lengths,
    gaps,
    directions,
    mask,
    *,
    seed: int | None = None,
    labels=None,
    policy: ClassicTrafficMorphingPolicy | None = None,
    **parameters,
):
    """Unified dispatcher for all five traditional baselines.

    Random, Random+, Classic Traffic Morphing, and Random Segmentation require an
    explicit seed.  Constant Rate is deterministic and ignores ``seed``.
    """
    key = str(method).strip().lower().replace(" ", "_")
    try:
        canonical = METHOD_ALIASES[key]
    except KeyError as error:
        raise ValueError(
            f"unknown baseline {method!r}; supported methods are "
            "Random, Random+, Constant Rate, Classic Traffic Morphing, and "
            "Random Segmentation"
        ) from error

    if canonical == "constant_rate":
        return apply_constant_rate(
            lengths, gaps, directions, mask, **parameters
        )
    if seed is None:
        raise ValueError(f"baseline {method!r} requires an explicit seed")
    if canonical == "random":
        return apply_random(
            lengths, gaps, directions, mask, seed=seed, **parameters
        )
    if canonical == "random_plus":
        return apply_random_plus(
            lengths,
            gaps,
            directions,
            mask,
            seed=seed,
            **parameters,
        )
    if canonical == "classic_traffic_morphing":
        if policy is None:
            raise ValueError("Classic Traffic Morphing requires a fitted policy")
        if labels is None:
            raise ValueError("Classic Traffic Morphing requires source labels")
        return apply_classic_traffic_morphing(
            lengths,
            gaps,
            directions,
            mask,
            labels=labels,
            policy=policy,
            seed=seed,
            **parameters,
        )
    if canonical == "random_segmentation":
        return apply_random_segmentation(
            lengths, gaps, directions, mask, seed=seed, **parameters
        )
    raise AssertionError(f"unhandled canonical baseline: {canonical}")


def summarize_overhead(stats: dict[str, np.ndarray]) -> dict[str, float]:
    """Reuse 04's per-flow summary and append baseline-specific quantities."""
    result = uap04.summarize_overhead(stats)
    for key in EXTRA_STAT_KEYS:
        result[key] = float(np.mean(stats[key]))
    return result


def _applications_equal(left, right) -> bool:
    return all(
        np.array_equal(a, b)
        for a, b in zip(left[:4], right[:4], strict=True)
    ) and all(
        np.array_equal(left[4][key], right[4][key]) for key in left[4]
    )


def _assert_application(application, input_shape):
    if len(application) != 7:
        raise AssertionError("baseline application must return seven elements")
    for array in application[:4]:
        if array.shape != input_shape:
            raise AssertionError("baseline changed the observation array shape")
    out_lengths, out_gaps, out_directions, out_mask = application[:4]
    uap04.validate_dataset(out_lengths, out_gaps, out_directions, out_mask)
    valid = out_mask.astype(bool)
    if not np.all(np.isfinite(out_lengths[valid])) or not np.all(
        np.isfinite(out_gaps[valid])
    ):
        raise AssertionError("baseline emitted non-finite features")
    if np.any(out_lengths[valid] < 0.0) or np.any(
        out_lengths[valid] > MAX_FRAME_LENGTH
    ):
        raise AssertionError("baseline emitted an invalid frame length")
    if np.any(out_gaps[valid] < 0.0):
        raise AssertionError("baseline emitted a negative gap")
    if not all(row["source_event_order_preserved"] for row in application[6]):
        raise AssertionError("source event order audit failed")
    if any(np.any(values < -1e-9) for values in application[4].values()):
        raise AssertionError("baseline reported negative overhead")


def self_test() -> dict[str, Any]:
    """Run a deterministic, dependency-light invariant test for all baselines."""
    lengths = np.asarray(
        [
            [180, 900, 220, 700, 300, 0, 0, 0],
            [220, 760, 190, 620, 850, 260, 0, 0],
            [160, 680, 240, 990, 410, 0, 0, 0],
            [200, 820, 270, 560, 920, 330, 0, 0],
        ],
        dtype=np.float64,
    )
    gaps = np.asarray(
        [
            [0.0, 0.31, 0.01, 0.22, 0.03, 0.0, 0.0, 0.0],
            [0.0, 0.18, 0.02, 0.27, 0.01, 0.16, 0.0, 0.0],
            [0.0, 0.24, 0.015, 0.29, 0.04, 0.0, 0.0, 0.0],
            [0.0, 0.21, 0.03, 0.26, 0.02, 0.14, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    directions = np.asarray(
        [
            [1, -1, 1, 1, -1, 0, 0, 0],
            [1, 1, -1, 1, -1, 1, 0, 0],
            [1, -1, 1, 1, -1, 0, 0, 0],
            [1, 1, -1, 1, -1, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    mask = (directions != 0).astype(np.uint8)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    policy = fit_classic_traffic_morphing(
        lengths,
        directions,
        mask,
        labels,
        bin_width=64,
        quantile_count=32,
    )
    applications = {
        "Constant Rate": apply_baseline(
            "Constant Rate", lengths, gaps, directions, mask
        ),
        "Random": apply_baseline(
            "Random", lengths, gaps, directions, mask, seed=101
        ),
        "Random+": apply_baseline(
            "Random+", lengths, gaps, directions, mask, labels=labels, seed=102
        ),
        "Classic Traffic Morphing": apply_baseline(
            "Classic Traffic Morphing",
            lengths,
            gaps,
            directions,
            mask,
            labels=labels,
            policy=policy,
            seed=103,
        ),
        "Random Segmentation": apply_baseline(
            "Random Segmentation", lengths, gaps, directions, mask, seed=104
        ),
    }
    for application in applications.values():
        _assert_application(application, lengths.shape)
    random_plus_metadata = applications["Random+"][5]["parameters"]
    if random_plus_metadata["persistent_mean_scope"] != "flow" or (
        random_plus_metadata["classification_labels_used_for_rate_identity"]
    ):
        raise AssertionError("Random+ implicitly used classification labels")

    repeated = {
        "Random": apply_baseline(
            "Random", lengths, gaps, directions, mask, seed=101
        ),
        "Random+": apply_baseline(
            "Random+", lengths, gaps, directions, mask, labels=labels, seed=102
        ),
        "Classic Traffic Morphing": apply_baseline(
            "Classic Traffic Morphing",
            lengths,
            gaps,
            directions,
            mask,
            labels=labels,
            policy=policy,
            seed=103,
        ),
        "Random Segmentation": apply_baseline(
            "Random Segmentation", lengths, gaps, directions, mask, seed=104
        ),
    }
    for name, replay in repeated.items():
        if not _applications_equal(applications[name], replay):
            raise AssertionError(f"{name} is not deterministic for a fixed seed")

    segmentation = applications["Random Segmentation"]
    if not all(row["outbound_payload_conserved"] for row in segmentation[6]):
        raise AssertionError("Random Segmentation payload conservation failed")
    classic = applications["Classic Traffic Morphing"]
    inbound = (directions == -1) & mask.astype(bool)
    if not np.array_equal(classic[0][inbound], lengths[inbound]):
        raise AssertionError("Classic Traffic Morphing changed inbound lengths")
    if np.any(classic[0][mask.astype(bool)] < lengths[mask.astype(bool)]):
        raise AssertionError("Classic Traffic Morphing decreased a real frame")

    return {
        "passed": True,
        "methods": list(applications),
        "fixed_seed_reproducibility_checked": list(repeated),
        "return_tuple_length": 7,
        "overhead": {
            name: summarize_overhead(application[4])
            for name, application in applications.items()
        },
    }


__all__ = [
    "ClassicTrafficMorphingPolicy",
    "METHOD_ALIASES",
    "apply_baseline",
    "apply_classic_traffic_morphing",
    "apply_constant_rate",
    "apply_random",
    "apply_random_plus",
    "apply_random_segmentation",
    "fit_classic_traffic_morphing",
    "self_test",
    "summarize_overhead",
]


def main():
    parser = argparse.ArgumentParser(
        description="Traditional baseline transformations for the IoT trace format"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run small deterministic invariant tests without training classifiers",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("this is a library module; use --self-test for its minimal test")
    print(json.dumps(self_test(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
