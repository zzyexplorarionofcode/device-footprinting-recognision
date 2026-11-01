"""Traffic features and physical-v5 injection for the adapted iPET port."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OMEGA_SECONDS = 0.1
TOTAL_TIME_SECONDS = 4.0
SILENCE_THRESHOLD_SECONDS = 5.0
FEATURE_ORDER = (
    "incoming_packet_count",
    "incoming_total_bytes",
    "outgoing_packet_count",
    "outgoing_total_bytes",
)


def _slot_index(timestamp: float, omega: float) -> int:
    """Return the stable slot index used by both training and runtime.

    Decimal omega values such as ``0.1`` are not represented exactly by binary
    floats.  Without the small snap-to-integer step, a timestamp that is
    exactly on a slot boundary can be represented as ``2.9999999999999996``
    and accidentally fall into the preceding slot.  The tolerance is kept in
    slot units and is deliberately much smaller than a packet-time interval.
    """

    omega = float(omega)
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("iPET omega必须是有限正数")
    quotient = float(timestamp) / omega
    nearest = round(quotient)
    if np.isclose(quotient, nearest, rtol=1e-12, atol=1e-9):
        quotient = float(nearest)
    return int(np.floor(quotient))


@dataclass(frozen=True)
class TraceEvent:
    timestamp: float
    length: float
    direction: int
    is_dummy: bool
    source_event_index: int | None


@dataclass(frozen=True)
class PerturbationPlan:
    """A shared long perturbation pool plus the row selected for one TLS flow."""

    values: np.ndarray
    start_index: int
    vector_count: int
    slot_count: int


def slotize_flows(
    lengths,
    gaps,
    directions,
    mask,
    *,
    total_time: float = TOTAL_TIME_SECONDS,
    omega: float = OMEGA_SECONDS,
):
    """Adapt packet-event flows to the authors' M_Sequence [slot, 4] tensor."""

    lengths = np.asarray(lengths, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    directions = np.asarray(directions)
    mask = np.asarray(mask, dtype=bool)
    slot_count = int(round(float(total_time) / float(omega)))
    features = np.zeros((len(lengths), slot_count, 4), dtype=np.float32)
    silences = np.zeros((len(lengths), 1), dtype=np.float32)
    for sample in range(len(lengths)):
        count = int(mask[sample].sum())
        if count == 0:
            continue
        timestamps = np.cumsum(gaps[sample, :count], dtype=np.float64)
        timestamps -= timestamps[0]
        slots = np.asarray(
            [_slot_index(timestamp, omega) for timestamp in timestamps],
            dtype=int,
        )
        for position, slot in enumerate(slots):
            if slot < 0 or slot >= slot_count:
                continue
            direction = int(directions[sample, position])
            column = 2 if direction == 1 else 0
            features[sample, slot, column] += 1.0
            features[sample, slot, column + 1] += float(lengths[sample, position])
    return features, silences


def _segment_bounds(timestamps: np.ndarray, threshold: float):
    starts = [0]
    for position in range(1, len(timestamps)):
        if timestamps[position] - timestamps[position - 1] > threshold:
            starts.append(position)
    ends = [position - 1 for position in starts[1:]] + [len(timestamps) - 1]
    return list(zip(starts, ends, strict=True))


def _make_ticks(
    timestamps: np.ndarray,
    omega: float,
    silence_threshold: float,
    dummy_directions=(-1, 1),
):
    ticks = []
    dummy_directions = tuple(int(direction) for direction in dummy_directions)
    if not dummy_directions or any(
        direction not in (-1, 1) for direction in dummy_directions
    ):
        raise ValueError("iPET dummy_directions必须是{-1, 1}的非空子集")
    for start_index, end_index in _segment_bounds(timestamps, silence_threshold):
        start = float(timestamps[start_index])
        end = float(timestamps[end_index]) + float(silence_threshold)
        if 1 in dummy_directions:
            for timestamp in np.arange(start + omega / 2.0, end, omega):
                ticks.append(
                    (float(timestamp), 1, _slot_index(timestamp, omega))
                )
        if -1 in dummy_directions:
            for timestamp in np.arange(start, end, omega):
                ticks.append(
                    (float(timestamp), -1, _slot_index(timestamp, omega))
                )
    return ticks


def _unpack_perturbations(perturbations) -> tuple[np.ndarray, int, dict]:
    if isinstance(perturbations, PerturbationPlan):
        values = np.asarray(perturbations.values)
        start_index = int(perturbations.start_index)
        plan_metadata = {
            "perturbation_vector_count": int(perturbations.vector_count),
            "perturbation_slot_count": int(perturbations.slot_count),
        }
    else:
        values = np.asarray(perturbations)
        start_index = 0
        plan_metadata = {
            "perturbation_vector_count": None,
            "perturbation_slot_count": None,
        }
    if values.ndim != 2 or values.shape[1] != 4 or len(values) == 0:
        raise ValueError("iPET扰动必须具有[非空slot, 4]形状")
    start_index %= len(values)
    return values, start_index, plan_metadata


def budget_training_slot_masks(lengths, gaps, directions, mask, budget, dummy_directions):
    """Project runtime tick eligibility to the existing 40-slot training proxy.

    A dummy may continue at most T seconds after the latest observed real packet.
    This is causal and guarantees wire extension <= T even if no more packets arrive.
    """
    count_slots = int(round(TOTAL_TIME_SECONDS / OMEGA_SECONDS))
    result = np.zeros((len(lengths), count_slots, 4), dtype=np.float32)
    for i in range(len(lengths)):
        count = int(np.asarray(mask[i]).sum())
        if not count:
            continue
        timestamps = np.cumsum(gaps[i, :count], dtype=float)
        timestamps -= timestamps[0]
        for timestamp, direction, slot in _make_ticks(
            timestamps, OMEGA_SECONDS, SILENCE_THRESHOLD_SECONDS, dummy_directions):
            latest = timestamps[np.searchsorted(timestamps, timestamp, side="right") - 1]
            if 0 <= slot < count_slots and timestamp <= latest + budget["max_wire_delay_seconds"] + 1e-9:
                col = 2 if direction == 1 else 0
                result[i, slot, col:col+2] = 1
    return result


def inject_official_ipet_trace(
    perturbations,
    lengths,
    gaps,
    directions,
    *,
    omega: float = OMEGA_SECONDS,
    silence_threshold: float = SILENCE_THRESHOLD_SECONDS,
    minimum_frame_length: float = 60.0,
    maximum_frame_length: float = 1514.0,
    dummy_directions=(-1, 1),
):
    """Apply iPET using a long pre-generated pool and one flow start offset."""

    perturbation_values, perturbation_start, plan_metadata = _unpack_perturbations(
        perturbations
    )
    lengths = np.asarray(lengths, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.int8)
    if not len(lengths):
        return [], {
            "dummy_packets": 0,
            "dummy_bytes": 0.0,
            "perturbation_rows_consumed": 0,
            "perturbation_pool_rows": int(len(perturbation_values)),
            "perturbation_start_index": int(perturbation_start),
            "perturbation_pool_wrapped": False,
            **plan_metadata,
        }

    timestamps = np.cumsum(gaps, dtype=np.float64)
    timestamps -= timestamps[0]
    timeline = [
        (float(timestamp), 0, int(position), int(direction), float(length))
        for position, (timestamp, direction, length) in enumerate(
            zip(timestamps, directions, lengths, strict=True)
        )
    ]
    timeline.extend(
        (timestamp, 1, slot, direction, -1.0)
        for timestamp, direction, slot in _make_ticks(
            timestamps, omega, silence_threshold, dummy_directions
        )
    )
    timeline.sort(key=lambda item: (item[0], item[1], item[2]))

    events = []
    perturbation_rows_consumed = 0
    dummy_packets = 0
    dummy_bytes = 0.0
    for timestamp, kind, source_index, direction, length in timeline:
        if kind == 0:
            events.append(
                TraceEvent(timestamp, length, direction, False, source_index)
            )
            continue

        row_index = (
            perturbation_start + source_index
        ) % len(perturbation_values)
        row = perturbation_values[row_index]
        # ``source_index`` is the slot id. Both active directions point at the
        # same row; advancing by max() therefore consumes one row per physical
        # slot instead of one row per direction tick. Absolute slot ids also
        # preserve the gap across long-silence segments, matching
        # ``slotize_flows`` and preventing the second segment from reusing the
        # rows assigned to the silent interval.
        perturbation_rows_consumed = max(
            perturbation_rows_consumed, source_index + 1
        )
        offset = 2 if direction == 1 else 0
        number_of_packets = int(row[offset])
        total_payload = float(row[offset + 1])
        if number_of_packets == 0:
            continue
        payload_per_packet = total_payload / number_of_packets
        frame_length = float(
            np.rint(np.clip(payload_per_packet, minimum_frame_length, maximum_frame_length))
        )
        for _ in range(number_of_packets):
            events.append(TraceEvent(timestamp, frame_length, direction, True, None))
            dummy_packets += 1
            dummy_bytes += frame_length

    events.sort(
        key=lambda event: (
            event.timestamp,
            event.is_dummy,
            event.source_event_index
            if event.source_event_index is not None
            else -1,
        )
    )
    pool_wrapped = (
        perturbation_start + perturbation_rows_consumed
        > len(perturbation_values)
    )
    return events, {
        "dummy_packets": int(dummy_packets),
        "dummy_bytes": float(dummy_bytes),
        "perturbation_rows_consumed": int(perturbation_rows_consumed),
        "perturbation_pool_rows": int(len(perturbation_values)),
        "perturbation_start_index": int(perturbation_start),
        "perturbation_pool_wrapped": bool(pool_wrapped),
        **plan_metadata,
    }


__all__ = [
    "FEATURE_ORDER",
    "OMEGA_SECONDS",
    "PerturbationPlan",
    "SILENCE_THRESHOLD_SECONDS",
    "TOTAL_TIME_SECONDS",
    "TraceEvent",
    "inject_official_ipet_trace",
    "slotize_flows",
]
