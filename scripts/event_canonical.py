"""Event-level canonical/actual transform with explicit truncation accounting.

Events are dictionaries with stable ``event_id`` and ``role``.  Trigger events
carry ``slot`` (1-based source anchor).  ``forward`` moves reference triggers to
actual anchors; ``inverse`` moves them back.  Both preserve event identity and
report events lost to the 32-slot truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Event:
    event_id: str
    role: str  # real / weakener / trigger
    source_index: int | None = None
    slot: int | None = None
    attrs: dict = field(default_factory=dict)


@dataclass
class TransformResult:
    events: list[Event]
    lost: list[Event]
    fallback_slots: list[int]


def _insert(events: Iterable[Event], triggers: list[Event], slots: list[int], window: int) -> TransformResult:
    body = [e for e in events if e.role != "trigger"]
    out: list[Event] = []
    inserted: set[int] = set()
    fallback: list[Event] = []
    for event in body:
        out.append(event)
        if event.role != "real" or event.source_index is None:
            continue
        anchor = event.source_index + 1
        if anchor in slots and anchor not in inserted:
            index = slots.index(anchor)
            out.append(triggers[index])
            inserted.add(anchor)
    for i, slot in enumerate(slots):
        if slot not in inserted:
            fallback.append(triggers[i])
            out.append(triggers[i])
    kept = out[:window]
    lost = out[window:]
    return TransformResult(kept, lost, [e.slot for e in fallback])


def forward(reference: list[Event], actual_slots: list[int], window: int = 32) -> TransformResult:
    triggers = [e for e in reference if e.role == "trigger"]
    if len(triggers) != len(actual_slots):
        raise ValueError("trigger count and actual slots must match")
    moved = [
        Event(e.event_id, e.role, e.source_index, actual_slots[i], dict(e.attrs))
        for i, e in enumerate(triggers)
    ]
    return _insert(reference, moved, actual_slots, window)


def inverse(actual: list[Event], reference_slots: list[int], window: int = 32) -> TransformResult:
    triggers = [e for e in actual if e.role == "trigger"]
    if len(triggers) != len(reference_slots):
        raise ValueError("trigger count and reference slots must match")
    moved = [
        Event(e.event_id, e.role, e.source_index, reference_slots[i], dict(e.attrs))
        for i, e in enumerate(triggers)
    ]
    return _insert(actual, moved, reference_slots, window)
