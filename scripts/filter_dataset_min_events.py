"""Derive a minimum-event subset from an existing dataset NPZ.

Motivation: the UNSW 2018 testbed produces many very short flows (median ~11
events).  With FADE's absolute per-flow budget, those flows carry a much larger
*relative* overhead than the paper's datasets.  Keeping only flows with at
least ``min-events`` events makes the overhead comparable while staying faithful
to the original traffic (no resampling, no re-extraction).

Classes that fall below ``min-flows`` or ``min-groups`` after filtering are
dropped, and labels are re-mapped to a contiguous ``0..K-1`` range.

Usage:
    python -B -m scripts.filter_dataset_min_events \
        data/processed/unsw2018_v1/flows_unsw2018_10class.npz \
        data/processed/unsw2018_v1/flows_unsw2018_9class.npz --min-events 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from data_pipeline import cic, pcap
except ImportError:  # Direct execution from the source directory.
    import cic
    import pcap


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(
    source: Path,
    output: Path,
    *,
    min_events: int = 16,
    min_flows: int = 50,
    min_groups: int = 2,
) -> dict[str, object]:
    data = np.load(source, allow_pickle=False)
    events = np.asarray(data["mask"]).sum(axis=1)
    keep = events >= int(min_events)
    if not np.any(keep):
        raise ValueError("no flow reaches the minimum event count")

    labels = np.asarray(data["labels"])
    groups = np.asarray(data["groups"]).astype(str)
    names = [str(name) for name in np.asarray(data["device_names"])]

    eligible: list[int] = []
    for label in range(len(names)):
        chosen = keep & (labels == label)
        if int(chosen.sum()) < int(min_flows):
            continue
        if len(np.unique(groups[chosen])) < int(min_groups):
            continue
        eligible.append(label)
    if len(eligible) < 2:
        raise ValueError("fewer than two classes survive the filter")

    label_map = {old: new for new, old in enumerate(eligible)}
    row_filter = keep & np.isin(labels, eligible)
    new_labels = np.asarray([label_map[int(value)] for value in labels[row_filter]], dtype=np.int16)

    dataset: dict[str, np.ndarray] = {}
    for key in (
        "lengths",
        "timestamps",
        "frame_ids",
        "source_ips",
        "destination_ips",
        "gaps",
        "directions",
        "mask",
    ):
        if key in data:
            dataset[key] = np.asarray(data[key])[row_filter]
    dataset["labels"] = new_labels
    dataset["groups"] = groups[row_filter]
    dataset["capture_ids"] = dataset["groups"].copy()
    dataset["device_names"] = np.asarray([names[old] for old in eligible])
    dataset["data_protocol"] = np.asarray(str(data["data_protocol"]))

    parent_manifest: dict[str, object] = {}
    if "extraction_manifest_json" in data:
        parent_manifest = json.loads(str(np.asarray(data["extraction_manifest_json"]).item()))
    derivation = {
        "parent_dataset": str(Path(source).resolve()),
        "parent_sha256": sha256_file(Path(source)),
        "min_events": int(min_events),
        "min_flows_per_class": int(min_flows),
        "min_groups_per_class": int(min_groups),
        "kept_classes": [names[old] for old in eligible],
        "dropped_classes": [name for index, name in enumerate(names) if index not in label_map],
        "flows": int(row_filter.sum()),
    }
    parent_manifest["derivation"] = derivation
    dataset["extraction_manifest_json"] = np.asarray(
        json.dumps(parent_manifest, ensure_ascii=False, separators=(",", ":"))
    )

    pcap.validate_dataset(dataset, expected_width=dataset["lengths"].shape[1])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cic.atomic_npz(output, dataset)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": str(dataset["data_protocol"]),
                "device_names": [str(name) for name in dataset["device_names"]],
                "class_count": len(eligible),
                "flows": int(row_filter.sum()),
                "derivation": derivation,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return derivation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-events", type=int, default=16)
    parser.add_argument("--min-flows", type=int, default=50)
    parser.add_argument("--min-groups", type=int, default=2)
    args = parser.parse_args()
    derivation = derive(
        args.source,
        args.output,
        min_events=args.min_events,
        min_flows=args.min_flows,
        min_groups=args.min_groups,
    )
    print(json.dumps(derivation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
