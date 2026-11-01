"""Audit raw captures: flow lengths, original bytes, and the FADE overhead ratio.

This is a dataset-screening tool, independent of any defense.  It answers one
question before investing in a new dataset:

    Given the paper's absolute per-flow budget (630 B / 1.5 s / 6 events),
    what fraction of the original traffic would that budget add?

If the ratio is far above the paper's operating point (~6-9%), the dataset is not
comparable and should not be used as a main result.

Usage:
    python -B -m scripts.audit_pcap_flows "E:\\path\\to\\captures" --limit 5
    python -B -m scripts.audit_pcap_flows one.pcapng
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

try:
    from data_pipeline import pcap as dataset_pcap
    from data_pipeline import unsw
except ImportError:  # Direct execution from the source directory.
    import pcap as dataset_pcap
    import unsw


ADDED_BYTES = 630.0
MIN_PACKETS = 4


def iter_captures(root: Path, temp_root: Path) -> Iterator[Path]:
    """Yield pcap/pcapng files; ``.tar.gz`` archives are extracted under ``temp_root``.

    ``temp_root`` is owned by the caller and must outlive the returned paths,
    because a ``TemporaryDirectory`` context would delete the extracted files
    before they are audited.
    """

    if root.is_file():
        if root.name.endswith(".tar.gz"):
            target = Path(tempfile.mkdtemp(dir=temp_root))
            with tarfile.open(root, "r:gz") as tar:
                tar.extractall(target, filter="data")
            yield from sorted(target.rglob("*.pcap*"))
            return
        yield root
        return

    for path in sorted(root.rglob("*")):
        if path.is_file() and (
            path.suffix in (".pcap", ".pcapng") or path.name.endswith(".tar.gz")
        ):
            yield from iter_captures(path, temp_root)


def audit_capture(path: Path) -> dict[str, object]:
    """Group records into bidirectional flows and report length/byte statistics."""

    flows: dict[tuple[object, ...], list[int]] = {}
    packets = 0
    ports: Counter[int] = Counter()
    for record in unsw.read_capture(path):
        packets += 1
        frame = record.frame
        if len(frame) < 14:
            continue
        # The real parser rejects frames whose MACs do not match ``device_mac``.
        # For a labeling-free audit we use the frame's own source MAC, which
        # yields direction +1 but keeps the bidirectional flow key intact.
        parsed = dataset_pcap.parse_packet(
            record.timestamp, frame, frame[6:12], 443, frame_id=record.frame_id
        )
        if parsed is None:
            continue
        bucket = flows.setdefault(parsed.flow_key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(parsed.frame_length)
        ports[int(parsed.destination_port)] += 1

    counts = np.asarray([value[0] for value in flows.values()], dtype=np.int64)
    byte_sums = np.asarray([value[1] for value in flows.values()], dtype=np.int64)
    eligible = counts >= MIN_PACKETS
    eligible_counts = counts[eligible]
    eligible_bytes = byte_sums[eligible]
    return {
        "capture": path.name,
        "packets": packets,
        "flows": int(len(counts)),
        "eligible_flows": int(eligible.sum()),
        "events_mean": float(eligible_counts.mean()) if len(eligible_counts) else 0.0,
        "events_median": float(np.median(eligible_counts)) if len(eligible_counts) else 0.0,
        "bytes_mean": float(eligible_bytes.mean()) if len(eligible_bytes) else 0.0,
        "bytes_median": float(np.median(eligible_bytes)) if len(eligible_bytes) else 0.0,
        "overhead_mean_pct": (
            ADDED_BYTES / float(eligible_bytes.mean()) * 100.0 if len(eligible_bytes) else 0.0
        ),
        "overhead_median_pct": (
            ADDED_BYTES / float(np.median(eligible_bytes)) * 100.0 if len(eligible_bytes) else 0.0
        ),
        "short_flow_fraction": (
            float((eligible_counts <= 10).mean()) if len(eligible_counts) else 0.0
        ),
        "top_ports": ports.most_common(5),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="only audit the first N captures")
    args = parser.parse_args(list(argv) if argv is not None else None)

    temp_root = Path(tempfile.mkdtemp(prefix="pcap_audit_"))
    try:
        captures = list(iter_captures(args.root, temp_root))
        if args.limit:
            captures = captures[: args.limit]
        if not captures:
            raise SystemExit("no pcap/pcapng files found")

        rows = [audit_capture(path) for path in captures]
        for row in rows:
            print(
                f"{row['capture']:<42} packets={row['packets']:>10,} flows={row['eligible_flows']:>7,} "
                f"events(mean/med)={row['events_mean']:5.1f}/{row['events_median']:4.0f} "
                f"bytes={row['bytes_mean']:>8,.0f} "
                f"overhead={row['overhead_mean_pct']:5.1f}%/{row['overhead_median_pct']:5.1f}% "
                f"short={row['short_flow_fraction']*100:4.0f}%"
            )

        total_flows = sum(int(row["eligible_flows"]) for row in rows)
        weighted_bytes = sum(
            float(row["bytes_mean"]) * int(row["eligible_flows"]) for row in rows
        )
        if total_flows:
            print(
                f"\naggregate: captures={len(rows)} flows={total_flows:,} "
                f"mean_bytes={weighted_bytes / total_flows:,.0f} "
                f"overhead_630B={ADDED_BYTES / (weighted_bytes / total_flows) * 100:.1f}%"
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
