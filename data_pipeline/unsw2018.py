"""Prepare the UNSW 2018 "IoT Traffic Traces" (a.k.a. Your Things) dataset.

The archive provides 20 daily captures of a *mixed* device testbed plus a
``List_Of_Devices.txt`` mapping device names to MAC addresses.  This adapter
turns those daily captures into the same ``[N, 32]`` flow representation used by
the CIC and UNSW-IoTraffic pipelines:

    lengths / gaps / directions / mask      fixed-width event arrays
    labels                                  device class (index into device_names)
    groups                                  capture identity = UTC day (file name)

Unlike the CIC extractor, this dataset is not TLS-only: the testbed carries
DNS/NTP/MQTT/HTTP and vendor-cloud traffic.  We therefore keep every TCP/UDP
flow with at least ``min_packets`` events and at least one outbound packet, and
only use the TLS marker for auditing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

try:  # Package import (python -m data_pipeline.unsw2018)
    from data_pipeline import cic
    from data_pipeline import pcap as dataset_pcap
    from data_pipeline import unsw
except ImportError:  # Direct execution from the source directory.
    import cic
    import pcap as dataset_pcap
    import unsw


PROTOCOL = "unsw2018-mixed-device-day-v1"
MAC_RE = re.compile(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}")
# Phones, tablets, laptops and the testbed gateway are not IoT device classes.
NON_IOT_HINTS = (
    "phone",
    "laptop",
    "macbook",
    "tablet",
    "galaxy",
    "iphone",
    "ipad",
    "android",
    "gateway",
    "router",
)


def load_device_list(path: Path) -> dict[str, tuple[bytes, ...]]:
    """Parse ``List_Of_Devices.txt`` into ``{name: (mac, ...)}``.

    A device may own several MAC addresses (e.g. wired + wireless); the file
    represents that with a continuation line that contains only a MAC and a
    connection type, so an empty name means "same device as the previous line".
    """

    devices: dict[str, list[bytes]] = {}
    current: str | None = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = MAC_RE.search(line)
        if match is None:
            continue
        mac = bytes.fromhex(match.group(0).replace(":", ""))
        name = line[: match.start()].strip()
        if name and not name.lower().startswith("list of devices"):
            current = name
        if current is None:
            continue
        bucket = devices.setdefault(current, [])
        if mac not in bucket:
            bucket.append(mac)
    return {name: tuple(macs) for name, macs in devices.items()}


def select_iot_devices(devices: Mapping[str, Sequence[bytes]]) -> list[str]:
    """Drop phones/laptops/gateway, keeping deterministic device order."""

    selected = []
    for name in sorted(devices):
        lowered = name.lower()
        if any(hint in lowered for hint in NON_IOT_HINTS):
            continue
        selected.append(name)
    return selected


def _iter_pcaps(archive: Path) -> Iterator[Path]:
    """Yield every ``*.pcap`` inside a daily ``.tar.gz`` (or the file itself)."""

    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.name.endswith(".tar.gz"):
        with tempfile.TemporaryDirectory() as temporary:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(temporary, filter="data")
            for path in sorted(Path(temporary).rglob("*.pcap")):
                yield path
        return
    yield archive


def extract_capture(
    path: Path,
    device_macs: Mapping[str, Sequence[bytes]],
    labels: Mapping[str, int],
    *,
    width: int = 32,
    min_packets: int = 4,
    timeout: float = 120.0,
) -> tuple[list[dataset_pcap.FlowSample], dict[str, int]]:
    """Sessionize one daily mixed capture and encode eligible flows.

    Direction is computed relative to the *observed* device MAC, so a device
    that owns two MACs (wired + wireless) still produces one bidirectional flow
    per connection.
    """

    mac_to_device = {
        mac: name for name, macs in device_macs.items() for mac in macs
    }
    sessions = cic.Sessionizer(timeout, width)
    samples: list[dataset_pcap.FlowSample] = []
    counts = {"records": 0, "sessions": 0, "accepted": 0, "non_tls": 0}
    day = path.name

    def finish(name: str, session: cic.Session) -> None:
        counts["sessions"] += 1
        if session.count < min_packets:
            return
        if not any(packet.direction == 1 for packet in session.prefix):
            return
        if not session.tls:
            counts["non_tls"] += 1
        samples.append(
            dataset_pcap._encode_flow(
                session.prefix,
                label=labels[name],
                group=day,
                max_packets=width,
            )
        )
        counts["accepted"] += 1

    # The 2018 archives are PCAPNG despite the .pcap suffix; ``unsw.read_capture``
    # auto-detects classic PCAP vs PCAPNG, so both container formats work here.
    for record in unsw.read_capture(path):
        counts["records"] += 1
        frame = record.frame
        if len(frame) < 14:
            continue
        # A selected-device-to-selected-device frame belongs to both endpoints.
        for mac in (frame[6:12], frame[0:6]):
            name = mac_to_device.get(mac)
            if name is None:
                continue
            packet = dataset_pcap.parse_packet(
                record.timestamp, frame, mac, 443, frame_id=record.frame_id
            )
            if packet is None:
                continue
            for finished in sessions.push((name, packet.flow_key), packet):
                finish(name, finished)

    for key, session in sessions.active.items():
        finish(key[0], session)
    return samples, counts


def _stack(
    samples: Sequence[dataset_pcap.FlowSample],
    device_names: Sequence[str],
    *,
    label_map: Mapping[int, int],
    protocol: str,
    manifest: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Stack per-flow samples into the shared dataset schema."""

    if not samples:
        raise RuntimeError("no eligible flows were extracted")
    labels = np.asarray([label_map[sample.label] for sample in samples], dtype=np.int16)
    groups = np.asarray([sample.group for sample in samples])
    dataset = {
        "lengths": np.stack([sample.lengths for sample in samples]),
        "timestamps": np.stack([sample.timestamps for sample in samples]),
        "frame_ids": np.stack([sample.frame_ids for sample in samples]),
        "source_ips": np.stack([sample.source_ips for sample in samples]),
        "destination_ips": np.stack([sample.destination_ips for sample in samples]),
        "gaps": np.stack([sample.gaps for sample in samples]),
        "directions": np.stack([sample.directions for sample in samples]),
        "mask": np.stack([sample.mask for sample in samples]),
        "labels": labels,
        "groups": groups,
        "capture_ids": groups.copy(),
        "device_names": np.asarray(tuple(device_names)),
        "data_protocol": np.asarray(protocol),
        "extraction_manifest_json": np.asarray(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        ),
    }
    dataset_pcap.validate_dataset(dataset, expected_width=dataset["lengths"].shape[1])
    return dataset


def build(
    raw_dir: Path,
    device_list: Path,
    output: Path,
    *,
    classes: int = 10,
    width: int = 32,
    min_packets: int = 4,
    timeout: float = 120.0,
) -> dict[str, object]:
    """Extract every daily archive, keep the ``classes`` largest IoT classes."""

    devices = load_device_list(device_list)
    names = select_iot_devices(devices)
    if classes < 2 or classes > len(names):
        raise ValueError(f"classes must be in [2, {len(names)}]")
    labels = {name: index for index, name in enumerate(names)}

    samples: list[dataset_pcap.FlowSample] = []
    audits: list[dict[str, object]] = []
    for archive in sorted(Path(raw_dir).glob("*.tar.gz")):
        for pcap_path in _iter_pcaps(archive):
            day_samples, counts = extract_capture(
                pcap_path,
                {name: devices[name] for name in names},
                labels,
                width=width,
                min_packets=min_packets,
                timeout=timeout,
            )
            audits.append(
                {
                    "archive": archive.name,
                    "pcap": pcap_path.name,
                    "counts": counts,
                    "accepted": len(day_samples),
                }
            )
            samples.extend(day_samples)
            print(
                f"[{archive.name}] {pcap_path.name}: {counts['accepted']} flows",
                flush=True,
            )
    if not samples:
        raise RuntimeError("no archives were processed")

    counts = np.bincount(
        np.asarray([sample.label for sample in samples], dtype=int), minlength=len(names)
    )
    # A class needs at least two independent capture days, otherwise the
    # capture-level train/test split cannot place it on both sides.
    groups_by_label: dict[int, set[str]] = {}
    for sample in samples:
        groups_by_label.setdefault(sample.label, set()).add(sample.group)
    eligible = [
        index
        for index in range(len(names))
        if counts[index] > 0 and len(groups_by_label.get(index, ())) >= 2
    ]
    if len(eligible) < 2:
        raise RuntimeError("fewer than two classes appear in at least two capture days")
    if len(eligible) < classes:
        print(
            f"[warn] only {len(eligible)} classes have >=2 capture days; "
            f"reducing class count from {classes} to {len(eligible)}",
            flush=True,
        )
    order = sorted(eligible, key=lambda index: (-int(counts[index]), names[index]))[:classes]
    keep = set(order)
    label_map = {old: new for new, old in enumerate(order)}
    kept = [sample for sample in samples if sample.label in keep]
    chosen_names = [names[index] for index in order]

    manifest = {
        "protocol": PROTOCOL,
        "source": "UNSW IoT Traffic Traces (2018), mixed daily captures",
        "device_list_sha256": hashlib.sha256(Path(device_list).read_bytes()).hexdigest(),
        "device_names": chosen_names,
        "class_count": len(chosen_names),
        "per_class_flows": {
            name: int(counts[index]) for name, index in zip(chosen_names, order)
        },
        "archives": audits,
        "width": width,
        "min_packets": min_packets,
        "timeout_seconds": float(timeout),
        "tls_required": False,
    }
    dataset = _stack(
        kept,
        chosen_names,
        label_map=label_map,
        protocol=PROTOCOL,
        manifest=manifest,
    )
    dataset["extraction_manifest_json"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cic.atomic_npz(output, dataset)
    output.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {output}: {len(dataset['labels'])} flows, {len(chosen_names)} classes")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="directory with daily *.tar.gz")
    parser.add_argument("--device-list", type=Path, required=True, help="List_Of_Devices.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--min-packets", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    build(
        args.raw_dir,
        args.device_list,
        args.output,
        classes=args.classes,
        width=args.width,
        min_packets=args.min_packets,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
