"""Prepare the Mon(IoT)r dataset (per-device, per-activity PCAP captures).

Layout::

    raw/<device>/<activity>/<YYYY-MM-DD_HH_MM_SS...>.pcap

Each capture contains one device's LAN traffic plus the gateway/cloud side.
The device IP is inferred from TCP SYNs (the client), falling back to the most
frequent private source IP; the device MAC is then read from packets whose
source IP is that device IP.  Only packets involving the device IP are kept,
which removes phone/other-LAN traffic from mixed activity captures.

Flows use the shared ``[N, 32]`` representation; labels are device names and
``groups`` are UTC dates, so day-grouped folds remain possible.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

try:
    from data_pipeline import cic, pcap, unsw
except ImportError:  # direct execution
    import cic
    import pcap
    import unsw

PROTOCOL = "mon-iot-per-device-day-v1"
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
PRIVATE_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
                    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
                    "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.")


def _ip_text(raw: bytes) -> str:
    return ".".join(map(str, raw))


def _parse_ips(frame: bytes):
    """Return (src_ip, dst_ip, protocol, tcp_flags) for an IPv4 frame."""

    located = pcap._ethernet_payload_offset(frame)
    if located is None:
        return None
    ethertype, offset = located
    if ethertype != 0x0800 or len(frame) < offset + 20:
        return None
    ihl = (frame[offset] & 0x0F) * 4
    if ihl < 20 or len(frame) < offset + ihl:
        return None
    protocol = int(frame[offset + 9])
    if protocol not in (6, 17):
        return None
    src = _ip_text(frame[offset + 12 : offset + 16])
    dst = _ip_text(frame[offset + 16 : offset + 20])
    flags = 0
    if protocol == 6 and len(frame) >= offset + ihl + 14:
        flags = int(frame[offset + ihl + 13])
    return src, dst, protocol, flags


def _device_ip(frames) -> str | None:
    syn_sources: collections.Counter[str] = collections.Counter()
    private_sources: collections.Counter[str] = collections.Counter()
    private_any: collections.Counter[str] = collections.Counter()
    for frame in frames:
        parsed = _parse_ips(frame)
        if parsed is None:
            continue
        src, dst, protocol, flags = parsed
        if protocol == 6 and (flags & 0x02) and not (flags & 0x10) and src.startswith(PRIVATE_PREFIXES):
            syn_sources[src] += 1
        if src.startswith(PRIVATE_PREFIXES):
            private_sources[src] += 1
        for candidate in (src, dst):
            if candidate.startswith(PRIVATE_PREFIXES):
                private_any[candidate] += 1
    for counter in (syn_sources, private_sources):
        if counter:
            return counter.most_common(1)[0][0]
    if private_any:
        return private_any.most_common(1)[0][0]
    return None


def _device_mac(frames, device_ip: str) -> bytes | None:
    macs: collections.Counter[bytes] = collections.Counter()
    for frame in frames:
        parsed = _parse_ips(frame)
        if parsed is None:
            continue
        src, _dst, _protocol, _flags = parsed
        if src == device_ip and len(frame) >= 12:
            macs[frame[6:12]] += 1
    return macs.most_common(1)[0][0] if macs else None


def extract_file(path: Path, label: int, *, width: int, min_packets: int, timeout: float):
    records = list(unsw.read_capture(path))
    if not records:
        return [], {"records": 0, "sessions": 0, "accepted": 0, "tls": 0, "device_ip": None}
    frames = [record.frame for record in records]
    device_ip = _device_ip(frames)
    if device_ip is None:
        return [], {"records": len(records), "sessions": 0, "accepted": 0, "tls": 0, "device_ip": None}
    device_mac = _device_mac(frames, device_ip)
    if device_mac is None:
        return [], {"records": len(records), "sessions": 0, "accepted": 0, "tls": 0, "device_ip": device_ip}
    match = DATE_RE.search(path.name)
    group = match.group(1) if match else path.parent.name
    sessions = cic.Sessionizer(timeout, width)
    samples = []
    counts = {"records": len(records), "sessions": 0, "accepted": 0, "tls": 0, "device_ip": device_ip}

    def finish(session):
        counts["sessions"] += 1
        if session.count < min_packets:
            return
        if not any(packet.direction == 1 for packet in session.prefix):
            return
        if session.tls:
            counts["tls"] += 1
        encoded = pcap._encode_flow(session.prefix, label=label, group=group, max_packets=width)
        samples.append(
            dataclasses.replace(encoded, lengths=np.minimum(encoded.lengths, 1514.0))
        )
        counts["accepted"] += 1

    for record in records:
        frame = record.frame
        parsed = _parse_ips(frame)
        if parsed is None:
            continue
        src, dst, _protocol, _flags = parsed
        if src != device_ip and dst != device_ip:
            continue
        packet = pcap.parse_packet(record.timestamp, frame, device_mac, 443, frame_id=record.frame_id)
        if packet is None:
            continue
        for done in sessions.push(packet.flow_key, packet):
            finish(done)
    for session in sessions.active.values():
        finish(session)
    return samples, counts


def build(raw_dir: Path, output: Path, *, classes: int, width: int, min_packets: int, timeout: float):
    devices = sorted(path.name for path in Path(raw_dir).iterdir() if path.is_dir())
    if classes > len(devices):
        raise ValueError(f"requested {classes} classes but only {len(devices)} device directories")
    labels = {name: index for index, name in enumerate(devices)}
    samples = []
    audits = []
    for device in devices:
        device_samples = []
        device_counts = collections.Counter()
        for path in sorted((Path(raw_dir) / device).rglob("*.pcap")):
            extracted, counts = extract_file(
                path, labels[device], width=width, min_packets=min_packets, timeout=timeout
            )
            device_samples.extend(extracted)
            for key, value in counts.items():
                if key != "device_ip":
                    device_counts[key] += value
        audits.append({"device": device, "samples": len(device_samples), "counts": dict(device_counts)})
        samples.extend(device_samples)
        print(f"[{device}] flows={len(device_samples)} packets={device_counts['records']} tls_sessions={device_counts['tls']}", flush=True)

    if not samples:
        raise RuntimeError("no Mon flows were extracted")
    per_label = np.bincount(np.asarray([sample.label for sample in samples]), minlength=len(devices))
    groups_by_label = collections.defaultdict(set)
    for sample in samples:
        groups_by_label[sample.label].add(sample.group)
    eligible = [index for index in range(len(devices)) if per_label[index] >= 50 and len(groups_by_label[index]) >= 2]
    if len(eligible) < classes:
        print(f"[warn] only {len(eligible)} devices pass >=50 flows and >=2 dates", flush=True)
    order = sorted(eligible, key=lambda index: (-int(per_label[index]), devices[index]))[:classes]
    mapping = {old: new for new, old in enumerate(order)}
    keep = [sample for sample in samples if sample.label in mapping]
    names = [devices[index] for index in order]

    dataset = {
        "lengths": np.stack([sample.lengths for sample in keep]),
        "timestamps": np.stack([sample.timestamps for sample in keep]),
        "frame_ids": np.stack([sample.frame_ids for sample in keep]),
        "source_ips": np.stack([sample.source_ips for sample in keep]),
        "destination_ips": np.stack([sample.destination_ips for sample in keep]),
        "gaps": np.stack([sample.gaps for sample in keep]),
        "directions": np.stack([sample.directions for sample in keep]),
        "mask": np.stack([sample.mask for sample in keep]),
        "labels": np.asarray([mapping[sample.label] for sample in keep], dtype=np.int16),
        "groups": np.asarray([sample.group for sample in keep]),
        "capture_ids": np.asarray([sample.group for sample in keep]),
        "device_names": np.asarray(names),
        "data_protocol": np.asarray(PROTOCOL),
    }
    manifest = {
        "protocol": PROTOCOL,
        "source": "Mon(IoT)r per-device captures",
        "devices": names,
        "per_class_flows": {devices[index]: int(per_label[index]) for index in order},
        "per_class_dates": {devices[index]: len(groups_by_label[index]) for index in order},
        "audits": audits,
        "width": width,
        "min_packets": min_packets,
        "timeout_seconds": float(timeout),
        "tls_required": False,
    }
    dataset["extraction_manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    pcap.validate_dataset(dataset, expected_width=width)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cic.atomic_npz(output, dataset)
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}: {len(keep)} flows, {len(names)} classes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--min-packets", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    build(args.raw_dir, args.output, classes=args.classes, width=args.width, min_packets=args.min_packets, timeout=args.timeout)


if __name__ == "__main__":
    main()
