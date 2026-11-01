"""Prepare the Deakin D-IoT daily captures (Linux cooked / SLL, link type 113).

Layout: ``raw/IoT_YYYY-MM-DD.pcap`` (one mixed daily capture), plus
``meta/macAddresses.csv`` mapping MAC -> device name.  Several device names own
two MACs (e.g. two plugs of the same model); names are merged into one class.

SLL only carries the *source* MAC, so direction is resolved by IP: a packet is
outbound (+1) when its source MAC belongs to the device, inbound (-1) when its
destination IP is a previously observed IP of that device.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import struct
from pathlib import Path

import numpy as np

try:
    from data_pipeline import cic, pcap, unsw
except ImportError:
    import cic
    import pcap
    import unsw

PROTOCOL = "deakin-iot-daily-sll-v1"
SLL_LINKTYPE = 113
ETH_HEADER = 14
SLL_HEADER = 16


def load_mac_map(path: Path) -> dict[bytes, str]:
    mapping = {}
    for row in csv.DictReader(Path(path).open(encoding="utf-8-sig")):
        mac = row["MAC Address"].strip().lower()
        name = row["Device Name"].strip()
        if not mac or not name:
            continue
        mapping[bytes.fromhex(mac.replace(":", ""))] = name
    return mapping


def _iter_sll(path: Path):
    data = Path(path).read_bytes()
    if len(data) < 24:
        return
    linktype = struct.unpack("<I", data[20:24])[0]
    if linktype != SLL_LINKTYPE:
        raise ValueError(f"unexpected link type {linktype}: {path}")
    offset = 24
    while offset + 16 <= len(data):
        ts_sec, ts_usec, included, _original = struct.unpack("<IIII", data[offset : offset + 16])
        offset += 16
        frame = data[offset : offset + included]
        offset += included
        if len(frame) < SLL_HEADER:
            continue
        source_mac = frame[6:12]
        ethertype = struct.unpack("!H", frame[14:16])[0]
        ip_offset = SLL_HEADER
        if ethertype in (0x8100, 0x88A8) and len(frame) >= 18:
            ethertype = struct.unpack("!H", frame[16:18])[0]
            ip_offset = 18
        if ethertype != 0x0800 or len(frame) < ip_offset + 20:
            continue
        ihl = (frame[ip_offset] & 0x0F) * 4
        if ihl < 20 or len(frame) < ip_offset + ihl:
            continue
        protocol = int(frame[ip_offset + 9])
        if protocol not in (6, 17):
            continue
        total_length = struct.unpack("!H", frame[ip_offset + 2 : ip_offset + 4])[0]
        source_ip = ".".join(map(str, frame[ip_offset + 12 : ip_offset + 16]))
        destination_ip = ".".join(map(str, frame[ip_offset + 16 : ip_offset + 20]))
        transport = ip_offset + ihl
        source_port, destination_port = struct.unpack("!HH", frame[transport : transport + 4])
        tcp_flags = int(frame[transport + 13]) if protocol == 6 and len(frame) >= transport + 20 else 0
        tcp_sequence = (
            struct.unpack("!I", frame[transport + 4 : transport + 8])[0]
            if protocol == 6 and len(frame) >= transport + 20
            else 0
        )
        yield {
            "timestamp": ts_sec + ts_usec / 1e6,
            "source_mac": source_mac,
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "protocol": protocol,
            "source_port": source_port,
            "destination_port": destination_port,
            "tcp_flags": tcp_flags,
            "tcp_sequence": tcp_sequence,
            "frame_length": float(min(1514, total_length + ETH_HEADER)),
            "is_tls_record": bool(
                protocol == 6
                and 443 in (source_port, destination_port)
                and len(frame) >= transport + 20
            ),
        }


def build(raw_dir: Path, mac_csv: Path, output: Path, *, devices, classes: int, width: int, min_packets: int, timeout: float, max_per_class_day: int):
    mac_to_name = load_mac_map(mac_csv)
    name_to_macs = collections.defaultdict(set)
    for mac, name in mac_to_name.items():
        name_to_macs[name].add(mac)
    selected = [name for name in devices if name in name_to_macs]
    if len(selected) < classes:
        raise ValueError(f"only {len(selected)} requested devices exist")
    selected = selected[:classes]
    mac_to_device = {mac: name for name in selected for mac in name_to_macs[name]}
    labels = {name: index for index, name in enumerate(selected)}

    ip_to_device: dict[str, str] = {}
    samples = []
    audits = []
    for path in sorted(Path(raw_dir).glob("IoT_*.pcap")):
        day = path.stem.replace("IoT_", "")
        # First pass: discover device IPs from outbound frames.
        for packet in _iter_sll(path):
            name = mac_to_device.get(packet["source_mac"])
            if name is not None and packet["source_ip"] != "0.0.0.0":
                ip_to_device.setdefault(packet["source_ip"], name)
        sessions = cic.Sessionizer(timeout, width)
        counts = {"packets": 0, "sessions": 0, "accepted": 0, "tls": 0}
        per_class_day: collections.Counter = collections.Counter()

        def finish(name, session):
            counts["sessions"] += 1
            if session.count < min_packets or not any(p.direction == 1 for p in session.prefix):
                return
            key = (labels[name], day)
            if per_class_day[key] >= max_per_class_day:
                return
            per_class_day[key] += 1
            if session.tls:
                counts["tls"] += 1
            samples.append(pcap._encode_flow(session.prefix, label=labels[name], group=day, max_packets=width))
            counts["accepted"] += 1

        frame_id = 0
        for packet in _iter_sll(path):
            counts["packets"] += 1
            frame_id += 1
            name = mac_to_device.get(packet["source_mac"])
            if name is not None:
                direction = 1
            else:
                name = ip_to_device.get(packet["destination_ip"])
                direction = -1
            if name is None:
                continue
            key = (
                name,
                packet["protocol"],
                *sorted(
                    (
                        (packet["source_ip"], packet["source_port"]),
                        (packet["destination_ip"], packet["destination_port"]),
                    )
                ),
            )
            parsed = pcap.ParsedPacket(
                flow_key=key,
                timestamp=packet["timestamp"],
                direction=direction,
                frame_length=packet["frame_length"],
                source_port=packet["source_port"],
                destination_port=packet["destination_port"],
                is_tls_record=packet["is_tls_record"],
                frame_id=frame_id,
                source_ip=packet["source_ip"],
                destination_ip=packet["destination_ip"],
                tcp_flags=packet["tcp_flags"],
                tcp_sequence=packet["tcp_sequence"],
            )
            for done in sessions.push(key, parsed):
                finish(name, done)
        for key, session in list(sessions.active.items()):
            finish(key[0], session)
        audits.append({"day": day, "counts": counts})
        print(f"[{day}] flows={counts['accepted']} packets={counts['packets']}", flush=True)
        if len(samples) == 0:
            continue

    if not samples:
        raise RuntimeError("no Deakin flows extracted")
    names = selected
    labels_arr = np.asarray([sample.label for sample in samples], dtype=np.int16)
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
        "labels": labels_arr,
        "groups": groups,
        "capture_ids": groups.copy(),
        "device_names": np.asarray(names),
        "data_protocol": np.asarray(PROTOCOL),
    }
    manifest = {
        "protocol": PROTOCOL,
        "source": "Deakin D-IoT daily SLL captures",
        "devices": names,
        "per_class_flows": {name: int((labels_arr == index).sum()) for index, name in enumerate(names)},
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
    print(f"Saved {output}: {len(samples)} flows, {len(names)} classes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--mac-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="comma separated device names in class order")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--min-packets", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-per-class-day", type=int, default=500)
    args = parser.parse_args()
    build(
        args.raw_dir,
        args.mac_csv,
        args.output,
        devices=[value.strip() for value in args.devices.split(",")],
        classes=args.classes,
        width=args.width,
        min_packets=args.min_packets,
        timeout=args.timeout,
        max_per_class_day=args.max_per_class_day,
    )


if __name__ == "__main__":
    main()
