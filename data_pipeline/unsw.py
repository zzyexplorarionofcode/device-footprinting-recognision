"""Prepare packet-level UNSW-IoTraffic datasets for Method32.

This module is intentionally separate from :mod:`dataset_pcap`.  The original
project data consists of directories of classic-PCAP captures, whereas the
UNSW archive uses one multi-month PCAPNG file per device.  In addition to
streaming PCAPNG, this adapter must split repeated five-tuples after inactivity
and create date-based groups so no UTC day is shared by train and test.

Only packet facts are used to build classifier inputs.  The flow-level CSVs are
not used to reconstruct packet sequences.  Every retained flow contains at
least one plausible TLS record on TCP/443, at least ``min_packets`` events, and
an outbound packet within the observed fixed-width prefix.
"""

# =============================================================================
# 中文阅读导图：UNSW-IoT 数据制备（一个设备一个多月份 PCAPNG）
# =============================================================================
# 与 pcap.py 的分工：pcap.py 只认 classic PCAP；本文件补上 PCAPNG 容器解析，
# 以及 UNSW 特有的“重复五元组如何断会话”和“如何按天分组避免泄漏”。
#
# 建议阅读顺序：
#   1. read_pcapng       按块流式读：SHB 决定字节序、IDB 定义接口、EPB 出包；
#                        选项 9 (if_tsresol) 决定时间戳单位，选项 14 是时区偏移；
#   2. parse_packet      复用 pcap.py 的协议解析，得到 ParsedPacket；
#   3. _session_boundary_reason / _session_terminal_reason
#                        会话断点：时间戳倒退、跨 UTC 日、120s 空闲、新 SYN 序列、
#                        RST、双向 FIN。前四者在“加入该包之前”判断，
#                        后两者在“加入该包之后”判断，保证 FIN/RST 留在本会话内；
#   4. FlowAccumulator  只保留前 max_packets 个包的定宽前缀，同时用全流计数器
#                       记录 packet_count/outbound/tls_like 等，供筛选与审计；
#   5. Reservoir         每类/每天最多留 capacity 条，蓄水池抽样保证内存有界；
#   6. extract_device    单设备主循环：读包→分流→断会话→交给 sample_consumer；
#   7. _dataset_from_samples / extract_balanced_dataset / extract_datasets
#                        把 FlowSample 列表 stack 成 [N,32] 四数组 + 审计数组。
#
# 复杂语法提示：
#   * PCAPNG 每个块的总长度写在块头与块尾两处，本文件两处都校验，
#     块长度必须是 4 的倍数且不超过 128 MB；
#   * 时间戳是 (high<<32)|low 的 64 位 ticks，需要除以 if_tsresol 再加重放偏移；
#   * 抽样用 random.Random（不是 NumPy），只依赖 Python 标准库，便于自测。

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import struct
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

try:
    from data_pipeline import pcap as dataset_pcap
except ImportError:  # Direct execution from the source directory.
    import pcap as dataset_pcap


PCAPNG_SECTION_HEADER = b"\x0a\x0d\x0d\x0a"
PCAPNG_BYTE_ORDER = {
    b"\x4d\x3c\x2b\x1a": "<",
    b"\x1a\x2b\x3c\x4d": ">",
}
PCAPNG_INTERFACE_DESCRIPTION = 0x00000001
PCAPNG_PACKET = 0x00000002
PCAPNG_SIMPLE_PACKET = 0x00000003
PCAPNG_ENHANCED_PACKET = 0x00000006
ETHERNET_LINK_TYPE = 1
MAX_BLOCK_BYTES = 128 * 1024 * 1024

DEFAULT_DEVICES_6 = (
    "iHome",
    "AmazonEcho",
    "WithingsSleepSensor",
    "TribySpeaker",
    "PixStarPhotoFrame",
    "AwairAirQuality",
)
DEFAULT_DEVICES_4 = DEFAULT_DEVICES_6[:4]
DEFAULT_DEVICES_10 = DEFAULT_DEVICES_6 + (
    "AugustDoorBell",
    "NetatmoWelcome",
    "PhilipsHue",
    "HPPrinter",
)

# This protocol version records the session boundaries and split unit used by
# the extracted UNSW arrays.  The source archive has one physical capture per
# device, so a filename is not a meaningful independent capture identifier.
UNSW_DATA_PROTOCOL = "unsw-tcp-session-day-v2"

TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04


@dataclass(frozen=True)
class PcapNgInterface:
    """Timestamp and link-layer parameters from one PCAPNG IDB."""

    link_type: int
    snap_length: int
    timestamp_scale: int
    timestamp_offset_seconds: int


@dataclass(frozen=True)
class CaptureRecord:
    """One packet plus its PCAPNG section identity.

    A section boundary always ends active sessions.  The published UNSW files
    can also contain timestamp-regressing capture runs inside one section, so
    extraction treats any global regression as an additional session boundary.
    """

    timestamp: float
    frame: bytes
    frame_id: int
    section_index: int


@dataclass
class FlowAccumulator:
    """Bounded packet prefix plus full-flow counters for one timed session."""

    observed_packets: list[dataset_pcap.ParsedPacket]
    first_timestamp: float
    last_timestamp: float
    first_frame_id: int
    last_frame_id: int
    packet_count: int = 0
    outbound_packets: int = 0
    inbound_packets: int = 0
    tls_like_records: int = 0
    syn_sequences: dict[int, set[int]] = field(default_factory=dict)
    fin_directions: set[int] = field(default_factory=set)

    @classmethod
    def start(
        cls, packet: dataset_pcap.ParsedPacket, max_packets: int
    ) -> "FlowAccumulator":
        accumulator = cls(
            observed_packets=[],
            first_timestamp=float(packet.timestamp),
            last_timestamp=float(packet.timestamp),
            first_frame_id=int(packet.frame_id),
            last_frame_id=int(packet.frame_id),
        )
        accumulator.add(packet, max_packets)
        return accumulator

    def add(self, packet: dataset_pcap.ParsedPacket, max_packets: int) -> None:
        if len(self.observed_packets) < int(max_packets):
            self.observed_packets.append(packet)
        else:
            # Keep the earliest fixed-width prefix even if a merged PCAPNG has
            # a few records that are slightly out of timestamp order.
            latest_index = max(
                range(len(self.observed_packets)),
                key=lambda index: (
                    self.observed_packets[index].timestamp,
                    self.observed_packets[index].frame_id,
                ),
            )
            latest = self.observed_packets[latest_index]
            if (packet.timestamp, packet.frame_id) < (
                latest.timestamp,
                latest.frame_id,
            ):
                self.observed_packets[latest_index] = packet
        self.packet_count += 1
        self.outbound_packets += int(packet.direction == 1)
        self.inbound_packets += int(packet.direction == -1)
        self.tls_like_records += int(packet.is_tls_record)
        if int(packet.tcp_flags) & TCP_SYN:
            self.syn_sequences.setdefault(int(packet.direction), set()).add(
                int(packet.tcp_sequence)
            )
        if int(packet.tcp_flags) & TCP_FIN:
            self.fin_directions.add(int(packet.direction))
        self.first_timestamp = min(self.first_timestamp, float(packet.timestamp))
        self.last_timestamp = max(self.last_timestamp, float(packet.timestamp))
        self.last_frame_id = max(self.last_frame_id, int(packet.frame_id))


class Reservoir:
    """Deterministic uniform reservoir that never stores every eligible flow."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.rng = random.Random(int(seed))
        self.items: list[dataset_pcap.FlowSample] = []
        self.seen = 0

    def offer(self, sample: dataset_pcap.FlowSample) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(sample)
            return
        replacement = self.rng.randrange(self.seen)
        if replacement < self.capacity:
            self.items[replacement] = sample


def _read_exact(handle: Any, count: int, path: Path, description: str) -> bytes:
    value = handle.read(int(count))
    if len(value) != int(count):
        raise ValueError(f"truncated {description}: {path}")
    return value


def _validate_block_length(total_length: int, path: Path) -> None:
    if (
        total_length < 12
        or total_length % 4 != 0
        or total_length > MAX_BLOCK_BYTES
    ):
        raise ValueError(f"invalid PCAPNG block length {total_length}: {path}")


def _parse_options(data: bytes, endian: str, path: Path) -> dict[int, list[bytes]]:
    options: dict[int, list[bytes]] = {}
    offset = 0
    while offset < len(data):
        if len(data) - offset < 4:
            if any(data[offset:]):
                raise ValueError(f"truncated PCAPNG option header: {path}")
            break
        code, value_length = struct.unpack(endian + "HH", data[offset : offset + 4])
        offset += 4
        padded_length = (int(value_length) + 3) & ~3
        if offset + padded_length > len(data):
            raise ValueError(f"truncated PCAPNG option value: {path}")
        value = data[offset : offset + int(value_length)]
        offset += padded_length
        if code == 0:
            if value_length != 0:
                raise ValueError(f"invalid PCAPNG end-of-options marker: {path}")
            break
        options.setdefault(int(code), []).append(value)
    return options


def _interface_from_body(body: bytes, endian: str, path: Path) -> PcapNgInterface:
    if len(body) < 8:
        raise ValueError(f"truncated PCAPNG interface description: {path}")
    link_type, _reserved, snap_length = struct.unpack(endian + "HHI", body[:8])
    options = _parse_options(body[8:], endian, path)
    timestamp_scale = 1_000_000
    resolutions = options.get(9, [])
    if resolutions:
        if len(resolutions) != 1 or len(resolutions[0]) != 1:
            raise ValueError(f"invalid PCAPNG if_tsresol option: {path}")
        encoded = resolutions[0][0]
        exponent = encoded & 0x7F
        timestamp_scale = 2**exponent if encoded & 0x80 else 10**exponent
    timestamp_offset = 0
    offsets = options.get(14, [])
    if offsets:
        if len(offsets) != 1 or len(offsets[0]) != 8:
            raise ValueError(f"invalid PCAPNG if_tsoffset option: {path}")
        timestamp_offset = int(struct.unpack(endian + "q", offsets[0])[0])
    return PcapNgInterface(
        link_type=int(link_type),
        snap_length=int(snap_length),
        timestamp_scale=int(timestamp_scale),
        timestamp_offset_seconds=timestamp_offset,
    )


def _timestamp(high: int, low: int, interface: PcapNgInterface) -> float:
    ticks = (int(high) << 32) | int(low)
    return (
        ticks / float(interface.timestamp_scale)
        + float(interface.timestamp_offset_seconds)
    )


def _packet_record(
    body: bytes,
    endian: str,
    interfaces: Sequence[PcapNgInterface],
    path: Path,
    frame_id: int,
    *,
    enhanced: bool,
) -> CaptureRecord:
    if enhanced:
        if len(body) < 20:
            raise ValueError(f"truncated PCAPNG enhanced packet block: {path}")
        interface_id, high, low, captured_length, original_length = struct.unpack(
            endian + "IIIII", body[:20]
        )
        packet_offset = 20
    else:
        if len(body) < 20:
            raise ValueError(f"truncated legacy PCAPNG packet block: {path}")
        interface_id, _drops, high, low, captured_length, original_length = struct.unpack(
            endian + "HHIIII", body[:20]
        )
        packet_offset = 20
    if interface_id >= len(interfaces):
        raise ValueError(f"PCAPNG packet references unknown interface: {path}")
    interface = interfaces[int(interface_id)]
    if interface.link_type != ETHERNET_LINK_TYPE:
        raise ValueError(
            f"only Ethernet PCAPNG is supported; link type={interface.link_type}: {path}"
        )
    captured_length = int(captured_length)
    original_length = int(original_length)
    if captured_length > original_length or captured_length > interface.snap_length:
        raise ValueError(f"invalid captured packet length in PCAPNG: {path}")
    packet_end = packet_offset + captured_length
    if packet_end > len(body):
        raise ValueError(f"truncated packet data in PCAPNG block: {path}")
    return CaptureRecord(
        timestamp=_timestamp(high, low, interface),
        frame=body[packet_offset:packet_end],
        frame_id=int(frame_id),
        section_index=-1,
    )


def read_pcapng(path: Path) -> Iterator[CaptureRecord]:
    """Yield Ethernet records from a PCAPNG file without loading it into RAM."""

    path = Path(path)
    with path.open("rb") as handle:
        endian: str | None = None
        interfaces: list[PcapNgInterface] = []
        frame_id = 0
        section_count = 0
        while True:
            header = handle.read(8)
            if not header:
                if section_count == 0:
                    raise ValueError(f"empty PCAPNG file: {path}")
                return
            if len(header) != 8:
                raise ValueError(f"truncated PCAPNG block header: {path}")

            if header[:4] == PCAPNG_SECTION_HEADER:
                byte_order_magic = _read_exact(
                    handle, 4, path, "PCAPNG section byte-order magic"
                )
                try:
                    endian = PCAPNG_BYTE_ORDER[byte_order_magic]
                except KeyError as error:
                    raise ValueError(f"invalid PCAPNG byte-order magic: {path}") from error
                total_length = int(struct.unpack(endian + "I", header[4:8])[0])
                _validate_block_length(total_length, path)
                remainder = _read_exact(
                    handle, total_length - 12, path, "PCAPNG section header"
                )
                if struct.unpack(endian + "I", remainder[-4:])[0] != total_length:
                    raise ValueError(f"PCAPNG section block lengths disagree: {path}")
                interfaces = []
                section_count += 1
                continue

            if endian is None:
                raise ValueError(f"PCAPNG data appears before a section header: {path}")
            block_type, total_length = struct.unpack(endian + "II", header)
            total_length = int(total_length)
            _validate_block_length(total_length, path)
            remainder = _read_exact(handle, total_length - 8, path, "PCAPNG block")
            if struct.unpack(endian + "I", remainder[-4:])[0] != total_length:
                raise ValueError(f"PCAPNG block lengths disagree: {path}")
            body = remainder[:-4]

            if block_type == PCAPNG_INTERFACE_DESCRIPTION:
                interfaces.append(_interface_from_body(body, endian, path))
            elif block_type in (PCAPNG_ENHANCED_PACKET, PCAPNG_PACKET):
                frame_id += 1
                record = _packet_record(
                    body,
                    endian,
                    interfaces,
                    path,
                    frame_id,
                    enhanced=block_type == PCAPNG_ENHANCED_PACKET,
                )
                yield CaptureRecord(
                    timestamp=record.timestamp,
                    frame=record.frame,
                    frame_id=record.frame_id,
                    section_index=section_count - 1,
                )
            elif block_type == PCAPNG_SIMPLE_PACKET:
                raise ValueError(
                    f"simple packet blocks lack timestamps and are unsupported: {path}"
                )


def read_capture(path: Path) -> Iterator[CaptureRecord]:
    """Auto-detect classic PCAP versus PCAPNG and stream packet records."""

    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic == PCAPNG_SECTION_HEADER:
        yield from read_pcapng(path)
        return
    if magic in dataset_pcap._PCAP_FORMATS:
        for record in dataset_pcap.read_pcap(path):
            yield CaptureRecord(
                timestamp=record.timestamp,
                frame=record.frame,
                frame_id=record.frame_id,
                section_index=0,
            )
        return
    raise ValueError(f"unsupported capture format: {path}")


def _device_capture(root: Path, device: str) -> tuple[Path, bytes]:
    roots = (Path(root), Path(root) / "pcaps")
    matches: list[Path] = []
    for candidate_root in roots:
        if candidate_root.is_dir():
            matches.extend(candidate_root.glob(f"{device}_*.pcap"))
            matches.extend(candidate_root.glob(f"{device}_*.pcapng"))
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected exactly one PCAP/PCAPNG for {device}, found {len(unique)}: {unique}"
        )
    path = unique[0]
    suffix = path.stem.rsplit("_", 1)[-1]
    try:
        mac = bytes.fromhex(suffix)
    except ValueError as error:
        raise ValueError(f"capture filename does not end with a hexadecimal MAC: {path}") from error
    if len(mac) != 6:
        raise ValueError(f"capture filename MAC must contain 6 bytes: {path}")
    return path, mac


def _date_group(device: str, timestamp: float) -> str:
    """Return the global UTC-day split unit for a flow.

    ``device`` remains in the signature for source compatibility with older
    callers, but it is deliberately not part of the identifier.  A per-device
    PCAP is one physical file in UNSW-IoTraffic; using its filename or device
    prefix would falsely present each file as an independent capture.
    """

    date = datetime.fromtimestamp(float(timestamp), timezone.utc).strftime("%Y-%m-%d")
    return date


def _utc_day(timestamp: float) -> str:
    """Return a stable UTC calendar-day key for session boundary checks."""

    return datetime.fromtimestamp(float(timestamp), timezone.utc).strftime("%Y-%m-%d")


def _syn_starts_new_session(
    flow: FlowAccumulator, packet: dataset_pcap.ParsedPacket
) -> bool:
    """Detect a new SYN while accepting same-sequence SYN retransmissions.

    A SYN-ACK from the opposite direction is part of the same handshake and is
    therefore accepted once for that direction.  A second SYN in an already
    observed direction starts a new connection only when its sequence differs;
    retransmitted SYNs keep the original sequence number.
    """

    if not (int(packet.tcp_flags) & TCP_SYN):
        return False
    seen = flow.syn_sequences.get(int(packet.direction), set())
    return bool(seen) and int(packet.tcp_sequence) not in seen


def _session_boundary_reason(
    flow: FlowAccumulator,
    packet: dataset_pcap.ParsedPacket,
    inactivity_timeout_seconds: float,
) -> str | None:
    """Return the pre-packet boundary reason, if this packet starts a session.

    The caller must finish ``flow`` before adding ``packet`` when a reason is
    returned.  TCP terminal flags are handled after adding their packet by
    :func:`_session_terminal_reason`, so the RST or second FIN remains part of
    the completed session.
    """

    if float(packet.timestamp) < float(flow.last_timestamp):
        return "timestamp_regression"
    if _utc_day(packet.timestamp) != _utc_day(flow.first_timestamp):
        return "utc_day"
    if (
        float(packet.timestamp) - float(flow.last_timestamp)
        > float(inactivity_timeout_seconds)
    ):
        return "inactivity"
    if _syn_starts_new_session(flow, packet):
        return "syn"
    return None


def _session_terminal_reason(
    flow: FlowAccumulator, packet: dataset_pcap.ParsedPacket
) -> str | None:
    """Return a post-packet boundary reason for RST or a bidirectional FIN."""

    flags = int(packet.tcp_flags)
    if flags & TCP_RST:
        return "rst"
    if flags & TCP_FIN and len(flow.fin_directions) >= 2:
        return "double_fin"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_device(
    root: Path,
    device: str,
    label: int,
    config: dataset_pcap.DatasetConfig,
    *,
    inactivity_timeout_seconds: float,
    reservoir_capacity: int,
    sampling_seed: int,
    progress_interval: int = 1_000_000,
    sample_consumer: Callable[[dataset_pcap.FlowSample], None] | None = None,
) -> tuple[list[dataset_pcap.FlowSample], dict[str, Any]]:
    """Extract, sessionize, audit, and reservoir-sample one UNSW device."""

    capture_path, device_mac = _device_capture(Path(root), device)
    reservoir = Reservoir(reservoir_capacity, sampling_seed)
    active: dict[tuple[object, ...], FlowAccumulator] = {}
    counters = {
        "pcap_records": 0,
        "device_mac_frames": 0,
        "ipv4_frames": 0,
        "device_mac_ipv4_frames": 0,
        "parsed_tcp_packets": 0,
        "parsed_udp_packets": 0,
        "parsed_tls_port_packets": 0,
        "tls_like_records": 0,
        "invalid_timestamp_records": 0,
        "nonmonotonic_flow_records": 0,
        "pcapng_sections": 0,
        "section_timestamp_regressions": 0,
        "capture_time_segments": 0,
        "global_timestamp_regressions": 0,
        "inactivity_boundaries": 0,
        "syn_boundaries": 0,
        "rst_boundaries": 0,
        "double_fin_boundaries": 0,
        "utc_day_boundaries": 0,
        "flow_timestamp_regressions": 0,
        "grouped_tls_port_sessions": 0,
        "excluded_below_min_packets": 0,
        "excluded_no_tls_like_record": 0,
        "excluded_no_outbound_packet_in_window": 0,
        "eligible_flows": 0,
    }
    group_eligible_counts: dict[str, int] = {}
    start_timestamp = datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp()
    end_timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()

    def finish(flow_key: tuple[object, ...], flow: FlowAccumulator) -> None:
        counters["grouped_tls_port_sessions"] += 1
        if flow.packet_count < config.min_packets:
            counters["excluded_below_min_packets"] += 1
            return
        if flow.tls_like_records < 1:
            counters["excluded_no_tls_like_record"] += 1
            return
        observed_packets = sorted(
            flow.observed_packets,
            key=lambda packet: (packet.timestamp, packet.frame_id),
        )
        if not any(packet.direction == 1 for packet in observed_packets):
            counters["excluded_no_outbound_packet_in_window"] += 1
            return
        group = _date_group(device, flow.first_timestamp)
        sample = dataset_pcap._encode_flow(
            observed_packets,
            label=label,
            group=group,
            max_packets=config.max_packets,
        )
        reservoir.offer(sample)
        if sample_consumer is not None:
            sample_consumer(sample)
        counters["eligible_flows"] += 1
        group_eligible_counts[group] = group_eligible_counts.get(group, 0) + 1

    print(f"[{device}] reading {capture_path.name}", flush=True)
    current_section: int | None = None
    current_section_last_timestamp: float | None = None
    previous_valid_timestamp: float | None = None

    def flush_active() -> None:
        for flow_key, flow in active.items():
            finish(flow_key, flow)
        active.clear()

    for record in read_capture(capture_path):
        if record.section_index != current_section:
            if current_section is not None:
                flush_active()
                if (
                    current_section_last_timestamp is not None
                    and record.timestamp < current_section_last_timestamp
                ):
                    counters["section_timestamp_regressions"] += 1
            current_section = int(record.section_index)
            counters["pcapng_sections"] += 1
            current_section_last_timestamp = float(record.timestamp)
            previous_valid_timestamp = None
        else:
            current_section_last_timestamp = max(
                float(current_section_last_timestamp), float(record.timestamp)
            )
        counters["pcap_records"] += 1
        if progress_interval and counters["pcap_records"] % progress_interval == 0:
            print(
                f"[{device}] {counters['pcap_records']:,} packets, "
                f"{counters['eligible_flows']:,} eligible sessions finished",
                flush=True,
            )
        if not (start_timestamp <= record.timestamp < end_timestamp):
            counters["invalid_timestamp_records"] += 1
            continue
        if previous_valid_timestamp is None:
            counters["capture_time_segments"] += 1
        elif record.timestamp < previous_valid_timestamp:
            # The published per-device files are merges.  Any timestamp
            # regression starts a fresh capture segment; even a small backward
            # step must not merge an identical five-tuple across source runs.
            flush_active()
            counters["capture_time_segments"] += 1
            counters["global_timestamp_regressions"] += 1
        previous_valid_timestamp = float(record.timestamp)
        frame = record.frame
        device_related = bool(
            len(frame) >= 12
            and (frame[:6] == device_mac or frame[6:12] == device_mac)
        )
        if device_related:
            counters["device_mac_frames"] += 1
        ethernet = dataset_pcap._ethernet_payload_offset(frame)
        is_ipv4 = ethernet is not None and ethernet[0] == dataset_pcap._IPV4_ETHERTYPE
        if is_ipv4:
            counters["ipv4_frames"] += 1
            if device_related:
                counters["device_mac_ipv4_frames"] += 1

        packet = dataset_pcap.parse_packet(
            record.timestamp,
            frame,
            device_mac,
            config.tls_port,
            frame_id=record.frame_id,
        )
        if packet is None:
            continue
        protocol = int(packet.flow_key[0])
        if protocol == dataset_pcap._TCP:
            counters["parsed_tcp_packets"] += 1
        elif protocol == dataset_pcap._UDP:
            counters["parsed_udp_packets"] += 1
        port_candidate = config.tls_port in (
            packet.source_port,
            packet.destination_port,
        )
        if port_candidate:
            counters["parsed_tls_port_packets"] += 1
        if packet.is_tls_record:
            counters["tls_like_records"] += 1
        if not port_candidate or protocol != dataset_pcap._TCP:
            continue

        flow = active.get(packet.flow_key)
        boundary_reason = (
            _session_boundary_reason(flow, packet, inactivity_timeout_seconds)
            if flow is not None
            else None
        )
        if boundary_reason is not None:
            if boundary_reason == "timestamp_regression":
                counters["nonmonotonic_flow_records"] += 1
                counters["flow_timestamp_regressions"] += 1
            else:
                counters[f"{boundary_reason}_boundaries"] += 1
            finish(packet.flow_key, flow)
            active.pop(packet.flow_key, None)
            flow = None
        if flow is None:
            flow = FlowAccumulator.start(packet, config.max_packets)
            active[packet.flow_key] = flow
        else:
            flow.add(packet, config.max_packets)

        terminal_reason = _session_terminal_reason(flow, packet)
        if terminal_reason is not None:
            counters[f"{terminal_reason}_boundaries"] += 1
            finish(packet.flow_key, flow)
            active.pop(packet.flow_key, None)

    flush_active()

    samples = sorted(
        reservoir.items,
        key=lambda sample: (
            str(sample.group),
            float(sample.timestamps[0]),
            int(sample.frame_ids[0]),
        ),
    )
    audit = {
        "device": device,
        "label": int(label),
        "capture": str(capture_path),
        "capture_name": capture_path.name,
        "capture_size_bytes": int(capture_path.stat().st_size),
        "capture_sha256": _sha256(capture_path),
        "device_mac": device_mac.hex(":"),
        "inactivity_timeout_seconds": float(inactivity_timeout_seconds),
        "reservoir_capacity": int(reservoir_capacity),
        "sampling_seed": int(sampling_seed),
        **counters,
        "parsed_device_ipv4_tcp_udp_packets": int(
            counters["parsed_tcp_packets"] + counters["parsed_udp_packets"]
        ),
        "sampled_flows_before_common_balance": int(len(samples)),
        "eligible_groups": int(len(group_eligible_counts)),
        "eligible_flows_by_group": dict(sorted(group_eligible_counts.items())),
    }
    print(
        f"[{device}] done: {counters['pcap_records']:,} packets, "
        f"{counters['eligible_flows']:,} eligible flows, "
        f"{len(samples):,} reservoir samples across {len(group_eligible_counts)} UTC days",
        flush=True,
    )
    return samples, audit


def _dataset_from_samples(
    samples: Sequence[dataset_pcap.FlowSample],
    devices: Sequence[str],
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not samples:
        raise RuntimeError("no UNSW samples were selected")
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
        "labels": np.asarray([sample.label for sample in samples], dtype=np.int16),
        "groups": groups,
        # There is one physical source file per UNSW device.  The reproducible
        # split unit is the global UTC day, so capture_ids intentionally match
        # groups rather than exposing a misleading filename identity.
        "capture_ids": groups.copy(),
        "device_names": np.asarray(tuple(devices)),
        "data_protocol": np.asarray(UNSW_DATA_PROTOCOL),
        "unsw_extraction_manifest_json": np.asarray(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        ),
    }
    dataset_pcap.validate_dataset(dataset, expected_width=dataset["lengths"].shape[1])
    return dataset


def _extract_balanced_samples(
    root: Path,
    *,
    devices: Sequence[str],
    target_flows_per_class: int,
    minimum_flows_per_class: int,
    sampling_seed: int,
    inactivity_timeout_seconds: float,
    max_packets: int,
) -> tuple[list[dataset_pcap.FlowSample], list[dict[str, Any]], int]:
    """Extract and balance an explicitly ordered UNSW device set.

    Labels are defined solely by ``devices`` order.  Keeping this common core
    prevents the 4/6/10-class protocols from silently drifting to different
    packet filters, session boundaries, or reservoir-sampling rules.
    """

    devices = tuple(str(device).strip() for device in devices)
    if len(devices) < 2 or any(not device for device in devices):
        raise ValueError("devices must contain at least two nonempty names")
    if len(set(devices)) != len(devices):
        raise ValueError("devices must not contain duplicates")
    if int(target_flows_per_class) < 1:
        raise ValueError("target_flows_per_class must be positive")
    if not 1 <= int(minimum_flows_per_class) <= int(target_flows_per_class):
        raise ValueError(
            "minimum_flows_per_class must lie in [1, target_flows_per_class]"
        )
    if int(max_packets) < 4:
        raise ValueError("max_packets must be at least four")
    if float(inactivity_timeout_seconds) <= 0.0:
        raise ValueError("inactivity_timeout_seconds must be positive")

    config = dataset_pcap.DatasetConfig(
        devices=devices,
        min_packets=4,
        max_packets=int(max_packets),
        tls_port=443,
        max_feature_gap_seconds=10.0,
        max_frame_length=1514.0,
    )
    by_class: list[list[dataset_pcap.FlowSample]] = []
    audits: list[dict[str, Any]] = []
    for label, device in enumerate(devices):
        samples, audit = extract_device(
            Path(root),
            device,
            label,
            config,
            inactivity_timeout_seconds=float(inactivity_timeout_seconds),
            reservoir_capacity=int(target_flows_per_class),
            sampling_seed=int(sampling_seed) + 1009 * label,
        )
        by_class.append(samples)
        audits.append(audit)

    common_count = min(len(samples) for samples in by_class)
    if common_count < int(minimum_flows_per_class):
        counts = {device: len(samples) for device, samples in zip(devices, by_class)}
        raise RuntimeError(
            f"only {common_count} common flows per class; minimum is "
            f"{minimum_flows_per_class}: {counts}"
        )
    balanced: list[dataset_pcap.FlowSample] = []
    for label, samples in enumerate(by_class):
        if len(samples) > common_count:
            rng = random.Random(int(sampling_seed) + 50_000 + label)
            indices = sorted(rng.sample(range(len(samples)), common_count))
            samples = [samples[index] for index in indices]
        balanced.extend(samples)
        audits[label]["selected_flows_after_common_balance"] = int(len(samples))
        audits[label]["selected_groups_after_common_balance"] = int(
            len({str(sample.group) for sample in samples})
        )

    return balanced, audits, int(common_count)


def _extraction_manifest(
    root: Path,
    *,
    schema: str,
    devices: Sequence[str],
    audits: Sequence[Mapping[str, Any]],
    common_count: int,
    target_flows_per_class: int,
    minimum_flows_per_class: int,
    sampling_seed: int,
    inactivity_timeout_seconds: float,
    max_packets: int,
) -> dict[str, Any]:
    """Build the shared, auditable extraction manifest."""

    devices = tuple(str(device) for device in devices)
    return {
        "schema": str(schema),
        "data_protocol": UNSW_DATA_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(Path(root).resolve()),
        "devices": list(devices),
        "class_count": int(len(devices)),
        "four_class_devices": list(devices[:4]) if len(devices) >= 4 else [],
        "class_order_is_label_order": True,
        "tls_port": 443,
        "min_packets": 4,
        "max_packets": int(max_packets),
        "inactivity_timeout_seconds": float(inactivity_timeout_seconds),
        "group_definition": "global UTC-date-of-first-flow-packet (no device prefix)",
        "capture_id_definition": (
            "capture_ids equal global UTC-day groups; source filenames are not "
            "independent captures because UNSW publishes one file per device"
        ),
        "session_boundary": (
            "the configured inactivity timeout, a new-sequence TCP SYN in an "
            "already observed direction, TCP RST, FIN from both directions, "
            "a UTC-day boundary, a PCAPNG section boundary, or any timestamp "
            "regression; same-sequence SYN retransmissions stay in-session; "
            "published per-device files may concatenate source captures out of "
            "chronological order"
        ),
        "sampling": {
            "method": "per-class deterministic uniform reservoir, then common-count balance",
            "seed": int(sampling_seed),
            "requested_per_class": int(target_flows_per_class),
            "selected_per_class": int(common_count),
            "minimum_required_per_class": int(minimum_flows_per_class),
        },
        "filters": [
            "Ethernet IPv4 TCP",
            "source or destination port 443",
            "at least one plausible TLS record header",
            f"at least four packets in the {float(inactivity_timeout_seconds):g}-second session",
            f"at least one device-outbound packet in the first {int(max_packets)} events",
            "timestamps restricted to [2015-01-01, 2020-01-01) UTC",
        ],
        "device_audits": list(audits),
    }


def extract_balanced_dataset(
    root: Path,
    *,
    devices: Sequence[str],
    target_flows_per_class: int,
    minimum_flows_per_class: int,
    sampling_seed: int = 20260904,
    inactivity_timeout_seconds: float = 120.0,
    max_packets: int = 32,
    schema: str = "unsw-iotraffic-method32-packet-dataset-v2-generic",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract one balanced dataset for an explicitly ordered device tuple."""

    devices = tuple(str(device) for device in devices)
    balanced, audits, common_count = _extract_balanced_samples(
        Path(root),
        devices=devices,
        target_flows_per_class=int(target_flows_per_class),
        minimum_flows_per_class=int(minimum_flows_per_class),
        sampling_seed=int(sampling_seed),
        inactivity_timeout_seconds=float(inactivity_timeout_seconds),
        max_packets=int(max_packets),
    )
    manifest = _extraction_manifest(
        Path(root),
        schema=schema,
        devices=devices,
        audits=audits,
        common_count=common_count,
        target_flows_per_class=int(target_flows_per_class),
        minimum_flows_per_class=int(minimum_flows_per_class),
        sampling_seed=int(sampling_seed),
        inactivity_timeout_seconds=float(inactivity_timeout_seconds),
        max_packets=int(max_packets),
    )
    return _dataset_from_samples(balanced, devices, manifest), manifest


def extract_datasets(
    root: Path,
    *,
    devices: Sequence[str] = DEFAULT_DEVICES_6,
    target_flows_per_class: int = 500,
    minimum_flows_per_class: int = 200,
    sampling_seed: int = 20260904,
    inactivity_timeout_seconds: float = 120.0,
    max_packets: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Extract one balanced six-class dataset and its exact four-class subset."""

    devices = tuple(str(device) for device in devices)
    if devices != DEFAULT_DEVICES_6:
        raise ValueError(
            "the reproducible UNSW 4/6-class protocol requires the documented "
            "six devices in their fixed order"
        )
    balanced, audits, common_count = _extract_balanced_samples(
        Path(root),
        devices=devices,
        target_flows_per_class=int(target_flows_per_class),
        minimum_flows_per_class=int(minimum_flows_per_class),
        sampling_seed=int(sampling_seed),
        inactivity_timeout_seconds=float(inactivity_timeout_seconds),
        max_packets=int(max_packets),
    )
    manifest = _extraction_manifest(
        Path(root),
        schema="unsw-iotraffic-method32-packet-dataset-v2",
        devices=devices,
        audits=audits,
        common_count=common_count,
        target_flows_per_class=int(target_flows_per_class),
        minimum_flows_per_class=int(minimum_flows_per_class),
        sampling_seed=int(sampling_seed),
        inactivity_timeout_seconds=float(inactivity_timeout_seconds),
        max_packets=int(max_packets),
    )
    six_class = _dataset_from_samples(balanced, devices, manifest)
    four_samples = [sample for sample in balanced if int(sample.label) < 4]
    four_manifest = {
        **manifest,
        "schema": "unsw-iotraffic-method32-packet-dataset-v2-4class-subset",
        "devices": list(devices[:4]),
        "class_count": 4,
        "derived_as_exact_label_prefix_of_six_class": True,
        "device_audits": audits[:4],
    }
    four_class = _dataset_from_samples(four_samples, devices[:4], four_manifest)
    return four_class, six_class, manifest


def extract_ten_class_dataset(
    root: Path,
    *,
    target_flows_per_class: int = 200,
    minimum_flows_per_class: int = 200,
    sampling_seed: int = 20260904,
    inactivity_timeout_seconds: float = 120.0,
    max_packets: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract the fixed ten-device protocol without touching 4/6-class data."""

    return extract_balanced_dataset(
        Path(root),
        devices=DEFAULT_DEVICES_10,
        target_flows_per_class=int(target_flows_per_class),
        minimum_flows_per_class=int(minimum_flows_per_class),
        sampling_seed=int(sampling_seed),
        inactivity_timeout_seconds=float(inactivity_timeout_seconds),
        max_packets=int(max_packets),
        schema="unsw-iotraffic-method32-packet-dataset-v2-10class",
    )


def _write_audit_csv(path: Path, audits: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for audit in audits:
        rows.append(
            {
                key: value
                for key, value in audit.items()
                if key not in {"eligible_flows_by_group"}
            }
        )
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> dict[str, Any]:
    """Exercise little-endian PCAPNG parsing and strict truncation failure."""

    def block(block_type: int, body: bytes) -> bytes:
        padded = body + b"\x00" * ((-len(body)) % 4)
        total = 12 + len(padded)
        return (
            struct.pack("<II", block_type, total)
            + padded
            + struct.pack("<I", total)
        )

    shb_body = bytes.fromhex("4d3c2b1a") + struct.pack("<HHq", 1, 0, -1)
    idb_body = struct.pack("<HHI", 1, 0, 65535) + struct.pack("<HH", 9, 1)
    idb_body += b"\x06\x00\x00\x00" + struct.pack("<HH", 0, 0)
    frame = bytes.fromhex("00112233445566778899aabb0800") + b"payload"
    ticks = 1_700_000_000_123_456
    epb_body = struct.pack(
        "<IIIII", 0, ticks >> 32, ticks & 0xFFFFFFFF, len(frame), len(frame)
    ) + frame
    section = block(0x0A0D0D0A, shb_body) + block(1, idb_body) + block(6, epb_body)
    payload = section + section

    with tempfile.TemporaryDirectory(prefix="unsw_pcapng_test_") as directory:
        valid = Path(directory) / "valid.pcap"
        valid.write_bytes(payload)
        records = list(read_pcapng(valid))
        if (
            len(records) != 2
            or records[0].frame != frame
            or records[0].frame_id != 1
            or records[1].frame_id != 2
            or [record.section_index for record in records] != [0, 1]
        ):
            raise AssertionError("PCAPNG record round trip failed")
        if abs(records[0].timestamp - ticks / 1_000_000) > 1e-6:
            raise AssertionError("PCAPNG timestamp decoding failed")
        truncated = Path(directory) / "truncated.pcap"
        truncated.write_bytes(payload[:-1])
        try:
            list(read_pcapng(truncated))
        except ValueError:
            pass
        else:
            raise AssertionError("truncated PCAPNG was accepted")
    return {
        "status": "passed",
        "records": 2,
        "sections": 2,
        "timestamp_resolution": "microseconds",
        "truncation_rejected": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("root", type=Path)
    inspect_parser.add_argument("--classes", type=int, choices=(6, 10), default=6)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("root", type=Path)
    extract_parser.add_argument("output_4class", type=Path)
    extract_parser.add_argument("output_6class", type=Path)
    extract_parser.add_argument("--audit-dir", type=Path, required=True)
    extract_parser.add_argument("--target-per-class", type=int, default=500)
    extract_parser.add_argument("--minimum-per-class", type=int, default=200)
    extract_parser.add_argument("--sampling-seed", type=int, default=20260904)
    extract_ten_parser = subparsers.add_parser(
        "extract-10", help="extract the fixed balanced ten-device dataset"
    )
    extract_ten_parser.add_argument("root", type=Path)
    extract_ten_parser.add_argument("output_10class", type=Path)
    extract_ten_parser.add_argument("--audit-dir", type=Path, required=True)
    extract_ten_parser.add_argument("--target-per-class", type=int, default=200)
    extract_ten_parser.add_argument("--minimum-per-class", type=int, default=200)
    extract_ten_parser.add_argument("--sampling-seed", type=int, default=20260904)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "self-test":
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.command == "inspect":
        devices = DEFAULT_DEVICES_10 if args.classes == 10 else DEFAULT_DEVICES_6
        for device in devices:
            path, mac = _device_capture(args.root, device)
            first = next(read_capture(path))
            print(
                f"{device}: {path.name}, {path.stat().st_size} bytes, "
                f"MAC={mac.hex(':')}, first={first.timestamp:.6f}, "
                f"frame_bytes={len(first.frame)}"
            )
        return
    if args.command == "extract":
        output_4class = args.output_4class.resolve()
        output_6class = args.output_6class.resolve()
        audit_dir = args.audit_dir.resolve()
        if output_4class == output_6class:
            raise ValueError("four-class and six-class outputs must be different")
        audit_dir.mkdir(parents=True, exist_ok=True)
        four_class, six_class, manifest = extract_datasets(
            args.root,
            target_flows_per_class=args.target_per_class,
            minimum_flows_per_class=args.minimum_per_class,
            sampling_seed=args.sampling_seed,
        )
        dataset_pcap.save_dataset(output_4class, four_class)
        dataset_pcap.save_dataset(output_6class, six_class)
        manifest_path = audit_dir / "unsw_extraction_manifest.json"
        audit_csv_path = audit_dir / "unsw_extraction_device_summary.csv"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_audit_csv(audit_csv_path, manifest["device_audits"])
        print(f"Saved four-class dataset: {output_4class}")
        print(f"Saved six-class dataset : {output_6class}")
        print(f"Saved extraction manifest: {manifest_path}")
        print(f"Saved extraction summary : {audit_csv_path}")
        return
    if args.command == "extract-10":
        output_10class = args.output_10class.resolve()
        audit_dir = args.audit_dir.resolve()
        audit_dir.mkdir(parents=True, exist_ok=True)
        ten_class, manifest = extract_ten_class_dataset(
            args.root,
            target_flows_per_class=args.target_per_class,
            minimum_flows_per_class=args.minimum_per_class,
            sampling_seed=args.sampling_seed,
        )
        dataset_pcap.save_dataset(output_10class, ten_class)
        manifest_path = audit_dir / "unsw_extraction_manifest.json"
        audit_csv_path = audit_dir / "unsw_extraction_device_summary.csv"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_audit_csv(audit_csv_path, manifest["device_audits"])
        print(f"Saved ten-class dataset: {output_10class}")
        print(f"Saved extraction manifest: {manifest_path}")
        print(f"Saved extraction summary : {audit_csv_path}")
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
