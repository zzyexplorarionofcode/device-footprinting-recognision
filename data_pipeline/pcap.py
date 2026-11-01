"""Dataset preparation for Method31 and the retained defense baselines.

This module has one responsibility: convert classic PCAP captures into the
fixed-width trace representation used by this project.  It deliberately does
not parse command-line arguments, print progress, read Method31 configuration
files, train classifiers, or apply a defense.  Those orchestration tasks live
in :mod:`method31_tools`.

Trace representation
--------------------
Each flow is represented by four arrays with shape ``[flow, event_position]``:

``lengths``
    Captured Ethernet frame lengths in bytes.
``gaps``
    Nonnegative time since the previous event in the same bidirectional flow.
``directions``
    ``+1`` for packets sent by the IoT device and ``-1`` for received packets.
``mask``
    ``1`` for a valid event and ``0`` for right-side padding.

Newly extracted datasets also carry audit-only ``timestamps``, ``frame_ids``,
``source_ips``, and ``destination_ips`` arrays with the same shape.  They map
each event back to Wireshark's ``frame.time_epoch``, ``frame.number``,
``ip.src``, and ``ip.dst`` fields.  The classifier and defenses still consume
only the four established arrays above, so these fields do not change Method31
or baseline inputs.

The companion one-dimensional arrays are ``labels`` (integer device class),
``groups`` (``device/capture`` identifier), and ``device_names``.  The schema
matches the existing ``flows_tls.npz`` files and the input schema accepted by
``defense_baselines``.  Newly extracted NPZ files also contain a scalar Unicode
``capture_summaries_json`` audit record.  It explains, per PCAP, how raw frames
became bidirectional candidates, why candidates were excluded, and which flows
were retained.  It is descriptive metadata and is never a model feature.

PCAP and TLS scope
------------------
The parser intentionally follows the project's original ``01_extract_flows``
semantics so old and new experiments remain comparable: classic PCAP only,
Ethernet link layer, IPv4, TCP/UDP five-tuples, and a TLS-candidate flow when at
least one TCP packet on the configured TLS port contains a plausible TLS record
header.  A dummy event used by a defense is only an abstract cover record; this
extractor never invents packets.
"""

from __future__ import annotations

# =============================================================================
# 中文阅读导图
# =============================================================================
# 本文件只负责“把原始抓包变成算法可以消费的数据”，不负责执行 Method31，
# 也不负责读取 method31_config.txt、训练攻击模型或打印实验表格。这样的边界
# 很重要：只要本文件的输出模式保持不变，Method31 与各 baseline 就可以共享
# 同一份数据，而无需互相导入各自的实现。
#
# 建议按下面的顺序阅读：
#   1. DatasetConfig：确定设备顺序、最少/最多包数和 TLS 端口；
#   2. read_pcap：从 classic PCAP 中流式读出“时间戳 + 原始帧”；
#   3. parse_packet：解析 Ethernet/VLAN/IPv4/TCP/UDP，并生成双向流键；
#   4. extract_device / extract_dataset：按设备与 capture 汇总合格 TLS 流；
#   5. validate_dataset：在数据进入算法前检查所有形状和语义约束；
#   6. split_by_capture：按完整 capture 划分训练/测试，避免同源流量泄漏；
#   7. make_features：把长度、方向与间隔编码为分类器使用的 2L 维特征。
#
# 项目统一用四个二维数组表示一批固定窗口流，形状均为 [N, L]：
#   lengths    —— Ethernet 抓获帧长度，单位 byte；
#   gaps       —— 同一双向流中与前一个事件的非负时间差，单位 second；
#   directions —— 设备发出为 +1，设备接收为 -1；
#   mask       —— 有效事件为 1，右侧补齐位置为 0。
# 新提取数据另带同形状timestamps/frame_ids/source_ips/destination_ips，分别
# 对应Wireshark frame.time_epoch/frame.number/ip.src/ip.dst。它们只服务
# CSV/HTML人工审计，不进入分类特征，也不传给Method31/baselines。
# 另外，labels 和 groups 都是一维长度 N：前者是设备类别，后者是
# “设备/capture”标识。groups 是无泄漏切分的关键，不能随意丢弃。
# capture_summaries_json则是单个Unicode JSON标量，记录每个PCAP的提取漏斗
# 和保留flow端点，用来在检查HTML开头解释“为什么得到这些flow”。它不是
# [N,L]数组，也不会被切片后送给分类器、Method31或baseline。

import csv
import json
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration required by the dataset layer.

    ``devices`` determines both extraction order and numeric class labels.  A
    change in this tuple is therefore a dataset-version change, not merely a
    cosmetic rename.  ``max_packets`` is also the classifier observation width.
    """

    devices: tuple[str, ...]
    min_packets: int = 4
    max_packets: int = 32
    tls_port: int = 443
    mac_filename: str = "_iotdevice-mac.txt"
    pcap_pattern: str = "*.pcap"
    max_feature_gap_seconds: float = 10.0
    max_frame_length: float = 1514.0


# PcapRecord 是最靠近文件格式的一层。这里不提前解析协议字段，既能让
# read_pcap 保持流式、低内存，也能把“文件容器错误”和“网络协议错误”分开定位。
@dataclass(frozen=True)
class PcapRecord:
    """One captured frame from a classic PCAP file."""

    timestamp: float
    frame: bytes
    # 1-based classic-PCAP record order; this matches Wireshark frame.number.
    frame_id: int = 0


# ParsedPacket 只保留后续分流和编码真正需要的信息。flow_key 已被规范化为
# 双向键，因此同一 TCP/UDP 会话的上行包和下行包会进入同一个列表。
@dataclass(frozen=True)
class ParsedPacket:
    """Packet fields needed to group frames into bidirectional flows."""

    flow_key: tuple[object, ...]
    timestamp: float
    direction: int
    frame_length: int
    source_port: int
    destination_port: int
    is_tls_record: bool
    # These audit fields preserve the per-packet orientation that is deliberately
    # removed when the bidirectional flow_key sorts its two endpoints.
    frame_id: int = 0
    source_ip: str = ""
    destination_ip: str = ""
    tcp_flags: int = 0
    tcp_sequence: int = 0


# FlowSample 是“单条流”的中间对象；等所有流收集完成后才由 extract_dataset
# stack 成二维矩阵。使用冻结 dataclass 可避免样本元信息被意外重新赋值。
@dataclass(frozen=True)
class FlowSample:
    """One fixed-width flow before samples are stacked into a dataset."""

    lengths: np.ndarray
    timestamps: np.ndarray
    frame_ids: np.ndarray
    source_ips: np.ndarray
    destination_ips: np.ndarray
    gaps: np.ndarray
    directions: np.ndarray
    mask: np.ndarray
    label: int
    group: str


# =============================================================================
# 协议常量
# =============================================================================
# Classic PCAP uses four magic values: little/big endian and micro/nanoseconds.
# Keeping the table explicit makes unsupported PCAP-NG files fail clearly rather
# than being silently misparsed as a classic header.
_PCAP_FORMATS: Mapping[bytes, tuple[str, int]] = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
}
_ETHERNET_LINK_TYPE = 1
_IPV4_ETHERTYPE = 0x0800
_VLAN_ETHERTYPES = {0x8100, 0x88A8}
_TCP = 6
_UDP = 17
_TLS_RECORD_TYPES = {20, 21, 22, 23}


def read_pcap(path: Path) -> Iterator[PcapRecord]:
    """Yield records from a classic Ethernet PCAP.

    The function is a generator, so a large capture is streamed instead of
    loaded into memory.  Truncated per-packet data is rejected because parsing a
    partial Ethernet/IP header could create a wrong flow key or direction.
    """

    # 读取策略是“严格失败”：损坏、截断或非 Ethernet 的文件不会被静默跳过。
    # 静默跳过虽然看起来方便，却可能让某个设备少掉一部分 capture，进而改变
    # 类别分布和最终防御指标，因此这里宁可尽早给出带路径的明确异常。
    path = Path(path)
    with path.open("rb") as handle:
        # classic PCAP 全局头固定为 24 byte；前 4 byte magic 同时决定端序与
        # 时间戳精度，20:24 则保存 link-layer network type。
        global_header = handle.read(24)
        if len(global_header) != 24:
            raise ValueError(f"PCAP global header is truncated: {path}")
        try:
            endian, time_scale = _PCAP_FORMATS[global_header[:4]]
        except KeyError as error:
            raise ValueError(
                f"unsupported PCAP format (PCAP-NG is not accepted): {path}"
            ) from error

        link_type = struct.unpack(endian + "I", global_header[20:24])[0]
        if link_type != _ETHERNET_LINK_TYPE:
            raise ValueError(
                f"only Ethernet PCAP is supported; link type={link_type}: {path}"
            )

        # 每个 record 是 16-byte 局部头加 captured_length byte 的帧体。
        # yield 让调用方逐包消费，不必把整个 PCAP 一次性读入内存。
        frame_id = 0
        while True:
            packet_header = handle.read(16)
            if not packet_header:
                return
            # Wireshark's frame.number is the 1-based record order within one
            # capture file.  Increment before yielding so both views use the
            # same identifier; filtered-out frames still consume an ID.
            frame_id += 1
            if len(packet_header) != 16:
                raise ValueError(f"truncated PCAP packet header: {path}")
            seconds, fraction, captured_length, _original_length = struct.unpack(
                endian + "IIII", packet_header
            )
            frame = handle.read(captured_length)
            if len(frame) != captured_length:
                raise ValueError(f"truncated PCAP packet body: {path}")
            yield PcapRecord(seconds + fraction / time_scale, frame, frame_id)


def _ethernet_payload_offset(frame: bytes) -> tuple[int, int] | None:
    """Return ``(ethertype, payload_offset)`` and handle one or more VLAN tags."""

    # Ethernet II 基本头为 14 byte。VLAN tag 每层再增加 4 byte；while 而非
    # 单个 if 使 802.1Q 与 QinQ 多层 tag 都能被正确越过。
    if len(frame) < 14:
        return None
    ethertype = struct.unpack("!H", frame[12:14])[0]
    offset = 14
    while ethertype in _VLAN_ETHERTYPES:
        if len(frame) < offset + 4:
            return None
        ethertype = struct.unpack("!H", frame[offset + 2 : offset + 4])[0]
        offset += 4
    return ethertype, offset


def parse_packet(
    timestamp: float,
    frame: bytes,
    device_mac: bytes,
    tls_port: int,
    *,
    frame_id: int = 0,
) -> ParsedPacket | None:
    """Parse a relevant Ethernet/IPv4/TCP-or-UDP frame.

    Direction is based on the configured device MAC, not on IP/port heuristics.
    The bidirectional flow key sorts endpoint tuples, so the two directions of a
    connection are grouped together.  Non-first IPv4 fragments are skipped
    because they do not contain a reliable transport header.
    """

    # 第一步只定位 Ethernet payload；任何长度不足或不支持的帧都返回 None，
    # 表示“与本数据集无关”，而不是把正常的非 IPv4 流量当成文件损坏。
    ethernet = _ethernet_payload_offset(frame)
    if ethernet is None:
        return None
    ethertype, ip_offset = ethernet
    if ethertype != _IPV4_ETHERTYPE or len(frame) < ip_offset + 20:
        return None

    # IPv4 首字节高 4 bit 是版本，低 4 bit 是 IHL（单位为 4 byte）。
    # transport_offset 必须由 IHL 计算，不能假设 IP 头永远没有 options。
    version = frame[ip_offset] >> 4
    ip_header_length = (frame[ip_offset] & 0x0F) * 4
    if version != 4 or ip_header_length < 20:
        return None
    transport_offset = ip_offset + ip_header_length
    if len(frame) < transport_offset + 4:
        return None

    # The low 13 bits are the IPv4 fragment offset.  A nonzero offset means the
    # transport ports are absent from this fragment, so it cannot be keyed here.
    fragment_field = struct.unpack("!H", frame[ip_offset + 6 : ip_offset + 8])[0]
    if fragment_field & 0x1FFF:
        return None

    protocol = int(frame[ip_offset + 9])
    if protocol not in (_TCP, _UDP):
        return None

    # 方向只根据设备 MAC 判定，不根据“443 端通常是服务器”等启发式规则。
    # 这样即使端口角色变化，+1/-1 的物理含义仍然稳定。
    destination_mac, source_mac = frame[0:6], frame[6:12]
    if source_mac == device_mac:
        direction = 1
    elif destination_mac == device_mac:
        direction = -1
    else:
        return None

    source_ip = frame[ip_offset + 12 : ip_offset + 16]
    destination_ip = frame[ip_offset + 16 : ip_offset + 20]
    # IPv4 bytes are converted without DNS/name resolution.  The strings thus
    # match Wireshark ip.src/ip.dst and retain this packet's direction, whereas
    # flow_key below sorts endpoints only for bidirectional aggregation.
    source_ip_text = ".".join(map(str, source_ip))
    destination_ip_text = ".".join(map(str, destination_ip))
    source_port, destination_port = struct.unpack(
        "!HH", frame[transport_offset : transport_offset + 4]
    )

    # 端口 443 只是候选条件；还要求 TCP payload 的前三个 byte 看起来像
    # TLS record header：content type 属于 20..23，major version 为 3。
    # 这是为复现旧提取器而保留的轻量识别，不等价于完整 TLS 协议解析。
    is_tls_record = False
    if protocol == _TCP and tls_port in (source_port, destination_port):
        # A valid TCP data-offset nibble is at least 5 (20 bytes).  We only need
        # the first three TLS bytes: content type and major/minor version.
        if len(frame) >= transport_offset + 20:
            tcp_header_length = (frame[transport_offset + 12] >> 4) * 4
            if tcp_header_length >= 20:
                payload_offset = transport_offset + tcp_header_length
                payload = frame[payload_offset:]
                is_tls_record = bool(
                    len(payload) >= 3
                    and payload[0] in _TLS_RECORD_TYPES
                    and payload[1] == 3
                )

    # 对两个 endpoint 排序后，A→B 与 B→A 得到同一个 flow_key；protocol
    # 放在最前面，避免端点相同的 TCP 和 UDP 流被错误合并。
    endpoint_a = (source_ip, int(source_port))
    endpoint_b = (destination_ip, int(destination_port))
    flow_key = (protocol, *sorted((endpoint_a, endpoint_b)))
    return ParsedPacket(
        flow_key=flow_key,
        timestamp=float(timestamp),
        direction=direction,
        # Match 01_extract_flows: the feature is the captured Ethernet frame
        # length, not an inferred IP/TLS payload size.
        frame_length=len(frame),
        source_port=int(source_port),
        destination_port=int(destination_port),
        is_tls_record=is_tls_record,
        frame_id=int(frame_id),
        source_ip=source_ip_text,
        destination_ip=destination_ip_text,
        tcp_flags=int(frame[transport_offset + 13]) if protocol == _TCP and len(frame) >= transport_offset + 20 else 0,
        tcp_sequence=struct.unpack("!I", frame[transport_offset + 4:transport_offset + 8])[0]
        if protocol == _TCP and len(frame) >= transport_offset + 20 else 0,
    )


def _read_device_mac(device_directory: Path, filename: str) -> bytes:
    """Read the Sentinel dataset's per-device MAC file."""

    # Sentinel 每个设备目录都有一个文本 MAC。去掉冒号后必须恰好解析成
    # 6 byte；这一检查可以捕获空文件、拼写错误和意外多写的字符。
    path = device_directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"device MAC file is missing: {path}")
    compact = path.read_text(encoding="utf-8").strip().replace(":", "")
    try:
        value = bytes.fromhex(compact)
    except ValueError as error:
        raise ValueError(f"invalid MAC address in {path}") from error
    if len(value) != 6:
        raise ValueError(f"MAC address must contain 6 bytes: {path}")
    return value


def _encode_flow(
    packets: Sequence[ParsedPacket],
    *,
    label: int,
    group: str,
    max_packets: int,
) -> FlowSample:
    """Encode one time-sorted variable-length flow as prefix-padded arrays."""

    # 分类器只观察前 max_packets 个事件。这里的截断仅定义特征窗口，不会
    # 回写或修改原 PCAP；不足的流在右侧以全零占位，并由 mask 区分。
    selected = list(packets[:max_packets])
    lengths = np.zeros(max_packets, dtype=np.float32)
    # Unix epoch量级的秒值约为1e9；若使用float32，相邻毫秒甚至秒级差异可能
    # 因有效位不足而消失。因此原始PCAP时间戳必须用float64独立保存。
    timestamps = np.zeros(max_packets, dtype=np.float64)
    # frame_id使用0作为padding哨兵；合法Wireshark frame.number从1开始。
    # IPv4点分十进制最长15字符，固定Unicode数组可安全写入NPZ而无需pickle。
    frame_ids = np.zeros(max_packets, dtype=np.int32)
    source_ips = np.full(max_packets, "", dtype="<U15")
    destination_ips = np.full(max_packets, "", dtype="<U15")
    gaps = np.zeros(max_packets, dtype=np.float32)
    directions = np.zeros(max_packets, dtype=np.int8)
    mask = np.zeros(max_packets, dtype=np.int8)

    # 第一包没有“流内前驱”，因此 gap 明确设为 0；后续包用时间戳差，
    # max(0, ...) 可防御极少数抓包时间戳倒退或同一时刻记录的问题。
    previous_time = selected[0].timestamp
    for position, packet in enumerate(selected):
        lengths[position] = packet.frame_length
        timestamps[position] = packet.timestamp
        frame_ids[position] = packet.frame_id
        source_ips[position] = packet.source_ip
        destination_ips[position] = packet.destination_ip
        gaps[position] = (
            max(0.0, packet.timestamp - previous_time) if position else 0.0
        )
        directions[position] = packet.direction
        mask[position] = 1
        previous_time = packet.timestamp
    # 使用关键字构造，避免将来增加审计字段时因位置参数错位而静默污染数据。
    return FlowSample(
        lengths=lengths,
        timestamps=timestamps,
        frame_ids=frame_ids,
        source_ips=source_ips,
        destination_ips=destination_ips,
        gaps=gaps,
        directions=directions,
        mask=mask,
        label=int(label),
        group=str(group),
    )


def extract_device(
    data_root: Path,
    device: str,
    label: int,
    config: DatasetConfig,
    *,
    capture_summaries: list[dict[str, object]] | None = None,
) -> list[FlowSample]:
    """Extract all eligible TLS-candidate flows for one device.

    A flow is retained only if it contains at least ``min_packets`` events, at
    least one plausible TLS record, and at least one outbound packet.  Capture
    identity is preserved in ``group`` for leakage-free train/test splitting.
    When ``capture_summaries`` is supplied, one JSON-safe diagnostic record is
    appended for every PCAP.  The diagnostics describe the complete extraction
    funnel and are audit metadata only; they never participate in features,
    labels, Method31, or any baseline.
    """

    # 一个设备目录对应一个数值 label。调用者按配置中的 devices 顺序传入
    # label，因此调整 devices 顺序会改变标签语义，应视为数据集版本变化。
    device_directory = Path(data_root) / device
    if not device_directory.is_dir():
        raise FileNotFoundError(f"device capture directory is missing: {device_directory}")
    device_mac = _read_device_mac(device_directory, config.mac_filename)

    samples: list[FlowSample] = []
    # sorted 保证不同机器、不同文件系统枚举目录时仍得到相同 capture 顺序，
    # 进而让流索引和基于索引的随机种子保持可复现。
    for pcap_path in sorted(device_directory.glob(config.pcap_pattern)):
        flows: dict[tuple[object, ...], list[ParsedPacket]] = {}
        # 这些计数器描述“原始PCAP如何逐层变成候选flow”。它们必须在真正的
        # 单次读取过程中采集；若HTML事后只看已保留CSV，就无法知道被过滤的
        # 帧和flow，也就不能可靠解释“为什么最后只有这些flow”。
        pcap_records = 0
        device_mac_frames = 0
        ipv4_frames = 0
        device_mac_ipv4_frames = 0
        parsed_tcp_packets = 0
        parsed_udp_packets = 0
        parsed_port_packets = 0
        tls_like_records = 0
        for record in read_pcap(pcap_path):
            pcap_records += 1
            frame = record.frame

            # MAC相关统计只检查Ethernet头中的源/目的地址；短于12 byte的损坏
            # 帧自然不计入。IPv4统计复用与解析器完全相同的VLAN感知辅助函数。
            # 同时保留“全部IPv4”和“设备相关IPv4”，可区分非IP背景流量与
            # 抓包点上属于其他主机的IPv4流量。
            device_related = bool(
                len(frame) >= 12
                and (frame[0:6] == device_mac or frame[6:12] == device_mac)
            )
            if device_related:
                device_mac_frames += 1
            ethernet = _ethernet_payload_offset(frame)
            is_ipv4 = ethernet is not None and ethernet[0] == _IPV4_ETHERTYPE
            if is_ipv4:
                ipv4_frames += 1
                if device_related:
                    device_mac_ipv4_frames += 1

            packet = parse_packet(
                record.timestamp,
                frame,
                device_mac,
                config.tls_port,
                frame_id=record.frame_id,
            )
            if packet is not None:
                flows.setdefault(packet.flow_key, []).append(packet)

                # flow_key首项是IP protocol number。端口计数针对配置中的
                # tls_port，并不声称这些包一定成功解析为TLS；后一计数才表示
                # TCP payload通过本项目的轻量TLS record-header判定。
                protocol = int(packet.flow_key[0])
                if protocol == _TCP:
                    parsed_tcp_packets += 1
                elif protocol == _UDP:
                    parsed_udp_packets += 1
                if config.tls_port in (packet.source_port, packet.destination_port):
                    parsed_port_packets += 1
                if packet.is_tls_record:
                    tls_like_records += 1

        # 一个 capture 中可能含很多普通 TCP/UDP 会话。只有同时满足以下条件
        # 的双向流才成为样本：包数足够、至少出现一个 TLS-like record、并且
        # 至少有一个设备上行包（Method31 的 trigger 需要以上行包为锚点）。
        excluded_below_min_packets = 0
        excluded_no_tls_like_record = 0
        excluded_no_outbound_packet = 0
        retained_flow_details: list[dict[str, object]] = []
        group = f"{device}/{pcap_path.stem}"
        for flow_key, packets in flows.items():
            packets.sort(key=lambda packet: packet.timestamp)
            if len(packets) < config.min_packets:
                excluded_below_min_packets += 1
                continue
            if not any(packet.is_tls_record for packet in packets):
                excluded_no_tls_like_record += 1
                continue
            if not any(packet.direction == 1 for packet in packets):
                excluded_no_outbound_packet += 1
                continue

            # 详情中的packet_count基于完整合格flow；max_packets只限制模型看到
            # 的固定窗口，不会把一条长flow切成多条，也不会改变这里的计数。
            # flow_key中的两个endpoint已排序，仅用于稳定展示；逐包真实方向仍
            # 由CSV/HTML事件表中的src_ip、dst_ip和direction审计。
            protocol, endpoint_1, endpoint_2 = flow_key

            def endpoint_text(endpoint: tuple[bytes, int]) -> str:
                address, port = endpoint
                return f"{'.'.join(map(str, address))}:{int(port)}"

            outbound_packets = sum(packet.direction == 1 for packet in packets)
            inbound_packets = sum(packet.direction == -1 for packet in packets)
            flow_tls_records = sum(packet.is_tls_record for packet in packets)
            retained_flow_details.append(
                {
                    "protocol": "TCP" if int(protocol) == _TCP else "UDP",
                    "endpoint_1": endpoint_text(endpoint_1),
                    "endpoint_2": endpoint_text(endpoint_2),
                    "packet_count": len(packets),
                    "outbound_packets": outbound_packets,
                    "inbound_packets": inbound_packets,
                    "tls_like_records": flow_tls_records,
                    "duration_seconds": max(
                        0.0, packets[-1].timestamp - packets[0].timestamp
                    ),
                    "first_frame_id": min(packet.frame_id for packet in packets),
                    "last_frame_id": max(packet.frame_id for packet in packets),
                }
            )
            samples.append(
                _encode_flow(
                    packets,
                    label=label,
                    group=group,
                    max_packets=config.max_packets,
                )
            )

        # collector是可选参数，保留extract_device原本“只返回样本”的轻量接口。
        # extract_dataset会传入collector并把结果保存成Unicode JSON标量；这样
        # NPZ仍可用allow_pickle=False安全加载，也不会为每条flow重复元数据。
        if capture_summaries is not None:
            capture_summaries.append(
                {
                    "device_name": device,
                    "capture": pcap_path.stem,
                    "group": group,
                    "pcap_file": pcap_path.name,
                    "pcap_records": pcap_records,
                    "device_mac_frames": device_mac_frames,
                    "ipv4_frames": ipv4_frames,
                    "device_mac_ipv4_frames": device_mac_ipv4_frames,
                    "parsed_device_ipv4_tcp_udp_packets": (
                        parsed_tcp_packets + parsed_udp_packets
                    ),
                    "parsed_tcp_packets": parsed_tcp_packets,
                    "parsed_udp_packets": parsed_udp_packets,
                    "parsed_tls_port_packets": parsed_port_packets,
                    "tls_like_records": tls_like_records,
                    "grouped_bidirectional_flows": len(flows),
                    "excluded_below_min_packets": excluded_below_min_packets,
                    "excluded_no_tls_like_record": excluded_no_tls_like_record,
                    "excluded_no_outbound_packet": excluded_no_outbound_packet,
                    "retained_flows": len(retained_flow_details),
                    "retained_flow_details": retained_flow_details,
                }
            )
    return samples


def extract_dataset(data_root: Path, config: DatasetConfig) -> dict[str, np.ndarray]:
    """Extract all configured devices and return the shared in-memory schema."""

    # enumerate 的位置就是类别编号；每个设备至少要贡献一条合格流，否则
    # 直接失败，避免不完整的类别集合在训练阶段才以难懂的错误暴露。
    samples: list[FlowSample] = []
    capture_summaries: list[dict[str, object]] = []
    for label, device in enumerate(config.devices):
        device_samples = extract_device(
            data_root,
            device,
            label,
            config,
            capture_summaries=capture_summaries,
        )
        if not device_samples:
            raise RuntimeError(f"no eligible TLS flow was extracted for {device}")
        samples.extend(device_samples)
    if not samples:
        raise RuntimeError("no eligible TLS flow was extracted")

    # np.stack 要求所有单流数组宽度相同，正好再次约束固定窗口模式。
    # 字符串数组由 NumPy 保存为 Unicode，不需要 pickle 即可安全加载。
    dataset = {
        "lengths": np.stack([sample.lengths for sample in samples]),
        # timestamps是审计附加字段。下游算法按名字显式取得原四个张量，因而
        # 不会把绝对时间误当成新分类特征；save_dataset会原样安全保存它。
        "timestamps": np.stack([sample.timestamps for sample in samples]),
        "frame_ids": np.stack([sample.frame_ids for sample in samples]),
        "source_ips": np.stack([sample.source_ips for sample in samples]),
        "destination_ips": np.stack([sample.destination_ips for sample in samples]),
        "gaps": np.stack([sample.gaps for sample in samples]),
        "directions": np.stack([sample.directions for sample in samples]),
        "mask": np.stack([sample.mask for sample in samples]),
        "labels": np.asarray([sample.label for sample in samples], dtype=np.int16),
        "groups": np.asarray([sample.group for sample in samples]),
        "device_names": np.asarray(config.devices),
        # Unicode JSON标量集中保存每个PCAP的提取漏斗。它只用于人工检查页，
        # 不沿flow维度展开，不会进入训练/测试或防御算法。
        "capture_summaries_json": np.asarray(
            json.dumps(capture_summaries, ensure_ascii=False, separators=(",", ":"))
        ),
    }
    validate_dataset(dataset, expected_width=config.max_packets)
    return dataset


def save_dataset(path: Path, dataset: Mapping[str, np.ndarray]) -> None:
    """Validate and save the dataset as a compressed, pickle-free NPZ."""

    # 先验证再写盘，保证任何由本函数生成的 NPZ 都满足公共输入契约。
    # np.savez_compressed 只保存数值/Unicode 数组，不保存任意 Python 对象。
    validate_dataset(dataset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in dataset.items()})


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Load a trusted numeric/string NPZ without enabling Python pickles."""

    # allow_pickle=False 是安全边界：数据文件不能借由 object array 触发任意
    # Python 对象反序列化。加载完成后再次验证，防止手工修改或旧格式混入。
    with np.load(Path(path), allow_pickle=False) as archive:
        dataset = {key: archive[key] for key in archive.files}
    validate_dataset(dataset)
    return dataset


def load_capture_summaries(
    dataset: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    """Decode and validate optional per-PCAP extraction diagnostics.

    Older NPZ files legitimately lack ``capture_summaries_json`` and therefore
    return an empty list.  A present field is validated strictly so the HTML
    never publishes an attractive but internally inconsistent explanation.
    """

    if "capture_summaries_json" not in dataset:
        return []
    encoded = np.asarray(dataset["capture_summaries_json"])
    if encoded.shape != () or encoded.dtype.kind not in {"U", "S"}:
        raise ValueError("capture_summaries_json must be one Unicode/byte string scalar")
    raw_value = encoded.item()
    if isinstance(raw_value, bytes):
        try:
            raw_value = raw_value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("capture_summaries_json is not UTF-8") from error
    try:
        decoded = json.loads(str(raw_value))
    except json.JSONDecodeError as error:
        raise ValueError("capture_summaries_json is not valid JSON") from error
    if not isinstance(decoded, list) or any(
        not isinstance(summary, dict) for summary in decoded
    ):
        raise ValueError("capture_summaries_json must decode to a list of objects")
    return decoded


def export_dataset_inspection_csv(
    dataset: Mapping[str, np.ndarray],
    output_directory: Path,
    *,
    filename_prefix: str = "flow_features",
    max_frame_length: float = 1514.0,
    max_gap_seconds: float = 10.0,
    timestamp: datetime | None = None,
) -> Path:
    """Export one human-inspection row per flow to a timestamped CSV file.

    The CSV deliberately contains the raw fixed-window representation, original
    PCAP frame IDs/timestamps/IP endpoints, and the exact classifier features
    produced by :func:`make_features`.  This lets a reviewer locate every event
    in Wireshark, verify each gap from adjacent timestamps, then independently
    verify the signed-length and logarithmic-gap features.

    Parameters
    ----------
    dataset:
        The validated in-memory dataset returned by :func:`extract_dataset` or
        :func:`load_dataset`.
    output_directory:
        Directory in which the CSV is created.  Parent directories are created
        when necessary.
    filename_prefix:
        Readable portion before the timestamp.  Path separators and characters
        forbidden in Windows filenames are rejected rather than rewritten.
    max_frame_length / max_gap_seconds:
        The same scaling constants that will be supplied to
        :func:`make_features` during evaluation.
    timestamp:
        Optional clock value for deterministic tests.  Normal callers omit it,
        causing the current local timestamp (including microseconds in the
        filename) to be used.

    Returns
    -------
    pathlib.Path
        Absolute or caller-relative path of the created CSV file.
    """

    # 先复用公共数据验证，保证CSV中的统计和逐位置字段不会建立在形状错位、
    # 非法mask或非连续label上。该函数不会修改传入dataset的任何数组。
    validate_dataset(dataset)
    required_audit_fields = {
        "timestamps",
        "frame_ids",
        "source_ips",
        "destination_ips",
    }
    missing_audit_fields = required_audit_fields - set(dataset)
    if missing_audit_fields:
        raise ValueError(
            "inspection CSV requires original PCAP packet audit fields; "
            f"missing {sorted(missing_audit_fields)}; re-extract the dataset "
            "with the current dataset_pcap.py"
        )
    prefix = str(filename_prefix).strip()
    if not prefix:
        raise ValueError("filename_prefix cannot be empty")
    forbidden = set('<>:"/\\|?*')
    if any(character in forbidden for character in prefix):
        raise ValueError("filename_prefix contains a forbidden filename character")

    # 默认使用系统本地时区的当前时间。加入微秒不仅满足“文件名带时间戳”，
    # 也避免同一秒内连续执行两次extract时后一个CSV覆盖前一个。
    clock = datetime.now().astimezone() if timestamp is None else timestamp
    timestamp_text = clock.strftime("%Y%m%d_%H%M%S_%f")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{prefix}_{timestamp_text}.csv"

    lengths = np.asarray(dataset["lengths"])
    timestamps = np.asarray(dataset["timestamps"], dtype=np.float64)
    frame_ids = np.asarray(dataset["frame_ids"])
    source_ips = np.asarray(dataset["source_ips"]).astype(str)
    destination_ips = np.asarray(dataset["destination_ips"]).astype(str)
    gaps = np.asarray(dataset["gaps"])
    directions = np.asarray(dataset["directions"])
    mask = np.asarray(dataset["mask"])
    labels = np.asarray(dataset["labels"], dtype=int)
    groups = np.asarray(dataset["groups"])
    valid = mask.astype(bool)
    flow_count, width = lengths.shape

    # make_features是正式攻击模型使用的唯一特征公式。直接调用而不是在CSV
    # 导出器里重写公式，可防止“用于检查的CSV”和“模型实际输入”悄悄分叉。
    features = make_features(
        lengths,
        gaps,
        directions,
        max_frame_length=max_frame_length,
        max_gap_seconds=max_gap_seconds,
    )
    if features.shape != (flow_count, 2 * width):
        raise AssertionError("feature matrix has an unexpected shape")

    device_names = (
        np.asarray(dataset["device_names"]).astype(str)
        if "device_names" in dataset
        else np.asarray([str(label) for label in np.unique(labels)])
    )

    # 前半部分是一流一行的摘要；后半部分按观察位置展开八个原始审计字段
    # 与两个分类特征。frame_id/src/dst让每个HTML事件可精确回查Wireshark。
    # 分类特征。列号使用01开始的人类习惯，同时对应数组中的0号位置。
    summary_header = [
        "flow_index",
        "device_name",
        "label",
        "group",
        "capture",
        "valid_event_count",
        "outbound_event_count",
        "inbound_event_count",
        "total_frame_bytes",
        "outbound_frame_bytes",
        "inbound_frame_bytes",
        "flow_duration_seconds",
        "mean_inter_event_gap_seconds",
        "max_gap_seconds",
        "first_direction",
        "last_direction",
    ]
    raw_header = [
        field
        for position in range(width)
        for field in (
            f"event_{position + 1:02d}_frame_id",
            f"event_{position + 1:02d}_timestamp_seconds",
            f"event_{position + 1:02d}_source_ip",
            f"event_{position + 1:02d}_destination_ip",
            f"event_{position + 1:02d}_length_bytes",
            f"event_{position + 1:02d}_gap_seconds",
            f"event_{position + 1:02d}_direction",
            f"event_{position + 1:02d}_valid",
        )
    ]
    feature_header = [
        field
        for position in range(width)
        for field in (
            f"feature_{position + 1:02d}_signed_length",
            f"feature_{position + 1:02d}_log_gap",
        )
    ]

    # utf-8-sig会写UTF-8 BOM，Windows Excel可直接识别中文设备名/路径；
    # newline=""是csv模块官方推荐方式，可避免Windows出现多余空行。
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(summary_header + raw_header + feature_header)
        for flow_index in range(flow_count):
            positions = np.flatnonzero(valid[flow_index])
            valid_lengths = lengths[flow_index, positions].astype(np.float64)
            valid_gaps = gaps[flow_index, positions].astype(np.float64)
            valid_directions = directions[flow_index, positions].astype(int)
            outbound = valid_directions == 1
            inbound = valid_directions == -1
            group = str(groups[flow_index])
            capture = group.split("/", 1)[1] if "/" in group else group
            label = int(labels[flow_index])
            device_name = (
                str(device_names[label])
                if 0 <= label < len(device_names)
                else str(label)
            )

            # duration是所有有效gap之和；平均inter-event gap排除第一事件约定的
            # 0值，避免短流的均值被“没有前驱”这一编码规则人为拉低。
            inter_event_gaps = valid_gaps[1:] if len(valid_gaps) > 1 else np.empty(0)
            summary_row = [
                flow_index,
                device_name,
                label,
                group,
                capture,
                int(len(positions)),
                int(np.sum(outbound)),
                int(np.sum(inbound)),
                float(np.sum(valid_lengths)),
                float(np.sum(valid_lengths[outbound])),
                float(np.sum(valid_lengths[inbound])),
                float(np.sum(valid_gaps)),
                float(np.mean(inter_event_gaps)) if len(inter_event_gaps) else 0.0,
                float(np.max(valid_gaps)) if len(valid_gaps) else 0.0,
                int(valid_directions[0]) if len(valid_directions) else "",
                int(valid_directions[-1]) if len(valid_directions) else "",
            ]
            raw_row = [
                value
                for position in range(width)
                for value in (
                    int(frame_ids[flow_index, position]),
                    float(timestamps[flow_index, position]),
                    str(source_ips[flow_index, position]),
                    str(destination_ips[flow_index, position]),
                    float(lengths[flow_index, position]),
                    float(gaps[flow_index, position]),
                    int(directions[flow_index, position]),
                    int(mask[flow_index, position]),
                )
            ]
            feature_row = [
                value
                for position in range(width)
                for value in (
                    float(features[flow_index, position]),
                    float(features[flow_index, width + position]),
                )
            ]
            writer.writerow(summary_row + raw_row + feature_row)

    return output_path


def _string_mask_violation(
    values: np.ndarray, mask: np.ndarray, *, expect_empty: bool, chunk: int = 1_000_000
) -> bool:
    """分块检查字符串数组在 mask 位置上的长度是否违规。

    等价于一次性执行 ``np.char.str_len(values[mask])``，但按行分块。
    UNSW 的单个 shard 可能有五百多万个有效事件；一次性 ``values[mask]``
    再加 ``str_len`` 会临时分配几百 MB 数组，在 16 GB 机器上足以让构建
    进程 OOM。分块后峰值被限制在每块（默认 100 万行，约 60 MB），
    判定语义与一次性检查完全相同。
    """
    for start in range(0, len(values), chunk):
        block = values[start : start + chunk]
        block_mask = mask[start : start + chunk]
        lengths = np.char.str_len(block[block_mask])
        if expect_empty:
            if np.any(lengths != 0):
                return True
        elif np.any(lengths == 0):
            return True
    return False


def validate_dataset(
    dataset: Mapping[str, np.ndarray], *, expected_width: int | None = None
) -> None:
    """Validate shapes, prefix masks, numeric domains, labels, and groups."""

    # 验证顺序刻意从“结构”走向“语义”：先检查字段与形状，再检查数值域，
    # 最后检查 mask 前缀、标签连续性和 capture 标识。这样错误信息更接近根因。
    required = {"lengths", "gaps", "directions", "mask", "labels", "groups"}
    missing = required - set(dataset)
    if missing:
        raise ValueError(f"dataset is missing fields: {sorted(missing)}")

    lengths, gaps, directions, mask = (
        np.asarray(dataset[name]) for name in ("lengths", "gaps", "directions", "mask")
    )
    if lengths.ndim != 2 or not (
        lengths.shape == gaps.shape == directions.shape == mask.shape
    ):
        raise ValueError("lengths/gaps/directions/mask must share a two-dimensional shape")
    if expected_width is not None and lengths.shape[1] != int(expected_width):
        raise ValueError(
            f"dataset width is {lengths.shape[1]}, expected {int(expected_width)}"
        )

    # timestamps是新提取NPZ中的可选审计字段，而不是算法输入必需字段。
    # 因此旧版NPZ没有它时仍可加载和评测；一旦存在，就必须与固定窗口同形。
    timestamps = None
    if "timestamps" in dataset:
        timestamps = np.asarray(dataset["timestamps"])
        if timestamps.shape != lengths.shape:
            raise ValueError("timestamps must have the same shape as flow arrays")

    # frame_ids/source_ips/destination_ips共同描述“这个事件对应PCAP中的哪一
    # 帧以及该帧当时的IP方向”。三者必须成组出现，避免只保存一半审计信息。
    packet_audit_keys = {"frame_ids", "source_ips", "destination_ips"}
    present_packet_audit_keys = packet_audit_keys & set(dataset)
    if present_packet_audit_keys and present_packet_audit_keys != packet_audit_keys:
        missing_audit_keys = packet_audit_keys - present_packet_audit_keys
        raise ValueError(
            f"packet audit fields must appear together; missing: {sorted(missing_audit_keys)}"
        )
    frame_ids = source_ips = destination_ips = None
    if present_packet_audit_keys:
        frame_ids = np.asarray(dataset["frame_ids"])
        source_ips = np.asarray(dataset["source_ips"]).astype(str)
        destination_ips = np.asarray(dataset["destination_ips"]).astype(str)
        if not (
            frame_ids.shape
            == source_ips.shape
            == destination_ips.shape
            == lengths.shape
        ):
            raise ValueError("packet audit fields must have the same shape as flow arrays")

    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"])
    if labels.ndim != 1 or groups.ndim != 1 or not (
        len(labels) == len(groups) == len(lengths)
    ):
        raise ValueError("labels and groups must be one-dimensional and match flow count")

    # 只在 mask=True 的位置检查长度、gap 和方向；右侧 padding 区本来就是
    # 占位空间，不应该被误认为真实的 0-byte、0 方向网络事件。
    valid = mask.astype(bool)
    if np.any(mask != valid.astype(mask.dtype)):
        raise ValueError("mask must contain only zero and one")
    if any(np.any(array[~valid] != 0) for array in (lengths, gaps, directions)):
        raise ValueError("masked padding lengths/gaps/directions must all be zero")
    if not np.all(np.isfinite(lengths[valid])) or not np.all(np.isfinite(gaps[valid])):
        raise ValueError("valid events contain a non-finite length or gap")
    if np.any(lengths[valid] <= 0.0) or np.any(gaps[valid] < 0.0):
        raise ValueError("valid lengths must be positive and valid gaps nonnegative")
    if np.any(~np.isin(directions[valid], (-1, 1))):
        raise ValueError("valid directions must be -1 or +1")

    if timestamps is not None:
        if not np.all(np.isfinite(timestamps[valid])):
            raise ValueError("valid events contain a non-finite timestamp")
        if np.any(timestamps[~valid] != 0.0):
            raise ValueError("padded timestamp positions must be zero")

        # 每条flow独立验证时间顺序和gap定义。gap以float32保存，而绝对时间戳
        # 是float64；容差只覆盖两者精度差，不容许真正的毫秒级计算错误。
        for row_index, row_valid in enumerate(valid):
            event_count = int(np.sum(row_valid))
            if event_count == 0:
                continue
            row_timestamps = timestamps[row_index, :event_count].astype(np.float64)
            row_gaps = gaps[row_index, :event_count].astype(np.float64)
            expected_gaps = np.concatenate(
                (np.asarray([0.0]), np.diff(row_timestamps))
            )
            if np.any(expected_gaps < 0.0):
                raise ValueError(f"timestamps decrease within flow {row_index}")
            if not np.allclose(row_gaps, expected_gaps, rtol=1e-6, atol=1e-6):
                maximum_difference = float(np.max(np.abs(row_gaps - expected_gaps)))
                raise ValueError(
                    f"gaps do not match timestamp differences in flow {row_index}; "
                    f"maximum difference is {maximum_difference}"
                )

    if frame_ids is not None and source_ips is not None and destination_ips is not None:
        if not np.issubdtype(frame_ids.dtype, np.integer):
            raise ValueError("frame_ids must use an integer dtype")
        if np.any(frame_ids[valid] <= 0) or np.any(frame_ids[~valid] != 0):
            raise ValueError("valid frame_ids must be positive and padded frame_ids zero")
        if _string_mask_violation(source_ips, valid, expect_empty=False) or _string_mask_violation(
            destination_ips, valid, expect_empty=False
        ):
            raise ValueError("valid packet audit IP addresses cannot be empty")
        if _string_mask_violation(source_ips, ~valid, expect_empty=True) or _string_mask_violation(
            destination_ips, ~valid, expect_empty=True
        ):
            raise ValueError("padded packet audit IP addresses must be empty")

    # Every flow must be a left-aligned prefix.  Method31 enumerates valid
    # positions in source order; an internal zero would make source indices and
    # observation positions ambiguous.
    for row in valid:
        count = int(np.sum(row))
        if not np.array_equal(np.flatnonzero(row), np.arange(count)):
            raise ValueError("each mask row must be a contiguous left-aligned prefix")

    # 标签必须为 0..K-1 连续整数，因为 trigger_codes 与 mapping 都用标签
    # 直接索引。若标签中间缺号，算法虽可能运行，却会让配置含义变得含糊。
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("labels must be nonnegative integers")
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        raise ValueError("labels must be contiguous integers starting at zero")
    if np.any(np.char.str_len(groups.astype(str)) == 0):
        raise ValueError("groups cannot contain an empty capture identifier")

    if "device_names" in dataset:
        names = np.asarray(dataset["device_names"])
        if names.ndim != 1 or len(names) != len(unique_labels):
            raise ValueError("device_names must contain exactly one name per label")

    # 元数据不是算法输入，但若它存在，就必须至少能在无pickle条件下解码为
    # JSON object列表；更具体的漏斗算术和CSV逐group一致性由HTML工具验证。
    if "capture_summaries_json" in dataset:
        load_capture_summaries(dataset)


def split_by_capture(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    train_ratio: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leakage-free train/test masks by assigning complete captures.

    The random generator is shared across classes to reproduce the project's
    established split semantics.  At least one capture per class is forced into
    both partitions.
    """

    # 这里返回的是布尔 mask，而不是复制后的数据。调用者可用同一 mask 对
    # lengths/gaps/directions/mask/labels/groups 同步切片，避免样本错位。
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.ndim != 1 or groups.shape != labels.shape:
        raise ValueError("labels and groups must be equal-length one-dimensional arrays")
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio must be strictly between zero and one")

    # 首先证明每个 capture 只属于一个类别。setdefault 返回已存在的值，
    # 因而能在一次遍历中同时建立映射并发现跨类别复用的 group 名。
    group_labels: dict[str, int] = {}
    mixed_groups = False
    for label, group in zip(labels, groups, strict=True):
        key = str(group)
        previous = group_labels.setdefault(key, int(label))
        if previous != int(label):
            mixed_groups = True

    if mixed_groups:
        # One mixed capture must move as a whole, even when it contains many devices.
        # Retry partitions only for class coverage; no features or model scores enter selection.
        unique_groups = np.unique(groups)
        count = max(1, min(len(unique_groups)-1, int(len(unique_groups)*train_ratio)))
        classes = np.unique(labels)
        for label in classes:
            if len(np.unique(groups[labels == label])) < 2:
                raise ValueError(f"label {label!r} has fewer than two independent captures")
        rng = np.random.default_rng(int(seed))
        for _ in range(4096):
            candidate = rng.permutation(unique_groups)[:count]
            train = np.isin(groups, candidate)
            if all(np.any(train & (labels == c)) and np.any(~train & (labels == c)) for c in classes):
                return train, ~train
        raise ValueError("could not split whole mixed captures with every class on both sides; change ratio/seed")

    # 每个类别独立选择 capture，但共享一个按固定顺序前进的 RNG，以匹配
    # 历史实验语义。每类至少保留一个训练 capture 和一个测试 capture。
    rng = np.random.default_rng(int(seed))
    train = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        class_groups = np.unique(groups[labels == label]).copy()
        if len(class_groups) < 2:
            raise ValueError(f"label {label!r} has fewer than two captures")
        rng.shuffle(class_groups)
        count = int(len(class_groups) * float(train_ratio))
        count = max(1, min(len(class_groups) - 1, count))
        train |= (labels == label) & np.isin(groups, class_groups[:count])
    # test 是 train 的逐位取反，因此每条流恰好进入一侧；随后用 group 集合
    # 交集做最终审计，确保同一 capture 没有跨越训练/测试边界。
    test = ~train
    if set(map(str, groups[train])) & set(map(str, groups[test])):
        raise AssertionError("capture leakage detected between train and test")
    return train, test


def make_features(
    lengths: np.ndarray,
    gaps: np.ndarray,
    directions: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    max_frame_length: float = 1514.0,
    max_gap_seconds: float = 10.0,
) -> np.ndarray:
    """Build the project's established classifier feature matrix.

    Signed normalized lengths preserve direction in one channel.  Log-scaled
    millisecond gaps form the second channel.  The result has ``2 * L`` columns
    and intentionally matches the historical four-class evaluation code.
    """

    # 输出形状为 [N, 2L]。前 L 维是带方向符号的归一化长度，后 L 维是
    # log1p 压缩后的毫秒间隔。该公式与旧四分类评测保持一致。
    lengths, gaps, directions = map(np.asarray, (lengths, gaps, directions))
    if lengths.ndim != 2 or not (lengths.shape == gaps.shape == directions.shape):
        raise ValueError("lengths/gaps/directions must share a two-dimensional shape")
    max_frame_length = float(max_frame_length)
    max_gap_seconds = float(max_gap_seconds)
    if max_frame_length <= 0.0 or max_gap_seconds <= 0.0:
        raise ValueError("feature scaling constants must be positive")
    # gap 先裁到 max_gap_seconds，防止极端长空闲期支配尺度；乘 1000 把秒
    # 转成毫秒后使用 log1p，使 0 保持为 0，同时压缩长尾分布。
    valid = (directions != 0) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != lengths.shape:
        raise ValueError("mask must have the same shape as flow arrays")
    visible_lengths = np.where(valid, lengths, 0.0)
    visible_gaps = np.where(valid, gaps, 0.0).astype(np.float64)
    # The observer's clock starts at the first emitted packet, not at a private send attempt.
    if visible_gaps.shape[1]:
        visible_gaps[:, 0] = 0.0
    if not np.all(np.isfinite(visible_lengths)) or not np.all(np.isfinite(visible_gaps)):
        raise ValueError("features contain nonfinite valid events")
    if np.any(visible_gaps < 0):
        raise ValueError("gaps must be nonnegative")
    # Write the two channels directly into the final buffer. Preserve float64
    # arithmetic and operation order without several full-size temporaries.
    features = np.empty((len(lengths), lengths.shape[1] * 2), dtype=np.float64)
    signed_lengths = features[:, :lengths.shape[1]]
    log_gaps = features[:, lengths.shape[1]:]
    signed_lengths[:] = np.where(valid, directions, 0)
    np.multiply(signed_lengths, visible_lengths, out=signed_lengths)
    np.divide(signed_lengths, max_frame_length, out=signed_lengths)
    np.minimum(visible_gaps, max_gap_seconds, out=log_gaps)
    np.multiply(log_gaps, 1000.0, out=log_gaps)
    np.log1p(log_gaps, out=log_gaps)
    return features


__all__ = [
    # 只把稳定公共接口列在 __all__ 中；以下划线开头的解析/编码辅助函数
    # 仍可在本模块内部使用，但不承诺给外部调用者长期兼容。
    "DatasetConfig",
    "FlowSample",
    "ParsedPacket",
    "PcapRecord",
    "extract_dataset",
    "extract_device",
    "export_dataset_inspection_csv",
    "load_capture_summaries",
    "load_dataset",
    "make_features",
    "parse_packet",
    "read_pcap",
    "save_dataset",
    "split_by_capture",
    "validate_dataset",
]
