"""Count every eligible UNSW session without balancing or writing a dataset.

Uses the production session/filter implementation. The one-row reservoir only
limits retained example data; eligible_flows and per-day counts cover the full
capture. Each completed device gets a separate audit for resumable inspection.
"""

# =============================================================================
# 中文阅读导图：先“数数”，再决定用哪 8 类
# =============================================================================
# UNSW 原始数据每个设备一个多月的大文件，直接抽取很贵；因此先只做计数审计：
#   * 复用生产版 extract_device（同一套会话/过滤逻辑），保证“数出来的”就是
#     “将来抽出来的”；
#   * reservoir_capacity=1 只保留 1 条样例，但 eligible_flows、按天计数
#     (eligible_flows_by_group) 覆盖整个抓包；
#   * 每个设备一个 JSON，进程池并行，大文件优先；
#   * 最后写 device_counts.csv 与 status=complete 的 manifest.json。
#
# 下游 unsw_main.py 会把这份 manifest 当作“源数据指纹”：实际抽取的计数
# 必须与这里逐设备一致，否则拒绝构建，防止源文件被替换。
#
# 用法：python -B -m scripts.audit_unsw_counts <UNSW根目录> <输出目录> --workers 3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from data_pipeline import pcap, unsw


def audit_device(args):
    root, output, device = args
    config = pcap.DatasetConfig(devices=(device,), min_packets=4, max_packets=32,
                                tls_port=443, max_feature_gap_seconds=10., max_frame_length=1514.)
    _, audit = unsw.extract_device(root, device, 0, config,
                                  inactivity_timeout_seconds=120., reservoir_capacity=1,
                                  sampling_seed=20260904)
    audit.update(count_only=True, data_protocol=unsw.UNSW_DATA_PROTOCOL,
                 scanned_at_utc=datetime.now(timezone.utc).isoformat(),
                 extractor_sha256=unsw._sha256(Path(unsw.__file__)),
                 parser_sha256=unsw._sha256(Path(pcap.__file__)))
    path = output / f"{device}.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    captures = sorted(args.root.glob("*.pcap*"))
    devices = [path.stem.rsplit("_", 1)[0] for path in captures]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("expected one MAC-suffixed capture per device")
    args.output.mkdir(parents=True, exist_ok=False)
    # Largest files start first; each device is streamed, so memory is bounded.
    captures.sort(key=lambda path: path.stat().st_size, reverse=True)
    jobs = [(args.root, args.output, path.stem.rsplit("_", 1)[0]) for path in captures]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        audits = list(pool.map(audit_device, jobs))
    audits.sort(key=lambda audit: audit["eligible_flows"], reverse=True)
    unsw._write_audit_csv(args.output / "device_counts.csv", audits)
    (args.output / "manifest.json").write_text(json.dumps({
        "status": "complete", "data_protocol": unsw.UNSW_DATA_PROTOCOL,
        "source_root": str(args.root.resolve()), "device_audits": audits,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for audit in audits:
        print(audit["device"], audit["eligible_flows"], audit["eligible_groups"], flush=True)


if __name__ == "__main__":
    main()
