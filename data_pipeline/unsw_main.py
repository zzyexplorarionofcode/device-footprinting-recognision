"""Build the UNSW main experiment: 8 large classes, capped TRAIN only.

All validation/test sessions are retained. Raw PCAP is streamed once per device;
per-date reservoirs bound training memory without balancing classes to a minimum.
"""

# =============================================================================
# 中文阅读导图：8 类主数据集的“一次成型”构建器
# =============================================================================
# 这个文件把 unsw.py（读包/会话化）和 partitions.py（冻结分区）串起来，
# 产出 data/processed/unsw_main8_v1/flows_unsw_8class.npz。
#
# 三步：
#   make_plan()     读计数审计的 eligible_flows_by_group（每设备每天多少条合格流），
#                   先按 UTC 日 75/25 分 outer-train/test，再在 75% 内 80/20 分
#                   train/validation；训练侧每类按天比例分配配额，总数不超过 cap。
#                   所有天数、原始计数、保留计数写进 plan，之后任何人都不能改。
#   extract_shard() 每个设备一个进程：流式读该设备 PCAPNG，训练日进蓄水池抽样，
#                   validation/test 日全部保留；核对“实际抽到的计数/文件摘要”
#                   必须等于审计值，否则拒绝继续（防止源数据被换过）。
#   build()         按列拼接所有 shard（避免同时解码两份全量数组），写入
#                   device_names、data_protocol、partition_manifest_json 等元数据，
#                   最后调用 frozen_partition/validate_dataset 做自检。
#
# 关键约束：输出目录必须尚不存在（mkdir(exist_ok=False)），防止半成品覆盖；
# 每个 shard 旁有同名 .json 审计，manifest.json 记录源码与数据 SHA-256。
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from . import pcap, unsw
from .partitions import PARTITION_PROTOCOL, frozen_partition, proportional_quotas

DEVICES = ('AugustDoorBell', 'iHome', 'NetatmoWelcome', 'AmazonEcho',
           'WithingsSleepSensor', 'TribySpeaker', 'PhilipsHue', 'PixStarPhotoFrame')


def make_plan(audits, *, cap=10000, outer_seed=42, inner_seed=731100, sampling_seed=20260922):
    by_device = {a['device']: a for a in audits}
    y, groups = [], []
    for label, device in enumerate(DEVICES):
        days = sorted(by_device[device]['eligible_flows_by_group'])
        y.extend([label] * len(days)); groups.extend(days)
    y, groups = np.asarray(y), np.asarray(groups)
    outer, test = pcap.split_by_capture(y, groups, train_ratio=.75, seed=outer_seed)
    inner, validation = pcap.split_by_capture(y[outer], groups[outer], train_ratio=.8, seed=inner_seed)
    days = {'train': sorted(set(groups[outer][inner])),
            'validation': sorted(set(groups[outer][validation])), 'test': sorted(set(groups[test]))}
    raw_counts = {role: [sum(by_device[d]['eligible_flows_by_group'].get(day, 0)
                            for day in day_list) for d in DEVICES] for role, day_list in days.items()}
    retained = {role: list(values) for role, values in raw_counts.items()}
    retained['train'] = [min(cap, n) for n in raw_counts['train']]
    return {'protocol': PARTITION_PROTOCOL, 'devices': list(DEVICES),
            'outer_train_ratio': .75, 'outer_seed': outer_seed,
            'inner_train_ratio': .8, 'inner_seed': inner_seed,
            'train_cap_per_class': cap, 'sampling_seed': sampling_seed,
            'sampling': 'training only; proportional per-day reservoirs; every training day retained',
            'days': days, 'raw_counts': raw_counts, 'retained_counts': retained}


def extract_shard(job):
    root, out, device, expected, plan = job
    label = DEVICES.index(device)
    day_counts = expected['eligible_flows_by_group']
    training_days = sorted(set(day_counts) & set(plan['days']['train']))
    quotas = proportional_quotas([day_counts[d] for d in training_days], plan['train_cap_per_class'])
    reservoirs = {day: unsw.Reservoir(int(n), plan['sampling_seed'] + 1009*label + i)
                  for i, (day, n) in enumerate(zip(training_days, quotas))}
    holdout_days = set(plan['days']['validation']) | set(plan['days']['test'])
    held_out = []

    def consume(sample):
        if sample.group in reservoirs:
            reservoirs[sample.group].offer(sample)
        elif sample.group in holdout_days:
            held_out.append(sample)
        else:
            raise ValueError(f'unexpected raw date {sample.group}; redo the count audit')

    config = pcap.DatasetConfig(devices=(device,), min_packets=4, max_packets=32,
                               tls_port=443, max_feature_gap_seconds=10., max_frame_length=1514.)
    # Label zero makes the per-device shard independently schema-valid.
    _, audit = unsw.extract_device(root, device, 0, config, inactivity_timeout_seconds=120.,
                                  reservoir_capacity=1, sampling_seed=plan['sampling_seed'],
                                  sample_consumer=consume)
    for key in ('capture_sha256', 'eligible_flows', 'eligible_flows_by_group'):
        if audit[key] != expected[key]:
            raise ValueError(f'{device}: raw count/source differs from audited input ({key})')
    samples = held_out + [s for r in reservoirs.values() for s in r.items]
    samples.sort(key=lambda s: (s.group, float(s.timestamps[0]), int(s.frame_ids[0])))
    audit.update(label=label, training_quotas=dict(zip(training_days, map(int, quotas))))
    data = unsw._dataset_from_samples(samples, (device,), {'devices': [device], 'class_count': 1})
    roles = {day: role for role, days in plan['days'].items() for day in days}
    data['partition_roles'] = np.asarray([roles[str(day)] for day in data['groups']])
    path = out / 'shards' / f'{label:02d}_{device}.npz'
    pcap.save_dataset(path, data)
    path.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    return str(path), audit


def build(root, output, audit_path, *, workers=3):
    audit_manifest = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit_manifest.get('status') != 'complete':
        raise ValueError('a complete raw count audit is required')
    audits = {a['device']: a for a in audit_manifest['device_audits']}
    plan = make_plan(list(audits.values()))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'shards').mkdir()
    (output / 'partition_plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print('Frozen raw counts:', plan['raw_counts'], flush=True)
    print('Retained counts:', plan['retained_counts'], flush=True)
    jobs = [(root, output, device, audits[device], plan) for device in DEVICES]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(extract_shard, jobs))
    # Combine one column at a time, avoiding two full decoded copies of all shards.
    columns = ('lengths', 'timestamps', 'frame_ids', 'source_ips', 'destination_ips',
               'gaps', 'directions', 'mask', 'groups', 'capture_ids', 'partition_roles')
    data = {}
    for key in columns:
        arrays = []
        for path, _ in results:
            with np.load(path, allow_pickle=False) as shard:
                arrays.append(shard[key])
        data[key] = np.concatenate(arrays)
    data['labels'] = np.concatenate([np.full(sum(plan['retained_counts'][r][i]
        for r in ('train', 'validation', 'test')), i, dtype=np.int16) for i in range(len(DEVICES))])
    manifest = {'schema': 'unsw-main-8class-v1', 'status': 'complete',
                'data_protocol': unsw.UNSW_DATA_PROTOCOL, 'devices': list(DEVICES), 'class_count': 8,
                'source_root': str(root.resolve()), 'device_audits': [a for _, a in results],
                'partition': plan, 'count_audit_sha256': unsw._sha256(audit_path),
                'source_sha256': {str(p): unsw._sha256(p) for p in
                    (Path(__file__), Path(unsw.__file__), Path(pcap.__file__), Path(__file__).with_name('partitions.py'))}}
    data.update(device_names=np.asarray(DEVICES), data_protocol=np.asarray(unsw.UNSW_DATA_PROTOCOL),
                partition_manifest_json=np.asarray(json.dumps(plan)),
                unsw_extraction_manifest_json=np.asarray(json.dumps(manifest)))
    frozen_partition(data, train_ratio=.75, seed=42, expected_protocol=PARTITION_PROTOCOL)
    pcap.validate_dataset(data)
    path = output / 'flows_unsw_8class.npz'
    pcap.save_dataset(path, data)
    manifest['dataset_sha256'] = unsw._sha256(path)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('COMPLETE', path, plan['retained_counts'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--count-audit', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    build(args.root, args.output, args.count_audit, workers=args.workers)


if __name__ == '__main__':
    main()
