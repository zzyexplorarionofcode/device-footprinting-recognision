"""Frozen group partitions and bounded, date-stratified training samples."""

# =============================================================================
# 中文阅读导图：已冻结分区的“只读校验器”
# =============================================================================
# 背景：UNSW 数据集在构建时就写好了 partition_roles（train/validation/test）以及
# partition_manifest_json（每天多少条、原始计数、保留计数、上限）。训练阶段
# 不允许再做随机流级切分，否则会把同一天的数据挪进训练集，破坏“按 UTC 日隔离”和
# “每类最多 1 万条”的语义。
#
# proportional_quotas(counts, limit)
#     在“每个非空日期至少保留 1 条”的前提下，把 limit 按比例分给各日期；
#     先各给 1 条，剩下按最大余数法分配，保证总数恰好等于 limit。
#
# frozen_partition(dataset, train_ratio, seed, expected_protocol)
#     返回 ({role: bool mask}, metadata)；旧数据没有 manifest 时返回 None。
#     校验链：manifest 与 roles 必须同时存在 → protocol 版本匹配 → seed/ratio
#     与冻结时一致 → 每个 role 都含全部类别 → 日期集合与 manifest 一致 →
#     计数与 manifest 一致 → validation/test 不允许被抽样 → 三个 role 的
#     日期两两不相交 → 训练计数等于 min(原始, 上限)。
#     任何一条不满足都直接报错，提示“重新构建数据集”而不是“换个种子试试”。
from __future__ import annotations

import json
import numpy as np

PARTITION_PROTOCOL = 'unsw-8class-train-cap10000-v1'


def proportional_quotas(counts, limit):
    """Allocate an exact cap, preserving every nonempty date when possible."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts <= 0) or limit < len(counts):
        raise ValueError('positive counts and capacity for every date are required')
    if counts.sum() <= limit:
        return counts.copy()
    remaining = counts - 1
    ideal = remaining * ((int(limit) - len(counts)) / float(remaining.sum()))
    quotas = 1 + np.floor(ideal).astype(np.int64)
    extra = int(limit) - int(quotas.sum())
    order = np.argsort(-(ideal - np.floor(ideal)), kind='stable')
    quotas[order[:extra]] += 1
    return quotas


def frozen_partition(dataset, *, train_ratio, seed, expected_protocol=None):
    """Return train/validation/test masks, or None for legacy dynamic splits.

    Never re-split already sampled data: doing so can move held-out dates into
    training and silently change the meaning of the training cap.
    """
    has_manifest = 'partition_manifest_json' in dataset
    if has_manifest != ('partition_roles' in dataset):
        raise ValueError('partition manifest and row roles must occur together')
    if not has_manifest:
        if expected_protocol is not None:
            raise ValueError('configured frozen partition is missing; rebuild the dataset')
        return None
    metadata = json.loads(str(np.asarray(dataset['partition_manifest_json']).item()))
    if metadata['protocol'] != PARTITION_PROTOCOL:
        raise ValueError('unsupported frozen partition protocol')
    if expected_protocol is not None and metadata['protocol'] != expected_protocol:
        raise ValueError('configured partition protocol mismatch')
    if int(seed) != metadata['outer_seed'] or not np.isclose(train_ratio, metadata['outer_train_ratio']):
        raise ValueError('cannot change the split of a frozen sampled dataset; rebuild it')
    labels = np.asarray(dataset['labels'])
    groups = np.asarray(dataset.get('capture_ids', dataset['groups'])).astype(str)
    roles = np.asarray(dataset['partition_roles']).astype(str)
    if roles.shape != labels.shape or not set(roles) <= {'train', 'validation', 'test'}:
        raise ValueError('invalid partition roles')
    masks = {role: roles == role for role in ('train', 'validation', 'test')}
    classes = np.unique(labels)
    for role, mask in masks.items():
        if not np.array_equal(np.unique(labels[mask]), classes):
            raise ValueError(f'every class must occur in {role}')
        if set(groups[mask]) != set(metadata['days'][role]):
            raise ValueError(f'{role} dates differ from frozen manifest')
        counts = np.bincount(labels[mask], minlength=len(classes)).tolist()
        if counts != metadata['retained_counts'][role]:
            raise ValueError(f'{role} counts differ from frozen manifest')
        if role != 'train' and counts != metadata['raw_counts'][role]:
            raise ValueError('validation/test rows must not be sampled')
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if set(groups[masks[left]]) & set(groups[masks[right]]):
            raise ValueError('a UTC day crosses frozen partitions')
    expected = np.minimum(metadata['raw_counts']['train'], metadata['train_cap_per_class']).tolist()
    if metadata['retained_counts']['train'] != expected:
        raise ValueError('training counts violate the per-class cap')
    return masks, metadata
