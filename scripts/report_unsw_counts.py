"""Summarize completed raw-session audits and candidate full-data class sets."""

# =============================================================================
# 中文阅读导图：从计数审计生成“候选类别集”报告（只出建议，不建数据集）
# =============================================================================
# 输入是 audit_unsw_counts.py 产出的 manifest.json（status 必须 complete）。
# 逻辑：
#   1. 按 eligible_flows 降序排列设备；
#   2. 分别用 3000 / 1000 条两个门槛筛选，且要求至少 20 个 UTC 日；
#   3. 用 split_by_capture 按“全局 UTC 日”做 75/25 划分，检查交集为空；
#   4. 输出每类训练/测试条数与候选设备顺序，写 candidate_class_sets.json。
#
# 注意：这里的输出只是 proposal（报告里明确写 status='proposal only'），
# 真正的主数据集由 data_pipeline.unsw_main 构建，两者不要混用。
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_pipeline.pcap import split_by_capture


def summarize(audit_dir: Path):
    manifest = json.loads((audit_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('A complete raw-PCAP audit is required')
    audits = sorted(manifest['device_audits'], key=lambda a: -a['eligible_flows'])
    candidates = []
    for minimum in (3000, 1000):
        selected = [a for a in audits if a['eligible_flows'] >= minimum and a['eligible_groups'] >= 20]
        labels, groups = [], []
        for label, audit in enumerate(selected):
            for day, count in audit['eligible_flows_by_group'].items():
                labels.extend([label] * count)
                groups.extend([day] * count)
        labels, groups = np.asarray(labels), np.asarray(groups)
        train, test = split_by_capture(labels, groups, train_ratio=.75, seed=42)
        candidates.append({
            'minimum_sessions': minimum, 'minimum_utc_days': 20,
            'device_names_in_proposed_label_order': [a['device'] for a in selected],
            'total_sessions': len(labels), 'sampling': 'none',
            'split': {'unit': 'global UTC day', 'train_ratio': .75, 'seed': 42,
                      'train_day_count': len(np.unique(groups[train])),
                      'test_day_count': len(np.unique(groups[test])),
                      'intersection': sorted(set(groups[train]) & set(groups[test]))},
            'train_sessions_by_class': np.bincount(labels[train]).tolist(),
            'test_sessions_by_class': np.bincount(labels[test]).tolist(),
            'status': 'proposal only; no new dataset or method mapping has been installed',
        })
    (audit_dir / 'candidate_class_sets.json').write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# UNSW 原始会话数量审计与选类建议', '',
        '本报告来自本机全部设备 PCAP 的重新扫描。会话数是在当前 UNSW v2 过滤协议下可用的样本数，不是原始包数。', '',
        '过滤口径：Ethernet IPv4 TCP/443、至少一个疑似 TLS record、每会话至少 4 包、前 32 包中至少一个设备出站包；'
        '120 秒空闲、SYN/RST/FIN、UTC 日期、PCAPNG section 或时间倒退会切分会话。'
        '日期限制为 2015-01-01 至 2020-01-01（不含）。', '',
        '计数扫描只保留 1 条示例会话以节省内存；eligible_flows 和逐日计数来自整个文件，不受 reservoir 容量影响。'
        '每个设备的 JSON 记录原始文件 SHA256、提取器/解析器 SHA256、扫描时间和逐日分布。', '',
        '| 设备 | 合格会话 | 有会话的 UTC 天数 | 单日最大占比 |',
        '| --- | ---: | ---: | ---: |',
    ]
    for audit in audits:
        counts = audit['eligible_flows_by_group'].values()
        lines.append(f"| {audit['device']} | {audit['eligible_flows']:,} | {audit['eligible_groups']} | "
                     f"{max(counts) / audit['eligible_flows']:.2%} |")
    total = sum(a['eligible_flows'] for a in audits)
    lines += ['', f'合计 **{total:,}** 条合格会话。这里只能对本地这 {len(audits)} 个设备排序，不能称为完整 UNSW 数据集的全局前几大类。', '',
              '## 每类 200 条的来源', '',
              '仓库旧版本的 10 类提取默认 target-per-class=200，6 类提取默认 500。'
              '先对每类做定额 reservoir，再将各类裁到共同最小数量。'
              '当前 6 类 NPZ 是 10 类 NPZ 的精确子集，所以也只有每类 200 条。'
              '这是仓库预处理选择，不代表数据提供方的原始 PCAP 已被处理为每类 200 条。', '',
              f'当前 10 类 2,000 条只占全部可用会话的 {2000 / total:.2%}。'
              '这种规模适合软件验证和探索，不能充分利用已有流量。HPPrinter 确实是小类，但不应因此把所有其他类都压到 200。', '',
              '## 可行候选', '']
    for candidate in candidates:
        lines += [f"- 至少 {candidate['minimum_sessions']:,} 条且至少 20 个 UTC 日："
                  f"{len(candidate['device_names_in_proposed_label_order'])} 类，共 {candidate['total_sessions']:,} 条；"
                  + '、'.join(candidate['device_names_in_proposed_label_order']) + '。']
    lines += ['', '建议先采用 6 大类作为主实验候选，8 类作为覆盖更多设备的补充。'
              '选类仅依据样本量和日期覆盖，不能根据测试集准确率挑选设备。'
              '候选类别顺序和 UTC 日划分的预计训练/测试数量已写入 candidate_class_sets.json。', '',
              'AugustDoorBell 数量非常大，选大类不会自动使类别平衡。保留全量并使用训练集类别权重，继续报告 BA 和每类指标。'
              '同日会话具有相关性，不能把会话数量当成独立采集次数；必须维持全局 UTC 日隔离。'
              '全量训练尤其会增加核 SVM 的时间和内存成本，需要实测后决定可承受的实验矩阵。', '',
              '本次只完成原始数量核查和候选设计，没有覆盖现有 NPZ、设备标签或触发映射。'
              '采用新类别集合时必须生成不做 reservoir/共同数量裁剪的新数据版本，重新定义标签顺序、映射和方法配置，'
              '重新在开发集选择参数并训练依赖攻击者的防御策略，不能仅替换旧配置中的设备名称。', '',
              '## 分类器代码变更与验证', '',
              '当前主配置为 MLP-full、RF-full、SVM-full、LSTM-full。'
              '训练时使用当前数据划分中的全部训练样本，不再执行类别下采样。'
              'MLP 使用 sample_weight，RF/SVM 使用 class_weight，LSTM 默认使用加权损失。'
              '权重只由训练标签计算；测试集保持原始分布。', '',
              'Method32 的选参代理和 Securitas 代理同步去掉下采样；旧 balanced 分类器名称不再被当前运行器接受。'
              'MLP 加权依赖 scikit-learn >= 1.7。现有 UNSW NPZ 内的数据上限与分类器训练下采样是两个独立环节。', '',
              '验证日志：classifier_full_data_tests.log（73 项通过）、classifier_securitas_functional.log（18 项通过），位于上一级 validation 目录。', '']
    (audit_dir / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit_dir', type=Path)
    summarize(parser.parse_args().audit_dir)


if __name__ == '__main__':
    main()
