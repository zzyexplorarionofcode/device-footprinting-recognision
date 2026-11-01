"""Glimpse-Seq64-Oracle-v2 的独立最小演示。

这个脚本只使用本目录中的包。它生成少量合成逐流数据，严格先划分训练/测试集，
再训练每台设备各自的 Isolation Forest 和 DDPG 策略，最后输出干净流量与防御后
流量的分类准确率及开销。合成数据仅用于验证代码通路，不是论文实验结果。

论文没有给出 gamma、rho、奖励阈值、训练迭代数、最大延迟和最大分片数，因此
命令行故意要求显式传入这些值，不把任何自选数字伪装成论文默认值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glimpse_seq64_oracle_v2 import (
    GlimpseConfig,
    GlimpseSeq64OracleV2,
    apply_policy,
    make_features,
)


def make_synthetic_data(samples_per_class: int, seed: int):
    """构造四类、每条32包的可分合成数据；只承担独立demo的冒烟测试。"""

    if samples_per_class < 8:
        raise ValueError("每类至少需要8条流，才能同时形成训练集和测试集")
    rng = np.random.default_rng(seed)
    all_lengths, all_gaps, all_directions, all_masks, all_labels = [], [], [], [], []
    for label in range(4):
        for _ in range(samples_per_class):
            # 不同类别具有不同平均包长与时间节奏，使分类器能识别设备。
            lengths = np.rint(np.clip(
                rng.normal(260.0 + 220.0 * label, 35.0, size=32), 60.0, 1450.0
            ))
            gaps = np.clip(
                rng.exponential(0.008 + 0.009 * label, size=32), 0.0, 1.0
            )
            gaps[0] = 0.0  # 与PCAP提取一致：首包没有流内前驱。
            directions = np.where(np.arange(32) % (label + 2) == 0, 1, -1)
            # 保证每条流都有足够的outbound包供最多10次动作使用。
            directions[:10] = 1
            all_lengths.append(lengths)
            all_gaps.append(gaps)
            all_directions.append(directions)
            all_masks.append(np.ones(32, dtype=bool))
            all_labels.append(label)
    return tuple(
        np.asarray(value)
        for value in (
            all_lengths,
            all_gaps,
            all_directions,
            all_masks,
            all_labels,
        )
    )


def stratified_split(labels: np.ndarray, seed: int):
    """每类按70%/30%划分，防止同一条测试流进入奖励模型或DDPG训练。"""

    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cut = max(2, int(round(0.7 * len(indices))))
        train.extend(indices[:cut])
        test.extend(indices[cut:])
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--reward-threshold", type=float, required=True)
    parser.add_argument("--training-iterations", type=int, required=True)
    parser.add_argument("--max-delay-seconds", type=float, required=True)
    parser.add_argument("--max-fragments", type=int, required=True)
    parser.add_argument("--target-update-period", type=int, required=True)
    parser.add_argument("--replay-buffer-capacity", type=int, required=True)
    parser.add_argument("--exploration-noise-stddev", type=float, required=True)
    parser.add_argument("--exploration-noise-damping", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frame-header-bytes", type=int, default=54)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--save-policy", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    config = GlimpseConfig(
        gamma=args.gamma,
        rho=args.rho,
        reward_threshold=args.reward_threshold,
        training_iterations=args.training_iterations,
        max_delay_seconds=args.max_delay_seconds,
        max_fragments=args.max_fragments,
        target_update_period=args.target_update_period,
        replay_buffer_capacity=args.replay_buffer_capacity,
        exploration_noise_stddev=args.exploration_noise_stddev,
        exploration_noise_damping=args.exploration_noise_damping,
        seed=args.seed,
        frame_header_bytes=args.frame_header_bytes,
    )
    lengths, gaps, directions, mask, labels = make_synthetic_data(
        args.samples_per_class, args.seed
    )
    train_indices, test_indices = stratified_split(labels, args.seed)

    # 攻击者分类器只在干净训练集拟合；这里的RF仅用于演示，不属于Glimpse本体。
    clean_features = make_features(lengths, gaps, directions, mask)
    classifier = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    classifier.fit(clean_features[train_indices], labels[train_indices])
    clean_accuracy = classifier.score(
        clean_features[test_indices], labels[test_indices]
    )

    policy = GlimpseSeq64OracleV2.fit(
        lengths,
        gaps,
        directions,
        mask,
        labels,
        config=config,
        train_indices=train_indices,
    )
    protected = apply_policy(
        policy,
        lengths[test_indices],
        gaps[test_indices],
        directions[test_indices],
        mask[test_indices],
        labels[test_indices],
    )
    p_lengths, p_gaps, p_directions, p_mask, costs, metadata = protected
    protected_features = make_features(p_lengths, p_gaps, p_directions, p_mask)
    protected_accuracy = classifier.score(protected_features, labels[test_indices])

    if args.save_policy:
        policy.save(args.save_policy)
    print("\n=== Glimpse-Seq64-Oracle-v2 独立demo（合成数据，不是论文结果） ===")
    print("显式配置:", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
    print(f"训练流/测试流: {len(train_indices)}/{len(test_indices)}")
    print(f"干净准确率: {clean_accuracy:.4f}")
    print(f"防御后准确率: {protected_accuracy:.4f}")
    print(f"防御后误识别率: {1.0 - protected_accuracy:.4f}")
    print(f"平均增加字节: {np.mean([row['added_bytes'] for row in costs]):.3f}")
    print(f"平均增加延迟(s): {np.mean([row['added_delay'] for row in costs]):.6f}")
    print("训练日志:", json.dumps(policy.training_log, ensure_ascii=False))
    print("适配元数据:", json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
