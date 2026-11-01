"""Regression checks for adapted physical-v5 iPET and its budgeted variant."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
for path in (PROJECT_DIR, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ipet_v2aligned as ipet  # noqa: E402
from ipet_v2aligned import model as aligned_model  # noqa: E402


def _synthetic_dataset(class_count: int):
    samples_per_class = 2
    sample_count = class_count * samples_per_class
    event_count = 32
    labels = np.repeat(np.arange(class_count), samples_per_class)
    lengths = np.zeros((sample_count, event_count), dtype=np.float64)
    gaps = np.zeros_like(lengths)
    directions = np.zeros((sample_count, event_count), dtype=np.int8)
    mask = np.zeros_like(directions, dtype=np.uint8)
    for sample in range(sample_count):
        count = 8 + sample % 3
        mask[sample, :count] = 1
        directions[sample, :count] = np.where(np.arange(count) % 2, -1, 1)
        lengths[sample, :count] = 80 + 11 * labels[sample] + np.arange(count)
        gaps[sample, :count] = 0.03
        gaps[sample, 0] = 0.0
    groups = np.asarray([f"capture-{sample}" for sample in range(sample_count)])
    return lengths, gaps, directions, mask, labels, groups


def _assert_dropout_active() -> None:
    discriminator = aligned_model.OfficialMSequenceDiscriminator(4, 2)
    generator = aligned_model.OfficialIPETGenerator(40)
    observed_modes = []

    def remember_mode(module, _inputs):
        observed_modes.append(bool(module.training))

    hook = discriminator.dropout.register_forward_pre_hook(remember_mode)
    try:
        aligned_model._train_generator_epochs(
            generator,
            discriminator,
            torch.zeros((4, 40, 4), dtype=torch.float32),
            torch.zeros((4, 1), dtype=torch.float32),
            torch.zeros(4, dtype=torch.int64),
            epochs=1,
            seed=123,
        )
    finally:
        hook.remove()
    if not observed_modes or not all(observed_modes):
        raise AssertionError("训练G时判别器Dropout未保持训练模式")
    if not all(parameter.requires_grad for parameter in discriminator.parameters()):
        raise AssertionError("训练G后判别器参数未恢复可训练状态")


def _exercise_class_count(class_count: int):
    arrays = _synthetic_dataset(class_count)
    policy = ipet.fit_baseline(
        *arrays[:4],
        arrays[4],
        np.arange(len(arrays[4])),
        groups=arrays[5],
        seed=7000 + class_count,
        discriminator_epochs=1,
        training_stages=2,
        generator_epochs=1,
        discriminator_retrain_epochs=1,
    )
    if len(policy.class_labels) != class_count:
        raise AssertionError("4/6分类训练没有复用同一类别无关实现")

    history = policy.training_metadata["training_history"]
    alignment = history["alignment"]
    expected = {
        "initial_discriminator_batch_size": 128,
        "generator_batch_size": 32,
        "discriminator_retrain_batch_size": 32,
        "nadam_lr": 0.001,
        "nadam_eps": 1e-7,
        "discriminator_dropout_active_for_generator": True,
        "discriminator_adam_reused_across_stages": True,
        "official_perturbation_vector_count": 20_000,
        "expected_flattened_pool_rows": 800_000,
    }
    for key, value in expected.items():
        if alignment.get(key) != value:
            raise AssertionError(f"对齐配置错误: {key}={alignment.get(key)!r}")

    stages = history["stages"]
    if len(stages) != 2:
        raise AssertionError("stage数量错误")
    previous_after = history["initial_discriminator_optimizer_steps"][1]
    for stage in stages:
        before = stage["discriminator_optimizer_steps_before_retrain"][0]
        after = stage["discriminator_optimizer_steps_after_retrain"][0]
        if before != previous_after or after <= before:
            raise AssertionError("判别器Adam step未跨stage连续累积")
        previous_after = stage["discriminator_optimizer_steps_after_retrain"][1]

    label = policy.class_labels[0]
    first = policy.generate(label, 9917)
    second = policy.generate(label, 9917)
    different_session = policy.generate(label, 9918)
    if first.values.shape != (800_000, 4):
        raise AssertionError(f"官方扰动池形状错误: {first.values.shape}")
    if first.values is not second.values:
        raise AssertionError("同一设备的20000向量扰动池没有复用")
    if first.start_index != second.start_index:
        raise AssertionError("相同session seed未选择相同扰动起点")
    if first.start_index == different_session.start_index:
        raise AssertionError("不同session seed未增加跨流扰动多样性")
    if policy.pool_generation_counts[label] != 1:
        raise AssertionError("同一设备的长扰动池被重复生成")

    source_count = int(arrays[3][0].sum())
    _, injection = ipet.inject_official_ipet_trace(
        first,
        arrays[0][0, :source_count],
        arrays[1][0, :source_count],
        arrays[2][0, :source_count],
        dummy_directions=policy.dummy_directions,
    )
    if injection["perturbation_rows_consumed"] <= 40:
        raise AssertionError("测试流没有覆盖旧版40行重复边界")
    if injection["perturbation_pool_wrapped"]:
        raise AssertionError("普通测试流意外耗尽800000行扰动池")

    budgeted = ipet.apply_budgeted_policy_v3(
        policy,
        *(value[:1] for value in arrays[:4]),
        arrays[4][:1],
        max_added_bytes=180.0,
        max_dummy_packets=2,
        base_seed=9917,
    )
    budget_stats = budgeted[4]
    budget_audit = budgeted[5][0]
    if budget_stats["dummy_bytes"][0] > 180.0 + 1e-9:
        raise AssertionError("iPET-physical-v5-budgeted测试违反字节预算")
    if budget_stats["dummy_count"][0] > 2:
        raise AssertionError("iPET-physical-v5-budgeted测试违反dummy数量预算")
    if budget_audit["method"] != ipet.IPET_BUDGETED_V3_METHOD:
        raise AssertionError("iPET-physical-v5-budgeted审计方法名错误")
    if not budget_audit["hard_budget_passed"]:
        raise AssertionError("iPET-physical-v5-budgeted预算审计未通过")
    if budget_audit["proposed_dummy_packets"] < budget_audit["dummy_packets"]:
        raise AssertionError("iPET-physical-v5-budgeted发出量超过原始提议量")

    result = {
        "classes": class_count,
        "class_labels": list(policy.class_labels),
        "pool_shape": list(first.values.shape),
        "pool_megabytes_float32": round(first.values.nbytes / 1_000_000, 3),
        "pool_generated_once_for_test_label": True,
        "different_sessions_use_different_vector_boundaries": True,
        "rows_consumed_by_probe_flow": injection["perturbation_rows_consumed"],
        "probe_flow_wrapped_pool": injection["perturbation_pool_wrapped"],
        "discriminator_adam_final_step": previous_after,
        "budgeted_v3_hard_budget_passed": True,
    }
    del first, second, different_session, policy
    gc.collect()
    return result


def main() -> None:
    _assert_dropout_active()
    results = [_exercise_class_count(class_count) for class_count in (4, 6, 10)]
    print(
        json.dumps(
            {
                "all_passed": True,
                "isolated_directory": str(HERE),
                "old_implementation_imported_as_code": False,
                "dropout_active_during_generator_training": True,
                "class_count_checks": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
