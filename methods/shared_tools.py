"""FADE evaluation helpers extracted from the original shared orchestration module.

No Method32 defense or policy-selection algorithm is included. The method32
attribute in capture-split helpers is the existing Securitas compatibility field.
"""

# =============================================================================
# 中文阅读导图：本文件是“被多个入口复用的实验积木”，不实现任何防御
# =============================================================================
# 调用方：
#   experiments/run_experiment.py        用 build_stratified_nested_masks、
#                                        _train_quick_attackers 构造嵌套重训练mask
#   experiment_support/rotation_history.py 用 _trigger_feature 计算码字距离
#   methods/method33_tools.py            保留了一份同名的数据集对齐/掩码实现，
#                                        目的是让独立 CLI 不必依赖实验编排层
#
# 四个关键能力：
#   SelectionSplit / build_policy_selection_split
#       无泄漏三段划分：selection 与 validation 都取自 outer-train，
#       outer-test 只用于最终评测，绝不参与候选选择；audit 记录 capture 名、
#       行数、交集和行号 SHA-256，使“无泄漏”可被独立复核。
#   validate_dataset_project_alignment
#       校验 NPZ 的 device_names、类别数与配置一致。设备顺序就是标签语义，
#       换顺序等同于换数据集版本。
#   build_stratified_nested_masks
#       每个类别只打乱一次，按比例取前缀；因此 r=10% 的受保护样本严格包含在
#       r=25% 中，曲线变化只来自比例，而不是“每个点换一批样本”。
#   _fit_full_attacker / _train_quick_attackers
#       统一超参的 MLP/RF/SVM 工厂，返回 (StandardScaler, {name: model})；
#       scaler 只在训练集拟合，测试特征只 transform。
#
# 复杂语法提示：_trigger_feature 把码字 ((l1,g1),(l2,g2)) 展平成向量，
# 供轮换模块计算码字间欧氏距离；它只服务“选下一个映射”，不参与在线防御。
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import hashlib
import json
import numpy as np
from data_pipeline import pcap as dataset_pcap


@dataclass(frozen=True)
class SelectionSplit:
    """Leakage-free three-way split used by offline policy selection.

    ``selection_indices`` and ``validation_indices`` are both subsets of the
    outer training partition.  ``outer_test_indices`` is carried only for the
    final frozen-policy evaluation and is never read by either candidate
    selector.  ``audit`` records the actual capture names so that this claim is
    independently checkable from the saved manifest.
    """

    # 全局切分对象中，这三组是原NPZ行号；传入候选选择器前会再构造一个不含
    # outer-test数据的局部视图，其outer_test_indices为空。audit保存capture名、
    # 行数、交集和行号hash，防止只凭一句“无泄漏”而无法核验。
    selection_indices: np.ndarray
    validation_indices: np.ndarray
    outer_test_indices: np.ndarray
    audit: Mapping[str, Any]


def validate_dataset_project_alignment(
    dataset: Mapping[str, np.ndarray], project: Any
) -> None:
    """Reject label/device-order mismatches before selection or evaluation.

    ``dataset_pcap.validate_dataset`` proves that labels are internally
    contiguous, but it cannot know which configured device each integer means.
    Trigger mappings are label-indexed, so silently swapping two device names
    would invalidate the experiment even though every array shape remained
    legal.  This cross-object check closes that gap.
    """

    dataset_pcap.validate_dataset(dataset, expected_width=project.dataset.max_packets)
    labels = np.asarray(dataset["labels"], dtype=int)
    expected_labels = np.arange(project.method32.class_count)
    if not np.array_equal(np.unique(labels), expected_labels):
        raise ValueError(
            "dataset labels do not match configured class count: "
            f"found {np.unique(labels).tolist()}, expected {expected_labels.tolist()}"
        )
    if "device_names" not in dataset:
        raise ValueError("dataset must include device_names for config-alignment validation")
    actual_devices = tuple(np.asarray(dataset["device_names"]).astype(str).tolist())
    if actual_devices != tuple(project.dataset.devices):
        raise ValueError(
            "dataset device_names order differs from configuration: "
            f"dataset={list(actual_devices)}, config={list(project.dataset.devices)}"
        )

    manifest_key = "unsw_extraction_manifest_json"
    if manifest_key in dataset:
        encoded = np.asarray(dataset[manifest_key])
        if encoded.shape != ():
            raise ValueError("unsw_extraction_manifest_json must be a scalar JSON string")
        try:
            manifest = json.loads(str(encoded.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("unsw_extraction_manifest_json is invalid") from error
        if not isinstance(manifest, Mapping):
            raise ValueError("UNSW extraction manifest must decode to an object")
        manifest_devices = tuple(str(value) for value in manifest.get("devices", ()))
        if manifest_devices != actual_devices:
            raise ValueError(
                "UNSW extraction manifest devices differ from dataset device_names"
            )
        manifest_class_count = manifest.get("class_count", len(manifest_devices))
        if int(manifest_class_count) != project.method32.class_count:
            raise ValueError(
                "UNSW extraction manifest class_count differs from configuration"
            )


def build_stratified_nested_masks(
    labels: np.ndarray,
    ratios: Sequence[float],
    *,
    seed: int,
) -> dict[float, np.ndarray]:
    """Build shared class-stratified masks nested across retraining ratios.

    A class is shuffled once, then each larger ratio takes a longer prefix of
    that same order.  Candidate identity is intentionally absent from the seed;
    all weakener candidates in one repeat must see the same retraining samples.
    """

    # 关键研究约束：seed 只由 repeat/协议预先决定，绝不混入候选弱化器 ID。
    # 因而同一轮中的所有候选都会在完全相同的重训练流上被比较。
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a nonempty one-dimensional array")
    ratios = tuple(float(value) for value in ratios)
    if any(not 0.0 <= value <= 1.0 for value in ratios):
        raise ValueError("ratios must lie in [0, 1]")
    # 每个类别只打乱一次；比例 r 取该固定顺序的前 round(n_k*r) 条。所以
    # 10% 样本天然包含于 25%，25% 又包含于 50%，减少曲线中的采样噪声。
    class_orders: dict[int, np.ndarray] = {}
    for label in np.unique(labels):
        local = np.flatnonzero(labels == label)
        rng = np.random.default_rng(int(seed) + 101 * int(label))
        class_orders[int(label)] = rng.permutation(local)
    result: dict[float, np.ndarray] = {}
    for ratio in ratios:
        selected = np.zeros(len(labels), dtype=bool)
        for order in class_orders.values():
            count = int(np.rint(len(order) * ratio))
            selected[order[:count]] = True
        result[ratio] = selected
    return result


def _index_sha256(indices: np.ndarray) -> str:
    """Hash one sorted index vector for compact experiment provenance."""

    # 显式使用little-endian int64，使同一行号集合在不同NumPy默认整数宽度或
    # CPU端序上仍得到同一个hash；hash用于核对，不替代CSV中的实际样本数。
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def build_policy_selection_split(
    dataset: Mapping[str, np.ndarray], project: Any
) -> SelectionSplit:
    """Use a frozen dataset partition, or build the historical capture split.

    The outer split is the final train/test boundary.  Inside outer-train, the
    historical Method32 protocol draws exactly
    ``validation_captures_per_class`` complete captures from every class for
    candidate validation; all remaining outer-train captures form
    selection-train.  Using a fixed capture count here is deliberate: replacing
    it with a flow ratio would silently change the data protocol and potentially
    the selected policy.
    """

    validate_dataset_project_alignment(dataset, project)
    labels = np.asarray(dataset["labels"], dtype=int)
    groups = np.asarray(dataset.get("capture_ids",dataset["groups"])).astype(str)

    from data_pipeline.partitions import frozen_partition
    frozen = frozen_partition(dataset, train_ratio=project.evaluation.capture_train_ratio,
                              seed=project.evaluation.split_seed)
    if frozen is not None:
        masks, metadata = frozen
        if project.selection.inner_split_seed != metadata['inner_seed']:
            raise ValueError('selection seed differs from frozen validation partition')
        indices = {role: np.flatnonzero(mask) for role, mask in masks.items()}
        audit = {
            'protocol': metadata['protocol'], 'frozen_partition': metadata,
            'selection_train_flows': len(indices['train']),
            'validation_flows': len(indices['validation']), 'outer_test_flows': len(indices['test']),
            'selection_train_capture_count': len(metadata['days']['train']),
            'validation_capture_count': len(metadata['days']['validation']),
            'outer_test_capture_count': len(metadata['days']['test']),
            'all_capture_intersections_empty': True, 'outer_test_used_for_selection': False,
            'capture_intersections': {'selection_validation': [], 'selection_outer_test': [], 'validation_outer_test': []},
            'selection_indices_sha256': _index_sha256(indices['train']),
            'validation_indices_sha256': _index_sha256(indices['validation']),
            'outer_test_indices_sha256': _index_sha256(indices['test']),
            'per_class': {device: {
                'label': label,
                'selection_train_captures': sorted(set(groups[masks['train'] & (labels == label)])),
                'validation_captures': sorted(set(groups[masks['validation'] & (labels == label)])),
                'outer_test_captures': sorted(set(groups[masks['test'] & (labels == label)])),
            } for label, device in enumerate(project.dataset.devices)},
        }
        return SelectionSplit(selection_indices=indices['train'], validation_indices=indices['validation'],
                              outer_test_indices=indices['test'], audit=audit)

    # 第一道边界由数据模块统一实现：同一个capture整体进入outer-train或
    # outer-test。最终测试capture此时就被封存，后续两个选择器只接收内部的
    # selection/validation行号，不会把outer-test特征用于候选打分。
    outer_train_mask, outer_test_mask = dataset_pcap.split_by_capture(
        labels,
        groups,
        train_ratio=project.evaluation.capture_train_ratio,
        seed=project.evaluation.split_seed,
    )
    outer_train_indices = np.flatnonzero(outer_train_mask)
    outer_test_indices = np.flatnonzero(outer_test_mask)

    # 历史实现先对capture名称排序，再让一个跨类别共享的RNG依次shuffle每类
    # capture。必须保留这个次序；改成“每类独立seed”虽然仍无泄漏，却不再是
    # 同一个可复现实验切分。
    rng = np.random.default_rng(project.selection.inner_split_seed)
    validation_groups: list[str] = []
    per_class: dict[str, Any] = {}
    count = project.selection.validation_captures_per_class
    global_validation = None
    mixed_capture_labels = any(len(np.unique(labels[groups == group])) > 1 for group in np.unique(groups))
    if mixed_capture_labels:
        candidates=np.unique(groups[outer_train_indices])
        if len(candidates)<=count:raise ValueError('not enough whole captures for inner train/validation')
        for _ in range(4096):
            chosen=rng.permutation(candidates)[:count]
            chosen_mask=np.isin(groups[outer_train_indices],chosen)
            if (len(np.unique(labels[outer_train_indices[chosen_mask]])) == len(project.dataset.devices)
                and len(np.unique(labels[outer_train_indices[~chosen_mask]])) == len(project.dataset.devices)):
                global_validation=set(chosen.tolist());break
        if global_validation is None:
            raise ValueError('Cannot cover every class with the requested whole validation captures; increase validation_captures_per_class')
    for label, device in enumerate(project.dataset.devices):
        class_indices = outer_train_indices[labels[outer_train_indices] == label]
        class_groups = np.asarray(sorted(set(groups[class_indices].tolist())), dtype=str)
        if global_validation is None and len(class_groups) <= count:
            raise ValueError(
                f"class {label} ({device}) has {len(class_groups)} outer-train captures; "
                f"need more than validation_captures_per_class={count}"
            )
        shuffled = class_groups.copy()
        rng.shuffle(shuffled)
        selected = sorted(set(class_groups.tolist()) & global_validation) if global_validation is not None else sorted(shuffled[:count].tolist())
        retained = sorted(set(class_groups.tolist()) - global_validation) if global_validation is not None else sorted(shuffled[count:].tolist())
        validation_groups.extend(selected)
        per_class[device] = {
            "label": int(label),
            "validation_captures": selected,
            "selection_train_captures": retained,
            "outer_test_captures": sorted(
                set(groups[outer_test_indices][labels[outer_test_indices] == label].tolist())
            ),
        }

    validation_mask_within_outer = np.isin(
        groups[outer_train_indices], np.asarray(validation_groups, dtype=str)
    )
    validation_indices = np.sort(outer_train_indices[validation_mask_within_outer])
    selection_indices = np.sort(outer_train_indices[~validation_mask_within_outer])

    selection_groups = set(groups[selection_indices].tolist())
    validation_group_set = set(groups[validation_indices].tolist())
    outer_test_groups = set(groups[outer_test_indices].tolist())
    intersections = {
        "selection_validation": sorted(selection_groups & validation_group_set),
        "selection_outer_test": sorted(selection_groups & outer_test_groups),
        "validation_outer_test": sorted(validation_group_set & outer_test_groups),
    }
    if any(intersections.values()):
        raise AssertionError(f"capture leakage detected: {intersections}")

    audit = {
        "protocol": "whole shared captures for inner validation" if mixed_capture_labels else "outer split, then fixed validation captures per class",
        "outer_split_seed": int(project.evaluation.split_seed),
        "inner_split_seed": int(project.selection.inner_split_seed),
        "validation_captures_per_class": int(count),
        "selection_train_flows": int(len(selection_indices)),
        "validation_flows": int(len(validation_indices)),
        "outer_test_flows": int(len(outer_test_indices)),
        "selection_train_capture_count": int(len(selection_groups)),
        "validation_capture_count": int(len(validation_group_set)),
        "outer_test_capture_count": int(len(outer_test_groups)),
        "capture_intersections": intersections,
        "all_capture_intersections_empty": True,
        "selection_indices_sha256": _index_sha256(selection_indices),
        "validation_indices_sha256": _index_sha256(validation_indices),
        "outer_test_indices_sha256": _index_sha256(outer_test_indices),
        "per_class": per_class,
        "outer_test_used_for_selection": False,
    }
    return SelectionSplit(
        selection_indices=selection_indices,
        validation_indices=validation_indices,
        outer_test_indices=np.sort(outer_test_indices),
        audit=audit,
    )


def _trigger_feature(
    code: Sequence[Sequence[float]], config: Any
) -> np.ndarray:
    """Convert one T-packet code into a dimensionless ``[length,gap,...]`` vector."""

    # 原始“字节”和“秒”量纲不同，直接做欧氏距离会让数值较大的字节维支配
    # 结果。分别除以固定Trigger总预算后，每包长度占比和gap占比处于可比尺度。
    array = np.asarray(code, dtype=np.float64)
    expected = (config.trigger_packet_count, 2)
    if array.shape != expected:
        raise ValueError(f"trigger code shape {array.shape} does not equal {expected}")
    return np.column_stack(
        (
            array[:, 0] / config.trigger_budget.bytes,
            array[:, 1] / config.trigger_budget.delay_seconds,
        )
    ).reshape(-1)


def _fit_full_attacker(name, features, labels, *, seed):
    """Fit one architecture on every row, with training-only class weights."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.utils.class_weight import compute_sample_weight

    if name == "MLP-full":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500,
                              early_stopping=False, random_state=int(seed))
        return model.fit(features, labels,
                         sample_weight=compute_sample_weight("balanced", labels))
    if name == "RF-full":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=int(seed) + 2,
            # Bound each experiment's CPU usage when datasets run concurrently.
            n_jobs=4,
        ).fit(features, labels)
    if name == "SVM-full":
        return SVC(class_weight="balanced", random_state=int(seed)).fit(features, labels)
    raise KeyError(name)


def _train_quick_attackers(features, labels, *, seed):
    """Train MLP, RF and SVM on all rows; no class undersampling."""
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    models = {name: _fit_full_attacker(name, scaled, labels, seed=seed)
              for name in ("MLP-full", "RF-full", "SVM-full")}
    return scaler, models
