"""Standalone aligned IoTGAN baseline for this project's TLS-flow protocol.

This file deliberately implements only the scope used by the paper comparison:

* device misidentification (IoTGAN Equation 4), not identity spoofing;
* outbound-only, functionality-preserving actions;
* non-negative padding and delay on outgoing real packets;
* one independently trained substitute/manipulator pair per black-box target;
* paper-described top-L feature refinement with Performance Gain reporting.

The paper's ``r*h`` term is retained only as the second input to the generator.
It is not used as a hard upper bound on the generated residual.  Generated
features are constrained only by the declared mutability mask and by physical
packet-length/gap bounds.

The IoTGAN paper does not disclose the feature-weighting algorithm, candidate L
values, exact network widths, loss implementation, optimizer, or stopping
thresholds.  Every such choice is recorded in policy metadata.  This module is
self-contained and does not import baselines.iotgan.

The hyphen in this requested filename means it cannot be imported with an
ordinary Python import statement.  It can be executed directly, or loaded with
importlib.util.spec_from_file_location.
"""

# =============================================================================
# 中文阅读导图：IoTGAN 对齐版（最“重”的生成式基线之一）
# =============================================================================
# 论文思想：训练一个替代分类器（substitute）逼近黑盒识别器，再训练一个生成器
# （manipulator）去改变流量特征，使替代分类器误判；训练完成后只把生成器
# 部署到流量上。本文件按论文 Algorithm 1 / 公式 (2)-(4) 复现，并明确缩到
# “设备误识别 + 仅上行 + 保功能”的范围。
#
# 结构（按训练顺序读）：
#   trace_feature_matrix()        四数组 → [N,64] 特征（与 make_features 同口径）
#   SubstituteClassifier()        替代分类器：MLP，学习黑盒目标的输出
#   ManipulativeModel()           生成器：输入当前特征，输出有界残差
#   _project_generated_raw()      物理投影：把残差变成整数字节/秒增量，
#                                 并限制在帧长、MTU、gap 上界内（可微）
#   _select_feature_subset()      按 Performance Gain 选 top-L 可修改特征：
#                                 候选子集在验证集上比较“降识别率 / 增开销”
#   IoTGANAlignedPolicy          训练好的策略对象（生成器+替代器+特征掩码+元数据）
#   fit_baseline / fit_target_specific_baselines
#                                 每个黑盒目标单独训练一套策略（target_specific）
#   apply_policy()                部署：只改被选中且允许修改的上行真实事件
#
# 关键点：
#   * 生成器输出不设 r*h 硬上界；r*h 只作为第二路输入（论文未披露细节，
#     这里按可复现的方式实现并把选择写入 metadata）；
#   * 所有随机性经 _set_seed/_state_digest 记录，策略保存/加载会校验
#     TRACE_CONSTRAINT_VERSION，防止用旧约束下训练的策略直接评测；
#   * baselines/high_budget.py 会在 fit/apply 期间临时替换
#     _project_generated_raw，把同一个可微投影换成统一预算版本。
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from experiment_support.metrics import timing_metrics, store_timing
import torch
from torch import nn

try:
    from . import common
except ImportError:  # Support: python baselines/iotgan_aligned.py
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    from baselines import common


SEED = common.SEED
OBSERVED_EVENTS = common.OBSERVED_EVENTS
FEATURE_COUNT = 2 * OBSERVED_EVENTS
SCHEMA_VERSION = 3
PAPER_PATH = Path(__file__).resolve().parents[1] / "docs/papers/IoTGAN.pdf"
METHOD_NAME = "IoTGAN-aligned-outbound-only-misidentification"
IOTGAN_OUTBOUND_METHOD = METHOD_NAME
DEFAULT_SUBSTITUTE_EPOCHS = 40
DEFAULT_MANIPULATOR_EPOCHS = 30
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_CANDIDATE_FEATURE_COUNTS = (8, 16, 24, 32, 48, 64)
DEFAULT_FIDELITY_TOLERANCE = 0.01
MAX_MULTIPLIER = 0.1
MAX_GAP_SECONDS = 10.0
GENERATOR_PARAMETERIZATION = "unbounded-monotone-residual-physical-projection"
LEGACY_GENERATOR_PARAMETERIZATION = "sigmoid-times-rh-hard-output-bound"
TRACE_CONSTRAINT_VERSION = "first-gap-zero-integer-bytes-v1"

REPRODUCTION_NOTICE = (
    "依据IoTGAN论文Algorithm 1及公式(2)-(4)独立实现；仅保留设备误识别，"
    "并采用论文实验范围之外、本文明确声明的outbound-only约束。每个黑盒"
    "目标必须独立训练策略。论文未披露的特征权重算法、网络和优化参数均记录"
    "在训练元数据中；r*h仅作为生成器输入，不作为生成器输出上限。"
)

__all__ = [
    "IoTGANAlignedPolicy",
    "OutboundOnlyIoTGANPolicy",
    "IOTGAN_OUTBOUND_METHOD",
    "METHOD_NAME",
    "REPRODUCTION_NOTICE",
    "apply_policy",
    "apply_target_specific_policy",
    "fit_baseline",
    "fit_outbound_only_baseline",
    "fit_target_specific_baselines",
    "load_policy",
    "run_self_test",
    "save_policy",
    "summarize_application",
    "trace_feature_matrix",
]


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _paper_sha256():
    return hashlib.sha256(PAPER_PATH.read_bytes()).hexdigest()


def _set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _state_digest(*states):
    digest = hashlib.sha256()
    for state in states:
        for name, value in sorted(state.items()):
            array = value.detach().cpu().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _target_seed(base_seed, target_name):
    digest = hashlib.sha256(str(target_name).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return int((int(base_seed) + offset) % (2**31 - 1))


def trace_feature_matrix(lengths, gaps, directions, mask):
    """Return the project's 64-feature pool used as IoTGAN traffic vector h."""

    lengths, gaps, directions, mask = common.validate_observation_arrays(
        lengths, gaps, directions, mask
    )
    common.validate_dataset(lengths, gaps, directions, mask)
    valid = np.asarray(mask, dtype=bool)
    # 输入表示真实帧；若原始包长已经是小数，既保持禁改包不变又输出整数字节
    # 不可能同时成立。因此在 IoTGAN 边界拒绝它，不静默修整其他方法的数据。
    if np.any(lengths[valid] != np.rint(lengths[valid])):
        raise ValueError("IoTGAN输入包长必须是整数字节")
    if np.any(lengths[valid] > common.MAX_FRAME_LENGTH):
        raise ValueError("IoTGAN输入包长超过1514字节，无法满足非减与帧长上界")
    if np.any(gaps[valid[:, 0], 0] != 0.0):
        raise ValueError("IoTGAN首包gap必须为0；首包发送前等待应单独统计")
    # 必须先提升精度再归一化，避免 float32 的中间舍入破坏字节网格。
    signed_lengths = (
        directions * np.asarray(lengths, dtype=np.float64) / common.MAX_FRAME_LENGTH
    )
    log_gaps = np.log1p(
        np.minimum(np.asarray(gaps, dtype=np.float64), MAX_GAP_SECONDS) * 1000.0
    )
    signed_lengths = np.asarray(signed_lengths, dtype=np.float64).copy()
    log_gaps = np.asarray(log_gaps, dtype=np.float64).copy()
    signed_lengths[~valid] = 0.0
    log_gaps[~valid] = 0.0
    return np.concatenate((signed_lengths, log_gaps), axis=1)


def _validate_functional_mutability_mask(
    functional_mutability_mask, directions, mask
):
    """Build a distinct functionality mask; validity and safety are not conflated.

    An event-level mask enables/disables both its packet-length and gap feature.
    A 64-column mask can control those two feature families independently.
    Regardless of the supplied mask, incoming and inactive events remain immutable.
    """

    directions = np.asarray(directions)
    valid = np.asarray(mask, dtype=bool)
    outbound = valid & (directions == 1)
    base = np.concatenate((outbound, outbound), axis=1)
    # gap[0] 没有流内前驱。必须在采样 r 和训练投影之前冻结，而不是仅在
    # apply_policy 最后清零，否则 G 会学会利用实际观测中不存在的特征。
    base[:, OBSERVED_EVENTS] = False
    if functional_mutability_mask is None:
        return base

    supplied = np.asarray(functional_mutability_mask, dtype=bool)
    if supplied.shape == directions.shape:
        supplied = np.concatenate((supplied, supplied), axis=1)
    elif supplied.shape != base.shape:
        raise ValueError(
            "functional_mutability_mask必须是[流,32]事件掩码或[流,64]特征掩码"
        )
    return base & supplied


def _iterate_batches(sample_count, batch_size, generator):
    order = torch.randperm(int(sample_count), generator=generator)
    for start in range(0, len(order), int(batch_size)):
        yield order[start : start + int(batch_size)]


class SubstituteClassifier(nn.Module):
    """Paper-required multi-layer fully connected sigmoid classifier."""

    def __init__(self, feature_count, class_count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(feature_count), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, int(class_count)),
        )

    def forward(self, features):
        return self.layers(features)

    def probabilities(self, features):
        return torch.sigmoid(self.forward(features))


class ManipulativeModel(nn.Module):
    """Outbound-only G_theta receiving h and r*h without an r*h output cap."""

    def __init__(self, feature_count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * int(feature_count), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, int(feature_count)),
        )

    def forward(
        self,
        normalized_features,
        normalized_noise,
        residual_direction=None,
        lower_bounds=None,
        upper_bounds=None,
    ):
        raw = self.layers(
            torch.cat((normalized_features, normalized_noise), dim=1)
        )
        # 公式 (3) 的 r*h 是 G 的第二输入，不是输出扰动的 10% 硬上限。
        # softplus 给出非负且无固定上限的残差，随后由方向、功能掩码、物理
        # 范围和共用字节投影决定能执行的动作；不要恢复旧 sigmoid*r*h 限制。
        residual_magnitude = torch.nn.functional.softplus(raw)
        if residual_direction is None:
            residual_direction = torch.ones_like(residual_magnitude)
        manipulated = (
            normalized_features + residual_direction * residual_magnitude
        )
        if lower_bounds is not None or upper_bounds is not None:
            if lower_bounds is None or upper_bounds is None:
                raise ValueError(
                    "lower_bounds and upper_bounds must be supplied together"
                )
            manipulated = torch.maximum(
                torch.minimum(manipulated, upper_bounds), lower_bounds
            )
        return manipulated


def _physical_projection_tensors(
    selected_features,
    selected_raw,
    normalized_features,
    feature_mean,
    feature_scale,
    selected_action_mask,
):
    """Return monotone directions and physical bounds for selected features."""

    selected = np.asarray(selected_features, dtype=np.int64)
    raw = torch.as_tensor(selected_raw, dtype=torch.float32)
    normalized = torch.as_tensor(normalized_features, dtype=torch.float32)
    mean = torch.as_tensor(feature_mean, dtype=torch.float32)
    scale = torch.as_tensor(feature_scale, dtype=torch.float32)
    action = torch.as_tensor(selected_action_mask, dtype=torch.bool)
    if raw.shape != normalized.shape or action.shape != normalized.shape:
        raise ValueError("selected raw/normalized/action arrays must have the same shape")
    if normalized.shape[1] != len(selected):
        raise ValueError("selected feature count does not match the feature tensors")

    direction = torch.zeros_like(normalized)
    lower = normalized.clone()
    upper = normalized.clone()
    maximum_log_gap = float(np.log1p(MAX_GAP_SECONDS * 1000.0))
    for column, feature_index in enumerate(selected.tolist()):
        # 第二道防线：直接调用底层投影并传入全 True 掩码也不能开放首包 gap。
        enabled = action[:, column] & (feature_index != OBSERVED_EVENTS)
        if feature_index < OBSERVED_EVENTS:
            # The outbound-only action mask enables only positive signed lengths.
            outgoing = enabled & (raw[:, column] > 0.0)
            direction[outgoing, column] = 1.0
            upper[outgoing, column] = (1.0 - mean[column]) / scale[column]
        else:
            direction[enabled, column] = 1.0
            upper[enabled, column] = (
                maximum_log_gap - mean[column]
            ) / scale[column]
    return direction, lower, upper


def _project_generated_raw(
    manipulated, selected_raw, selected_features, feature_mean, feature_scale,
    selected_action_mask, *, straight_through=False,
):
    """两版共用的执行投影：整数字节、非减、方向与禁改约束一次完成。

    在物理字节域向下取整（不足 1 B 的 padding 不执行），然后夹到原包长和
    1514 B 之间；不能对负的 signed length 直接 floor，否则入站会多加字节。
    训练前向和推理都使用这条规则。训练时只把 floor 的梯度近似为恒等映射
    （STE），真实前向仍是离散字节；物理夹紧和功能掩码的梯度约束保持有效。
    """
    device = manipulated.device
    original = torch.as_tensor(selected_raw, dtype=torch.float64, device=device)
    mean = torch.as_tensor(feature_mean, dtype=torch.float64, device=device)
    scale = torch.as_tensor(feature_scale, dtype=torch.float64, device=device)
    indices = torch.as_tensor(selected_features, dtype=torch.int64, device=device)
    action = torch.as_tensor(selected_action_mask, dtype=torch.bool, device=device)
    action = action & (indices != OBSERVED_EVENTS)
    # 反标准化用 float64，避免可修改范围很小时浮点误差放大到字节域。
    raw = manipulated.to(torch.float64) * scale + mean
    is_length = indices < OBSERVED_EVENTS
    source_bytes = torch.round(original.abs() * common.MAX_FRAME_LENGTH)
    continuous_bytes = torch.maximum(
        (raw.abs() * common.MAX_FRAME_LENGTH).clamp(max=common.MAX_FRAME_LENGTH),
        source_bytes,
    )
    integer_bytes = torch.floor(continuous_bytes)
    if straight_through:
        integer_bytes = continuous_bytes + (integer_bytes - continuous_bytes).detach()
    signed_lengths = original.sign() * integer_bytes / common.MAX_FRAME_LENGTH
    # 大于 10 s 的原始 gap 在特征入口已截断；apply 会保留原始 gap 下界。
    # 此处只投影分类器可见的 log-gap，预处理上界并非功能安全阈值。
    log_gaps = torch.maximum(
        raw.clamp(max=float(np.log1p(MAX_GAP_SECONDS * 1000))), original,
    )
    projected = torch.where(is_length, signed_lengths, log_gaps)
    return torch.where(action, projected, original)


def _project_for_substitute(
    manipulated, selected_raw, selected_features, feature_mean, feature_scale,
    selected_action_mask,
):
    """让训练损失和训练翻转率基于实际可输出的整数包长，而非连续候选。"""
    raw = _project_generated_raw(
        manipulated, selected_raw, selected_features, feature_mean, feature_scale,
        selected_action_mask, straight_through=True,
    )
    mean = torch.as_tensor(feature_mean, dtype=raw.dtype, device=raw.device)
    scale = torch.as_tensor(feature_scale, dtype=raw.dtype, device=raw.device)
    return ((raw - mean) / scale).to(manipulated.dtype)


def _trace_constraint_metadata():
    """声明执行语义，区分旧连续策略与本次修复后的训练产物。"""
    return {
        "version": TRACE_CONSTRAINT_VERSION,
        "first_gap": "immutable_zero_relative_to_first_packet",
        "packet_lengths": "floor_physical_bytes_clamped_to_source_and_1514",
        "training_quantization_gradient": "straight_through_floor",
        "overhead_basis": "emitted_integer_lengths_and_preserved_source_gaps",
        "initial_wait_and_window_buffering_included": False,
        "tls_pcap_execution_verified": False,
    }


def _validate_saved_constraints(metadata, *, allow_legacy_constraints):
    """旧权重可显式用于诊断，但不能静默被当作满足新训练约束的策略。"""
    schema = metadata.get("schema_version")
    training = metadata.get("training", {})
    if schema not in (2, SCHEMA_VERSION):
        raise ValueError("不支持的IoTGAN策略schema；必须重新训练")
    if training.get("generator_parameterization", {}).get("name") != GENERATOR_PARAMETERIZATION:
        raise ValueError("策略文件未声明当前无r*h输出上限的生成器参数化")
    trained_version = training.get("trace_constraints", {}).get("version")
    if schema != SCHEMA_VERSION or trained_version != TRACE_CONSTRAINT_VERSION:
        message = (
            "旧IoTGAN策略未按首包gap冻结和整数字节投影训练，必须重新训练；"
            "仅诊断回放可显式传入allow_legacy_constraints=True，旧实验结果不可沿用"
        )
        if not allow_legacy_constraints:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _standardize_fit(features):
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _print_training_progress(label, completed, total, loss):
    """Render a low-overhead terminal progress bar without adding a dependency."""

    total = int(total)
    completed = int(completed)
    interval = max(1, total // 20)
    if completed != total and completed % interval:
        return
    width = 24
    filled = round(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if completed == total else ""
    print(
        f"\r{label} [{bar}] {completed:>3}/{total} epoch  loss={loss:.5f}",
        end=ending,
        flush=True,
    )


def _train_substitute(
    normalized_features,
    encoded_labels,
    class_count,
    *,
    epochs,
    batch_size,
    learning_rate,
    seed,
    progress_label=None,
):
    _set_seed(seed)
    features = torch.as_tensor(normalized_features, dtype=torch.float32)
    labels = torch.as_tensor(encoded_labels, dtype=torch.int64)
    one_hot = nn.functional.one_hot(labels, num_classes=int(class_count)).to(
        torch.float32
    )
    model = SubstituteClassifier(features.shape[1], class_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = torch.Generator().manual_seed(int(seed) + 17)
    history = []
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for batch in _iterate_batches(len(features), batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, one_hot[batch]
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if progress_label is not None:
            _print_training_progress(
                progress_label, epoch + 1, epochs, history[-1]
            )
    return model, history


def _stratified_selection_split(encoded_labels, seed, validation_fraction=0.2):
    """Deterministic internal holdout used only to select L without test leakage."""

    labels = np.asarray(encoded_labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    training = []
    validation = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label).copy()
        rng.shuffle(rows)
        if len(rows) < 2:
            training.extend(rows.tolist())
            continue
        validation_count = max(1, int(round(len(rows) * validation_fraction)))
        validation_count = min(validation_count, len(rows) - 1)
        validation.extend(rows[:validation_count].tolist())
        training.extend(rows[validation_count:].tolist())
    if not validation:
        all_rows = np.arange(len(labels), dtype=int)
        return all_rows, all_rows, "training-fidelity-fallback"
    return (
        np.asarray(sorted(training), dtype=int),
        np.asarray(sorted(validation), dtype=int),
        "stratified-20-percent-internal-holdout",
    )


def _model_fidelity(model, normalized_features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.as_tensor(normalized_features, dtype=torch.float32)
        )
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return float(np.mean(predictions == np.asarray(labels))), predictions


def _measure_inference_seconds_per_flow(model, normalized_features):
    """Measure c from the paper as median substitute inference wall time."""

    probe = torch.as_tensor(normalized_features, dtype=torch.float32)
    if not len(probe):
        raise ValueError("特征选择计时数据不能为空")
    if len(probe) < 1024:
        repetitions = int(np.ceil(1024 / len(probe)))
        probe = probe.repeat((repetitions, 1))[:1024]
    elif len(probe) > 4096:
        probe = probe[:4096]
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(probe)
        elapsed = []
        for _ in range(9):
            started = time.perf_counter()
            model(probe)
            elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed) / len(probe))


def _performance_gain(current_accuracy, previous_accuracy, current_cost, previous_cost):
    """Implement the Performance Gain expression printed in Section IV-B."""

    if current_accuracy <= 0.0 or current_cost <= 0.0:
        return None
    relative_accuracy = (current_accuracy - previous_accuracy) / current_accuracy
    relative_cost = (current_cost - previous_cost) / current_cost
    if relative_cost <= 1e-12:
        return None
    return float((relative_accuracy - relative_cost) / relative_cost)


def _resolve_candidate_counts(candidate_feature_counts):
    counts = sorted(
        {
            int(value)
            for value in candidate_feature_counts
            if 1 <= int(value) <= FEATURE_COUNT
        }
    )
    if FEATURE_COUNT not in counts:
        counts.append(FEATURE_COUNT)
    if not counts:
        raise ValueError("candidate_feature_counts至少需要一个1到64之间的值")
    return tuple(counts)


def _select_feature_subset(
    raw_features,
    encoded_labels,
    class_count,
    *,
    candidate_feature_counts,
    fidelity_tolerance,
    epochs,
    batch_size,
    learning_rate,
    seed,
    progress_label=None,
):
    """Weight all features, retrain top-L models, and report Performance Gain.

    The paper does not define its weight-training algorithm.  Here importance is
    the mean absolute first-layer weight of a full-pool substitute.  L selection
    first retains candidates within fidelity_tolerance of the best validation
    fidelity, then chooses the candidate with the best non-negative Performance
    Gain; if none has one, it chooses the fastest high-fidelity candidate.
    """

    training_rows, validation_rows, split_description = _stratified_selection_split(
        encoded_labels, int(seed) + 31
    )
    full_mean, full_scale = _standardize_fit(raw_features[training_rows])
    full_training = (raw_features[training_rows] - full_mean) / full_scale
    full_model, full_history = _train_substitute(
        full_training,
        np.asarray(encoded_labels)[training_rows],
        class_count,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=int(seed) + 43,
        progress_label=(
            None if progress_label is None else f"{progress_label} | feature weights"
        ),
    )
    first_linear = full_model.layers[0]
    importance = (
        first_linear.weight.detach().abs().mean(dim=0).cpu().numpy().astype(float)
    )
    ranking = np.argsort(-importance, kind="stable")

    candidates = []
    previous = None
    for position, feature_count in enumerate(
        _resolve_candidate_counts(candidate_feature_counts)
    ):
        selected = ranking[:feature_count]
        mean, scale = _standardize_fit(raw_features[training_rows][:, selected])
        normalized_training = (
            raw_features[training_rows][:, selected] - mean
        ) / scale
        model, history = _train_substitute(
            normalized_training,
            np.asarray(encoded_labels)[training_rows],
            class_count,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=int(seed) + 101 + position,
            progress_label=(
                None
                if progress_label is None
                else f"{progress_label} | candidate L={feature_count}"
            ),
        )
        normalized_validation = (
            raw_features[validation_rows][:, selected] - mean
        ) / scale
        fidelity, _ = _model_fidelity(
            model,
            normalized_validation,
            np.asarray(encoded_labels)[validation_rows],
        )
        cost = _measure_inference_seconds_per_flow(model, normalized_validation)
        gain = None
        if previous is not None:
            gain = _performance_gain(
                fidelity,
                previous["fidelity"],
                cost,
                previous["inference_seconds_per_flow"],
            )
        row = {
            "L": int(feature_count),
            "selected_feature_indices": selected.astype(int).tolist(),
            "fidelity": fidelity,
            "inference_seconds_per_flow": cost,
            "performance_gain": gain,
            "final_training_loss": float(history[-1]),
        }
        candidates.append(row)
        previous = row

    best_fidelity = max(row["fidelity"] for row in candidates)
    high_fidelity = [
        row
        for row in candidates
        if row["fidelity"] >= best_fidelity - float(fidelity_tolerance)
    ]
    non_negative_gain = [
        row
        for row in high_fidelity
        if row["performance_gain"] is not None
        and row["performance_gain"] >= 0.0
    ]
    if non_negative_gain:
        chosen = max(
            non_negative_gain,
            key=lambda row: (
                row["performance_gain"],
                -row["inference_seconds_per_flow"],
                -row["L"],
            ),
        )
        chosen_reason = "best-non-negative-performance-gain-within-fidelity-band"
    else:
        chosen = min(
            high_fidelity,
            key=lambda row: (row["inference_seconds_per_flow"], row["L"]),
        )
        chosen_reason = "fastest-candidate-within-fidelity-band"

    return np.asarray(chosen["selected_feature_indices"], dtype=int), {
        "feature_pool_size": FEATURE_COUNT,
        "weight_training_algorithm": "mean-absolute-first-layer-weight",
        "weight_training_full_model_final_loss": float(full_history[-1]),
        "feature_importance": importance.tolist(),
        "feature_ranking": ranking.astype(int).tolist(),
        "candidate_results": candidates,
        "internal_split": split_description,
        "fidelity_tolerance": float(fidelity_tolerance),
        "selected_L": int(chosen["L"]),
        "selection_reason": chosen_reason,
        "paper_disclosure_note": (
            "论文未披露权重训练算法、候选L和精确选择规则；这里的选择已显式记录。"
        ),
    }


class IoTGANAlignedPolicy:
    """One outbound-only misidentification policy bound to one black-box target."""

    method_name = METHOD_NAME
    reproduction_notice = REPRODUCTION_NOTICE

    def __init__(
        self,
        generator_state,
        substitute_state,
        selected_features,
        feature_mean,
        feature_scale,
        class_labels,
        black_box_target,
        *,
        seed=SEED,
        training_metadata=None,
    ):
        self.generator_state = {
            name: value.detach().cpu().clone()
            for name, value in generator_state.items()
        }
        self.substitute_state = {
            name: value.detach().cpu().clone()
            for name, value in substitute_state.items()
        }
        self.selected_features = np.asarray(selected_features, dtype=np.int64).copy()
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64).copy()
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64).copy()
        if self.selected_features.ndim != 1 or not len(self.selected_features):
            raise ValueError("selected_features必须是非空一维数组")
        if len(np.unique(self.selected_features)) != len(self.selected_features):
            raise ValueError("selected_features不能重复")
        if self.selected_features.min() < 0 or self.selected_features.max() >= FEATURE_COUNT:
            raise ValueError("selected_features超出64维特征池")
        expected = (len(self.selected_features),)
        if self.feature_mean.shape != expected or self.feature_scale.shape != expected:
            raise ValueError("feature_mean/scale维数必须与selected_features一致")
        if np.any(self.feature_scale <= 0.0):
            raise ValueError("feature_scale必须为正")
        self.class_labels = tuple(_python_scalar(value) for value in class_labels)
        if len(self.class_labels) < 2:
            raise ValueError("替代模型至少需要两个类别")
        if not str(black_box_target).strip():
            raise ValueError("black_box_target不能为空")
        self.black_box_target = str(black_box_target)
        self.seed = int(seed)
        self.training_metadata = dict(training_metadata or {})
        self._generator = None
        self._substitute = None

    @classmethod
    def fit(
        cls,
        lengths,
        gaps,
        directions,
        mask,
        black_box_labels,
        train_indices,
        *,
        black_box_target,
        groups=None,
        functional_mutability_mask=None,
        seed=SEED,
        substitute_epochs=DEFAULT_SUBSTITUTE_EPOCHS,
        manipulator_epochs=DEFAULT_MANIPULATOR_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate=DEFAULT_LEARNING_RATE,
        candidate_feature_counts=DEFAULT_CANDIDATE_FEATURE_COUNTS,
        fidelity_tolerance=DEFAULT_FIDELITY_TOLERANCE,
        progress=False,
        progress_label=None,
    ):
        if min(substitute_epochs, manipulator_epochs, batch_size) < 1:
            raise ValueError("epoch与batch size必须为正整数")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate必须为正")
        if fidelity_tolerance < 0.0:
            raise ValueError("fidelity_tolerance不能为负")
        if not str(black_box_target).strip():
            raise ValueError("必须提供本策略绑定的black_box_target名称")
        progress_label = (
            str(progress_label)
            if progress_label is not None
            else f"[IoTGAN][{black_box_target}]"
        )
        progress = bool(progress)

        selected_arrays, indices, training_groups = common.training_view(
            lengths,
            gaps,
            directions,
            mask,
            train_indices,
            groups=groups,
        )
        black_box_labels = np.asarray(black_box_labels)
        if black_box_labels.ndim != 1 or len(black_box_labels) != len(
            np.asarray(lengths)
        ):
            raise ValueError("black_box_labels必须与全部输入流一一对应")
        selected_black_box_labels = black_box_labels[indices]
        class_labels = np.unique(selected_black_box_labels)
        if len(class_labels) < 2:
            raise ValueError("训练数据中的黑盒输出至少需要两个类别")
        label_to_index = {
            _python_scalar(label): position
            for position, label in enumerate(class_labels)
        }
        encoded = np.asarray(
            [label_to_index[_python_scalar(value)] for value in selected_black_box_labels],
            dtype=np.int64,
        )
        raw_features = trace_feature_matrix(*selected_arrays)

        full_functional_mask = _validate_functional_mutability_mask(
            functional_mutability_mask,
            np.asarray(directions),
            np.asarray(mask),
        )
        training_action_mask = full_functional_mask[indices]

        chosen_features, selection_metadata = _select_feature_subset(
            raw_features,
            encoded,
            len(class_labels),
            candidate_feature_counts=candidate_feature_counts,
            fidelity_tolerance=fidelity_tolerance,
            epochs=substitute_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=int(seed) + 1000,
            progress_label=progress_label if progress else None,
        )
        chosen_raw = raw_features[:, chosen_features]
        feature_mean, feature_scale = _standardize_fit(chosen_raw)
        normalized = (chosen_raw - feature_mean) / feature_scale

        substitute, substitute_history = _train_substitute(
            normalized,
            encoded,
            len(class_labels),
            epochs=substitute_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=int(seed) + 2000,
            progress_label=(
                None if not progress else f"{progress_label} | substitute"
            ),
        )
        substitute_fidelity, substitute_predictions_np = _model_fidelity(
            substitute, normalized, encoded
        )

        tensor_raw = torch.as_tensor(chosen_raw, dtype=torch.float32)
        tensor_normalized = torch.as_tensor(normalized, dtype=torch.float32)
        tensor_scale = torch.as_tensor(feature_scale, dtype=torch.float32)
        tensor_action_mask = torch.as_tensor(
            training_action_mask[:, chosen_features], dtype=torch.float32
        )
        physical_projection = _physical_projection_tensors(
            chosen_features,
            chosen_raw,
            normalized,
            feature_mean,
            feature_scale,
            training_action_mask[:, chosen_features],
        )
        substitute_predictions = torch.as_tensor(
            substitute_predictions_np, dtype=torch.int64
        )
        clean_prediction_one_hot = nn.functional.one_hot(
            substitute_predictions, num_classes=len(class_labels)
        ).to(torch.float32)

        _set_seed(int(seed) + 3000)
        generator = ManipulativeModel(len(chosen_features))
        generator_optimizer = torch.optim.Adam(
            generator.parameters(), lr=float(learning_rate)
        )
        generator_rng = torch.Generator().manual_seed(int(seed) + 3001)
        noise_rng = torch.Generator().manual_seed(int(seed) + 3002)
        substitute.eval()
        for parameter in substitute.parameters():
            parameter.requires_grad_(False)

        generator_history = []
        for epoch in range(int(manipulator_epochs)):
            generator.train()
            epoch_losses = []
            for batch in _iterate_batches(len(indices), batch_size, generator_rng):
                r = (
                    torch.rand(
                        (len(batch), len(chosen_features)),
                        generator=noise_rng,
                        dtype=torch.float32,
                    )
                    * MAX_MULTIPLIER
                    * tensor_action_mask[batch]
                )
                noise_normalized = (r * tensor_raw[batch]) / tensor_scale
                generator_optimizer.zero_grad(set_to_none=True)
                manipulated = generator(
                    tensor_normalized[batch],
                    noise_normalized,
                    physical_projection[0][batch],
                    physical_projection[1][batch],
                    physical_projection[2][batch],
                )
                manipulated = _project_for_substitute(
                    manipulated, chosen_raw[batch.numpy()], chosen_features,
                    feature_mean, feature_scale, tensor_action_mask[batch],
                )
                manipulated_logits = substitute(manipulated)
                # 公式 (4) 对应增大干净决策与扰动后决策间的经验损失。
                # 这里保留已声明的负 BCE 选型；损失增大不保证 argmax 翻转，
                # 所以下面另统计整数投影后的替代模型翻转率。
                loss = -nn.functional.binary_cross_entropy_with_logits(
                    manipulated_logits, clean_prediction_one_hot[batch]
                )
                loss.backward()
                generator_optimizer.step()
                epoch_losses.append(float(loss.detach()))
            generator_history.append(float(np.mean(epoch_losses)))
            if progress:
                _print_training_progress(
                    f"{progress_label} | generator",
                    epoch + 1,
                    manipulator_epochs,
                    generator_history[-1],
                )

        generator.eval()
        with torch.no_grad():
            evaluation_rng = torch.Generator().manual_seed(int(seed) + 4000)
            r = (
                torch.rand(
                    (len(indices), len(chosen_features)),
                    generator=evaluation_rng,
                    dtype=torch.float32,
                )
                * MAX_MULTIPLIER
                * tensor_action_mask
            )
            manipulated = generator(
                tensor_normalized,
                (r * tensor_raw) / tensor_scale,
                *physical_projection,
            )
            manipulated = _project_for_substitute(
                manipulated, chosen_raw, chosen_features, feature_mean,
                feature_scale, tensor_action_mask,
            )
            manipulated_predictions = torch.argmax(substitute(manipulated), dim=1)
        training_misidentification_rate = float(
            torch.mean(
                (manipulated_predictions != substitute_predictions).to(torch.float32)
            )
        )

        metadata = {
            "trace_constraints": _trace_constraint_metadata(),
            "implementation_origin": "paper-guided-independent-aligned-reimplementation",
            "paper": {
                "local_path": str(PAPER_PATH.resolve()),
                "sha256": _paper_sha256(),
                "algorithm": "Algorithm 1",
                "objective": "Equation (4) device misidentification only",
            },
            "declared_scope": {
                "outbound_only": True,
                "device_misidentification": True,
                "identity_spoofing": False,
                "feature_decreases": False,
                "dummy_packets": False,
            },
            "black_box_target": str(black_box_target),
            "target_specific_policy_required": True,
            "black_box_outputs_used_instead_of_ground_truth": True,
            "training_flows": int(len(indices)),
            "training_groups": training_groups,
            "class_counts_from_black_box": {
                str(_python_scalar(label)): int(
                    np.sum(selected_black_box_labels == label)
                )
                for label in class_labels
            },
            "feature_adapter": {
                "feature_pool_count": FEATURE_COUNT,
                "features": [
                    "32 signed Ethernet-frame lengths / 1514",
                    "32 log1p(inter-arrival-gap-ms) values",
                ],
                "selected_feature_indices": chosen_features.astype(int).tolist(),
            },
            "feature_selection": selection_metadata,
            "functional_safety": {
                "validity_mask_separate_from_functional_mutability_mask": True,
                "incoming_features_immutable": True,
                "only_non_negative_outgoing_padding_and_delay": True,
                "mutable_training_feature_fraction": float(
                    np.mean(training_action_mask[:, chosen_features])
                ),
            },
            "generator_parameterization": {
                "name": GENERATOR_PARAMETERIZATION,
                "rh_role": "second_generator_input_only",
                "maximum_r": float(MAX_MULTIPLIER),
                "sigmoid_times_rh_output_bound": False,
                "output_residual_cap_standard_deviations": None,
                "physical_projection": (
                    "outbound packet lengths and gaps can only increase within "
                    "their valid physical ranges; disabled features stay unchanged"
                ),
            },
            "paper_unspecified_engineering_choices": {
                "substitute_hidden_layers": [128, 64],
                "manipulator_hidden_layers": [128, 128],
                "hidden_activation": "ReLU",
                "optimizer": "Adam",
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "substitute_epochs": int(substitute_epochs),
                "manipulator_epochs": int(manipulator_epochs),
                "loss": "binary cross entropy consistent with sigmoid output",
                "feature_normalization": "training mean/std",
            },
            "training_history": {
                "substitute_binary_cross_entropy": substitute_history,
                "manipulator_negative_binary_cross_entropy": generator_history,
                "substitute_fidelity_to_black_box": substitute_fidelity,
                "training_misidentification_rate_against_substitute": (
                    training_misidentification_rate
                ),
            },
        }
        return cls(
            deepcopy(generator.state_dict()),
            deepcopy(substitute.state_dict()),
            chosen_features,
            feature_mean,
            feature_scale,
            class_labels,
            black_box_target,
            seed=seed,
            training_metadata=metadata,
        )

    @property
    def training_constraints_match_runtime(self):
        # 保存诊断加载的旧权重时也保留旧训练记录，不能靠重新保存伪装成已重训。
        return self.training_metadata.get("trace_constraints", {}).get("version") == TRACE_CONSTRAINT_VERSION

    @property
    def weights_sha256(self):
        return _state_digest(self.generator_state, self.substitute_state)

    def _generator_model(self):
        if self._generator is None:
            model = ManipulativeModel(len(self.selected_features))
            model.load_state_dict(self.generator_state)
            model.eval()
            self._generator = model
        return self._generator

    def _substitute_model(self):
        if self._substitute is None:
            model = SubstituteClassifier(
                len(self.selected_features), len(self.class_labels)
            )
            model.load_state_dict(self.substitute_state)
            model.eval()
            self._substitute = model
        return self._substitute

    def assert_target(self, black_box_target):
        if str(black_box_target) != self.black_box_target:
            raise ValueError(
                "IoTGAN策略与目标模型不匹配："
                f"策略绑定{self.black_box_target!r}，请求的是{str(black_box_target)!r}。"
                "每个黑盒目标必须独立训练IoTGAN。"
            )

    def transform_features(
        self,
        raw_features,
        directions,
        mask,
        seed,
        *,
        functional_mutability_mask=None,
    ):
        raw_features = np.asarray(raw_features, dtype=np.float64)
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)
        if raw_features.shape[1] != FEATURE_COUNT:
            raise ValueError("raw_features必须为64维")
        directions = np.asarray(directions)
        mask = np.asarray(mask)
        if directions.shape != mask.shape or directions.shape != (
            len(raw_features),
            OBSERVED_EVENTS,
        ):
            raise ValueError("directions/mask形状与流数量不一致")
        full_action_mask = _validate_functional_mutability_mask(
            functional_mutability_mask, directions, mask
        )
        selected_action_mask = full_action_mask[:, self.selected_features]
        rng = np.random.default_rng(int(seed))
        r_full = (
            rng.uniform(0.0, MAX_MULTIPLIER, size=raw_features.shape)
            * full_action_mask
        )
        selected_raw = raw_features[:, self.selected_features]
        selected_r = r_full[:, self.selected_features]
        normalized = (selected_raw - self.feature_mean) / self.feature_scale
        noise_normalized = (selected_r * selected_raw) / self.feature_scale
        physical_projection = _physical_projection_tensors(
            self.selected_features,
            selected_raw,
            normalized,
            self.feature_mean,
            self.feature_scale,
            selected_action_mask,
        )
        with torch.no_grad():
            manipulated = self._generator_model()(
                torch.as_tensor(normalized, dtype=torch.float32),
                torch.as_tensor(noise_normalized, dtype=torch.float32),
                *physical_projection,
            )
        # 与训练共用执行投影，返回可由整数字节重建的特征；禁改维度保留原值。
        manipulated_raw = _project_generated_raw(
            manipulated, selected_raw, self.selected_features,
            self.feature_mean, self.feature_scale, selected_action_mask,
        ).cpu().numpy()
        result = raw_features.copy()
        result[:, self.selected_features] = manipulated_raw
        return result, r_full, full_action_mask

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "runtime_trace_constraints": _trace_constraint_metadata(),
            "training_constraints_match_runtime": self.training_constraints_match_runtime,
            "method": self.method_name,
            "reproduction_status": self.reproduction_notice,
            "paper": {
                "title": (
                    "IoTGAN: GAN Powered Camouflage Against Machine Learning "
                    "Based IoT Device Identification"
                ),
                "local_path": str(PAPER_PATH.resolve()),
                "sha256": _paper_sha256(),
            },
            "scope": {
                "outbound_only": True,
                "objective": "device_misidentification_equation_4",
                "identity_spoofing_supported": False,
                "feature_decreases_supported": False,
                "dummy_packets_supported": False,
            },
            "black_box_target": self.black_box_target,
            "target_specific_policy_required": True,
            "class_labels": list(self.class_labels),
            "feature_pool_count": FEATURE_COUNT,
            "selected_features": self.selected_features.astype(int).tolist(),
            "generator_parameterization": self.training_metadata.get(
                "generator_parameterization", {}
            ),
            "seed": self.seed,
            "weights_sha256": self.weights_sha256,
            "training": self.training_metadata,
        }


# Compatibility name for active comparison scripts.  It refers only to the
# aligned outbound-only implementation; no feature-space policy is exposed.
OutboundOnlyIoTGANPolicy = IoTGANAlignedPolicy


def fit_baseline(
    lengths,
    gaps,
    directions,
    mask,
    black_box_labels,
    train_indices,
    *,
    black_box_target,
    **kwargs,
):
    """Fit one policy for exactly one named black-box target."""

    return IoTGANAlignedPolicy.fit(
        lengths,
        gaps,
        directions,
        mask,
        black_box_labels,
        train_indices,
        black_box_target=black_box_target,
        **kwargs,
    )


def fit_outbound_only_baseline(*args, **kwargs):
    return fit_baseline(*args, **kwargs)


def fit_target_specific_baselines(
    lengths,
    gaps,
    directions,
    mask,
    black_box_outputs_by_target,
    train_indices,
    *,
    seed=SEED,
    **kwargs,
):
    """Train a distinct IoTGAN policy for every black-box target.

    black_box_outputs_by_target must map target model names to one predicted
    class per input flow.  Returning a mapping makes the target-policy relation
    explicit and prevents a single RF-trained policy from silently standing in
    for SVM, neural-network, or other targets.
    """

    if not hasattr(black_box_outputs_by_target, "items"):
        raise TypeError("black_box_outputs_by_target必须是目标名到预测数组的映射")
    items = list(black_box_outputs_by_target.items())
    if not items:
        raise ValueError("至少需要一个黑盒目标")
    progress = bool(kwargs.pop("progress", False))
    policies = {}
    for target_position, (target_name, black_box_outputs) in enumerate(items, start=1):
        name = str(target_name)
        if not name.strip():
            raise ValueError("黑盒目标名称不能为空")
        if name in policies:
            raise ValueError(f"黑盒目标名称重复: {name!r}")
        progress_label = f"[IoTGAN][target {target_position}/{len(items)}: {name}]"
        if progress:
            print(f"{progress_label} 开始训练", flush=True)
        policies[name] = fit_baseline(
            lengths,
            gaps,
            directions,
            mask,
            black_box_outputs,
            train_indices,
            black_box_target=name,
            seed=_target_seed(seed, name),
            progress=progress,
            progress_label=progress_label,
            **kwargs,
        )
        if progress:
            print(f"{progress_label} 完成", flush=True)
    return policies


def _decode_features(features, source_directions, source_mask):
    features = np.asarray(features, dtype=np.float64)
    signed_lengths = features[:OBSERVED_EVENTS]
    log_gaps = features[OBSERVED_EVENTS:]
    valid = np.asarray(source_mask, dtype=bool)
    lengths = np.clip(
        np.abs(signed_lengths) * common.MAX_FRAME_LENGTH,
        0.0,
        common.MAX_FRAME_LENGTH,
    )
    # transform_features 已向下投影到整数网格。这里 round 只消除除以/乘以
    # 1514 的逆变换误差，不能再 floor，否则可能把合法的 N B 误减为 N-1 B。
    lengths = np.rint(lengths)
    maximum_log_gap = np.log1p(MAX_GAP_SECONDS * 1000.0)
    gaps = np.expm1(np.clip(log_gaps, 0.0, maximum_log_gap)) / 1000.0
    directions = np.asarray(source_directions).copy()
    lengths[~valid] = 0.0
    gaps[~valid] = 0.0
    directions[~valid] = 0
    return lengths, gaps, directions


def apply_policy(
    policy,
    lengths,
    gaps,
    directions,
    mask,
    labels=None,
    *,
    black_box_target,
    functional_mutability_mask=None,
    base_seed=SEED,
):
    """Apply an aligned policy after enforcing target and action compatibility."""

    if not isinstance(policy, IoTGANAlignedPolicy):
        raise TypeError("policy必须是IoTGANAlignedPolicy")
    policy.assert_target(black_box_target)
    lengths, gaps, directions, mask = common.validate_observation_arrays(
        lengths, gaps, directions, mask
    )
    common.validate_dataset(lengths, gaps, directions, mask)
    sample_count = len(lengths)
    labels_array = None if labels is None else np.asarray(labels)
    if labels_array is not None and len(labels_array) != sample_count:
        raise ValueError("labels数量与输入流不一致")

    full_functional_mask = _validate_functional_mutability_mask(
        functional_mutability_mask, directions, mask
    )
    raw_features = trace_feature_matrix(lengths, gaps, directions, mask)
    out_lengths = np.zeros_like(lengths, dtype=np.float64)
    out_gaps = np.zeros_like(gaps, dtype=np.float64)
    out_directions = np.asarray(directions).copy()
    out_mask = np.asarray(mask).copy()
    stats = common.empty_stats(sample_count)
    audit_rows = []
    seed_rng = np.random.default_rng(int(base_seed))

    for sample in range(sample_count):
        session_seed = int(seed_rng.integers(0, np.iinfo(np.int64).max))
        sample_functional_mask = full_functional_mask[sample : sample + 1]
        manipulated, multiplier, action_mask = policy.transform_features(
            raw_features[sample],
            directions[sample : sample + 1],
            mask[sample : sample + 1],
            session_seed,
            functional_mutability_mask=sample_functional_mask,
        )
        decoded_lengths, decoded_gaps, decoded_directions = _decode_features(
            manipulated[0], directions[sample], mask[sample]
        )
        valid = np.asarray(mask[sample], dtype=bool)
        outgoing = valid & (np.asarray(directions[sample]) == 1)
        incoming = valid & (np.asarray(directions[sample]) == -1)
        length_allowed = action_mask[0, :OBSERVED_EVENTS]
        gap_allowed = action_mask[0, OBSERVED_EVENTS:]

        # 字节已经在 transform_features 中量化；此处保留原始物理事件的
        # 非减下界（尤其是预处理截断前超过 10 s 的 gap），再恢复禁改字段。
        # 开销必须在这些执行修正之后由最终输出减原始输入计算。
        decoded_lengths[outgoing] = np.maximum(
            decoded_lengths[outgoing], np.asarray(lengths[sample])[outgoing]
        )
        decoded_gaps[outgoing] = np.maximum(
            decoded_gaps[outgoing], np.asarray(gaps[sample])[outgoing]
        )
        decoded_lengths[incoming] = np.asarray(lengths[sample])[incoming]
        decoded_gaps[incoming] = np.asarray(gaps[sample])[incoming]
        decoded_lengths[valid & ~length_allowed] = np.asarray(lengths[sample])[
            valid & ~length_allowed
        ]
        decoded_gaps[valid & ~gap_allowed] = np.asarray(gaps[sample])[
            valid & ~gap_allowed
        ]

        out_lengths[sample] = decoded_lengths
        out_gaps[sample] = decoded_gaps
        out_directions[sample] = decoded_directions
        length_delta = decoded_lengths - np.asarray(lengths[sample], dtype=np.float64)
        gap_delta = decoded_gaps - np.asarray(gaps[sample], dtype=np.float64)
        changed = valid & (
            (np.abs(length_delta) > 1e-7) | (np.abs(gap_delta) > 1e-10)
        )

        if not np.array_equal(decoded_directions[valid], directions[sample][valid]):
            raise AssertionError("aligned IoTGAN改变了真实事件方向")
        if np.any(np.abs(length_delta[incoming]) > 1e-9) or np.any(
            np.abs(gap_delta[incoming]) > 1e-12
        ):
            raise AssertionError("aligned IoTGAN改变了incoming事件")
        if np.any(length_delta[outgoing] < -1e-9) or np.any(
            gap_delta[outgoing] < -1e-12
        ):
            raise AssertionError("aligned IoTGAN产生了特征减少")
        if np.any(np.abs(length_delta[valid & ~length_allowed]) > 1e-9):
            raise AssertionError("aligned IoTGAN修改了功能掩码禁止的包长")
        if np.any(np.abs(gap_delta[valid & ~gap_allowed]) > 1e-12):
            raise AssertionError("aligned IoTGAN修改了功能掩码禁止的间隔")

        stats["real_added_bytes"][sample] = np.sum(length_delta[valid])
        stats["real_added_delay"][sample] = np.sum(gap_delta[valid])
        store_timing(stats,sample,timing_metrics(np.asarray(gaps[sample])[valid],
            np.cumsum(decoded_gaps[valid]),np.ones(np.count_nonzero(valid),dtype=bool)),sample_count)
        stats["outbound_real_added_bytes"][sample] = np.sum(
            length_delta[outgoing]
        )
        stats["outbound_real_added_delay"][sample] = np.sum(gap_delta[outgoing])
        # All inbound and dummy statistics intentionally remain zero.

        audit_rows.append(
            {
                "method": METHOD_NAME,
                "black_box_target": policy.black_box_target,
                "policy_target_verified": True,
                "source_label": (
                    _python_scalar(labels_array[sample])
                    if labels_array is not None
                    else "not-provided"
                ),
                "session_seed": session_seed,
                "runtime_uses_source_label": False,
                "outbound_only": True,
                "feature_decreases": 0,
                "dummy_packets": 0,
                "actions_taken": int(np.sum(changed)),
                "padding_actions": int(np.sum(valid & (length_delta > 1e-7))),
                "delay_actions": int(np.sum(valid & (gap_delta > 1e-10))),
                "real_added_bytes": float(np.sum(length_delta[valid])),
                "real_added_delay": float(np.sum(gap_delta[valid])),
                "maximum_r": float(np.max(multiplier)),
                "functionally_disabled_length_features_unchanged": True,
                "functionally_disabled_gap_features_unchanged": True,
            }
        )

    metadata = {
        "method": METHOD_NAME,
        "reproduction_status": REPRODUCTION_NOTICE,
        "objective": "device_misidentification_equation_4",
        "black_box_target": policy.black_box_target,
        "policy_target_verified": True,
        "runtime_uses_source_label": False,
        "outbound_only": True,
        "feature_decreases_supported": False,
        "identity_spoofing_supported": False,
        "dummy_packets_supported": False,
        "functional_mutability_mask_enforced": True,
        "trace_constraints": _trace_constraint_metadata(),
        "training_constraints_match_runtime": policy.training_constraints_match_runtime,
    }
    return (
        out_lengths,
        out_gaps,
        out_directions,
        out_mask,
        stats,
        metadata,
        audit_rows,
    )


def apply_target_specific_policy(
    policies,
    target_name,
    lengths,
    gaps,
    directions,
    mask,
    labels=None,
    **kwargs,
):
    """Select and apply only the policy trained for target_name."""

    name = str(target_name)
    if name not in policies:
        raise KeyError(
            f"没有为黑盒目标{name!r}训练IoTGAN；禁止复用其他目标的策略"
        )
    return apply_policy(
        policies[name],
        lengths,
        gaps,
        directions,
        mask,
        labels,
        black_box_target=name,
        **kwargs,
    )


def summarize_application(application):
    rows = application[6]
    return {
        "overhead": common.summarize_overhead(application[4]),
        "flows": len(rows),
        "black_box_target": rows[0]["black_box_target"] if rows else None,
        "policy_target_verified": bool(
            all(row["policy_target_verified"] for row in rows)
        ),
        "runtime_uses_source_label": False,
        "outbound_only": True,
        "feature_decreases": 0,
        "identity_spoofing": False,
    }


def save_policy(path, policy):
    if not isinstance(policy, IoTGANAlignedPolicy):
        raise TypeError("只能保存IoTGANAlignedPolicy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(policy.to_dict(), ensure_ascii=False), dtype=np.str_
        ),
        "selected_features": policy.selected_features,
        "feature_mean": policy.feature_mean,
        "feature_scale": policy.feature_scale,
    }
    for prefix, state in (
        ("generator", policy.generator_state),
        ("substitute", policy.substitute_state),
    ):
        for name, value in state.items():
            arrays[f"{prefix}__{name.replace('.', '__dot__')}"] = (
                value.detach().cpu().numpy()
            )
    np.savez_compressed(path, **arrays)


def load_policy(path, *, allow_legacy_constraints=False):
    """默认只接受新约束训练的策略；旧策略诊断必须显式选择并产生警告。"""
    with np.load(path, allow_pickle=False) as content:
        metadata = json.loads(str(content["metadata_json"]))
        if metadata.get("method") != METHOD_NAME:
            raise ValueError("策略文件不是aligned IoTGAN")
        _validate_saved_constraints(
            metadata, allow_legacy_constraints=allow_legacy_constraints,
        )
        states = {"generator": {}, "substitute": {}}
        for key in content.files:
            for prefix in states:
                marker = f"{prefix}__"
                if key.startswith(marker):
                    name = key[len(marker) :].replace("__dot__", ".")
                    states[prefix][name] = torch.from_numpy(content[key].copy())
        policy = IoTGANAlignedPolicy(
            states["generator"],
            states["substitute"],
            content["selected_features"].copy(),
            content["feature_mean"].copy(),
            content["feature_scale"].copy(),
            metadata["class_labels"],
            metadata["black_box_target"],
            seed=metadata["seed"],
            training_metadata=metadata.get("training", {}),
        )
    if policy.weights_sha256 != metadata["weights_sha256"]:
        raise ValueError("aligned IoTGAN权重摘要不匹配")
    return policy


def _synthetic_dataset():
    sample_count = 18
    lengths = np.zeros((sample_count, OBSERVED_EVENTS), dtype=np.float64)
    gaps = np.zeros_like(lengths)
    directions = np.zeros((sample_count, OBSERVED_EVENTS), dtype=np.int8)
    mask = np.zeros_like(directions, dtype=np.uint8)
    labels = np.repeat(np.asarray([0, 1, 2]), 6)
    for sample in range(sample_count):
        count = 10 + (sample % 4)
        mask[sample, :count] = 1
        directions[sample, :count] = np.where(np.arange(count) % 3 == 1, -1, 1)
        lengths[sample, :count] = (
            100 + 70 * labels[sample] + 3 * np.arange(count) + sample
        )
        gaps[sample, :count] = 0.008 + 0.006 * labels[sample]
        gaps[sample, 0] = 0.0
    groups = np.asarray([f"capture-{sample}" for sample in range(sample_count)])
    return lengths, gaps, directions, mask, labels, groups


def run_self_test():
    # A zero-weight generator emits softplus(0), which must be able to exceed a
    # tiny r*h input.  This directly guards against reintroducing the removed
    # sigmoid-times-r*h hard output bound.
    probe_generator = ManipulativeModel(1)
    with torch.no_grad():
        for parameter in probe_generator.parameters():
            parameter.zero_()
        probe_feature = torch.zeros((1, 1), dtype=torch.float32)
        probe_rh = torch.full((1, 1), 1e-4, dtype=torch.float32)
        probe_output = probe_generator(probe_feature, probe_rh)
    if not float(probe_output[0, 0] - probe_feature[0, 0]) > float(
        torch.abs(probe_rh[0, 0])
    ):
        raise AssertionError("单向IoTGAN仍把r*h错误地用作生成器输出上限")

    lengths, gaps, directions, mask, labels, groups = _synthetic_dataset()
    train_indices = np.asarray(
        list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)),
        dtype=int,
    )
    test_indices = np.asarray([5, 11, 17], dtype=int)
    # Event 2 is outgoing in this synthetic trace and is explicitly immutable.
    functional_mask = np.ones_like(mask, dtype=bool)
    functional_mask[:, 2] = False
    black_box_outputs = {
        "rf-oracle": labels.copy(),
        "svm-oracle": (labels + 1) % 3,
    }
    policies = fit_target_specific_baselines(
        lengths,
        gaps,
        directions,
        mask,
        black_box_outputs,
        train_indices,
        groups=groups,
        functional_mutability_mask=functional_mask,
        seed=1234,
        substitute_epochs=3,
        manipulator_epochs=3,
        batch_size=8,
        candidate_feature_counts=(8, 32, 64),
    )
    if set(policies) != set(black_box_outputs):
        raise AssertionError("没有为每个目标生成独立策略")
    if policies["rf-oracle"].black_box_target != "rf-oracle":
        raise AssertionError("策略没有绑定正确目标")
    for policy in policies.values():
        parameterization = policy.training_metadata.get(
            "generator_parameterization", {}
        )
        if parameterization.get("name") != GENERATOR_PARAMETERIZATION:
            raise AssertionError("策略没有记录当前生成器参数化")
        if parameterization.get("sigmoid_times_rh_output_bound") is not False:
            raise AssertionError("策略元数据仍声明sigmoid×r*h输出上限")

    test_functional_mask = functional_mask[test_indices]
    application = apply_target_specific_policy(
        policies,
        "rf-oracle",
        lengths[test_indices],
        gaps[test_indices],
        directions[test_indices],
        mask[test_indices],
        labels[test_indices],
        functional_mutability_mask=test_functional_mask,
        base_seed=7000,
    )
    valid = mask[test_indices].astype(bool)
    incoming = valid & (directions[test_indices] == -1)
    outgoing = valid & (directions[test_indices] == 1)
    if not np.array_equal(application[0][incoming], lengths[test_indices][incoming]):
        raise AssertionError("自测试发现incoming包长被修改")
    if not np.array_equal(application[1][incoming], gaps[test_indices][incoming]):
        raise AssertionError("自测试发现incoming间隔被修改")
    if np.any(application[0][outgoing] < lengths[test_indices][outgoing]):
        raise AssertionError("自测试发现包长减少")
    if np.any(application[1][outgoing] < gaps[test_indices][outgoing]):
        raise AssertionError("自测试发现间隔减少")
    if not np.array_equal(application[0][:, 2], lengths[test_indices, 2]):
        raise AssertionError("功能安全掩码禁止的包长发生变化")
    if not np.array_equal(application[1][:, 2], gaps[test_indices, 2]):
        raise AssertionError("功能安全掩码禁止的间隔发生变化")

    mismatch_rejected = False
    try:
        apply_policy(
            policies["rf-oracle"],
            lengths[test_indices],
            gaps[test_indices],
            directions[test_indices],
            mask[test_indices],
            black_box_target="svm-oracle",
        )
    except ValueError:
        mismatch_rejected = True
    if not mismatch_rejected:
        raise AssertionError("没有拒绝目标不匹配的策略")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "iotgan_aligned_rf_policy.npz"
        save_policy(path, policies["rf-oracle"])
        loaded = load_policy(path)
        replay = apply_policy(
            loaded,
            lengths[test_indices],
            gaps[test_indices],
            directions[test_indices],
            mask[test_indices],
            labels[test_indices],
            black_box_target="rf-oracle",
            functional_mutability_mask=test_functional_mask,
            base_seed=7000,
        )
        for left, right in zip(application[:4], replay[:4], strict=True):
            if not np.array_equal(left, right):
                raise AssertionError("策略序列化往返结果不一致")

    return {
        "all_passed": True,
        "standalone_from_legacy_iotgan": True,
        "scope": "outbound-only device-misidentification",
        "targets": sorted(policies),
        "separate_policy_per_target": True,
        "cross_target_policy_rejected": mismatch_rejected,
        "feature_selection": {
            target: policy.training_metadata["feature_selection"]["selected_L"]
            for target, policy in policies.items()
        },
        "functional_mutability_mask_enforced": True,
        "generator_parameterization": GENERATOR_PARAMETERIZATION,
        "sigmoid_times_rh_output_bound_removed": True,
        "feature_decreases": 0,
        "serialization_roundtrip": True,
        "application_summary": summarize_application(application),
    }


def main():
    parser = argparse.ArgumentParser(
        description="独立aligned IoTGAN：outbound-only设备误识别"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("当前独立入口仅支持--self-test")
    print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
