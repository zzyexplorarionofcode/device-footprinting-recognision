"""Bidirectional IoTGAN variant for this project's TLS-flow protocol.

This module keeps the aligned implementation's device-misidentification,
target-specific substitute/manipulator training, top-L feature selection, and
non-negative functionality constraint, but allows actions on real packets in
both directions.  It is intentionally a separate variant: the project's
outbound-only ``iotGAN-aligned.py`` remains unchanged.

Signed packet lengths require direction-aware residuals.  Increasing an
incoming packet's physical length means making its negative signed-length
feature more negative, whereas gaps always increase in the positive direction.
The paper defines ``r*h`` as an input to G; it does not define it as a hard
output budget.  The active parameterization therefore never multiplies G's
output by ``r*h``.  This project may optionally impose one global cap on the
standardized residual as an explicitly declared overhead-control setting.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from experiment_support.metrics import timing_metrics, store_timing
import torch


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from baselines import common, iotgan as outbound


METHOD_NAME = "IoTGAN-aligned-bidirectional-misidentification"
IOTGAN_BIDIRECTIONAL_METHOD = METHOD_NAME
REPRODUCTION_NOTICE = (
    "基于独立核验的aligned IoTGAN实现；恢复incoming与outgoing双向真实包的"
    "非减包长/延迟动作，仅保留设备误识别。双向方向感知残差是本项目适配，"
    "不得与outbound-only结果混称。"
)
MAX_MULTIPLIER = outbound.MAX_MULTIPLIER
MAX_GAP_SECONDS = outbound.MAX_GAP_SECONDS
OBSERVED_EVENTS = outbound.OBSERVED_EVENTS
FEATURE_COUNT = outbound.FEATURE_COUNT
SCHEMA_VERSION = outbound.SCHEMA_VERSION
LEGACY_GENERATOR_PARAMETERIZATION = "sigmoid-times-rh-hard-output-bound"
GENERATOR_PARAMETERIZATION = "unbounded-monotone-residual-physical-projection"
DEFAULT_MANIPULATOR_EPOCHS = 30
DEFAULT_GENERATOR_LEARNING_RATE = 1e-3
DEFAULT_GENERATOR_BATCH_SIZE = 32
DEFAULT_OUTPUT_RESIDUAL_CAP_STANDARD_DEVIATIONS = None


def _bidirectional_mutability_mask(
    functional_mutability_mask, directions, mask
):
    directions = np.asarray(directions)
    valid = np.asarray(mask, dtype=bool)
    if directions.shape != valid.shape:
        raise ValueError("directions/mask形状不一致")
    if np.any(valid & ~np.isin(directions, (-1, 1))):
        raise ValueError("有效事件方向必须是-1或1")
    base = np.concatenate((valid, valid), axis=1)
    # 双向仅扩大允许动作的网络方向；首包仍没有流内前驱，不允许修改 gap[0]。
    # 自定义事件/特征掩码只能收紧此约束，不能重新开放它。
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


class LegacyBoundedBidirectionalManipulativeModel(outbound.ManipulativeModel):
    """Compatibility model for policies saved before the output-bound fix."""

    def forward(self, normalized_features, normalized_noise):
        raw = self.layers(
            torch.cat((normalized_features, normalized_noise), dim=1)
        )
        multiplier = torch.sigmoid(raw)
        return normalized_features + multiplier * normalized_noise


class BidirectionalManipulativeModel(outbound.ManipulativeModel):
    """G(h, r*h) with no sigmoid-times-r*h output limit.

    ``r*h`` remains an input exactly as described by the paper.  Network output
    controls an unbounded residual magnitude.  Direction, mutability, and the
    physical packet/gap limits are supplied by the caller so training and
    runtime application see the same feasible transformation.
    """

    def forward(
        self,
        normalized_features,
        normalized_noise,
        residual_direction=None,
        lower_bounds=None,
        upper_bounds=None,
        residual_cap_standard_deviations=None,
    ):
        raw = self.layers(
            torch.cat((normalized_features, normalized_noise), dim=1)
        )
        if residual_direction is None:
            residual_direction = torch.sign(normalized_noise)
        if residual_cap_standard_deviations is None:
            residual_magnitude = torch.nn.functional.softplus(raw)
        else:
            cap = float(residual_cap_standard_deviations)
            if not np.isfinite(cap) or cap <= 0.0:
                raise ValueError("residual cap must be a positive finite value")
            # Smoothly map the free network output into the declared global
            # cap.  This is deliberately not the old sigmoid * (r*h) rule:
            # r*h remains only the second input to G.
            residual_magnitude = cap * torch.sigmoid(raw)
        manipulated = (
            normalized_features + residual_direction * residual_magnitude
        )
        if lower_bounds is not None or upper_bounds is not None:
            if lower_bounds is None or upper_bounds is None:
                raise ValueError("lower_bounds and upper_bounds must be supplied together")
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
    """Return differentiable monotone directions and physical feature bounds."""

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
        enabled = action[:, column] & (feature_index != OBSERVED_EVENTS)
        if feature_index < OBSERVED_EVENTS:
            incoming = enabled & (raw[:, column] < 0.0)
            outgoing = enabled & (raw[:, column] > 0.0)
            direction[outgoing, column] = 1.0
            upper[outgoing, column] = (1.0 - mean[column]) / scale[column]
            direction[incoming, column] = -1.0
            lower[incoming, column] = (-1.0 - mean[column]) / scale[column]
        else:
            direction[enabled, column] = 1.0
            upper[enabled, column] = (
                maximum_log_gap - mean[column]
            ) / scale[column]
    return direction, lower, upper


class IoTGANBidirectionalPolicy(outbound.IoTGANAlignedPolicy):
    method_name = METHOD_NAME
    reproduction_notice = REPRODUCTION_NOTICE

    @classmethod
    def fit(cls, *args, **kwargs):
        # Reuse the audited feature-selection and substitute-model procedure.
        # Its one-epoch placeholder G is discarded below; the active G is then
        # trained with the physical projection used at runtime.
        requested_manipulator_epochs = int(
            kwargs.get("manipulator_epochs", DEFAULT_MANIPULATOR_EPOCHS)
        )
        requested_learning_rate = float(
            kwargs.get("learning_rate", DEFAULT_GENERATOR_LEARNING_RATE)
        )
        requested_batch_size = int(
            kwargs.get("batch_size", DEFAULT_GENERATOR_BATCH_SIZE)
        )
        requested_output_cap = kwargs.get(
            "output_residual_cap_standard_deviations",
            DEFAULT_OUTPUT_RESIDUAL_CAP_STANDARD_DEVIATIONS,
        )
        progress = bool(kwargs.get("progress", False))
        progress_label = str(kwargs.get("progress_label", "[IoTGAN-bidirectional]"))
        base_kwargs = dict(kwargs)
        base_kwargs.pop("output_residual_cap_standard_deviations", None)
        base_kwargs["manipulator_epochs"] = 1
        # The parent trains a one-epoch placeholder generator which this v2
        # policy immediately replaces.  Its stage label makes that distinction
        # visible while still reporting the lengthy shared feature selection.
        base_kwargs["progress"] = progress
        base_kwargs["progress_label"] = f"{progress_label} | shared setup"
        old_mask_builder = outbound._validate_functional_mutability_mask
        old_manipulator = outbound.ManipulativeModel
        outbound._validate_functional_mutability_mask = (
            _bidirectional_mutability_mask
        )
        outbound.ManipulativeModel = BidirectionalManipulativeModel
        try:
            policy = super().fit(*args, **base_kwargs)
        finally:
            outbound._validate_functional_mutability_mask = old_mask_builder
            outbound.ManipulativeModel = old_manipulator
        if len(args) < 6:
            raise TypeError("bidirectional fit requires six positional data arguments")
        selected_arrays, selected_indices, _ = common.training_view(
            args[0],
            args[1],
            args[2],
            args[3],
            args[5],
            groups=kwargs.get("groups"),
        )
        supplied_functional_mask = kwargs.get("functional_mutability_mask")
        selected_functional_mask = (
            None
            if supplied_functional_mask is None
            else np.asarray(supplied_functional_mask)[selected_indices]
        )
        policy = retrain_generator(
            policy,
            *selected_arrays,
            generator_epochs=requested_manipulator_epochs,
            generator_learning_rate=requested_learning_rate,
            batch_size=requested_batch_size,
            functional_mutability_mask=selected_functional_mask,
            output_residual_cap_standard_deviations=requested_output_cap,
            progress=progress,
            progress_label=f"{progress_label} | bidirectional generator",
        )
        policy.training_metadata["declared_scope"].update(
            {
                "outbound_only": False,
                "bidirectional": True,
            }
        )
        policy.training_metadata["functional_safety"].update(
            {
                "incoming_features_immutable": False,
                "only_non_negative_outgoing_padding_and_delay": False,
                "only_non_negative_bidirectional_padding_and_delay": True,
                "direction_aware_signed_length_residual": True,
            }
        )
        policy.training_metadata["implementation_variant"] = (
            "bidirectional-direction-aware-G-output-with-declared-global-cap"
        )
        return policy

    def _generator_model(self):
        if self._generator is None:
            parameterization = self.training_metadata.get(
                "generator_parameterization", {}
            ).get("name", LEGACY_GENERATOR_PARAMETERIZATION)
            model_class = (
                BidirectionalManipulativeModel
                if parameterization == GENERATOR_PARAMETERIZATION
                else LegacyBoundedBidirectionalManipulativeModel
            )
            model = model_class(len(self.selected_features))
            model.load_state_dict(self.generator_state)
            model.eval()
            self._generator = model
        return self._generator

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
        full_action_mask = _bidirectional_mutability_mask(
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
        model = self._generator_model()
        with torch.no_grad():
            if isinstance(model, BidirectionalManipulativeModel):
                projection = _physical_projection_tensors(
                    self.selected_features,
                    selected_raw,
                    normalized,
                    self.feature_mean,
                    self.feature_scale,
                    selected_action_mask,
                )
                manipulated = model(
                    torch.as_tensor(normalized, dtype=torch.float32),
                    torch.as_tensor(noise_normalized, dtype=torch.float32),
                    *projection,
                    residual_cap_standard_deviations=self.training_metadata.get(
                        "generator_parameterization", {}
                    ).get("output_residual_cap_standard_deviations"),
                )
            else:
                manipulated = model(
                    torch.as_tensor(normalized, dtype=torch.float32),
                    torch.as_tensor(noise_normalized, dtype=torch.float32),
                )
        # 物理字节先取绝对值投影，再恢复 signed length 的符号。这样入站
        # padding 对应更负的特征值，且训练与应用遵守同一整数执行规则。
        manipulated_raw = outbound._project_generated_raw(
            manipulated, selected_raw, self.selected_features,
            self.feature_mean, self.feature_scale, selected_action_mask,
        ).cpu().numpy()
        result = raw_features.copy()
        result[:, self.selected_features] = manipulated_raw
        return result, r_full, full_action_mask

    def to_dict(self):
        result = super().to_dict()
        result["method"] = METHOD_NAME
        result["reproduction_status"] = REPRODUCTION_NOTICE
        result["scope"].update(
            {
                "outbound_only": False,
                "bidirectional": True,
                "active_directions": ["incoming", "outgoing"],
            }
        )
        result["direction_aware_signed_length_residual"] = True
        result["generator_parameterization"] = self.training_metadata.get(
            "generator_parameterization",
            {"name": LEGACY_GENERATOR_PARAMETERIZATION},
        )
        return result


def fit_baseline(*args, **kwargs):
    return IoTGANBidirectionalPolicy.fit(*args, **kwargs)


def fit_target_specific_baselines(
    lengths,
    gaps,
    directions,
    mask,
    black_box_outputs_by_target,
    train_indices,
    *,
    seed=outbound.SEED,
    **kwargs,
):
    if not hasattr(black_box_outputs_by_target, "items"):
        raise TypeError("black_box_outputs_by_target必须是目标名到预测数组的映射")
    items = list(black_box_outputs_by_target.items())
    progress = bool(kwargs.pop("progress", False))
    policies = {}
    for target_position, (target_name, black_box_outputs) in enumerate(items, start=1):
        name = str(target_name)
        if not name.strip() or name in policies:
            raise ValueError(f"黑盒目标名称为空或重复: {name!r}")
        progress_label = (
            f"[IoTGAN-bidirectional][target {target_position}/{len(items)}: {name}]"
        )
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
            seed=outbound._target_seed(seed, name),
            progress=progress,
            progress_label=progress_label,
            **kwargs,
        )
        if progress:
            print(f"{progress_label} 完成", flush=True)
    if not policies:
        raise ValueError("至少需要一个黑盒目标")
    return policies


def retrain_generator(
    policy,
    lengths,
    gaps,
    directions,
    mask,
    *,
    generator_epochs,
    generator_learning_rate,
    batch_size=DEFAULT_GENERATOR_BATCH_SIZE,
    functional_mutability_mask=None,
    output_residual_cap_standard_deviations=(
        DEFAULT_OUTPUT_RESIDUAL_CAP_STANDARD_DEVIATIONS
    ),
    progress=False,
    progress_label="[IoTGAN-bidirectional]",
):
    """Retrain only G while freezing feature selection, D, loss, and r budget."""

    if not isinstance(policy, IoTGANBidirectionalPolicy):
        raise TypeError("policy必须是IoTGANBidirectionalPolicy")
    if min(int(generator_epochs), int(batch_size)) < 1:
        raise ValueError("generator_epochs和batch_size必须为正整数")
    if float(generator_learning_rate) <= 0.0:
        raise ValueError("generator_learning_rate必须为正")
    if output_residual_cap_standard_deviations is not None:
        output_residual_cap_standard_deviations = float(
            output_residual_cap_standard_deviations
        )
        if (
            not np.isfinite(output_residual_cap_standard_deviations)
            or output_residual_cap_standard_deviations <= 0.0
        ):
            raise ValueError(
                "output_residual_cap_standard_deviations must be positive and finite"
            )
    lengths, gaps, directions, mask = common.validate_observation_arrays(
        lengths, gaps, directions, mask
    )
    common.validate_dataset(lengths, gaps, directions, mask)
    raw_features = outbound.trace_feature_matrix(lengths, gaps, directions, mask)
    action_mask = _bidirectional_mutability_mask(
        functional_mutability_mask, directions, mask
    )
    selected = policy.selected_features
    selected_raw = raw_features[:, selected]
    normalized = (selected_raw - policy.feature_mean) / policy.feature_scale
    selected_action_mask = action_mask[:, selected]

    tensor_raw = torch.as_tensor(selected_raw, dtype=torch.float32)
    tensor_normalized = torch.as_tensor(normalized, dtype=torch.float32)
    tensor_scale = torch.as_tensor(policy.feature_scale, dtype=torch.float32)
    tensor_action_mask = torch.as_tensor(
        selected_action_mask, dtype=torch.float32
    )
    projection = _physical_projection_tensors(
        selected,
        selected_raw,
        normalized,
        policy.feature_mean,
        policy.feature_scale,
        selected_action_mask,
    )
    substitute = policy._substitute_model()
    substitute.eval()
    for parameter in substitute.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        clean_predictions = torch.argmax(
            substitute(tensor_normalized), dim=1
        )
    clean_one_hot = torch.nn.functional.one_hot(
        clean_predictions, num_classes=len(policy.class_labels)
    ).to(torch.float32)

    # Match the original G initialization and random streams exactly.  Thus
    # epoch count and G learning rate are the only trial-varying quantities.
    outbound._set_seed(int(policy.seed) + 3000)
    generator = BidirectionalManipulativeModel(len(selected))
    optimizer = torch.optim.Adam(
        generator.parameters(), lr=float(generator_learning_rate)
    )
    batch_rng = torch.Generator().manual_seed(int(policy.seed) + 3001)
    noise_rng = torch.Generator().manual_seed(int(policy.seed) + 3002)
    history = []
    for epoch in range(int(generator_epochs)):
        generator.train()
        losses = []
        for batch in outbound._iterate_batches(
            len(lengths), int(batch_size), batch_rng
        ):
            r = (
                torch.rand(
                    (len(batch), len(selected)),
                    generator=noise_rng,
                    dtype=torch.float32,
                )
                * MAX_MULTIPLIER
                * tensor_action_mask[batch]
            )
            noise_normalized = (r * tensor_raw[batch]) / tensor_scale
            optimizer.zero_grad(set_to_none=True)
            manipulated = generator(
                tensor_normalized[batch],
                noise_normalized,
                *(value[batch] for value in projection),
                residual_cap_standard_deviations=(
                    output_residual_cap_standard_deviations
                ),
            )
            manipulated = outbound._project_for_substitute(
                manipulated, selected_raw[batch.numpy()], selected,
                policy.feature_mean, policy.feature_scale, tensor_action_mask[batch],
            )
            manipulated_logits = substitute(manipulated)
            loss = -torch.nn.functional.binary_cross_entropy_with_logits(
                manipulated_logits, clean_one_hot[batch]
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if progress:
            outbound._print_training_progress(
                progress_label, epoch + 1, generator_epochs, history[-1]
            )

    generator.eval()
    with torch.no_grad():
        evaluation_rng = torch.Generator().manual_seed(int(policy.seed) + 4000)
        r = (
            torch.rand(
                (len(lengths), len(selected)),
                generator=evaluation_rng,
                dtype=torch.float32,
            )
            * MAX_MULTIPLIER
            * tensor_action_mask
        )
        manipulated = generator(
            tensor_normalized,
            (r * tensor_raw) / tensor_scale,
            *projection,
            residual_cap_standard_deviations=(
                output_residual_cap_standard_deviations
            ),
        )
        manipulated = outbound._project_for_substitute(
            manipulated, selected_raw, selected, policy.feature_mean,
            policy.feature_scale, tensor_action_mask,
        )
        manipulated_predictions = torch.argmax(substitute(manipulated), dim=1)
    misidentification_rate = float(
        torch.mean(
            (manipulated_predictions != clean_predictions).to(torch.float32)
        )
    )

    metadata = deepcopy(policy.training_metadata)
    # retrain_generator 确实重训了 G，因此可以更新训练约束版本；普通 load/save
    # 绝不补写这个字段，以免将历史连续权重误标成已通过整数投影训练。
    metadata["trace_constraints"] = outbound._trace_constraint_metadata()
    metadata["generator_parameterization"] = {
        "name": GENERATOR_PARAMETERIZATION,
        "paper_r_role": "input noise r*h only",
        "sigmoid_times_rh_output_bound": False,
        "output_residual": (
            "softplus network output when uncapped; cap*sigmoid(network output) "
            "when capped; never multiplied by r*h"
        ),
        "output_residual_cap_standard_deviations": (
            output_residual_cap_standard_deviations
        ),
        "cap_selected_globally": (
            output_residual_cap_standard_deviations is not None
        ),
        "runtime_constraints": [
            "only enabled real packet/gap features",
            "packet length and gap never decrease",
            f"Ethernet frame length <= {common.MAX_FRAME_LENGTH:g} bytes",
            f"gap <= {MAX_GAP_SECONDS:g} seconds",
        ],
    }
    metadata["generator_hyperparameter_retraining"] = {
        "feature_selection_frozen": True,
        "selected_feature_indices_unchanged": selected.astype(int).tolist(),
        "substitute_weights_frozen": True,
        "substitute_weights_sha256": outbound._state_digest(
            policy.substitute_state
        ),
        "generator_hidden_architecture_unchanged": True,
        "generator_output_parameterization": GENERATOR_PARAMETERIZATION,
        "sigmoid_times_rh_output_bound_removed": True,
        "generator_loss_unchanged": (
            "negative binary_cross_entropy_with_logits against clean prediction"
        ),
        "maximum_r_unchanged": float(MAX_MULTIPLIER),
        "output_residual_cap_standard_deviations": (
            output_residual_cap_standard_deviations
        ),
        "batch_size": int(batch_size),
        "generator_epochs": int(generator_epochs),
        "generator_learning_rate": float(generator_learning_rate),
        "generator_initialization_seed": int(policy.seed) + 3000,
        "batch_seed": int(policy.seed) + 3001,
        "noise_seed": int(policy.seed) + 3002,
    }
    metadata.setdefault("paper_unspecified_engineering_choices", {}).update(
        {
            "manipulator_epochs": int(generator_epochs),
            "learning_rate": float(generator_learning_rate),
            "batch_size": int(batch_size),
            "generator_output_parameterization": GENERATOR_PARAMETERIZATION,
            "output_residual_cap_standard_deviations": (
                output_residual_cap_standard_deviations
            ),
        }
    )
    metadata["training_history"][
        "manipulator_negative_binary_cross_entropy"
    ] = history
    metadata["training_history"][
        "training_misidentification_rate_against_substitute"
    ] = misidentification_rate
    return IoTGANBidirectionalPolicy(
        deepcopy(generator.state_dict()),
        deepcopy(policy.substitute_state),
        selected.copy(),
        policy.feature_mean.copy(),
        policy.feature_scale.copy(),
        policy.class_labels,
        policy.black_box_target,
        seed=policy.seed,
        training_metadata=metadata,
    )


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
    base_seed=outbound.SEED,
):
    if not isinstance(policy, IoTGANBidirectionalPolicy):
        raise TypeError("policy必须是IoTGANBidirectionalPolicy")
    policy.assert_target(black_box_target)
    lengths, gaps, directions, mask = common.validate_observation_arrays(
        lengths, gaps, directions, mask
    )
    common.validate_dataset(lengths, gaps, directions, mask)
    sample_count = len(lengths)
    labels_array = None if labels is None else np.asarray(labels)
    if labels_array is not None and len(labels_array) != sample_count:
        raise ValueError("labels数量与输入流不一致")

    full_functional_mask = _bidirectional_mutability_mask(
        functional_mutability_mask, directions, mask
    )
    raw_features = outbound.trace_feature_matrix(lengths, gaps, directions, mask)
    out_lengths = np.zeros_like(lengths, dtype=np.float64)
    out_gaps = np.zeros_like(gaps, dtype=np.float64)
    out_directions = np.asarray(directions).copy()
    out_mask = np.asarray(mask).copy()
    stats = common.empty_stats(sample_count)
    audit_rows = []
    seed_rng = np.random.default_rng(int(base_seed))

    for sample in range(sample_count):
        session_seed = int(seed_rng.integers(0, np.iinfo(np.int64).max))
        manipulated, r_values, action_mask = policy.transform_features(
            raw_features[sample],
            directions[sample : sample + 1],
            mask[sample : sample + 1],
            session_seed,
            functional_mutability_mask=full_functional_mask[sample : sample + 1],
        )
        decoded_lengths, decoded_gaps, decoded_directions = outbound._decode_features(
            manipulated[0], directions[sample], mask[sample]
        )
        valid = np.asarray(mask[sample], dtype=bool)
        outgoing = valid & (np.asarray(directions[sample]) == 1)
        incoming = valid & (np.asarray(directions[sample]) == -1)
        length_allowed = action_mask[0, :OBSERVED_EVENTS]
        gap_allowed = action_mask[0, OBSERVED_EVENTS:]

        # 先保留原始物理下界与禁改事件，再据最终输出计算开销；不能使用
        # G 的连续候选计费，也不能将首包之前的等待算作 gap[0] 增量。
        decoded_lengths[valid] = np.maximum(
            decoded_lengths[valid], np.asarray(lengths[sample])[valid]
        )
        decoded_gaps[valid] = np.maximum(
            decoded_gaps[valid], np.asarray(gaps[sample])[valid]
        )
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
            raise AssertionError("双向IoTGAN改变了真实事件方向")
        if np.any(length_delta[valid] < -1e-9) or np.any(
            gap_delta[valid] < -1e-12
        ):
            raise AssertionError("双向IoTGAN产生了特征减少")
        if np.any(np.abs(length_delta[valid & ~length_allowed]) > 1e-9):
            raise AssertionError("双向IoTGAN修改了功能掩码禁止的包长")
        if np.any(np.abs(gap_delta[valid & ~gap_allowed]) > 1e-12):
            raise AssertionError("双向IoTGAN修改了功能掩码禁止的间隔")

        stats["real_added_bytes"][sample] = np.sum(length_delta[valid])
        stats["real_added_delay"][sample] = np.sum(gap_delta[valid])
        store_timing(stats,sample,timing_metrics(np.asarray(gaps[sample])[valid],
            np.cumsum(decoded_gaps[valid]),np.ones(np.count_nonzero(valid),dtype=bool)),sample_count)
        stats["outbound_real_added_bytes"][sample] = np.sum(length_delta[outgoing])
        stats["outbound_real_added_delay"][sample] = np.sum(gap_delta[outgoing])
        stats["inbound_real_added_bytes"][sample] = np.sum(length_delta[incoming])
        stats["inbound_real_added_delay"][sample] = np.sum(gap_delta[incoming])
        audit_rows.append(
            {
                "method": METHOD_NAME,
                "black_box_target": policy.black_box_target,
                "policy_target_verified": True,
                "source_label": (
                    outbound._python_scalar(labels_array[sample])
                    if labels_array is not None
                    else "not-provided"
                ),
                "session_seed": session_seed,
                "runtime_uses_source_label": False,
                "outbound_only": False,
                "bidirectional": True,
                "feature_decreases": 0,
                "dummy_packets": 0,
                "actions_taken": int(np.sum(changed)),
                "outbound_actions": int(np.sum(changed & outgoing)),
                "inbound_actions": int(np.sum(changed & incoming)),
                "padding_actions": int(np.sum(valid & (length_delta > 1e-7))),
                "delay_actions": int(np.sum(valid & (gap_delta > 1e-10))),
                "real_added_bytes": float(np.sum(length_delta[valid])),
                "real_added_delay": float(np.sum(gap_delta[valid])),
                "maximum_r": float(np.max(r_values)),
                "direction_aware_signed_length_residual": True,
            }
        )

    metadata = {
        "method": METHOD_NAME,
        "reproduction_status": REPRODUCTION_NOTICE,
        "objective": "device_misidentification_equation_4",
        "black_box_target": policy.black_box_target,
        "policy_target_verified": True,
        "runtime_uses_source_label": False,
        "outbound_only": False,
        "bidirectional": True,
        "active_directions": ["incoming", "outgoing"],
        "feature_decreases_supported": False,
        "identity_spoofing_supported": False,
        "dummy_packets_supported": False,
        "functional_mutability_mask_enforced": True,
        "trace_constraints": outbound._trace_constraint_metadata(),
        "training_constraints_match_runtime": policy.training_constraints_match_runtime,
        "direction_aware_signed_length_residual": True,
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
    policies, target_name, lengths, gaps, directions, mask, labels=None, **kwargs
):
    name = str(target_name)
    if name not in policies:
        raise KeyError(f"没有为黑盒目标{name!r}训练双向IoTGAN")
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


def save_policy(path, policy):
    if not isinstance(policy, IoTGANBidirectionalPolicy):
        raise TypeError("只能保存IoTGANBidirectionalPolicy")
    outbound.save_policy(path, policy)


def load_policy(path, *, allow_legacy_constraints=False):
    """与单向版共用版本校验，阻止旧连续策略静默进入新评估。"""
    with np.load(path, allow_pickle=False) as content:
        metadata = json.loads(str(content["metadata_json"]))
        if metadata.get("method") != METHOD_NAME:
            raise ValueError("策略文件不是bidirectional aligned IoTGAN")
        outbound._validate_saved_constraints(
            metadata, allow_legacy_constraints=allow_legacy_constraints,
        )
        states = {"generator": {}, "substitute": {}}
        for key in content.files:
            for prefix in states:
                marker = f"{prefix}__"
                if key.startswith(marker):
                    name = key[len(marker) :].replace("__dot__", ".")
                    states[prefix][name] = torch.from_numpy(content[key].copy())
        policy = IoTGANBidirectionalPolicy(
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
        raise ValueError("双向IoTGAN权重摘要不匹配")
    return policy


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
        "outbound_only": False,
        "bidirectional": True,
        "feature_decreases": 0,
        "identity_spoofing": False,
    }


def run_self_test():
    lengths, gaps, directions, mask, labels, groups = outbound._synthetic_dataset()
    train_indices = np.asarray(
        list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)), dtype=int
    )
    test_indices = np.asarray([5, 11, 17], dtype=int)
    functional_mask = np.ones_like(mask, dtype=bool)
    functional_mask[:, 2] = False
    policies = fit_target_specific_baselines(
        lengths,
        gaps,
        directions,
        mask,
        {"rf-oracle": labels.copy(), "svm-oracle": (labels + 1) % 3},
        train_indices,
        groups=groups,
        functional_mutability_mask=functional_mask,
        seed=1234,
        substitute_epochs=3,
        manipulator_epochs=3,
        batch_size=8,
        candidate_feature_counts=(64,),
    )
    for policy in policies.values():
        parameterization = policy.training_metadata.get(
            "generator_parameterization", {}
        )
        if parameterization.get("name") != GENERATOR_PARAMETERIZATION:
            raise AssertionError("self-test policy did not use the active G parameterization")
        if parameterization.get("sigmoid_times_rh_output_bound") is not False:
            raise AssertionError("self-test policy retained the removed sigmoid*r*h bound")

    probe = BidirectionalManipulativeModel(2)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        original = torch.zeros((1, 2), dtype=torch.float32)
        tiny_noise = torch.full((1, 2), 0.01, dtype=torch.float32)
        direction = torch.ones_like(original)
        projected = probe(
            original,
            tiny_noise,
            direction,
            torch.zeros_like(original),
            torch.full_like(original, 10.0),
        )
    if not torch.all(projected - original > tiny_noise):
        raise AssertionError("G output is still bounded by the r*h input magnitude")
    capped = probe(
        original,
        tiny_noise,
        direction,
        torch.zeros_like(original),
        torch.full_like(original, 10.0),
        residual_cap_standard_deviations=0.25,
    )
    if not torch.all(capped - original <= 0.25 + 1e-7):
        raise AssertionError("declared standardized residual cap was exceeded")

    tuned = retrain_generator(
        policies["rf-oracle"],
        lengths[train_indices],
        gaps[train_indices],
        directions[train_indices],
        mask[train_indices],
        generator_epochs=2,
        generator_learning_rate=0.003,
        batch_size=8,
        functional_mutability_mask=functional_mask[train_indices],
    )
    if not np.array_equal(
        tuned.selected_features, policies["rf-oracle"].selected_features
    ):
        raise AssertionError("仅重训G时特征选择发生变化")
    if outbound._state_digest(
        tuned.substitute_state
    ) != outbound._state_digest(policies["rf-oracle"].substitute_state):
        raise AssertionError("仅重训G时替代模型权重发生变化")
    if len(
        tuned.training_metadata["training_history"][
            "manipulator_negative_binary_cross_entropy"
        ]
    ) != 2:
        raise AssertionError("仅重训G没有记录正确轮数")
    application = apply_target_specific_policy(
        policies,
        "rf-oracle",
        lengths[test_indices],
        gaps[test_indices],
        directions[test_indices],
        mask[test_indices],
        labels[test_indices],
        functional_mutability_mask=functional_mask[test_indices],
        base_seed=7000,
    )
    valid = mask[test_indices].astype(bool)
    incoming = valid & (directions[test_indices] == -1)
    outgoing = valid & (directions[test_indices] == 1)
    if np.any(application[0][valid] < lengths[test_indices][valid]):
        raise AssertionError("自测试发现双向包长减少")
    if np.any(application[1][valid] < gaps[test_indices][valid]):
        raise AssertionError("自测试发现双向间隔减少")
    if not np.any(application[0][incoming] > lengths[test_indices][incoming]):
        raise AssertionError("自测试没有产生incoming包长动作")
    if not np.any(application[0][outgoing] > lengths[test_indices][outgoing]):
        raise AssertionError("自测试没有产生outgoing包长动作")
    if not np.array_equal(application[0][:, 2], lengths[test_indices, 2]):
        raise AssertionError("功能掩码禁止的包长发生变化")
    if not np.array_equal(application[1][:, 2], gaps[test_indices, 2]):
        raise AssertionError("功能掩码禁止的间隔发生变化")
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
        pass
    else:
        raise AssertionError("没有拒绝目标不匹配的策略")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "iotgan_bidirectional_policy.npz"
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
            functional_mutability_mask=functional_mask[test_indices],
            base_seed=7000,
        )
        if not np.array_equal(application[0], replay[0]) or not np.array_equal(
            application[1], replay[1]
        ):
            raise AssertionError("双向策略保存后回放不一致")
    return {
        "passed": True,
        "generator_parameterization": GENERATOR_PARAMETERIZATION,
        "sigmoid_times_rh_output_bound_removed": True,
        "output_exceeds_rh_probe": True,
        "declared_output_cap_probe": True,
        "saved_default_generator_epochs": DEFAULT_MANIPULATOR_EPOCHS,
        "saved_default_generator_learning_rate": DEFAULT_GENERATOR_LEARNING_RATE,
        "saved_default_generator_batch_size": DEFAULT_GENERATOR_BATCH_SIZE,
        "targets": sorted(policies),
        "incoming_actions": int(
            sum(row["inbound_actions"] for row in application[6])
        ),
        "outgoing_actions": int(
            sum(row["outbound_actions"] for row in application[6])
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.error("当前命令行接口仅支持--self-test")
    print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
