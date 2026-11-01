"""Class-count-agnostic adapted physical-v5 PyTorch iPET implementation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from baselines import common
from experiment_support.metrics import timing_metrics, store_timing

from .model import (
    DEFAULT_DISCRIMINATOR_EPOCHS,
    DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS,
    DEFAULT_GENERATION_BATCH_SIZE,
    DEFAULT_GENERATOR_EPOCHS,
    DEFAULT_TRAINING_STAGES,
    LATENT_DIM,
    MAX_PACKETS_PER_OMEGA,
    MAX_PAYLOAD_PER_OMEGA,
    OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE,
    OFFICIAL_IPET_BATCH_SIZE,
    OFFICIAL_NADAM_EPS,
    OFFICIAL_NADAM_LR,
    OFFICIAL_PERTURBATION_VECTOR_COUNT,
    OfficialIPETGenerator,
    generate_perturbation_pool,
    train_official_ipet,
)
from .traffic import (
    FEATURE_ORDER,
    OMEGA_SECONDS,
    PerturbationPlan,
    SILENCE_THRESHOLD_SECONDS,
    TOTAL_TIME_SECONDS,
    inject_official_ipet_trace,
    slotize_flows,
)


OBSERVED_EVENTS = common.OBSERVED_EVENTS
SEED = common.SEED
SCHEMA_VERSION = 5
IPET_METHOD = "iPET-physical-slots-v5"
IPET_OUTBOUND_METHOD = "iPET-outbound-physical-slots-v5"
# The public adapter key remains ``ipet_budgeted_v3`` for compatibility, while
# the artifact/audit name states the physical-v5 execution semantics.
IPET_BUDGETED_V3_METHOD = "iPET-physical-v5-budgeted"
PHYSICAL_RUNTIME_MARKER = "physical-slot-projection-v5"
UPSTREAM_REPOSITORY = "https://github.com/akshayeshenoi/ipet"
UPSTREAM_COMMIT = "ccbe4f7477fb5da6fc09986c5c01eda8b11fedf8"
UPSTREAM_TAG = "v0.1.3"
REPRODUCTION_NOTICE = (
    "明确标注为adapted physical-v5的隔离PyTorch实现，不声称上游TensorFlow原样复现；"
    "G与D重训练batch=32；G训练时保留D的Dropout；"
    "NAdam使用TensorFlow 2.5默认lr/eps；D的Adam跨stage复用；"
    "每个设备类别预生成20000个40x4扰动矩阵；训练、池生成和运行时共用物理slot投影。"
)
OUTBOUND_REPRODUCTION_NOTICE = (
    REPRODUCTION_NOTICE
    + " 本项目单向变体仅屏蔽incoming生成通道并停用incoming dummy tick。"
)


def validate_dataset(lengths, gaps, directions, mask):
    return common.validate_dataset(lengths, gaps, directions, mask)


def _python_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _state_digest(generator_states) -> str:
    digest = hashlib.sha256()
    for label in sorted(generator_states, key=lambda item: str(item)):
        digest.update(str(label).encode("utf-8"))
        for name, value in sorted(generator_states[label].items()):
            array = value.detach().cpu().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class OfficialCodeIPETPolicy:
    """Device-specific generators and their official-size perturbation pools."""

    method_name = IPET_METHOD
    reproduction_notice = REPRODUCTION_NOTICE
    feature_mask = (1.0, 1.0, 1.0, 1.0)
    dummy_directions = (-1, 1)
    active_direction_names = ("incoming", "outgoing")

    def __init__(
        self,
        class_labels,
        generator_states,
        *,
        slot_count=40,
        max_packets=MAX_PACKETS_PER_OMEGA,
        max_payload=MAX_PAYLOAD_PER_OMEGA,
        seed=SEED,
        perturbation_vector_count=OFFICIAL_PERTURBATION_VECTOR_COUNT,
        generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
        training_metadata=None,
    ):
        self.class_labels = tuple(_python_scalar(value) for value in class_labels)
        self.generator_states = {
            _python_scalar(label): {
                name: value.detach().cpu().clone()
                for name, value in state.items()
            }
            for label, state in generator_states.items()
        }
        if set(self.class_labels) != set(self.generator_states):
            raise ValueError("iPET generator labels与class_labels不一致")
        self.slot_count = int(slot_count)
        self.max_packets = float(max_packets)
        self.max_payload = float(max_payload)
        self.seed = int(seed)
        self.perturbation_vector_count = int(perturbation_vector_count)
        self.generation_batch_size = int(generation_batch_size)
        if self.perturbation_vector_count < 1 or self.generation_batch_size < 1:
            raise ValueError("扰动矩阵数量和生成batch必须为正整数")
        self.training_metadata = dict(training_metadata or {})
        self._generators = {}
        self._perturbation_pools = {}
        self._pool_generation_counts = {label: 0 for label in self.class_labels}

    @classmethod
    def fit(
        cls,
        lengths,
        gaps,
        directions,
        mask,
        labels,
        train_indices,
        *,
        groups=None,
        seed=SEED,
        discriminator_epochs=DEFAULT_DISCRIMINATOR_EPOCHS,
        training_stages=DEFAULT_TRAINING_STAGES,
        generator_epochs=DEFAULT_GENERATOR_EPOCHS,
        discriminator_retrain_epochs=DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS,
        perturbation_vector_count=OFFICIAL_PERTURBATION_VECTOR_COUNT,
        generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
        progress=False,
        device="cpu",
        shared_budget=None,
    ):
        selected, indices, training_groups = common.training_view(
            lengths,
            gaps,
            directions,
            mask,
            train_indices,
            groups=groups,
        )
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(np.asarray(lengths)):
            raise ValueError("labels必须是一维数组且与输入流数量一致")
        selected_labels = labels[indices]
        slot_features, silences = slotize_flows(*selected)
        slot_masks = None
        if shared_budget is not None:
            from experiment_support.budget import PhysicalBudget
            from .traffic import budget_training_slot_masks
            shared_budget = PhysicalBudget(**shared_budget).to_dict()
            slot_masks = budget_training_slot_masks(*selected, shared_budget, cls.dummy_directions)
        trained = train_official_ipet(
            slot_features,
            silences,
            selected_labels,
            seed=seed,
            discriminator_epochs=discriminator_epochs,
            training_stages=training_stages,
            generator_epochs=generator_epochs,
            discriminator_retrain_epochs=discriminator_retrain_epochs,
            feature_mask=cls.feature_mask,
            progress=bool(progress),
            device=device,
            budget=shared_budget, slot_masks=slot_masks,
        )
        class_counts = {
            str(_python_scalar(label)): int(np.sum(selected_labels == label))
            for label in np.unique(selected_labels)
        }
        metadata = {
            "implementation_origin": (
                "authors-official-code-v2aligned-adapted-physical-v5"
            ),
            "upstream_exact_reproduction": False,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_tag": UPSTREAM_TAG,
            "upstream_license": "GPL-3.0",
            "backend_adaptation": "TensorFlow-2.5-to-PyTorch-v2aligned",
            "action_constraint": {
                "feature_mask": list(cls.feature_mask),
                "dummy_directions": list(cls.dummy_directions),
                "active_directions": list(cls.active_direction_names),
            },
            "training_flows": int(len(indices)),
            "fit_received_explicit_training_indices": True,
            "training_groups": training_groups,
            "class_counts": class_counts,
            "slot_representation": {
                "total_time_seconds": TOTAL_TIME_SECONDS,
                "omega_seconds": OMEGA_SECONDS,
                "slot_count": int(slot_features.shape[1]),
                "feature_order": list(FEATURE_ORDER),
                "source_representation": (
                    "first-32 Ethernet-frame events per TLS flow"
                ),
                "silence_feature": "zero because flows are not contiguous day windows",
                "boundary_rule": (
                    "timestamps on an omega boundary use that boundary's slot"
                ),
            },
            "physical_slot_semantics": {
                "marker": PHYSICAL_RUNTIME_MARKER,
                "training_pool_runtime_projection_shared": True,
                "count_rounding": "nearest-even integral packet count",
                "count_zero_is_noop": True,
                "directions_share_slot_row": True,
                "frame_rounding": "nearest-even after min/max frame clamp",
                "long_silence_slot_ids": "absolute relative-to-flow timeline",
            },
            "official_alignment": {
                "initial_discriminator_batch_size": (
                    OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE
                ),
                "generator_batch_size": OFFICIAL_IPET_BATCH_SIZE,
                "discriminator_retrain_batch_size": OFFICIAL_IPET_BATCH_SIZE,
                "nadam_lr": OFFICIAL_NADAM_LR,
                "nadam_eps": OFFICIAL_NADAM_EPS,
                "discriminator_dropout_active_for_generator": True,
                "discriminator_adam_reused_across_stages": True,
                "perturbation_vectors_per_device": int(
                    perturbation_vector_count
                ),
                "flattened_rows_per_device": int(
                    perturbation_vector_count * slot_features.shape[1]
                ),
                "per_flow_start": (
                    "session-seeded vector boundary in the shared device pool"
                ),
            },
            "training_history": trained["training"],
        }
        return cls(
            trained["class_labels"],
            trained["generators"],
            slot_count=slot_features.shape[1],
            seed=seed,
            perturbation_vector_count=perturbation_vector_count,
            generation_batch_size=generation_batch_size,
            training_metadata=metadata,
        )

    @property
    def weights_sha256(self):
        return _state_digest(self.generator_states)

    @property
    def pool_generation_counts(self):
        return dict(self._pool_generation_counts)

    def _generator(self, label):
        label = _python_scalar(label)
        if label not in self.generator_states:
            raise ValueError(f"没有设备类别{label!r}的iPET generator")
        if label not in self._generators:
            model = OfficialIPETGenerator(
                self.slot_count,
                max_packets=self.max_packets,
                max_payload=self.max_payload,
                feature_mask=self.feature_mask,
            )
            model.load_state_dict(self.generator_states[label])
            model.eval()
            self._generators[label] = model
        return self._generators[label]

    def _pool_seed(self, label) -> int:
        label = _python_scalar(label)
        class_index = self.class_labels.index(label)
        return self.seed + 50_000_003 + 104_729 * class_index

    def _pool(self, label) -> np.ndarray:
        label = _python_scalar(label)
        if label not in self._perturbation_pools:
            self._perturbation_pools[label] = generate_perturbation_pool(
                self._generator(label),
                self._pool_seed(label),
                vector_count=self.perturbation_vector_count,
                generation_batch_size=self.generation_batch_size,
            )
            self._pool_generation_counts[label] += 1
        return self._perturbation_pools[label]

    def generate(self, label, seed):
        """Return a plan over the shared 20,000-vector device pool.

        The upstream gateway consumes a long pre-generated device sequence.  The
        project stores independent TLS flows, so a deterministic session seed
        selects a vector boundary rather than restarting every flow at row zero.
        """

        pool = self._pool(label)
        rng = np.random.default_rng(int(seed))
        vector_index = int(rng.integers(0, self.perturbation_vector_count))
        return PerturbationPlan(
            values=pool,
            start_index=vector_index * self.slot_count,
            vector_count=self.perturbation_vector_count,
            slot_count=self.slot_count,
        )

    def to_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "method": self.method_name,
            "reproduction_status": self.reproduction_notice,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "commit": UPSTREAM_COMMIT,
                "tag": UPSTREAM_TAG,
                "license": "GPL-3.0",
            },
            "runtime_semantics": {
                "physical_slot_projection": PHYSICAL_RUNTIME_MARKER,
                "count_zero_is_noop": True,
                "directions_share_slot_row": True,
                "frame_rounding": "nearest-even after min/max frame clamp",
                "slot_boundary_rule": "stable omega boundary assignment",
                "long_silence_rows_preserved": True,
                "pool_cursor_rule": (
                    "start_index plus max absolute slot id plus one; "
                    "shared direction rows count once"
                ),
                "device_specific_generator": True,
                "runtime_uses_device_label": True,
                "perturbation_pre_generated": True,
                "perturbation_vectors_per_device": self.perturbation_vector_count,
                "flattened_pool_rows_per_device": (
                    self.perturbation_vector_count * self.slot_count
                ),
                "per_flow_start": "session-seeded vector boundary",
                "active_directions": list(self.active_direction_names),
                "generator_feature_mask": list(self.feature_mask),
                "dummy_directions": list(self.dummy_directions),
                "omega_seconds": OMEGA_SECONDS,
                "silence_threshold_seconds": SILENCE_THRESHOLD_SECONDS,
                "real_events_modified": False,
                "real_events_deleted_or_reordered": False,
                "max_observed_events": OBSERVED_EVENTS,
            },
            "class_labels": list(self.class_labels),
            "slot_count": self.slot_count,
            "max_packets": self.max_packets,
            "max_payload": self.max_payload,
            "seed": self.seed,
            "generation_batch_size": self.generation_batch_size,
            "generator_weights_sha256": self.weights_sha256,
            "training": self.training_metadata,
        }


class OutboundOnlyOfficialCodeIPETPolicy(OfficialCodeIPETPolicy):
    """Project-authorized outgoing-only constraint over the aligned iPET port."""

    method_name = IPET_OUTBOUND_METHOD
    reproduction_notice = OUTBOUND_REPRODUCTION_NOTICE
    feature_mask = (0.0, 0.0, 1.0, 1.0)
    dummy_directions = (1,)
    active_direction_names = ("outgoing",)


def fit_baseline(
    lengths,
    gaps,
    directions,
    mask,
    labels,
    train_indices,
    *,
    groups=None,
    seed=SEED,
    discriminator_epochs=DEFAULT_DISCRIMINATOR_EPOCHS,
    training_stages=DEFAULT_TRAINING_STAGES,
    generator_epochs=DEFAULT_GENERATOR_EPOCHS,
    discriminator_retrain_epochs=DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS,
    perturbation_vector_count=OFFICIAL_PERTURBATION_VECTOR_COUNT,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    progress=False,
    device="cpu",
    shared_budget=None,
):
    return OfficialCodeIPETPolicy.fit(
        lengths,
        gaps,
        directions,
        mask,
        labels,
        train_indices,
        groups=groups,
        seed=seed,
        discriminator_epochs=discriminator_epochs,
        training_stages=training_stages,
        generator_epochs=generator_epochs,
        discriminator_retrain_epochs=discriminator_retrain_epochs,
        perturbation_vector_count=perturbation_vector_count,
        generation_batch_size=generation_batch_size,
        progress=progress,
        device=device,
        shared_budget=shared_budget,
    )


def fit_outbound_only_baseline(
    lengths,
    gaps,
    directions,
    mask,
    labels,
    train_indices,
    *,
    groups=None,
    seed=SEED,
    discriminator_epochs=DEFAULT_DISCRIMINATOR_EPOCHS,
    training_stages=DEFAULT_TRAINING_STAGES,
    generator_epochs=DEFAULT_GENERATOR_EPOCHS,
    discriminator_retrain_epochs=DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS,
    perturbation_vector_count=OFFICIAL_PERTURBATION_VECTOR_COUNT,
    generation_batch_size=DEFAULT_GENERATION_BATCH_SIZE,
    progress=False,
    device="cpu",
    shared_budget=None,
):
    return OutboundOnlyOfficialCodeIPETPolicy.fit(
        lengths,
        gaps,
        directions,
        mask,
        labels,
        train_indices,
        groups=groups,
        seed=seed,
        discriminator_epochs=discriminator_epochs,
        training_stages=training_stages,
        generator_epochs=generator_epochs,
        discriminator_retrain_epochs=discriminator_retrain_epochs,
        perturbation_vector_count=perturbation_vector_count,
        generation_batch_size=generation_batch_size,
        progress=progress,
        device=device,
        shared_budget=shared_budget,
    )


def _validate_budgeted_v3_limits(max_added_bytes, max_dummy_packets):
    """Validate the explicit per-flow hard budget used by the v3 variant."""

    max_added_bytes = float(max_added_bytes)
    if not np.isfinite(max_added_bytes) or max_added_bytes < 0.0:
        raise ValueError("iPET-physical-v5-budgeted max_added_bytes必须是有限非负数")
    if isinstance(max_dummy_packets, (bool, np.bool_)):
        raise ValueError("iPET-physical-v5-budgeted max_dummy_packets必须是非负整数")
    packet_limit = int(max_dummy_packets)
    if packet_limit != max_dummy_packets or packet_limit < 0:
        raise ValueError("iPET-physical-v5-budgeted max_dummy_packets必须是非负整数")
    return max_added_bytes, packet_limit


def _apply_policy(
    policy,
    lengths,
    gaps,
    directions,
    mask,
    labels,
    *,
    base_seed=SEED,
    max_added_bytes=None,
    max_dummy_packets=None,
    reported_method=None,
    max_wire_delay_seconds=None,
):
    lengths, gaps, directions, mask = validate_dataset(
        lengths, gaps, directions, mask
    )
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(lengths):
        raise ValueError("iPET应用阶段必须提供每条流的设备标签")
    sample_count, observed_events = lengths.shape
    out_lengths = np.zeros_like(lengths, dtype=np.float64)
    out_gaps = np.zeros_like(gaps, dtype=np.float64)
    out_directions = np.zeros_like(directions)
    out_mask = np.zeros_like(mask)
    stats = common.empty_stats(sample_count)
    audit_rows = []
    seed_generator = np.random.default_rng(int(base_seed))

    for sample in range(sample_count):
        source_count = int(np.sum(mask[sample]))
        session_seed = int(seed_generator.integers(0, np.iinfo(np.int64).max))
        perturbations = policy.generate(labels[sample], session_seed)
        events, injection = inject_official_ipet_trace(
            perturbations,
            lengths[sample, :source_count],
            gaps[sample, :source_count],
            directions[sample, :source_count],
            minimum_frame_length=common.DUMMY_MIN_FRAME_LENGTH,
            maximum_frame_length=common.DUMMY_MAX_FRAME_LENGTH,
            dummy_directions=policy.dummy_directions,
        )
        proposed_dummy_packets = int(injection["dummy_packets"])
        proposed_dummy_bytes = float(injection["dummy_bytes"])
        budget_rejected_packets = 0
        budget_rejected_bytes = 0.0
        if max_added_bytes is not None or max_dummy_packets is not None:
            if max_added_bytes is None or max_dummy_packets is None:
                raise ValueError(
                    "iPET-physical-v5-budgeted必须同时设置字节与dummy包预算"
                )
            emitted_events = []
            emitted_dummy_packets = 0
            emitted_dummy_bytes = 0.0
            # 按网络时间顺序逐个决策，只依赖已经消耗的预算；不读取未来事件。
            latest_real_timestamp = 0.0
            for event in events:
                if not event.is_dummy:
                    latest_real_timestamp = event.timestamp
                    emitted_events.append(event)
                    continue
                fits = (
                    (max_wire_delay_seconds is None or event.timestamp <= latest_real_timestamp + max_wire_delay_seconds + 1e-9)
                    and emitted_dummy_packets < max_dummy_packets
                    and emitted_dummy_bytes + event.length <= max_added_bytes + 1e-9
                )
                if fits:
                    emitted_events.append(event)
                    emitted_dummy_packets += 1
                    emitted_dummy_bytes += event.length
                else:
                    budget_rejected_packets += 1
                    budget_rejected_bytes += event.length
            events = emitted_events
        else:
            emitted_dummy_packets = proposed_dummy_packets
            emitted_dummy_bytes = proposed_dummy_bytes

        real_events = [event for event in events if not event.is_dummy]
        if [event.source_event_index for event in real_events] != list(
            range(source_count)
        ):
            raise AssertionError("iPET适配器删除、复制或重排了真实事件")
        for position, event in enumerate(real_events):
            if event.direction != int(directions[sample, position]):
                raise AssertionError("iPET适配器改变了真实事件方向")
            if event.length != float(lengths[sample, position]):
                raise AssertionError("iPET适配器改变了真实事件长度")

        retained = events[:observed_events]
        store_timing(stats,sample,timing_metrics(gaps[sample,:source_count],
            [event.timestamp for event in events],[not event.is_dummy for event in events]),sample_count)
        previous_timestamp = 0.0
        retained_real = 0
        retained_dummy = 0
        for position, event in enumerate(retained):
            out_lengths[sample, position] = event.length
            out_gaps[sample, position] = max(
                0.0, event.timestamp - previous_timestamp
            )
            out_directions[sample, position] = event.direction
            out_mask[sample, position] = 1
            previous_timestamp = event.timestamp
            if event.is_dummy:
                retained_dummy += 1
            else:
                retained_real += 1

        if retained:out_gaps[sample,0]=0.0
        outbound_dummy = [
            event for event in events if event.is_dummy and event.direction == 1
        ]
        inbound_dummy = [
            event for event in events if event.is_dummy and event.direction == -1
        ]
        allowed_dummy_directions = set(policy.dummy_directions)
        if any(
            event.direction not in allowed_dummy_directions
            for event in events
            if event.is_dummy
        ):
            raise AssertionError("iPET产生了策略约束以外方向的dummy包")
        stats["dummy_count"][sample] = emitted_dummy_packets
        stats["dummy_bytes"][sample] = emitted_dummy_bytes
        stats["retained_dummy_count"][sample] = retained_dummy
        stats["displaced_real_packets"][sample] = source_count - retained_real
        stats["outbound_dummy_count"][sample] = len(outbound_dummy)
        stats["outbound_dummy_bytes"][sample] = sum(
            event.length for event in outbound_dummy
        )
        stats["inbound_dummy_count"][sample] = len(inbound_dummy)
        stats["inbound_dummy_bytes"][sample] = sum(
            event.length for event in inbound_dummy
        )
        audit_rows.append(
            {
                "method": reported_method or policy.method_name,
                "active_directions": list(policy.active_direction_names),
                "source_label": _python_scalar(labels[sample]),
                "session_seed": session_seed,
                "source_events_seen": source_count,
                "source_lookahead_events": 0,
                "network_event_count": len(events),
                "network_real_event_count": len(real_events),
                "network_real_order_preserved": True,
                "dummy_packets": emitted_dummy_packets,
                "dummy_bytes": emitted_dummy_bytes,
                "proposed_dummy_packets": proposed_dummy_packets,
                "proposed_dummy_bytes": proposed_dummy_bytes,
                "budget_rejected_dummy_packets": budget_rejected_packets,
                "budget_rejected_dummy_bytes": budget_rejected_bytes,
                "hard_budget_max_added_bytes": max_added_bytes,
                "hard_budget_max_dummy_packets": max_dummy_packets,
                "hard_budget_max_wire_delay_seconds": max_wire_delay_seconds,
                "hard_budget_passed": bool(
                    max_added_bytes is None
                    or (
                        emitted_dummy_bytes <= max_added_bytes + 1e-9
                        and emitted_dummy_packets <= max_dummy_packets
                        and (max_wire_delay_seconds is None or stats['wire_completion_delay_seconds'][sample] <= max_wire_delay_seconds + 1e-9)
                    )
                ),
                "outbound_dummy_packets": len(outbound_dummy),
                "inbound_dummy_packets": len(inbound_dummy),
                "retained_event_count": len(retained),
                "retained_dummy_count": retained_dummy,
                "real_events_outside_first_32": source_count - retained_real,
                "perturbation_generated_before_trace_application": True,
                "perturbation_vector_count": injection[
                    "perturbation_vector_count"
                ],
                "perturbation_pool_rows": injection["perturbation_pool_rows"],
                "perturbation_start_index": injection[
                    "perturbation_start_index"
                ],
                "perturbation_rows_consumed": injection[
                    "perturbation_rows_consumed"
                ],
                "perturbation_pool_wrapped": injection[
                    "perturbation_pool_wrapped"
                ],
                "runtime_uses_device_label": True,
                "max_real_padding": 0.0,
                "max_real_delay": 0.0,
            }
        )
    return (
        out_lengths,
        out_gaps,
        out_directions,
        out_mask,
        stats,
        audit_rows,
    )


def apply_policy(
    policy,
    lengths,
    gaps,
    directions,
    mask,
    labels,
    *,
    base_seed=SEED,
):
    """Apply the unmodified aligned iPET policy without a global flow budget."""

    return _apply_policy(
        policy,
        lengths,
        gaps,
        directions,
        mask,
        labels,
        base_seed=base_seed,
    )


def apply_shared_budget_policy(policy, lengths, gaps, directions, mask, labels, *, shared_budget, base_seed=SEED):
    from experiment_support.budget import PhysicalBudget
    budget = PhysicalBudget(**shared_budget)
    recorded = policy.training_metadata.get("training_history", {}).get("shared_budget")
    if recorded != budget.to_dict():
        raise ValueError("iPET policy was not trained for this shared budget; retrain")
    return _apply_policy(policy, lengths, gaps, directions, mask, labels,
        base_seed=base_seed, max_added_bytes=budget.max_added_bytes,
        max_dummy_packets=budget.max_added_packets,
        max_wire_delay_seconds=budget.max_wire_delay_seconds,
        reported_method="iPET-physical-v5-shared-budget-v1")


def apply_budgeted_policy_v3(
    policy,
    lengths,
    gaps,
    directions,
    mask,
    labels,
    *,
    max_added_bytes,
    max_dummy_packets,
    base_seed=SEED,
):
    """Apply iPET proposals under a causal, absolute per-flow hard budget.

    The generator and its proposal order are unchanged.  Dummy events are
    considered chronologically and emitted only when both remaining budgets can
    accommodate the complete frame.  Real events are never removed or changed.
    """

    byte_limit, packet_limit = _validate_budgeted_v3_limits(
        max_added_bytes, max_dummy_packets
    )
    application = _apply_policy(
        policy,
        lengths,
        gaps,
        directions,
        mask,
        labels,
        base_seed=base_seed,
        max_added_bytes=byte_limit,
        max_dummy_packets=packet_limit,
        reported_method=IPET_BUDGETED_V3_METHOD,
    )
    stats = application[4]
    if np.any(stats["dummy_bytes"] > byte_limit + 1e-9):
        raise AssertionError("iPET-physical-v5-budgeted违反逐流新增字节硬预算")
    if np.any(stats["dummy_count"] > packet_limit):
        raise AssertionError("iPET-physical-v5-budgeted违反逐流dummy包硬预算")
    return application


def summarize_application(application):
    rows = application[5]
    return {
        "04_compatible_overhead": common.summarize_overhead(application[4]),
        "stats_keys_match_04": set(application[4]) == set(common.STAT_KEYS),
        "flows": len(rows),
        "source_lookahead_events": int(
            max((row["source_lookahead_events"] for row in rows), default=0)
        ),
        "network_real_order_preserved": bool(
            all(row["network_real_order_preserved"] for row in rows)
        ),
        "runtime_uses_device_label": True,
        "active_directions": rows[0]["active_directions"] if rows else [],
        "max_dummy_packets_observed": int(
            max((row["dummy_packets"] for row in rows), default=0)
        ),
        "perturbation_vector_count": (
            int(rows[0]["perturbation_vector_count"]) if rows else 0
        ),
        "perturbation_pool_rows": (
            int(rows[0]["perturbation_pool_rows"]) if rows else 0
        ),
        "any_perturbation_pool_wrap": bool(
            any(row["perturbation_pool_wrapped"] for row in rows)
        ),
    }


def save_policy(path, policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "metadata_json": np.asarray(
            json.dumps(policy.to_dict(), ensure_ascii=False), dtype=np.str_
        )
    }
    for class_index, label in enumerate(policy.class_labels):
        for name, value in policy.generator_states[label].items():
            safe_name = name.replace(".", "__dot__")
            arrays[f"generator_{class_index}__{safe_name}"] = (
                value.detach().cpu().numpy()
            )
    np.savez_compressed(path, **arrays)


def load_policy(path):
    with np.load(path, allow_pickle=False) as content:
        metadata = json.loads(str(content["metadata_json"]))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "iPET execution changed to physical slots v5; "
                "legacy weights must be retrained"
            )
        semantics = metadata.get("runtime_semantics", {})
        if semantics.get("physical_slot_projection") != PHYSICAL_RUNTIME_MARKER:
            raise ValueError(
                "iPET策略缺少physical-v5运行时语义标记；"
                "旧权重不能复用，必须使用新output重新训练"
            )
        policy_classes = {
            IPET_METHOD: OfficialCodeIPETPolicy,
            IPET_OUTBOUND_METHOD: OutboundOnlyOfficialCodeIPETPolicy,
        }
        policy_class = policy_classes.get(metadata.get("method"))
        if policy_class is None:
            raise ValueError("策略文件不是iPET PyTorch v2-aligned格式")
        states = {}
        for class_index, label in enumerate(metadata["class_labels"]):
            prefix = f"generator_{class_index}__"
            state = {}
            for key in content.files:
                if key.startswith(prefix):
                    name = key[len(prefix) :].replace("__dot__", ".")
                    state[name] = torch.from_numpy(content[key].copy())
            states[_python_scalar(label)] = state
    runtime_semantics = metadata.get("runtime_semantics", {})
    if runtime_semantics.get("generator_feature_mask") not in (
        None,
        list(policy_class.feature_mask),
    ):
        raise ValueError("iPET策略文件的生成通道约束与方法类型不一致")
    if runtime_semantics.get("dummy_directions") not in (
        None,
        list(policy_class.dummy_directions),
    ):
        raise ValueError("iPET策略文件的dummy方向约束与方法类型不一致")
    policy = policy_class(
        metadata["class_labels"],
        states,
        slot_count=metadata["slot_count"],
        max_packets=metadata["max_packets"],
        max_payload=metadata["max_payload"],
        seed=metadata["seed"],
        perturbation_vector_count=runtime_semantics[
            "perturbation_vectors_per_device"
        ],
        generation_batch_size=metadata.get(
            "generation_batch_size", DEFAULT_GENERATION_BATCH_SIZE
        ),
        training_metadata=metadata.get("training", {}),
    )
    if policy.weights_sha256 != metadata["generator_weights_sha256"]:
        raise ValueError("iPET generator权重摘要不匹配")
    return policy


__all__ = [
    "DEFAULT_DISCRIMINATOR_EPOCHS",
    "DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS",
    "DEFAULT_GENERATOR_EPOCHS",
    "DEFAULT_TRAINING_STAGES",
    "IPET_METHOD",
    "IPET_OUTBOUND_METHOD",
    "IPET_BUDGETED_V3_METHOD",
    "PHYSICAL_RUNTIME_MARKER",
    "OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE",
    "OFFICIAL_IPET_BATCH_SIZE",
    "OFFICIAL_NADAM_EPS",
    "OFFICIAL_NADAM_LR",
    "OFFICIAL_PERTURBATION_VECTOR_COUNT",
    "OUTBOUND_REPRODUCTION_NOTICE",
    "OutboundOnlyOfficialCodeIPETPolicy",
    "REPRODUCTION_NOTICE",
    "OfficialCodeIPETPolicy",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "apply_policy",
    "apply_budgeted_policy_v3",
    "apply_shared_budget_policy",
    "fit_baseline",
    "fit_outbound_only_baseline",
    "inject_official_ipet_trace",
    "load_policy",
    "save_policy",
    "summarize_application",
    "validate_dataset",
]
