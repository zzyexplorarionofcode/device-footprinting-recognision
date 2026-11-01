"""Adapted physical-v5 PyTorch iPET model and training code.

This module is the maintained implementation; the earlier PyTorch port has
been removed. The physical projection and other adaptation semantics are
documented in the sibling README.
"""

from __future__ import annotations

from math import ceil

import numpy as np
import torch
from torch import nn


LATENT_DIM = 20
MAX_PACKETS_PER_OMEGA = 4.0
MAX_PAYLOAD_PER_OMEGA = 100.0
DEFAULT_TRAINING_STAGES = 5
DEFAULT_GENERATOR_EPOCHS = 3
DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS = 2
DEFAULT_DISCRIMINATOR_EPOCHS = 100

# Keras Model.fit defaults to 32 when batch_size is omitted.  The authors omit
# it in both generator fitting and discriminator re-training.
OFFICIAL_IPET_BATCH_SIZE = 32
OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE = 128

# TensorFlow 2.5 tf.keras.optimizers.Nadam defaults used by optimizer='nadam'.
OFFICIAL_NADAM_LR = 0.001
OFFICIAL_NADAM_EPS = 1e-7
OFFICIAL_NADAM_BETAS = (0.9, 0.999)
OFFICIAL_NADAM_MOMENTUM_DECAY = 0.004

# ipet-pert-add/ipet_pert_add.py sets num_samples = int(2e4).
OFFICIAL_PERTURBATION_VECTOR_COUNT = 20_000
DEFAULT_GENERATION_BATCH_SIZE = 2_048


def physical_slot_projection(values, *, straight_through=False):
    """Count/bytes actually emitted by an Ethernet slot (v5 execution semantics).

    Both discriminator training and pool generation consume this projection.
    The STE changes derivatives only; the forward values are integral packets/bytes.
    A zero count is a true no-op. Active directions are already masked by G.
    """
    pairs=values.reshape(*values.shape[:-1],2,2)
    proposed_count=pairs[...,0].clamp(min=0)
    rounded_count=torch.round(proposed_count)
    count=proposed_count+(rounded_count-proposed_count).detach() if straight_through else rounded_count
    frame=(pairs[...,1].clamp(min=0)/count.clamp(min=1)).clamp(min=60,max=1514)
    integer_frame=torch.round(frame)
    if straight_through:integer_frame=frame+(integer_frame-frame).detach()
    total=count*integer_frame
    return torch.stack((count,total),dim=-1).reshape_as(values)


def budget_training_projection(values, budget=None, slot_mask=None, *, straight_through=False):
    """Same chronological complete-frame admission as runtime, in M_Sequence space.

    Pool generation intentionally retains native proposals: a flow may span several
    vectors, and its budget must never reset at a vector boundary. The generator
    loss and discriminator retraining see admitted perturbations, not the pool.
    """
    projected = physical_slot_projection(values, straight_through=straight_through)
    if budget is None:
        return projected
    from experiment_support.budget import PhysicalBudget
    limits = PhysicalBudget(**budget)
    if slot_mask is not None:
        projected = projected * slot_mask
    pairs = projected.reshape(len(values), -1, 2)
    remaining_b = torch.full((len(values),), float(limits.max_added_bytes), device=values.device, dtype=values.dtype)
    remaining_n = torch.full_like(remaining_b, float(limits.max_added_packets))
    accepted = []
    for pair in pairs.unbind(dim=1):
        count, total = pair[:, 0], pair[:, 1]
        frame = total / count.clamp(min=1)
        fit = torch.floor((remaining_b / frame.clamp(min=60) + 1e-7)).clamp(min=0)
        used = torch.minimum(count, torch.minimum(remaining_n, fit))
        accepted.append(torch.stack((used, used * frame), dim=-1))
        remaining_b = (remaining_b - used * frame).clamp(min=0)
        remaining_n = (remaining_n - used).clamp(min=0)
    admitted = torch.stack(accepted, dim=1).reshape_as(projected)
    # Same STE convention as physical rounding; exact admitted forward values.
    return projected + (admitted - projected).detach() if straight_through else admitted


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _print_training_progress(label: str, completed: int, total: int, loss: float) -> None:
    """Render a compact terminal progress bar without changing training state."""

    interval = max(1, int(total) // 20)
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


class OfficialMSequenceDiscriminator(nn.Module):
    """Authors' default M_Sequence discriminator architecture."""

    def __init__(self, feature_count: int, class_count: int):
        super().__init__()
        self.feature_count = int(feature_count)
        self.class_count = int(class_count)
        self.bilstm_1 = nn.LSTM(
            input_size=self.feature_count,
            hidden_size=8,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_2 = nn.LSTM(
            input_size=16,
            hidden_size=8,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(17, 128)
        self.dropout = nn.Dropout(0.2)
        self.softmax_logits = nn.Linear(128, self.class_count)

    def forward(self, main_features: torch.Tensor, silence: torch.Tensor):
        sequence, _ = self.bilstm_1(main_features)
        _, (hidden, _) = self.bilstm_2(sequence)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=1)
        silence = silence.reshape(-1, 1)
        combined = torch.cat((encoded, silence), dim=1)
        combined = torch.tanh(self.fc(combined))
        combined = self.dropout(combined)
        return self.softmax_logits(combined)


class OfficialIPETGenerator(nn.Module):
    """Authors' Dense -> sigmoid -> reshape -> UpScale generator."""

    def __init__(
        self,
        slot_count: int,
        feature_count: int = 4,
        max_packets: float = MAX_PACKETS_PER_OMEGA,
        max_payload: float = MAX_PAYLOAD_PER_OMEGA,
        feature_mask=(1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        if feature_count != 4:
            raise ValueError("iPET slot必须恰好包含四个方向/计数特征")
        self.slot_count = int(slot_count)
        self.feature_count = int(feature_count)
        self.max_packets = float(max_packets)
        self.max_payload = float(max_payload)
        feature_mask = np.asarray(feature_mask, dtype=np.float32)
        if feature_mask.shape != (self.feature_count,):
            raise ValueError("iPET feature_mask必须包含四个特征通道")
        if np.any((feature_mask < 0.0) | (feature_mask > 1.0)):
            raise ValueError("iPET feature_mask必须位于[0, 1]")
        self.adversarial_noise = nn.Linear(
            LATENT_DIM, self.slot_count * self.feature_count, bias=True
        )
        scale = torch.tensor(
            [self.max_packets, self.max_payload] * 2,
            dtype=torch.float32,
        ) * torch.from_numpy(feature_mask)
        self.register_buffer(
            "upscale_multiplier", scale.reshape(1, 1, self.feature_count)
        )

    def forward(self, noise: torch.Tensor):
        perturbations = torch.sigmoid(self.adversarial_noise(noise))
        perturbations = perturbations.reshape(
            -1, self.slot_count, self.feature_count
        )
        return perturbations * self.upscale_multiplier


def _iterate_minibatches(
    features: torch.Tensor,
    silences: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    order = torch.randperm(len(features), generator=generator)
    for start in range(0, len(order), int(batch_size)):
        indices = order[start : start + int(batch_size)]
        yield features[indices], silences[indices], labels[indices]


def _official_discriminator_loss(discriminator, logits, labels):
    loss = nn.functional.cross_entropy(logits, labels)
    input_kernel_penalty = (
        discriminator.bilstm_1.weight_ih_l0.square().sum()
        + discriminator.bilstm_1.weight_ih_l0_reverse.square().sum()
    )
    return loss + 0.01 * input_kernel_penalty


def _optimizer_step_range(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    steps = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            steps.append(int(step.item() if hasattr(step, "item") else step))
    return (min(steps), max(steps)) if steps else (0, 0)


def _train_discriminator_epochs(
    discriminator: OfficialMSequenceDiscriminator,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    silences: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    seed: int,
    batch_size: int = OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE,
    progress_label: str | None = None,
):
    """Train D without recreating its optimizer.

    Passing the optimizer in is important: Adam moments must survive every
    iPET stage, matching the single loaded discriminator in the TF program.
    """

    discriminator.train()
    device = next(discriminator.parameters()).device
    rng = torch.Generator().manual_seed(int(seed))
    losses = []
    for epoch in range(int(epochs)):
        epoch_losses = []
        for batch_features, batch_silences, batch_labels in _iterate_minibatches(
            features, silences, labels, batch_size, rng
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = discriminator(batch_features.to(device), batch_silences.to(device))
            loss = _official_discriminator_loss(
                discriminator, logits, batch_labels.to(device)
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        if progress_label is not None:
            _print_training_progress(progress_label, epoch + 1, epochs, losses[-1])
    return losses


def _train_generator_epochs(
    generator: OfficialIPETGenerator,
    discriminator: OfficialMSequenceDiscriminator,
    features: torch.Tensor,
    silences: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    seed: int,
    batch_size: int = OFFICIAL_IPET_BATCH_SIZE,
    progress_label: str | None = None,
    budget=None,
    slot_masks=None,
):
    """Train G while freezing D weights but retaining D's active Dropout."""

    # Keras model.fit keeps Dropout in training mode even though the nested
    # discriminator has trainable=False.  Therefore use train(), not eval().
    discriminator.train()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    generator.train()
    device = next(generator.parameters()).device
    optimizer = torch.optim.NAdam(
        generator.parameters(),
        lr=OFFICIAL_NADAM_LR,
        betas=OFFICIAL_NADAM_BETAS,
        eps=OFFICIAL_NADAM_EPS,
        momentum_decay=OFFICIAL_NADAM_MOMENTUM_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()
    rng = torch.Generator().manual_seed(int(seed))
    noise_rng = torch.Generator().manual_seed(int(seed) + 1)
    losses = []
    try:
        for epoch in range(int(epochs)):
            order = torch.randperm(len(features), generator=rng)
            epoch_losses = []
            for start in range(0, len(order), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                batch_features = features[indices].to(device)
                batch_silences = silences[indices].to(device)
                batch_labels = labels[indices].to(device)
                noise = torch.randn(
                    len(indices),
                    LATENT_DIM,
                    generator=noise_rng,
                    dtype=torch.float32,
                )
                optimizer.zero_grad(set_to_none=True)
                perturbed = batch_features + budget_training_projection(
                    generator(noise.to(device)), budget,
                    None if slot_masks is None else slot_masks[indices].to(device),
                    straight_through=True)
                logits = discriminator(perturbed, batch_silences)
                loss = -loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            losses.append(float(np.mean(epoch_losses)))
            if progress_label is not None:
                _print_training_progress(progress_label, epoch + 1, epochs, losses[-1])
    finally:
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
    return losses


def generate_perturbation_pool(
    generator: OfficialIPETGenerator,
    seed: int,
    *,
    vector_count: int = OFFICIAL_PERTURBATION_VECTOR_COUNT,
    generation_batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
) -> np.ndarray:
    """Generate and concatenate the official number of perturbation vectors.

    The returned array has ``vector_count * slot_count`` rows and four columns,
    exactly like upstream ``predict.py`` followed by ``reshape(-1, 4)``.
    Batched inference bounds temporary memory without changing the method.
    """

    vector_count = int(vector_count)
    generation_batch_size = int(generation_batch_size)
    if vector_count < 1 or generation_batch_size < 1:
        raise ValueError("vector_count和generation_batch_size必须为正整数")
    generator.eval()
    device = next(generator.parameters()).device
    noise_rng = torch.Generator().manual_seed(int(seed))
    outputs = []
    with torch.no_grad():
        for start in range(0, vector_count, generation_batch_size):
            count = min(generation_batch_size, vector_count - start)
            noise = torch.randn(
                count, LATENT_DIM, generator=noise_rng, dtype=torch.float32
            )
            outputs.append(physical_slot_projection(generator(noise.to(device))).cpu())
    perturbations = torch.cat(outputs, dim=0)
    return perturbations.reshape(-1, generator.feature_count).numpy()


def generate_perturbation(
    generator: OfficialIPETGenerator,
    seed: int,
) -> np.ndarray:
    """Compatibility helper for one 40-row vector; runtime uses the pool API."""

    return generate_perturbation_pool(generator, seed, vector_count=1)


def train_official_ipet(
    features: np.ndarray,
    silences: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    discriminator_epochs: int = DEFAULT_DISCRIMINATOR_EPOCHS,
    training_stages: int = DEFAULT_TRAINING_STAGES,
    generator_epochs: int = DEFAULT_GENERATOR_EPOCHS,
    discriminator_retrain_epochs: int = DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS,
    max_packets: float = MAX_PACKETS_PER_OMEGA,
    max_payload: float = MAX_PAYLOAD_PER_OMEGA,
    feature_mask=(1.0, 1.0, 1.0, 1.0),
    progress: bool = False,
    device: str = "cpu",
    budget=None,
    slot_masks=None,
):
    """Train the stage-wise device-specific iPET generators."""

    if device not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be 'cpu', 'cuda' or 'auto'")
    requested_device = device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("iPET device='cuda' requires a CUDA-enabled PyTorch runtime")
    features = np.asarray(features, dtype=np.float32)
    silences = np.asarray(silences, dtype=np.float32).reshape(-1, 1)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 3 or features.shape[2] != 4:
        raise ValueError("iPET features必须具有[样本, slot, 4]形状")
    if len(features) != len(silences) or len(features) != len(labels):
        raise ValueError("iPET features/silences/labels数量不一致")
    if min(
        discriminator_epochs,
        training_stages,
        generator_epochs,
        discriminator_retrain_epochs,
    ) < 1:
        raise ValueError("iPET所有训练轮数必须为正整数")

    class_labels = np.unique(labels)
    if len(class_labels) < 2:
        raise ValueError("iPET判别器至少需要两个设备类别")
    label_to_index = {
        value.item() if hasattr(value, "item") else value: index
        for index, value in enumerate(class_labels)
    }
    encoded = np.asarray(
        [
            label_to_index[value.item() if hasattr(value, "item") else value]
            for value in labels
        ],
        dtype=np.int64,
    )

    _set_seed(seed)
    tensor_features = torch.from_numpy(features)
    tensor_silences = torch.from_numpy(silences)
    tensor_labels = torch.from_numpy(encoded)
    tensor_slot_masks = None if slot_masks is None else torch.as_tensor(slot_masks, dtype=torch.float32)
    discriminator = OfficialMSequenceDiscriminator(4, len(class_labels)).to(device)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    initial_losses = _train_discriminator_epochs(
        discriminator,
        discriminator_optimizer,
        tensor_features,
        tensor_silences,
        tensor_labels,
        discriminator_epochs,
        seed + 100,
        batch_size=OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE,
        progress_label="[iPET][initial discriminator]" if progress else None,
    )
    optimizer_steps_after_initial = _optimizer_step_range(discriminator_optimizer)

    stage_history = []
    final_generators = {}
    for stage in range(int(training_stages)):
        if progress:
            print(
                f"[iPET][stage {stage + 1}/{training_stages}] 训练 "
                f"{len(class_labels)} 个设备生成器",
                flush=True,
            )
        generators = {}
        generator_losses = {}
        for class_index, class_label in enumerate(class_labels):
            torch.manual_seed(int(seed) + 1000 * (stage + 1) + class_index)
            generator = OfficialIPETGenerator(
                features.shape[1],
                max_packets=max_packets,
                max_payload=max_payload,
                feature_mask=feature_mask,
            ).to(device)
            class_rows = tensor_labels == class_index
            generator_losses[str(class_label)] = _train_generator_epochs(
                generator,
                discriminator,
                tensor_features[class_rows],
                tensor_silences[class_rows],
                tensor_labels[class_rows],
                generator_epochs,
                seed + 2000 * (stage + 1) + class_index,
                budget=budget,
                slot_masks=None if tensor_slot_masks is None else tensor_slot_masks[class_rows],
                progress_label=(
                    None
                    if not progress
                    else f"[iPET][stage {stage + 1}/{training_stages}] "
                    f"generator device {class_index + 1}/{len(class_labels)}: {class_label}"
                ),
            )
            key = class_label.item() if hasattr(class_label, "item") else class_label
            generators[key] = generator

        perturbed_batches = []
        perturbed_silences = []
        perturbed_labels = []
        for class_index, class_label in enumerate(class_labels):
            key = class_label.item() if hasattr(class_label, "item") else class_label
            rows = tensor_labels == class_index
            count = int(rows.sum())
            noise_rng = torch.Generator().manual_seed(
                int(seed) + 3000 * (stage + 1) + class_index
            )
            noise = torch.randn(count, LATENT_DIM, generator=noise_rng)
            generators[key].eval()
            with torch.no_grad():
                # Generate in the same host RNG call as before, but bound device memory.
                chunks = []
                row_masks = None if tensor_slot_masks is None else tensor_slot_masks[rows]
                if budget is None:
                    # Preserve the native CPU/GPU batching behavior verbatim.
                    perturbations = torch.cat([
                        physical_slot_projection(generators[key](chunk.to(device))).cpu()
                        for chunk in noise.split(OFFICIAL_IPET_BATCH_SIZE)
                    ]) if device != "cpu" else physical_slot_projection(generators[key](noise))
                else:
                    for start in range(0, count, OFFICIAL_IPET_BATCH_SIZE):
                        end = start + OFFICIAL_IPET_BATCH_SIZE
                        chunks.append(budget_training_projection(
                            generators[key](noise[start:end].to(device)), budget,
                            None if row_masks is None else row_masks[start:end].to(device)).cpu())
                    perturbations = torch.cat(chunks)
            perturbed_batches.append(tensor_features[rows] + perturbations)
            perturbed_silences.append(tensor_silences[rows])
            perturbed_labels.append(tensor_labels[rows])

        optimizer_steps_before_retrain = _optimizer_step_range(
            discriminator_optimizer
        )
        retrain_losses = _train_discriminator_epochs(
            discriminator,
            discriminator_optimizer,
            torch.cat(perturbed_batches),
            torch.cat(perturbed_silences),
            torch.cat(perturbed_labels),
            discriminator_retrain_epochs,
            seed + 4000 * (stage + 1),
            batch_size=OFFICIAL_IPET_BATCH_SIZE,
            progress_label=(
                None
                if not progress
                else f"[iPET][stage {stage + 1}/{training_stages}] discriminator retrain"
            ),
        )
        optimizer_steps_after_retrain = _optimizer_step_range(discriminator_optimizer)
        final_generators = generators
        stage_history.append(
            {
                "stage": stage,
                "generator_negative_cross_entropy": generator_losses,
                "discriminator_retrain_cross_entropy": retrain_losses,
                "discriminator_optimizer_steps_before_retrain": list(
                    optimizer_steps_before_retrain
                ),
                "discriminator_optimizer_steps_after_retrain": list(
                    optimizer_steps_after_retrain
                ),
                "discriminator_dropout_active_during_generator_training": True,
            }
        )

    return {
        "class_labels": tuple(
            value.item() if hasattr(value, "item") else value
            for value in class_labels
        ),
        "generators": {
            label: {key: value.detach().cpu().clone() for key, value in generator.state_dict().items()}
            for label, generator in final_generators.items()
        },
        "training": {
            "shared_budget": budget,
            "budget_training_domain": "existing first-4-second M_Sequence proxy; runtime audits complete prefix",
            "device": device,
            "requested_device": requested_device,
            "initial_discriminator_cross_entropy": initial_losses,
            "initial_discriminator_optimizer_steps": list(
                optimizer_steps_after_initial
            ),
            "stages": stage_history,
            "alignment": {
                "initial_discriminator_batch_size": (
                    OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE
                ),
                "generator_batch_size": OFFICIAL_IPET_BATCH_SIZE,
                "discriminator_retrain_batch_size": OFFICIAL_IPET_BATCH_SIZE,
                "nadam_lr": OFFICIAL_NADAM_LR,
                "nadam_eps": OFFICIAL_NADAM_EPS,
                "discriminator_dropout_active_for_generator": True,
                "discriminator_adam_reused_across_stages": True,
                "official_perturbation_vector_count": (
                    OFFICIAL_PERTURBATION_VECTOR_COUNT
                ),
                "expected_flattened_pool_rows": (
                    OFFICIAL_PERTURBATION_VECTOR_COUNT * features.shape[1]
                ),
                "discriminator_batches_per_epoch": ceil(
                    len(features) / OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE
                ),
            },
        },
    }


__all__ = [
    "DEFAULT_DISCRIMINATOR_EPOCHS",
    "DEFAULT_DISCRIMINATOR_RETRAIN_EPOCHS",
    "DEFAULT_GENERATION_BATCH_SIZE",
    "DEFAULT_GENERATOR_EPOCHS",
    "DEFAULT_TRAINING_STAGES",
    "LATENT_DIM",
    "MAX_PACKETS_PER_OMEGA",
    "MAX_PAYLOAD_PER_OMEGA",
    "OFFICIAL_IPET_BATCH_SIZE",
    "OFFICIAL_INITIAL_DISCRIMINATOR_BATCH_SIZE",
    "OFFICIAL_NADAM_EPS",
    "OFFICIAL_NADAM_LR",
    "OFFICIAL_PERTURBATION_VECTOR_COUNT",
    "OfficialIPETGenerator",
    "OfficialMSequenceDiscriminator",
    "generate_perturbation",
    "generate_perturbation_pool",
    "train_official_ipet",
]
