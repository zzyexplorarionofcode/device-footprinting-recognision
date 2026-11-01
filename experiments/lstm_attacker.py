"""LSTM attacker for the existing signed-length/log-gap features.

The runner fits its usual StandardScaler on training data only. This estimator
receives those scaled [N, 2L] features and pairs column t with column L+t. It
receives the common first-gap-zero observer features and does not reinterpret zeros as padding.
All L positions, including the existing padded positions, remain observable.
"""

# =============================================================================
# 中文阅读导图：第四个攻击者（LSTM），与 MLP/RF/SVM 共用同一套输入协议
# =============================================================================
# 输入不是原始序列，而是 run_experiment 已经算好的 2L 维特征：
#   前 L 维 = 带符号归一化包长，后 L 维 = log1p(毫秒 gap)，都已过 StandardScaler。
# 本文件把它重排成 [N, L, 2] 的序列：第 t 个时间步是 (长度_t, gap_t)。
#
# 关键点：
#   * LSTMConfig：隐藏维/层数/轮数/批大小/学习率/类别权重/设备（cpu|cuda|auto），
#     __post_init__ 在构造期做完整校验，训练循环里不再重复判断；
#   * _SequenceClassifier：单层/多层 LSTM + 线性读出头，取最后一层最后时刻的隐状态；
#   * LSTMAttacker：继承 ClassifierMixin/BaseEstimator，因此可被 joblib 保存、
#     可进入统一评测循环（fit/predict/predict_proba 三个方法）；
#   * fit() 里用 torch.random.fork_rng(devices=[]) 只种子化 CPU 生成器，
#     不污染 CUDA 全局随机状态；shuffle 用私有 NumPy RNG，别人无法干扰；
#     _cpu_threads 是 contextmanager，异常路径也会把线程数恢复；
#   * __getstate__ 保存前把权重复制到 CPU，避免在纯 CPU 机器上反序列化 CUDA 张量失败。
#
# 注意：训练是固定轮数、全训练集、无测试集早停；类别权重默认 balanced。
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn


@dataclass(frozen=True)
class LSTMConfig:
    """Starting hyperparameters, not selected using any published test result."""

    hidden_size: int = 64
    num_layers: int = 1
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_clip: float = 5.0
    class_weight: str | None = "balanced"
    torch_threads: int = 1
    device: str = "cpu"

    def resolved_device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("LSTM device='cuda' requires a CUDA-enabled PyTorch runtime")
        return self.device

    def __post_init__(self):
        for name in ("hidden_size", "num_layers", "max_epochs", "batch_size", "torch_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"invalid {name}: {value}")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be 'cpu', 'cuda' or 'auto'")

    def metadata(self):
        return {
            **asdict(self), "device": self.resolved_device(),
            "requested_device": self.device, "bidirectional": False,
            "input_channels": ["signed_normalized_length", "log_gap"],
            "input_scaling": "runner StandardScaler fitted on full training data only",
            "padding": "retain all positions from the common fixed observation window",
            "first_gap": "zero_relative_to_first_observed_packet", "readout": "last layer final hidden state",
            "optimizer": "Adam", "loss": "cross_entropy",
            "training": "fixed epochs; all training rows; no test-set early stopping",
        }


def sequence_features(features, expected_width=None):
    """[length_0..length_L-1, gap_0..gap_L-1] -> [N,L,2]."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[1] % 2:
        raise ValueError("LSTM input must have shape [N, 2L] with L >= 1")
    if expected_width is not None and values.shape[1] != expected_width:
        raise ValueError("prediction feature width differs from training")
    if not np.isfinite(values).all():
        raise ValueError("LSTM input contains NaN or infinity")
    length = values.shape[1] // 2
    return np.stack((values[:, :length], values[:, length:]), axis=-1)


class _SequenceClassifier(nn.Module):
    def __init__(self, config, class_count):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2, hidden_size=config.hidden_size,
            num_layers=config.num_layers, batch_first=True, device="cpu", dtype=torch.float32,
        )
        self.output = nn.Linear(config.hidden_size, class_count, device="cpu", dtype=torch.float32)

    def forward(self, features):
        _, (hidden, _) = self.lstm(features)
        return self.output(hidden[-1])


@contextmanager
def _cpu_threads(count):
    previous = torch.get_num_threads()
    try:
        if previous != count:
            torch.set_num_threads(count)
        yield
    finally:
        if torch.get_num_threads() != previous:
            torch.set_num_threads(previous)


class LSTMAttacker(ClassifierMixin, BaseEstimator):
    """fit/predict/predict_proba classifier; joblib preserves learned weights.

CPU initialization is inside fork_rng and shuffling uses a private NumPy RNG.
Training and inference restore the caller's thread count. No other attacker's
or defense's random stream is consumed. Intended for sequential runner calls.
"""

    def __init__(self, *, config=None, seed=42):
        self.config = config
        self.seed = seed

    def __getstate__(self):
        state = super().__getstate__().copy()
        if "model_" in state:
            # CUDA storages cannot be unpickled on CPU-only hosts. Copy rather
            # than moving the live estimator, which may still be in use.
            state["model_"] = deepcopy(state["model_"]).cpu()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if "model_" in state:
            # auto may fall back to CPU; an explicit cuda request still fails
            # clearly when CUDA is unavailable.
            device = self.config_.resolved_device()
            self.model_.to(device)
            self.device_ = device

    def fit(self, features, labels):
        config = self.config if self.config is not None else LSTMConfig()
        if not isinstance(config, LSTMConfig):
            raise TypeError("config must be an LSTMConfig")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        sequence = sequence_features(features)
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(sequence) or not len(labels):
            raise ValueError("labels must contain one value per nonempty training row")
        if labels.dtype.kind not in "iu":
            raise ValueError("device labels must be integers")
        classes, encoded = np.unique(labels, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("LSTM classification requires at least two training classes")
        rng = np.random.default_rng(int(self.seed))
        device = torch.device(config.resolved_device())
        inputs = torch.from_numpy(sequence)
        targets = torch.from_numpy(encoded.astype(np.int64))
        weight = None
        if config.class_weight == "balanced":
            counts = np.bincount(encoded, minlength=len(classes))
            weight = torch.tensor(len(labels) / (len(classes) * counts), dtype=torch.float32)
        losses = []
        # Seed only the CPU generator; torch.manual_seed would also touch CUDA.
        with _cpu_threads(config.torch_threads), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(self.seed))
            # Initialize on CPU to preserve the seed/initial weights across backends.
            model = _SequenceClassifier(config, len(classes)).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            )
            criterion = nn.CrossEntropyLoss(
                weight=None if weight is None else weight.to(device), reduction="none")
            model.train()
            for _ in range(config.max_epochs):
                total_loss = 0.0
                indices = rng.permutation(len(labels))
                for start in range(0, len(labels), config.batch_size):
                    batch = torch.from_numpy(indices[start:start + config.batch_size].astype(np.int64, copy=False))
                    optimizer.zero_grad(set_to_none=True)
                    # Keep the full dataset in host RAM; GPU memory is bounded by a batch.
                    loss = criterion(model(inputs[batch].to(device)), targets[batch].to(device)).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite LSTM training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(batch)
                losses.append(total_loss / len(labels))
            model.eval()
        self.config_ = config
        self.device_ = str(device)
        self.classes_ = classes
        self.n_features_in_ = sequence.shape[1] * 2
        self.n_samples_seen_ = len(labels)
        self.n_iter_ = config.max_epochs
        self.loss_curve_ = losses
        self.model_ = model
        return self

    def predict_proba(self, features):
        check_is_fitted(self, ("model_", "classes_", "config_"))
        sequence = sequence_features(features, self.n_features_in_)
        if len(sequence) == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float32)
        outputs = []
        device = next(self.model_.parameters()).device
        with _cpu_threads(self.config_.torch_threads), torch.inference_mode():
            for start in range(0, len(sequence), self.config_.batch_size):
                batch = torch.from_numpy(sequence[start:start + self.config_.batch_size])
                outputs.append(self.model_(batch.to(device)).softmax(dim=1).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]
