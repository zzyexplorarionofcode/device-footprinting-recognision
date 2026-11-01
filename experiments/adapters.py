"""把现有防御实现接到统一的 ``check/fit/apply`` 接口。

为什么需要适配器
----------------
各方法本身的生命周期不同：Method32读取已选择配置；传统方法大多无需训练；
Classic Traffic Morphing要从训练集拟合转换分布；iPET和Glimpse各训练一套设备
策略；IoTGAN则必须为每种被查询识别器分别训练策略。如果把这些差异直接写进
主循环，运行器会迅速变成大量特殊分支。适配器将差异封装后，主循环始终只做：

``adapter.check(context)`` → ``adapter.fit(context)`` →
``adapter.apply(context, 'train'/'test')``。

边界约束
--------
本文件只传递配置、训练索引和随机种子，不重写算法动作，不替换损失函数，也不
为了开销匹配凭空添加delay、dummy或其他原方法没有的机制。任何非原生预算调整
都必须注册为独立、带版本的方法键，明确写入experiment_config.py并进入manifest。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from experiment_support.metrics import TIMING_KEYS
from experiment_support.budget import PhysicalBudget, PROTOCOL as BUDGET_PROTOCOL
from experiment_support.provenance import (
    policy_descriptor,
    validate_reuse,
    write_provenance,
)


# ---------------------------------------------------------------------------
# 稳定路径与延迟依赖
# ---------------------------------------------------------------------------
# 只有用户真正启用iPET或Glimpse时，才把对应独立目录加入sys.path并导入重型
# 依赖。这使仅运行Method32/传统方法时无需初始化TensorFlow-Agents。
PROJECT = Path(__file__).resolve().parents[1]
OUR_METHOD = PROJECT / "methods"
IPET_DIR = PROJECT / "baselines/ipet"
GLIMPSE_DIR = PROJECT / "baselines/glimpse"
for path in (PROJECT, OUR_METHOD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from methods import shared_tools  # noqa: E402
from baselines import common, traditional  # noqa: E402


def safe_name(value: str) -> str:
    """把用户显示名转换成跨平台安全、稳定的策略文件名。"""

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")


def file_sha256(path: Path) -> str:
    """把策略/配置摘要写进manifest，保证以后能确认输入未变。"""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _policy_stem(name: str) -> str:
    """Return a safe policy filename stem, rejecting empty names."""

    stem = safe_name(name)
    if not stem:
        raise ValueError(f"方法名 {name!r} 不能生成有效策略文件名")
    return stem


def _provenance_sources(module: Any) -> list[Path]:
    """Return only the explicit adapter and implementation source files."""

    paths = [PROJECT/name for name in (
        'experiments/adapters.py','experiments/run_experiment.py','experiments/lstm_attacker.py',
        'data_pipeline/pcap.py','data_pipeline/partitions.py','baselines/common.py','experiment_support/metrics.py',
        'experiment_support/provenance.py','experiment_support/budget.py','baselines/high_budget.py')]
    if 'iotgan' in str(getattr(module,'__file__','')).lower():
        paths += [PROJECT/'baselines/iotgan_aligned.py',PROJECT/'baselines/iotgan_aligned_bidirectional.py']
    module_path = getattr(module, "__file__", None)
    if module_path:
        candidate = Path(module_path).resolve()
        if candidate.is_file() and candidate not in paths:
            paths.append(candidate)
        # iPET exposes its public API from a package __init__, while defaults
        # and runtime code live in these explicit sibling modules.  Include
        # them when present without recursively discovering an unknown tree.
        if candidate.name == "__init__.py":
            for sibling_name in ("baseline.py", "model.py", "traffic.py", "core.py"):
                sibling = candidate.with_name(sibling_name)
                if sibling.is_file() and sibling not in paths:
                    paths.append(sibling)
    return paths


def _policy_seed(context: Mapping[str, Any], params: Mapping[str, Any]) -> int:
    return int(params.get("policy_seed", context["evaluation"]["policy_seed"]))


def _cache_params(
    params: Mapping[str, Any],
    fit_params: Mapping[str, Any],
    *,
    mode: str,
    policy_seed: int,
    target: str | None = None,
) -> dict[str, Any]:
    """Build semantic cache parameters, excluding the reuse control switch."""

    semantic = dict(params)
    semantic.pop("reuse_existing_policy", None)
    semantic["mode"] = mode
    semantic["policy_seed"] = int(policy_seed)
    semantic["fit"] = dict(fit_params)
    if target is not None:
        semantic["target"] = str(target)
    return semantic


def _policy_descriptor(
    context: Mapping[str, Any],
    method_key: str,
    params: Mapping[str, Any],
    module: Any,
    *,
    targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a cache descriptor from explicit context and source files."""

    return policy_descriptor(
        context,
        method_key,
        params,
        _provenance_sources(module),
        targets=targets,
    )


def _validate_cached_policy(
    path: Path,
    descriptor: Mapping[str, Any],
) -> None:
    """Fail closed before loading an existing policy artifact."""

    validate_reuse(path, descriptor)


def _write_policy_provenance(path: Path, descriptor: Mapping[str, Any]) -> None:
    write_provenance(path, descriptor)


def split_view(context: Mapping[str, Any], part: str):
    """取得outer-train或outer-test的四个逐包数组和标签。

    context只保存一份全局数组；本函数按预先冻结的索引切片。适配器不能自行
    重新划分数据，从结构上保证所有方法看到完全相同的样本身份。
    """

    indices = context[f"{part}_indices"]
    return tuple(array[indices] for array in context["arrays"]), context["labels"][indices]


class Adapter:
    """所有方法适配器的最小公共协议。

    ``target_specific=False`` 表示同一策略面对全部识别器；IoTGAN覆写为True。
    ``seed_offset`` 是用户显式设置的方法内随机流偏移，不依赖方法列表位置，
    所以增删另一个方法不会悄悄改变已有方法的随机实现。
    """

    target_specific = False

    def __init__(self, spec: Mapping[str, Any]):
        self.key = str(spec["key"])
        self.name = str(spec.get("name", self.key))
        self.params = dict(spec.get("params", {}))
        self.seed_offset = int(spec.get("seed_offset", 0))
        self.metadata: dict[str, Any] = {}

    def fit(self, context: Mapping[str, Any], output_dir: Path) -> None:
        """只允许使用outer-train；无需训练的方法保持空实现。"""

    def check(self, context: Mapping[str, Any]) -> None:
        """快速校验依赖和配置；不得启动模型训练或写文件。"""

    def apply(
        self,
        context: Mapping[str, Any],
        part: str,
        *,
        seed: int,
        target: str | None = None,
    ):
        raise NotImplementedError


class IdentityAdapter(Adapter):
    """无防御控制组：复制输入数组，并返回全零开销。"""

    def apply(self, context, part, *, seed, target=None):
        del seed, target
        arrays, _ = split_view(context, part)
        copied = tuple(np.asarray(value).copy() for value in arrays)
        stats=common.empty_stats(len(copied[0]))
        stats.update({key:np.zeros(len(copied[0])) for key in TIMING_KEYS})
        return (*copied, stats, {"identity": True}, [])




class Method33Adapter(Adapter):
    """读取已配置的Method33高预算策略并应用指定Trigger位置/码字映射。"""

    def check(self, context):
        from methods import method33_tools

        configs = self.params.get("configs_by_class")
        if configs is None:
            configured = self.params.get("config")
        else:
            configured = configs.get(context["class_count"], configs.get(str(context["class_count"])))
        if configured is None:
            raise ValueError(f"没有为{context['class_count']}分类提供Method33配置")
        config_path = Path(configured).resolve()
        self.project = method33_tools.load_config(config_path)
        method33_tools.validate_dataset_project_alignment(context["dataset"], self.project)
        expected = context["feature"]
        actual = self.project.dataset
        if not np.isclose(expected["max_frame_length"], actual.max_frame_length) or not np.isclose(
            expected["max_gap_seconds"], actual.max_feature_gap_seconds
        ):
            raise ValueError("统一特征归一化参数与Method33配置不一致")
        split = context["spec"]["split"]
        if split.get("unit", "capture") != "capture" or int(split["seed"]) != int(
            self.project.evaluation.split_seed
        ) or not np.isclose(
            float(split["train_ratio"]), self.project.evaluation.capture_train_ratio
        ):
            raise ValueError(
                "Method33配置是按特定capture划分选择的；通用实验必须使用同一train_ratio/split seed"
            )
        self.metadata = {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "runtime_protocol": json.loads(config_path.read_text(encoding="utf-8-sig")).get("runtime_protocol", {"parameter_status":"legacy_unbound"}),
            "adapter_changes_algorithm": False,
        }

    def fit(self, context, output_dir):
        del context, output_dir

    def apply(self, context, part, *, seed, target=None):
        del target
        from methods import method33_tools

        indices = context[f"{part}_indices"]
        flow_count = len(context["labels"])
        subset = {
            key: (
                value[indices]
                if key != "device_names" and isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == flow_count
                else value
            )
            for key, value in context["dataset"].items()
        }
        return method33_tools.apply_configured_method(
            subset,
            self.project,
            seed=seed,
            mapping=self.params.get(f"{part}_mapping", "old" if part == "train" else "next"),
            include_weakener=bool(self.params.get("include_weakener", True)),
            include_trigger=bool(self.params.get("include_trigger", True)),
        )


class TraditionalAdapter(Adapter):
    """传统方法统一入口。

    Random、Random+、Constant Rate和Random Segmentation不拟合策略；Classic
    Traffic Morphing例外，其长度转换分布只能由outer-train估计。``fit``和
    ``apply``参数分开保存，防止把拟合参数误传到运行阶段。
    """

    def check(self, context):
        del context
        self.variant = str(self.params["variant"])
        allowed = {
            "random", "random_plus", "constant_rate",
            "classic_traffic_morphing", "random_segmentation",
        }
        if self.variant not in allowed:
            raise ValueError(f"traditional.variant必须属于{sorted(allowed)}")

    def fit(self, context, output_dir):
        del output_dir
        self.policy = None
        if self.variant == "classic_traffic_morphing":
            fit_params = dict(self.params.get("fit", {}))
            self.policy = traditional.fit_classic_traffic_morphing(
                context["arrays"][0],
                context["arrays"][2],
                context["arrays"][3],
                context["labels"],
                training_indices=context["train_indices"],
                **fit_params,
            )
        self.metadata = {
            "variant": self.variant,
            "fit_uses_outer_train_only": True,
            "adapter_changes_algorithm": False,
        }

    def apply(self, context, part, *, seed, target=None):
        del target
        arrays, labels = split_view(context, part)
        kwargs = dict(self.params.get("apply", {}))
        if self.variant == "classic_traffic_morphing":
            kwargs.update(labels=labels, policy=self.policy)
        return traditional.apply_baseline(self.variant, *arrays, seed=seed, **kwargs)


class IPETAdapter(Adapter):
    """PyTorch v2-aligned iPET；默认仅单向，双向必须显式配置。

    fit得到每设备生成器和官方规模扰动池，随后立即保存并重新加载一次。这既
    检查策略可复现性，也确保正式应用使用的正是磁盘中被manifest记录的版本。
    适配器不会给iPET添加其原本没有的delay或额外dummy机制。
    """

    def check(self, context):
        del context
        if str(IPET_DIR) not in sys.path:
            sys.path.insert(0, str(IPET_DIR))
        import ipet_v2aligned as ipet

        self.module = ipet
        self.mode = str(self.params.get("mode", "outbound"))
        if self.mode not in {"outbound", "bidirectional"}:
            raise ValueError("iPET mode只能是outbound或bidirectional")

    def fit(self, context, output_dir):
        ipet = self.module
        fit = ipet.fit_outbound_only_baseline if self.mode == "outbound" else ipet.fit_baseline
        fit_params = dict(self.params.get("fit", {}))
        fit_params.setdefault("progress", True)
        # Cache provenance must distinguish CPU training from CUDA training.
        if fit_params.get("device") == "auto":
            import torch
            fit_params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        path = output_dir / "policies" / f"{_policy_stem(self.name)}.npz"
        policy_seed = _policy_seed(context, self.params)
        descriptor_params = _cache_params(
            self.params,
            fit_params,
            mode=self.mode,
            policy_seed=policy_seed,
        )
        descriptor = _policy_descriptor(context, self.key, descriptor_params, ipet)
        reuse_requested = bool(self.params.get("reuse_existing_policy", True))
        reused = reuse_requested and path.is_file()
        if reused:
            _validate_cached_policy(path, descriptor)
            self.policy = ipet.load_policy(path)
            print(f"[{self.name}] 复用已保存策略：{path.name}", flush=True)
        else:
            self.policy = fit(
                *context["arrays"],
                context["labels"],
                context["train_indices"],
                groups=context["groups"],
                seed=policy_seed,
                **fit_params,
            )
            ipet.save_policy(path, self.policy)
            self.policy = ipet.load_policy(path)
            _write_policy_provenance(path, descriptor)
            print(f"[{self.name}] 策略已保存并回读验证：{path.name}", flush=True)
        self.metadata = {
            "mode": self.mode,
            "policy": str(path.resolve()),
            "policy_sha256": file_sha256(path),
            "provenance": str(f"{path}.provenance.json"),
            "provenance_sha256": file_sha256(Path(f"{path}.provenance.json")),
            "reused_existing_policy": reused,
            "adapter_changes_algorithm": False,
        }

    def apply(self, context, part, *, seed, target=None):
        del target
        arrays, labels = split_view(context, part)
        return self.module.apply_policy(self.policy, *arrays, labels, base_seed=seed)


class IPETBudgetedV3Adapter(IPETAdapter):
    """iPET提议策略的逐流硬预算v3变体。

    训练过程和生成器与v2aligned完全相同；应用阶段按时间顺序接受能同时满足
    字节与dummy数量硬上限的完整帧。预算不会从测试集估计，必须显式写入配置。
    """

    def check(self, context):
        super().check(context)
        budget = dict(self.params.get("budget", {}))
        required = {"max_added_bytes", "max_dummy_packets"}
        missing = sorted(required - set(budget))
        if missing:
            raise ValueError(f"iPET-budgeted-v3缺少预算字段：{missing}")
        extra = sorted(set(budget) - required)
        if extra:
            raise ValueError(f"iPET-budgeted-v3未知预算字段：{extra}")
        byte_limit = float(budget["max_added_bytes"])
        packet_raw = budget["max_dummy_packets"]
        if not np.isfinite(byte_limit) or byte_limit < 0.0:
            raise ValueError("iPET-budgeted-v3 max_added_bytes必须是有限非负数")
        if isinstance(packet_raw, (bool, np.bool_)):
            raise ValueError("iPET-budgeted-v3 max_dummy_packets必须是非负整数")
        packet_limit = int(packet_raw)
        if packet_limit != packet_raw or packet_limit < 0:
            raise ValueError("iPET-budgeted-v3 max_dummy_packets必须是非负整数")
        self.budget = {
            "max_added_bytes": byte_limit,
            "max_dummy_packets": packet_limit,
            "max_added_delay_seconds": 0.0,
        }

    def fit(self, context, output_dir):
        super().fit(context, output_dir)
        self.metadata.update(
            {
                "variant": "iPET-budgeted-v3",
                "base_policy_method": self.policy.method_name,
                "hard_budget": self.budget,
                "budget_enforcement": "causal chronological accept-if-complete-frame-fits",
                "adapter_changes_algorithm": True,
            }
        )

    def apply(self, context, part, *, seed, target=None):
        del target
        arrays, labels = split_view(context, part)
        return self.module.apply_budgeted_policy_v3(
            self.policy,
            *arrays,
            labels,
            max_added_bytes=self.budget["max_added_bytes"],
            max_dummy_packets=self.budget["max_dummy_packets"],
            base_seed=seed,
        )


class IoTGANAdapter(Adapter):
    """每个目标识别器训练一套IoTGAN策略，禁止跨目标复用。

    context中的black_box_outputs来自只在干净outer-train上训练的四个查询目标。
    fit_target_specific_baselines会为每个键建立独立代理模型和生成器。apply必须
    收到target名称，并由底层实现再次核验策略绑定关系。
    """

    target_specific = True

    def prepare_policy_for_save(self, policy):
        """Variant hook; native policies require no extra resource metadata."""

    def check(self, context):
        del context
        from baselines import iotgan, iotgan_bidirectional

        self.mode = str(self.params.get("mode", "outbound"))
        if self.mode not in {"outbound", "bidirectional"}:
            raise ValueError("IoTGAN mode只能是outbound或bidirectional")
        self.module = iotgan if self.mode == "outbound" else iotgan_bidirectional

    def fit(self, context, output_dir):
        fit_params = dict(self.params.get("fit", {}))
        fit_params.setdefault("progress", True)
        targets = tuple(context["black_box_outputs"])
        paths = {
            target: output_dir / "policies" / f"{_policy_stem(self.name)}__{_policy_stem(target)}.npz"
            for target in targets
        }
        if len({str(path.resolve()) for path in paths.values()}) != len(paths):
            raise ValueError(
                f"{self.name} 的 target 名称在 safe_name 后发生策略路径冲突；"
                "请使用唯一可区分的 target 名称"
            )
        policy_seed = _policy_seed(context, self.params)
        reuse_requested = bool(self.params.get("reuse_existing_policy", True))
        all_present = bool(paths) and all(path.is_file() for path in paths.values())
        any_present = any(path.exists() for path in paths.values())
        if reuse_requested and any_present and not all_present:
            raise ValueError(
                f"{self.name} 的 IoTGAN 策略缓存不完整；请使用新的 output 或禁用 reuse"
            )
        descriptors = {
            target: _policy_descriptor(
                context,
                self.key,
                _cache_params(
                    self.params,
                    fit_params,
                    mode=self.mode,
                    policy_seed=policy_seed,
                    target=target,
                ),
                self.module,
                targets=context["black_box_outputs"],
            )
            for target in targets
        }
        reused = reuse_requested and all_present
        if reused:
            for target, path in paths.items():
                _validate_cached_policy(path, descriptors[target])
            self.policies = {
                target: self.module.load_policy(path) for target, path in paths.items()
            }
            print(f"[{self.name}] 复用 {len(self.policies)} 套已保存策略", flush=True)
        else:
            self.policies = self.module.fit_target_specific_baselines(
                *context["arrays"],
                context["black_box_outputs"],
                context["train_indices"],
                groups=context["groups"],
                seed=policy_seed,
                **fit_params,
            )
            for target, policy in self.policies.items():
                self.prepare_policy_for_save(policy)
                path = paths[target]
                self.module.save_policy(path, policy)
                self.policies[target] = self.module.load_policy(path)
                _write_policy_provenance(path, descriptors[target])
                print(f"[{self.name}][{target}] 策略已保存并回读验证：{path.name}", flush=True)
        path_metadata = {
            target: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "provenance": str(f"{path}.provenance.json"),
                "provenance_sha256": file_sha256(Path(f"{path}.provenance.json")),
            }
            for target, path in paths.items()
        }
        self.metadata = {
            "mode": self.mode,
            "target_specific": True,
            "policies": path_metadata,
            "reused_existing_policies": reused,
            "adapter_changes_algorithm": False,
        }

    def apply(self, context, part, *, seed, target=None):
        if target is None:
            raise ValueError("IoTGAN必须指定其对应的目标识别器")
        arrays, labels = split_view(context, part)
        return self.module.apply_target_specific_policy(
            self.policies, target, *arrays, labels, base_seed=seed
        )


class GlimpseAdapter(Adapter):
    """Glimpse-Seq64-Oracle适配器；仅启用时导入TensorFlow依赖。

    Glimpse论文未明确的实验者参数必须在配置中逐项给出。这里先构造严格的
    GlimpseConfig做范围校验，再只用outer-train拟合每设备IF奖励模型和DDPG
    Actor。Oracle应用是确定性的，因此repeat主要反映攻击者重训练随机性。
    """

    def check(self, context):
        if str(GLIMPSE_DIR) not in sys.path:
            sys.path.insert(0, str(GLIMPSE_DIR))
        from glimpse_seq64_oracle_v2 import GlimpseConfig, GlimpseSeq64OracleV2

        required_python = (3, 11)
        current_python = sys.version_info[:2]
        if current_python != required_python:
            raise RuntimeError(
                "Glimpse-Seq64-Oracle-v2 requires Python 3.11 with "
                "TensorFlow 2.15 / TF-Agents 0.19; current interpreter is "
                f"Python {current_python[0]}.{current_python[1]}. "
                "Create/use a Python 3.11 virtual environment and install "
                "baselines/glimpse/requirements.txt."
            )
        numpy_major = int(np.__version__.split(".", 1)[0])
        if numpy_major >= 2:
            raise RuntimeError(
                "Glimpse-Seq64-Oracle-v2 requires numpy<2 because it pins "
                "tensorflow==2.15.0; current NumPy is "
                f"{np.__version__}. Use a separate Python 3.11 / NumPy 1.26 "
                "environment for the complete iPET + IoTGAN + Glimpse case."
            )
        try:
            import tensorflow as tf
            import tf_agents
        except ImportError as error:
            raise RuntimeError(
                "Glimpse-Seq64-Oracle-v2 requires tensorflow==2.15.0 and "
                "tf-agents==0.19.0. Install the Glimpse requirements in this "
                "Python 3.11 environment before starting the experiment."
            ) from error
        if not tf.__version__.startswith("2.15."):
            raise RuntimeError(
                "Glimpse-Seq64-Oracle-v2 requires tensorflow==2.15.0; found "
                f"{tf.__version__}."
            )
        if not tf_agents.__version__.startswith("0.19."):
            raise RuntimeError(
                "Glimpse-Seq64-Oracle-v2 requires tf-agents==0.19.0; found "
                f"{tf_agents.__version__}."
            )

        config = dict(self.params["config"])
        config.setdefault("seed", int(context["evaluation"]["policy_seed"]))
        self.module = __import__("glimpse_seq64_oracle_v2", fromlist=["apply_policy"])
        self.policy_class = GlimpseSeq64OracleV2
        self.policy_config = GlimpseConfig(**config)
        # Run the same packet-level validation as fit before any adapter trains.
        self.module.validate_packet_inputs(*context["arrays"], config=self.policy_config)

    def fit(self, context, output_dir):
        GlimpseSeq64OracleV2 = self.policy_class
        self.policy = GlimpseSeq64OracleV2.fit(
            *context["arrays"],
            context["labels"],
            config=self.policy_config,
            train_indices=context["train_indices"],
            progress=True,
        )
        path = output_dir / "policies" / f"{_policy_stem(self.name)}.pkl"
        self.policy.save(path)
        config_params = dict(self.params.get("config", {}))
        policy_seed = int(config_params.get("seed", context["evaluation"]["policy_seed"]))
        descriptor = _policy_descriptor(
            context,
            self.key,
            _cache_params(
                self.params,
                {"config": config_params},
                mode="oracle",
                policy_seed=policy_seed,
            ),
            self.module,
        )
        _write_policy_provenance(path, descriptor)
        print(f"[{self.name}] 策略已保存：{path.name}", flush=True)
        self.metadata = {
            "policy": str(path.resolve()),
            "policy_sha256": file_sha256(path),
            "provenance": str(f"{path}.provenance.json"),
            "provenance_sha256": file_sha256(Path(f"{path}.provenance.json")),
            "oracle_complete_window": True,
            "adapter_changes_algorithm": False,
        }

    def apply(self, context, part, *, seed, target=None):
        del seed, target
        arrays, labels = split_view(context, part)
        return self.module.apply_policy(self.policy, *arrays, labels)


# ---------------------------------------------------------------------------
class SharedBudgetMixin:
    """Explicit constrained variants; the native adapters remain available."""

    def check(self, context):
        self.shared_budget = PhysicalBudget(**self.params["shared_budget"])
        if isinstance(self, IPETAdapter):
            self.params["fit"] = dict(self.params.get("fit", {}))
            self.params["fit"]["shared_budget"] = self.shared_budget.to_dict()
        if isinstance(self, GlimpseAdapter):
            self.params["config"] = dict(self.params["config"])
            self.params["config"]["shared_budget"] = self.shared_budget.to_dict()
        super().check(context)
        if isinstance(self, Method33Adapter):
            project = self.project
            method = getattr(project, "method33", None) or getattr(project, "method32", None)
            hard = method.hard_budget
            if (hard.bytes > self.shared_budget.max_added_bytes
                or hard.delay_seconds > self.shared_budget.max_wire_delay_seconds
                or hard.dummy_packets > self.shared_budget.max_added_packets):
                raise ValueError("Method32/33 configuration exceeds shared budget; reselect a feasible config/codebook, do not truncate triggers")

    def fit(self, context, output_dir):
        super().fit(context, output_dir)
        self.metadata.update(shared_budget=self.shared_budget.to_dict(),
            budget_protocol=BUDGET_PROTOCOL, adapter_changes_algorithm=True,
            budget_adaptation="resource constraints only; native architecture, objective and action families retained")

    def audit_application(self, app):
        self.shared_budget.audit(overhead_arrays(app))
        return app

    def apply(self, context, part, *, seed, target=None):
        return self.audit_application(super().apply(context, part, seed=seed, target=target))

    def legacy_budget(self):
        from baselines.high_budget import Budget
        b = self.shared_budget
        return Budget(b.max_added_bytes, b.max_wire_delay_seconds, b.max_added_packets)




class SharedMethod33Adapter(SharedBudgetMixin, Method33Adapter):
    pass


class SharedTraditionalAdapter(SharedBudgetMixin, TraditionalAdapter):
    def apply(self, context, part, *, seed, target=None):
        from baselines.high_budget import apply_traditional
        arrays, labels = split_view(context, part)
        kwargs = dict(self.params.get("apply", {}))
        if self.variant == "classic_traffic_morphing":
            kwargs.update(labels=labels, policy=self.policy)
        return self.audit_application(apply_traditional(self.variant, *arrays,
            seed=seed, budget=self.legacy_budget(), **kwargs))


class SharedIPETAdapter(SharedBudgetMixin, IPETAdapter):
    def apply(self, context, part, *, seed, target=None):
        arrays, labels = split_view(context, part)
        return self.audit_application(self.module.apply_shared_budget_policy(
            self.policy, *arrays, labels, shared_budget=self.shared_budget.to_dict(), base_seed=seed))


class SharedIoTGANAdapter(SharedBudgetMixin, IoTGANAdapter):
    def prepare_policy_for_save(self, policy):
        policy.training_metadata['shared_budget'] = self.shared_budget.to_dict()
        policy.training_metadata['budget_protocol'] = BUDGET_PROTOCOL
        policy.training_metadata['budget_enforced_during_training'] = True

    def fit(self, context, output_dir):
        from baselines.high_budget import iotgan_budget
        with iotgan_budget(self.legacy_budget()):
            super().fit(context, output_dir)

    def apply(self, context, part, *, seed, target=None):
        from baselines.high_budget import iotgan_budget
        if target in self.policies:
            record = self.policies[target].training_metadata
            if record.get('shared_budget') != self.shared_budget.to_dict() or not record.get('budget_enforced_during_training'):
                raise ValueError('IoTGAN policy was not trained for this shared budget; retrain')
        with iotgan_budget(self.legacy_budget()):
            return super().apply(context, part, seed=seed, target=target)


class SharedGlimpseAdapter(SharedBudgetMixin, GlimpseAdapter):
    pass


# 方法注册表：新增baseline时唯一需要扩展的中央位置
# ---------------------------------------------------------------------------
ADAPTERS = {
    "none": IdentityAdapter,
    "method33": Method33Adapter,
    "traditional": TraditionalAdapter,
    "ipet_v2": IPETAdapter,
    "ipet_budgeted_v3": IPETBudgetedV3Adapter,
    "iotgan": IoTGANAdapter,
    "glimpse_oracle": GlimpseAdapter,
    "method33_shared_budget_v1": SharedMethod33Adapter,
    "traditional_shared_budget_v1": SharedTraditionalAdapter,
    "ipet_shared_budget_v1": SharedIPETAdapter,
    "iotgan_shared_budget_v1": SharedIoTGANAdapter,
    "glimpse_shared_budget_v1": SharedGlimpseAdapter,
}


def build_adapters(method_specs, shared_budget=None):
    """过滤enabled条目、检查显示名唯一性并构造适配器实例。

    显示名会成为CSV分组键，因此禁止重复；key则决定调用哪种适配器。同一key
    可以出现多次，例如用不同原生参数同时评价两组Random Rate设置。
    """

    enabled = [spec for spec in method_specs if bool(spec.get("enabled", True))]
    names = [str(spec.get("name", spec["key"])) for spec in enabled]
    if len(names) != len(set(names)):
        raise ValueError("启用方法的name不能重复")
    stems = [_policy_stem(name) for name in names]
    if len(stems) != len(set(stems)):
        raise ValueError(
            "启用方法的name在safe_name后发生策略路径冲突；请使用唯一名称"
        )
    result = []
    for spec in enabled:
        key = str(spec["key"])
        if key not in ADAPTERS:
            raise KeyError(f"未知方法{key!r}；可用方法：{sorted(ADAPTERS)}")
        if shared_budget is not None:
            budget = PhysicalBudget(**shared_budget).to_dict()
            if not issubclass(ADAPTERS[key], SharedBudgetMixin):
                raise ValueError(f"{key} is not an explicit shared-budget variant")
            params = dict(spec.get("params", {}))
            if "shared_budget" in params and params["shared_budget"] != budget:
                raise ValueError("Per-method override conflicts with the common budget")
            params["shared_budget"] = budget
            spec = {**spec, "params": params}
        result.append(ADAPTERS[key](spec))
    if not result:
        raise ValueError("至少需要启用一个方法")
    return result


def overhead_arrays(application) -> dict[str, np.ndarray]:
    """把不同方法的开销格式统一成四个逐流数组。

    Method32、traditional、iPET和IoTGAN返回公共stats字典；Glimpse返回逐流
    FlowCost字典列表。这里只统一统计口径，不修改防御输出。缺少某种原生动作
    时对应数组为0，例如IoTGAN没有dummy、Glimpse没有本项目意义下的dummy。
    """

    costs = application[4]
    if isinstance(costs, Mapping):
        zeros = np.zeros(len(application[0]), dtype=float)
        get = lambda key: np.asarray(costs.get(key, zeros), dtype=float)
        missing=set(TIMING_KEYS)-set(costs)
        if missing:raise ValueError(f'legacy application lacks physical timing metrics: {sorted(missing)}; regenerate it')
        audits = application[6] if len(application) > 6 else application[5] if len(application) > 5 and isinstance(application[5], list) else []
        rejected_dummy = np.asarray([r.get('budget_rejected_dummy_packets', r.get('rejected_dummy_proposals', 0)) for r in audits], dtype=float) if audits else zeros
        return {
            "added_bytes": get("real_added_bytes") + get("dummy_bytes"),
            **{key:get(key) for key in TIMING_KEYS},
            "dummy_packets": get("dummy_count"),
            "added_fragments": get("additional_real_segments"),
            "added_packets": get("dummy_count") + get("additional_real_segments"),
            "budget_rejected_dummy_packets": rejected_dummy,
            "budget_projected_actions": zeros,
            "budget_rejected_actions": zeros,
        }
    rows = list(costs)  # Glimpse使用逐流FlowCost字典。
    if any(set(TIMING_KEYS)-set(row) for row in rows):
        raise ValueError('legacy application lacks physical timing metrics; regenerate it')
    return {
        "added_bytes": np.asarray([row.get("added_bytes", 0.0) for row in rows]),
        **{key:np.asarray([row[key] for row in rows]) for key in TIMING_KEYS},
        "dummy_packets": np.zeros(len(rows), dtype=float),
        "added_fragments": np.asarray([row.get("added_fragments", 0.0) for row in rows]),
        "added_packets": np.asarray([row.get("added_fragments", 0.0) for row in rows]),
        "budget_rejected_dummy_packets": np.zeros(len(rows), dtype=float),
        "budget_projected_actions": np.asarray([row.get('budget_projected_actions', 0) for row in rows]),
        "budget_rejected_actions": np.asarray([row.get('budget_rejected_actions', 0) for row in rows]),
    }
