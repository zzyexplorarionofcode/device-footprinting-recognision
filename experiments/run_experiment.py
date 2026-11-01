"""配置驱动的统一指纹防御实验运行器。

本文件只承担“实验编排”，不实现任何防御算法。它把一轮实验拆成六步：

1. 从 :mod:`experiment_config` 读取一个或多个实验 case；
2. 加载统一NPZ数据，并在capture级划分outer-train与outer-test；
3. 把原始流编码成共同的Seq64特征；
4. 通过 :mod:`adapters` 调用各防御方法的原始实现；
5. 按0%、10%……等比例把受保护训练流混入攻击者训练集并重新训练；
6. 保存逐次观测、逐模型汇总、总体汇总、开销和可追溯manifest。

这里固定的公平协议包括：相同outer split、相同Seq64特征、相同的已配置识别器、
相同嵌套重训练样本、相同模型种子和相同指标。适配器只解决函数签名不同的
问题，不修改baseline的动作、损失、网络或训练思想。

重要术语
--------
``outer-train``
    防御策略拟合和攻击者训练可以使用的capture集合。
``outer-test``
    仅用于最终评价的互斥capture集合，任何策略拟合都不能读取其统计量。
``retraining_ratio``
    攻击者训练集中被受保护版本替换的流比例。0表示攻击者完全没有见过防御
    流，1表示攻击者训练集全部为受保护流；测试集在所有比例下都全部受保护。
``target_specific``
    IoTGAN必须针对每一种被查询的识别器单独训练策略，不能把RF策略拿去直接
    评价SVM。普通方法则由一套共享策略面对全部已配置识别器。
"""

from __future__ import annotations

# 标准库：CLI、CSV/JSON输出、配置动态载入、哈希与运行时间记录。
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
import os
import traceback
import tempfile
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

# 数值与分类器依赖。所有攻击者都在本文件中使用同一组定义。
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 路径初始化
# ---------------------------------------------------------------------------
# 使用文件自身位置推导路径，而不是依赖“从哪个目录启动Python”。因此可以在
# 项目根目录、universal_experiment目录或IDE中直接运行同一入口。
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUR_METHOD = PROJECT / "methods"
for path in (HERE, PROJECT, OUR_METHOD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pipeline import pcap as dataset_pcap  # noqa: E402
from methods import shared_tools  # noqa: E402
from experiments.adapters import ADAPTERS, build_adapters, overhead_arrays  # noqa: E402
from experiment_support.metrics import TIMING_KEYS
from experiment_support.checkpoints import Checkpoints, run_guard


# 模型名称同时是CSV中的稳定键。不要只改这里的名字；如需更换模型族，还要
# 同步修改 fit_one_model 和 shared_tools._train_quick_attackers。
MODEL_NAMES = ("MLP-full", "RF-full", "SVM-full")
# All models use every training row and inverse-frequency class weights.
AVAILABLE_MODEL_NAMES = MODEL_NAMES + ("LSTM-full",)
# Keep surviving architectures' historical application seeds after removing RF-balanced.
MODEL_SEED_SLOTS = {"MLP-full": 0, "RF-full": 3, "SVM-full": 2, "LSTM-full": 4}


def resolve_attacker_models(model_names=None):
    """Validate selection and preserve stable model/IoTGAN seed order."""
    if model_names is None:
        return MODEL_NAMES
    if isinstance(model_names, str) or not isinstance(model_names, (list, tuple)):
        raise ValueError("attacker_models must be a nonempty list/tuple of model names")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("attacker_models must be nonempty and contain no duplicates")
    unknown = set(model_names) - set(AVAILABLE_MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown attacker models: {sorted(unknown)}")
    return tuple(name for name in AVAILABLE_MODEL_NAMES if name in model_names)


def lstm_configuration(params=None):
    """Import PyTorch only when the optional LSTM is requested."""
    from experiments.lstm_attacker import LSTMConfig

    return LSTMConfig(**dict(params or {}))


def sha256(path: Path) -> str:
    """分块计算文件SHA-256。

    数据集和结果表的摘要会写入manifest。以后即使文件名相同，只要内容发生
    改动，摘要就不同，能够避免误把两轮不同输入当成同一实验。
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    """递归把配置转换成JSON可序列化值。

    Python的 ``json`` 不认识NumPy标量/数组和Path。这里仅改变表示形式，不
    改变数值，从而可以把用户配置原样快照到manifest中。
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """保存字典行表，并自动合并所有出现过的列。

    ``utf-8-sig`` 带BOM，目的是让Windows Excel直接双击时仍能正确显示中文。
    空表通常意味着上游循环未运行，所以这里主动报错而不是留下“成功”的空
    文件。字段按第一次出现的次序排列，结果可读且可复现。
    """

    rows = list(rows)
    if not rows:
        raise ValueError(f"不能写空结果表：{path}")
    fields = list(dict.fromkeys(str(key) for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stratified_sample_split(labels, train_ratio: float, seed: int):
    """按类别分层进行样本级切分。

    这是没有独立capture时的备选方案。它保证每个类别至少有一条训练样本和
    一条测试样本，但无法阻止同一capture中的高度相似流落入两侧，因此正式
    PCAP实验应优先使用 ``split.unit='capture'``。
    """

    labels = np.asarray(labels, dtype=int)
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("sample级train_ratio必须严格位于(0,1)")
    rng = np.random.default_rng(seed)
    train = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            raise ValueError(f"类别{label}不足两条样本，无法同时进入训练集和测试集")
        # floor使比例语义稳定；随后裁到[1,n-1]，防止任何类别在一侧消失。
        count = max(1, min(len(indices) - 1, int(np.floor(len(indices) * train_ratio))))
        train[indices[:count]] = True
    return train, ~train


def fit_one_model(name, train_x, train_y, seed, *, lstm_params=None):
    """训练一台指定架构的指纹识别器。

    该函数专门服务IoTGAN：IoTGAN对MLP、RF和SVM分别拥有不同策略，所以一次
    只训练与当前策略绑定的目标模型。StandardScaler始终只在当前训练特征上
    拟合，绝不读取测试特征。所有模型均使用完整训练集和类别平衡权重。
    """

    if name == "LSTM-full":
        from experiments.lstm_attacker import LSTMAttacker

        scaler = StandardScaler().fit(train_x)
        model = LSTMAttacker(config=lstm_configuration(lstm_params), seed=seed)
        model.fit(scaler.transform(train_x), train_y)
        return scaler, model

    scaler = StandardScaler().fit(train_x)
    scaled = scaler.transform(train_x)
    model = shared_tools._fit_full_attacker(name, scaled, np.asarray(train_y), seed=seed)
    return scaler, model


def train_attackers(train_x, train_y, seed, *, model_names=None, lstm_params=None):
    """Return the usual (scaler, models) bundle, optionally appending an LSTM.

    Keeping the bundle intact across a rotation evaluates the same fitted model
    at the boundary. joblib can persist the whole bundle, including LSTM weights.
    """
    selected = resolve_attacker_models(model_names)
    if all(name in selected for name in MODEL_NAMES):
        # Share one implementation with the standalone Method32 evaluation.
        scaler, models = shared_tools._train_quick_attackers(train_x, train_y, seed=seed)
    else:
        scaler = StandardScaler().fit(train_x)
        models = {}
        scaled = scaler.transform(train_x)
        for name in selected:
            if name in MODEL_NAMES:
                models[name] = shared_tools._fit_full_attacker(
                    name, scaled, np.asarray(train_y), seed=seed)
    if "LSTM-full" in selected:
        from experiments.lstm_attacker import LSTMAttacker

        models["LSTM-full"] = LSTMAttacker(
            config=lstm_configuration(lstm_params), seed=seed,
        ).fit(scaler.transform(train_x), train_y)
    return scaler, models


def evaluate(train_x, train_y, test_x, test_y, seed, target=None, *, model_names=None, lstm_params=None,
             fitted_bundle=None, on_result=None):
    """训练攻击者并在受保护测试集上计算ACC与BA。

    ``target is None`` 时训练全部已配置攻击者；指定target时仅训练IoTGAN
    对应的目标架构。Balanced Accuracy是各类别召回率的宏平均，能避免样本
    数量较多的设备支配结果；普通Accuracy同时保留用于计算误识别率。
    """

    if fitted_bundle is not None:
        scaler, models = fitted_bundle
        if target is not None:
            models = {target: models[target]}
    elif target is None:
        scaler, models = train_attackers(
            train_x, train_y, seed, model_names=model_names, lstm_params=lstm_params,
        )
    else:
        scaler, model = fit_one_model(target, train_x, train_y, seed, lstm_params=lstm_params)
        models = {target: model}
    scaled_test = scaler.transform(test_x)
    results = {}
    for name, model in models.items():
        results[name] = {
            "accuracy": float(accuracy_score(test_y, prediction := model.predict(scaled_test))),
            "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        }
        if on_result is not None:
            on_result(name, results[name])
    return results


def save_partial_results(output_dir, observations, performance):
    """Publish completed metrics atomically; never mark a partial run complete."""
    from experiment_support.provenance import atomic_json
    output_dir = Path(output_dir)
    descriptor, temporary = tempfile.mkstemp(prefix='.observations-', suffix='.csv', dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        write_csv(temporary, observations)
        os.replace(temporary, output_dir/'observations.partial.csv')
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(output_dir/'progress.json', {
        'status': 'partial', 'completed_observations': len(observations),
        'last_completed': observations[-1], 'performance': performance,
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'note': 'Partial metrics; validated model and stage checkpoints are stored under checkpoints/.',
    })


def checkpoint_contract(spec):
    from experiment_support.provenance import runtime_versions
    semantic = {k:v for k,v in spec.items() if k not in
                {'output_dir','output_root','run_id','_resume'}}
    paths = {}
    def visit(value):
        if isinstance(value, Path) and value.is_file():
            paths[str(value.resolve())] = sha256(value)
        elif isinstance(value,str) and value.endswith(('.json','.txt','.npz','.py','.yaml','.yml')):
            candidate=Path(value)
            if candidate.is_file(): paths[str(candidate.resolve())]=sha256(candidate)
        elif isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, (tuple,list)):
            for child in value: visit(child)
    visit(semantic)
    # Dataset may be configured as a string rather than a Path.
    paths[str(Path(spec['dataset']).resolve())] = sha256(Path(spec['dataset']))
    sources = {str(p.relative_to(PROJECT)):sha256(p)
               for folder in ('methods','data_pipeline','baselines','experiments','experiment_support')
               for p in (PROJECT/folder).rglob('*.py')
               if not {'provenance','validation','outputs','__pycache__'}.intersection(p.parts)}
    return jsonable({'spec':semantic,'inputs':paths,'source':sources,'software':runtime_versions(),
                     'lstm':lstm_configuration(spec['evaluation'].get('lstm')).metadata()
                     if 'LSTM-full' in resolve_attacker_models(spec['evaluation'].get('attacker_models')) else None})


def summarize(rows):
    """由逐次观测生成两种层级的汇总表。

    第一张表按“方法×比例×模型”统计跨repeat均值与标准差，便于查看某一种
    识别器；第二张表进一步汇合所有模型，形成论文曲线常用的总体结果。同时
    报告 ``strongest_attacker``，即已配置模型中平均BA最高、对防御最不利
    的模型。这里动态读取方法名和比例，因此增加类别或方法无需修改汇总代码。
    """

    def mean_std(values):
        values = np.asarray(values, dtype=float)
        return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    per_model, overall = [], []
    # 先聚合最细粒度的三元组。一个组中通常包含repeat_count条记录。
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["retraining_ratio"], row["model"])].append(row)
    for (method, ratio, model), selected in sorted(groups.items()):
        ba, ba_std = mean_std([row["balanced_accuracy"] for row in selected])
        acc, acc_std = mean_std([row["accuracy"] for row in selected])
        per_model.append({
            "method": method, "retraining_ratio": ratio, "model": model,
            "repeats": len(selected), "balanced_accuracy_mean": ba,
            "balanced_accuracy_std": ba_std, "accuracy_mean": acc,
            "accuracy_std": acc_std, "misclassification_rate_mean": 1.0 - acc,
        })
    # 再跨模型汇总。注意总体标准差混合了“模型差异”和“重复差异”，所以正式
    # 分析模型稳定性时应优先读取per_model_summary.csv。
    for method, ratio in sorted({(row["method"], row["retraining_ratio"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method and row["retraining_ratio"] == ratio]
        ba, ba_std = mean_std([row["balanced_accuracy"] for row in selected])
        acc, acc_std = mean_std([row["accuracy"] for row in selected])
        model_means = [
            row["balanced_accuracy_mean"] for row in per_model
            if row["method"] == method and row["retraining_ratio"] == ratio
        ]
        overall.append({
            "method": method, "retraining_ratio": ratio, "observations": len(selected),
            "balanced_accuracy_mean_all_models": ba,
            "balanced_accuracy_std_all_models": ba_std,
            "accuracy_mean_all_models": acc, "accuracy_std_all_models": acc_std,
            "misclassification_rate_mean_all_models": 1.0 - acc,
            "strongest_attacker_balanced_accuracy": max(model_means),
        })
    return per_model, overall


def prepare_context(spec):
    """读取一个case并构造后续步骤共享的只读上下文。

    输入NPZ必须通过 ``dataset_pcap.load_dataset`` 的完整schema校验。类别数由
    labels自动推断；``expected_classes`` 只是防止用户误选文件。capture切分
    时，同一group只能出现在训练或测试一侧。最终的clean_all是所有方法共同
    面对的64维特征：前32维为带方向归一化长度，后32维为对数包间隔。

    返回的context同时保存全局数组和train/test索引。学习式方法用全局数组加
    显式train_indices拟合，可在其内部审计“是否只消费outer-train”；评测循环
    则直接使用已切好的clean_train/clean_test，减少重复计算。
    """

    dataset_path = Path(spec["dataset"]).resolve()
    data = dataset_pcap.load_dataset(dataset_path)
    if 'data_protocol' not in data and not spec.get('allow_unversioned_dataset',False):
        raise ValueError('Unversioned dataset rejected: regenerate with data_pipeline, or explicitly set allow_unversioned_dataset=True for legacy exploratory work')
    labels = np.asarray(data["labels"], dtype=int)
    groups = np.asarray(data.get("capture_ids",data["groups"])).astype(str)
    expected_protocol=spec.get('data_protocol')
    if expected_protocol and str(data.get('data_protocol','')) != expected_protocol:
        raise ValueError('dataset extraction protocol mismatch; regenerate the configured dataset')
    class_count = len(np.unique(labels))
    expected = spec.get("expected_classes")
    if expected is not None and class_count != int(expected):
        raise ValueError(f"数据实际为{class_count}分类，不是配置声明的{expected}分类")
    split = dict(spec["split"])
    from data_pipeline.partitions import frozen_partition
    frozen = frozen_partition(data, train_ratio=float(split['train_ratio']), seed=int(split['seed']),
                              expected_protocol=split.get('partition_protocol'))
    validation = np.zeros(len(labels), dtype=bool)
    partition_metadata = None
    if frozen is not None:
        if split.get('unit', 'capture') != 'capture':
            raise ValueError('frozen UTC-day partitions require split.unit=capture')
        masks, partition_metadata = frozen
        train, validation, test = masks['train'], masks['validation'], masks['test']
        if split.get('train_cap_per_class', partition_metadata['train_cap_per_class']) != partition_metadata['train_cap_per_class']:
            raise ValueError('configured training cap differs from frozen dataset')
    elif split.get("unit", "capture") == "capture":
        # capture级划分是默认方案。同一抓包中的时间邻近流高度相关，若按流随机
        # 切分会造成来源泄漏并高估识别器与防御效果。
        train, test = dataset_pcap.split_by_capture(
            labels, groups, train_ratio=float(split["train_ratio"]), seed=int(split["seed"])
        )
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("发现capture跨outer-train/test泄漏")
    elif split["unit"] == "sample":
        train, test = stratified_sample_split(labels, float(split["train_ratio"]), int(split["seed"]))
    else:
        raise ValueError("split.unit只能是capture或sample")
    # 两个归一化上界属于共同特征协议，不能针对某个baseline单独设置。
    feature = dict(spec.get("feature", {}))
    feature.setdefault("max_frame_length", 1514.0)
    feature.setdefault("max_gap_seconds", 10.0)
    arrays = tuple(np.asarray(data[key]) for key in ("lengths", "gaps", "directions", "mask"))
    clean_all = dataset_pcap.make_features(
        arrays[0], arrays[1], arrays[2], arrays[3],
        max_frame_length=float(feature["max_frame_length"]),
        max_gap_seconds=float(feature["max_gap_seconds"]),
    )
    train_indices, test_indices = np.flatnonzero(train), np.flatnonzero(test)
    return {
        "spec": spec, "dataset_path": dataset_path, "dataset": data,
        "arrays": arrays, "labels": labels, "groups": groups,
        "class_count": class_count, "train_indices": train_indices,
        "test_indices": test_indices, "feature": feature,
        "validation_indices": np.flatnonzero(validation), "partition_metadata": partition_metadata,
        "evaluation": dict(spec["evaluation"]), "clean_all": clean_all,
        "clean_train": clean_all[train_indices], "clean_test": clean_all[test_indices],
        "train_labels": labels[train_indices], "test_labels": labels[test_indices],
    }


def validate_application(application, expected_shape):
    """检查不同防御输出是否仍可进入共同识别器。

    各方法的metadata/audit结构可以不同，但前四项必须依次是lengths、gaps、
    directions、mask，且都保持输入 ``[N,32]`` 形状。这里只做跨方法共同的
    最小校验；Method32等实现还会在自身内部执行更严格的预算与网络约束检查。
    """

    if len(application) < 5:
        raise ValueError("防御输出至少需要lengths/gaps/directions/mask/costs")
    if any(np.asarray(value).shape != expected_shape for value in application[:4]):
        raise ValueError("防御输出形状与输入观察窗口不同")
    if not all(np.all(np.isfinite(value)) for value in application[:2]):
        raise ValueError("防御输出包含NaN或无穷值")


def _run_case(spec, *, dry_run=False, overwrite=False):
    """执行EXPERIMENTS列表中的一个实验case。

    ``dry_run`` 只完成配置、依赖、数据、划分和方法参数检查，不拟合策略也不
    写结果；应在每次修改配置后先运行。``overwrite`` 默认False，防止同名目录
    中的正式结果被无意替换。函数成功时返回与磁盘manifest相同的字典。
    """

    # 阶段A：建立共同数据上下文，并将配置中的method字典实例化为适配器。
    context = prepare_context(spec)
    model_names = resolve_attacker_models(context["evaluation"].get("attacker_models"))
    lstm_params = context["evaluation"].get("lstm")
    lstm_config = lstm_configuration(lstm_params) if "LSTM-full" in model_names else None
    if lstm_config is not None:
        print(f"[{spec['name']}] LSTM device={lstm_config.resolved_device()}", flush=True)
    adapters = build_adapters(spec["methods"], shared_budget=spec.get("shared_budget"))
    output_dir = Path(spec["output_dir"]).resolve()
    manifest_path = output_dir / "manifest.json"
    # check()只做快速静态检查。尤其会验证Method32配置与设备顺序、特征上界、
    # capture划分完全匹配，也会在启用学习式baseline时检查相应依赖可导入。
    for adapter in adapters:
        adapter.check(context)
    print(
        f"\n[{spec['name']}] {context['class_count']}分类，train={len(context['train_indices'])}，"
        f"test={len(context['test_indices'])}，方法={[adapter.name for adapter in adapters]}",
        flush=True,
    )
    if dry_run:
        print("配置、依赖、数据、划分和方法参数检查通过（dry-run不训练）。")
        return None
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"结果已存在：{manifest_path}；确认后可用--overwrite")

    # 阶段B：正式运行前才创建输出目录；dry-run保证不留下半成品。
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = Checkpoints(output_dir, checkpoint_contract(spec), resume=spec.get('_resume',False))
    checkpoints.get('split_indices', lambda: {part:context[part+'_indices']
                    for part in ('train','test','validation') if part+'_indices' in context})
    evaluation = context["evaluation"]
    ratios = tuple(float(value) for value in evaluation["retraining_ratios"])
    if not ratios or len(set(ratios)) != len(ratios) or any(not 0 <= value <= 1 for value in ratios):
        raise ValueError("retraining_ratios必须是[0,1]内不重复的非空序列")

    # 阶段C：只有IoTGAN需要query标签。目标模型只在干净outer-train上训练。
    # API要求输出数组与全部流等长，所以这里会对clean_all预测；但IoTGAN.fit
    # 同时收到train_indices，实际拟合只索引outer-train部分，测试预测不会使用。
    started = time.perf_counter()
    performance = []
    if any(adapter.target_specific for adapter in adapters):
        phase_started = time.perf_counter()
        context['black_box_outputs'] = {}
        for name in model_names:
            def query_prediction():
                scaler, model = checkpoints.get('query_model/'+name, lambda: fit_one_model(
                    name,context['clean_train'],context['train_labels'],
                    int(evaluation['query_model_seed']),lstm_params=lstm_params))
                return model.predict(scaler.transform(context['clean_all']))
            context['black_box_outputs'][name] = checkpoints.get('query_predictions/'+name,query_prediction)
        performance.append({"stage": "query_attackers", "seconds": time.perf_counter() - phase_started})

    # 阶段D：每种学习式防御只拟合一次并冻结。repeat改变应用随机性、重训练
    # mask和攻击模型种子，不反复“挑最好的一次防御策略”。
    for adapter in adapters:
        print(f"[{spec['name']}][{adapter.name}] 拟合/载入策略", flush=True)
        phase_started = time.perf_counter()
        def fit_policy():
            policy_dir=Path(tempfile.mkdtemp(prefix='policy-',dir=output_dir))
            adapter.fit(context,policy_dir)
            return {key:value for key,value in vars(adapter).items() if not isinstance(value,ModuleType)}
        state=checkpoints.get('policy/'+adapter.name,fit_policy)
        vars(adapter).update(state)
        performance.append({"stage": "policy_fit", "method": adapter.name,
                            "seconds": time.perf_counter() - phase_started})

    # observations是最细粒度主表；mask_audit证明
    # 各方法在同一repeat、同一比例下使用完全相同的受保护训练样本身份。
    observations, mask_audit = [], []
    costs = {adapter.name: defaultdict(list) for adapter in adapters}
    budget_audit_rows = []
    target_costs = {}
    repeat_count = int(evaluation["repeat_count"])
    if repeat_count < 1:
        raise ValueError("repeat_count必须为正整数")
    seed_step = int(evaluation["seed_step"])
    # 阶段E：重复评测。seed_step让每轮种子彼此分离，同时保持可完全复现。
    for repeat in range(repeat_count):
        # Lifetime is exactly one repeat: identical clean data, configuration and seed.
        # Reuse weights only; each defense still gets its own protected test predictions.
        step = repeat * seed_step
        mask_seed = int(evaluation["mask_seed"]) + step
        model_seed = int(evaluation["model_seed"]) + step
        # 每类只打乱一次，比例越大取越长的前缀，因此10%严格包含在25%中，
        # 25%又包含在50%中。这样曲线变化来自比例，而非每点换了一批随机样本。
        masks = shared_tools.build_stratified_nested_masks(
            context["train_labels"], ratios, seed=mask_seed
        )
        for ratio in ratios:
            mask_audit.append({
                "repeat": repeat, "retraining_ratio": ratio,
                "protected_train_flows": int(np.sum(masks[ratio])),
                "per_class": np.bincount(
                    context["train_labels"][masks[ratio]], minlength=context["class_count"]
                ).tolist(),
            })
        # 各方法共用mask_seed/model_seed。seed_offset只隔离方法自己的随机流，
        # 且由配置显式决定；在METHODS列表中调整顺序不会改变任何方法的种子。
        for adapter in adapters:
            base_seed = int(evaluation["apply_seed"]) + step + adapter.seed_offset
            # IoTGAN逐目标循环；其他方法只应用一次，再由已配置攻击者共享。
            targets = model_names if adapter.target_specific else (None,)
            for target in targets:
                # Appending LSTM or selecting a subset preserves historical
                # target-specific application seeds.
                target_index = MODEL_SEED_SLOTS[target] if target is not None else 0
                train_seed = base_seed + 1000 * target_index
                test_seed = train_seed + int(evaluation["test_apply_seed_offset"])
                # train/test应用使用不同种子，防止攻击者记住测试流的同一次随机
                # realization。Method32还通过配置分别使用old/next Trigger映射。
                phase_started = time.perf_counter()
                application_key=f'application/{repeat}/{adapter.name}/{target}'
                def apply_and_extract():
                    train_app = adapter.apply(context, "train", seed=train_seed, target=target)
                    test_app = adapter.apply(context, "test", seed=test_seed, target=target)
                    width = context["arrays"][0].shape[1]
                    validate_application(train_app, (len(context["train_indices"]), width))
                    validate_application(test_app, (len(context["test_indices"]), width))
                    return (dataset_pcap.make_features(*train_app[:4], **context['feature']),
                            dataset_pcap.make_features(*test_app[:4], **context['feature']),
                            overhead_arrays(test_app),
                            overhead_arrays(train_app) if spec.get('shared_budget') is not None else None)
                protected_train, protected_test, application_costs, training_costs = checkpoints.get(application_key,apply_and_extract)
                if spec.get("shared_budget") is not None:
                    from experiment_support.budget import PhysicalBudget
                    for split_name, split_costs in (('train',training_costs),('test',application_costs)):
                        audit = PhysicalBudget(**spec["shared_budget"]).audit(split_costs)
                        budget_audit_rows.append(dict(method=adapter.name, repeat=repeat, target=target,
                            split=split_name, flow_applications=len(split_costs['added_bytes']),
                            protocol=audit['protocol'], **{f'budget_{k}':v for k,v in audit['limits'].items()}, violation_count=0, violation_rate=0.,
                            **{f'cap_hit_rate_{k}':v for k,v in audit['cap_hit_rates'].items()},
                            **{f'max_{k}':v for k,v in audit['maxima'].items()}))
                bucket = target_costs.setdefault((adapter.name, target), defaultdict(list))
                for key, values in application_costs.items():
                    bucket[key].append(values)
                performance.append({"stage": "apply", "method": adapter.name, "repeat": repeat,
                                    "target": target, "seconds": time.perf_counter() - phase_started})
                # 论文比较以测试流开销为准。IoTGAN有各目标的专用策略，因此开销表
                # 会汇总“测试流×目标模型×repeat”的全部实际应用。
                for key, values in application_costs.items():
                    costs[adapter.name][key].append(values)
                for ratio in ratios:
                    phase_started = time.perf_counter()
                    selected = masks[ratio]
                    # “重训练比例r”的精确定义：复制干净训练矩阵，并仅把mask为
                    # True的行替换为相同原始流的受保护版本。样本数和标签不变。
                    if not np.any(selected):
                        mixed = context["clean_train"]
                    else:
                        mixed = context["clean_train"].copy()
                        mixed[selected] = protected_train[selected]
                    def record_result(model, metrics):
                        observations.append({
                            "class_count": context["class_count"], "method": adapter.name,
                            "repeat": repeat, "retraining_ratio": ratio,
                            "no_retraining": ratio == 0.0, "model": model, **metrics,
                            "misclassification_rate": 1.0 - metrics["accuracy"],
                            "train_flows": len(context["train_indices"]),
                            "protected_train_flows": int(np.sum(selected)),
                            "test_flows": len(context["test_indices"]),
                            "split_seed": spec["split"]["seed"], "mask_seed": mask_seed,
                            "model_seed": model_seed, "defense_train_seed": train_seed,
                            "defense_test_seed": test_seed,
                            "policy_target": target or "shared",
                            "train_mapping":adapter.params.get('train_mapping','not-applicable'),
                            "test_mapping":adapter.params.get('test_mapping','not-applicable'),
                            "first_packet_wait_observable":False,
                        })
                        save_partial_results(output_dir, observations, performance)
                        print(f"[{adapter.name}][repeat {repeat + 1}/{repeat_count}] "
                              f"重训练={ratio:>5.0%} [{model}] BA={metrics['balanced_accuracy']:.4f} 已保存", flush=True)
                    from experiment_support.provenance import array_digest
                    training_key=f'attacker/{model_seed}/{array_digest(mixed)}'
                    cache_hit=all(checkpoints.has(training_key+'/'+name) for name in ((target,) if target else model_names))
                    for name in ((target,) if target else model_names):
                        result_key=f'result/{repeat}/{adapter.name}/{target}/{ratio}/{name}'
                        def compute_metrics():
                            scaler, model=checkpoints.get(training_key+'/'+name, lambda: fit_one_model(
                                name,mixed,context['train_labels'],model_seed,lstm_params=lstm_params))
                            prediction=model.predict(scaler.transform(protected_test))
                            return {'accuracy':float(accuracy_score(context['test_labels'],prediction)),
                                    'balanced_accuracy':float(balanced_accuracy_score(context['test_labels'],prediction))}
                        record_result(name,checkpoints.get(result_key,compute_metrics))
                    performance.append({"stage": "attacker_evaluation", "method": adapter.name,
                                        "repeat": repeat, "target": target, "ratio": ratio,
                                        "clean_model_cache_hit": cache_hit,
                                        "seconds": time.perf_counter() - phase_started})
            for ratio in ratios:
                current = [row for row in observations if row["method"] == adapter.name and row["repeat"] == repeat and row["retraining_ratio"] == ratio]
                print(
                    f"[{adapter.name}][repeat {repeat + 1}/{repeat_count}] "
                    f"重训练={ratio:>5.0%}，{len(model_names)}模型平均BA="
                    f"{np.mean([row['balanced_accuracy'] for row in current]):.4f}",
                    flush=True,
                )

    # 阶段F：汇总与落盘。所有CSV先写完并计算摘要，最后才写manifest；因此
    # manifest存在可被视作该case完整结束的标志。
    per_model, ratio_summary = summarize(observations)
    overhead_summary = []
    for method, items in costs.items():
        row = {"method": method}
        if spec.get("shared_budget") is not None:
            row.update({f"budget_{k}":v for k,v in spec["shared_budget"].items()}, budget_violation_count=0, budget_violation_rate=0.)
        for key, arrays in items.items():
            values = np.concatenate(arrays).astype(float)
            row[f"mean_{key}"] = float(np.mean(values))
            row[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"p95_{key}"] = float(np.percentile(values, 95))
            row[f"max_{key}"] = float(np.max(values))
        row["flow_applications"] = int(sum(len(value) for value in items["added_bytes"]))
        overhead_summary.append(row)

    overhead_by_target = []
    for (method, target), items in target_costs.items():
        row = dict(method=method, target=target or 'shared')
        if spec.get("shared_budget") is not None:
            row.update({f"budget_{k}":v for k,v in spec["shared_budget"].items()})
        for key, arrays in items.items():
            values = np.concatenate(arrays).astype(float)
            row[f'mean_{key}'] = float(values.mean())
            row[f'p95_{key}'] = float(np.percentile(values, 95))
            row[f'max_{key}'] = float(values.max())
        row['flow_applications'] = sum(len(v) for v in items['added_bytes'])
        overhead_by_target.append(row)

    paths = {
        "observations": output_dir / "observations.csv",
        "per_model_summary": output_dir / "per_model_summary.csv",
        "ratio_summary": output_dir / "ratio_summary.csv",
        "overhead_summary": output_dir / "overhead_summary.csv",
        "overhead_by_target": output_dir / "overhead_by_target.csv",
    }
    for key, rows in (
        ("observations", observations), ("per_model_summary", per_model),
        ("ratio_summary", ratio_summary),
        ("overhead_summary", overhead_summary),
        ("overhead_by_target", overhead_by_target),
    ):
        write_csv(paths[key], rows)
    if budget_audit_rows:
        paths['budget_audit'] = output_dir / 'budget_audit.csv'
        write_csv(paths['budget_audit'], budget_audit_rows)
    split_path=output_dir/'split_indices.npz'
    np.savez_compressed(split_path,train=context['train_indices'],test=context['test_indices'],
                        validation=context.get('validation_indices', np.empty(0, dtype=np.int64)))
    paths['split_indices']=split_path
    paths['checkpoints']=checkpoints.index_path
    # manifest保留完整用户配置、输入摘要、划分审计、随机种子、方法策略文件、
    # 软件版本和输出摘要，是日后判断“两次结果为何不同”的首要依据。
    manifest = {
        "schema": "universal-defense-experiment-v2", "status": "complete",
        "experiment": jsonable(spec), "dataset_sha256": sha256(context["dataset_path"]),
        "class_count": context["class_count"],
        "outer_split": {
            "group_semantics": spec["split"].get("group_semantics",spec["split"].get("unit","capture")),
            "train_flows": len(context["train_indices"]),
            "test_flows": len(context["test_indices"]),
            "validation_flows": len(context.get("validation_indices", ())),
            "frozen_partition": context.get('partition_metadata'),
            "capture_intersection": sorted(
                set(context["groups"][context["train_indices"]])
                & set(context["groups"][context["test_indices"]])
            ) if spec["split"].get("unit", "capture") == "capture" else "not-applicable",
        },
        "protocol": {
            "version":"first-packet-observer-v2",
            "data_protocol":str(context.get("dataset",{}).get("data_protocol","legacy-unversioned")),
            "source_packet_width":int(context["arrays"][0].shape[1]),
            "source_scope":"extracted source packet prefix; complete transformed prefix events",
            "first_packet_wait_observable":False,
            "timing_metrics":list(TIMING_KEYS),
            "feature": f"{context['arrays'][0].shape[1]} signed normalized lengths + {context['arrays'][0].shape[1]} log gaps",
            "attacker_models": list(model_names), "nested_retraining_masks": True,
            "attacker_training": "full-data-class-weight-v1",
            "shared_split_masks_and_model_seeds": True,
            "iotgan_target_specific_policies": True, "mask_audit": mask_audit,
        },
        "methods": {adapter.name: jsonable(adapter.metadata) for adapter in adapters},
        "elapsed_seconds": time.perf_counter() - started,
        "performance": performance,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "files": {key: str(path.resolve()) for key, path in paths.items()},
    }
    if lstm_config is not None:
        import torch

        manifest["protocol"]["lstm"] = {
            **lstm_config.metadata(), "sequence_length": context["clean_train"].shape[1] // 2,
            "source_sha256": sha256(HERE / "lstm_attacker.py"),
        }
        manifest["software"]["torch"] = torch.__version__
        manifest["software"]["torch_cuda"] = torch.version.cuda
        if lstm_config.resolved_device() == "cuda":
            manifest["software"]["cuda_device"] = torch.cuda.get_device_name()
    from experiment_support.provenance import runtime_versions
    manifest['software'].update(runtime_versions())
    manifest['source_sha256']={str(path.relative_to(PROJECT)):sha256(path)
        for folder in ('methods','data_pipeline','baselines','experiments','experiment_support')
        for path in (PROJECT/folder).rglob('*.py')
        if not {'provenance','validation','outputs','__pycache__'}.intersection(path.parts)}
    manifest["file_sha256"] = {key: sha256(path) for key, path in paths.items()}
    from experiment_support.provenance import atomic_json
    atomic_json(manifest_path,manifest)
    print(f"[{spec['name']}] 完成，结果：{output_dir}", flush=True)
    return manifest


def run_case(spec, *, dry_run=False, overwrite=False, new_run=True, resume=False):
    """Create an isolated run by default, then publish its lifecycle atomically.

    output_dir is the parent for new runs. Explicit overwrite or new_run=False
    retains the legacy exact-directory API. Never mutate the caller's spec.
    """
    from experiment_support.provenance import atomic_json
    evaluation=spec['evaluation']
    ratios=tuple(evaluation['retraining_ratios'])
    if not ratios or len(set(ratios))!=len(ratios) or any(not np.isfinite(r) or not 0<=r<=1 for r in ratios):
        raise ValueError('retraining_ratios must be unique finite values in [0,1]')
    repeats=evaluation['repeat_count']
    if isinstance(repeats,bool) or not isinstance(repeats,(int,np.integer)) or repeats<1:
        raise ValueError('repeat_count must be a positive integer')
    if evaluation['test_apply_seed_offset']==0:raise ValueError('train/test application seeds must differ')
    output=Path(spec['output_dir']).resolve()
    if not resume and (overwrite or not new_run) and (output/'checkpoints/index.json').exists():
        raise ValueError('该目录已有检查点；使用 --resume 复用，或使用默认方式创建新运行')
    if resume:
        if overwrite:
            raise ValueError('resume 与 overwrite 不能同时使用')
        if not output.is_dir():
            raise FileNotFoundError(f'恢复目录不存在：{output}')
        Checkpoints(output,checkpoint_contract(spec),resume=True)
        spec={**spec,'_resume':True}
    if output.is_relative_to(PROJECT/'artifacts/legacy'):
        raise ValueError('Archived results are read-only; choose artifacts/runs/<new-name> and a current protocol configuration')
    if dry_run:
        print(f"[{spec['name']}] 产物根目录：{output}；正式运行默认创建独立子目录", flush=True)
        return _run_case(spec,dry_run=True)
    output.mkdir(parents=True,exist_ok=True)
    if new_run and not overwrite and not resume:
        parent = output
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')
        # mkdtemp allocates atomically, including for simultaneous invocations.
        output = Path(tempfile.mkdtemp(prefix=stamp, dir=parent))
        spec = {**spec, 'output_dir': output, 'output_root': parent, 'run_id': output.name}
    print(f"[{spec['name']}] 本次产物目录：{output}", flush=True)
    manifest=output/'manifest.json';lock=output/'.run.lock'
    with run_guard(output):
        if resume and lock.exists():
            try: stale=json.loads(lock.read_text())
            except (ValueError,OSError): stale={}
            # Acquiring the OS guard proves no new-format writer is alive.
            if isinstance(stale,dict) and stale.get('guard_schema')==1:
                lock.unlink()
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(descriptor,json.dumps({'pid':os.getpid(),'guard_schema':1}).encode());os.close(descriptor)
        started=False
        try:
            if manifest.exists():
                old=json.loads(manifest.read_text(encoding='utf-8'))
                if old.get('status')=='complete' and not overwrite and not resume:
                    raise FileExistsError(f'completed run exists: {manifest}; use a new output directory')
                atomic_json(output/f'manifest.previous.{time.time_ns()}.json',old)
            atomic_json(manifest,{'status':'running','pid':os.getpid(),'experiment':jsonable(spec)})
            started=True
            return _run_case(spec,overwrite=True)
        except BaseException as error:
            if started:atomic_json(manifest,{'status':'failed','experiment':jsonable(spec),
                'error':str(error),'exception':type(error).__name__,'traceback':traceback.format_exc()})
            raise
        finally:
            lock.unlink(missing_ok=True)


def load_config(path: Path):
    """把任意Python配置文件作为独立模块加载并读取EXPERIMENTS。

    使用Python配置而不是复杂命令行，是为了让Path、元组和多case配置保持直观。
    配置模块只负责声明数据，不应在导入时启动训练或产生文件。
    """

    specification = importlib.util.spec_from_file_location("user_experiment_config", path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    experiments = getattr(module, "EXPERIMENTS", None)
    if not isinstance(experiments, (list, tuple)) or not experiments:
        raise ValueError("配置必须定义非空EXPERIMENTS列表")
    return list(experiments)


def main(argv=None):
    """命令行入口：列方法、检查配置、筛选case或正式执行。"""

    # Windows重定向输出时显式使用UTF-8，避免中文实验进度乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/experiments/current.py")
    parser.add_argument("--case", help="只运行EXPERIMENTS中指定name")
    parser.add_argument("--dry-run", action="store_true", help="只校验配置、数据和注册，不训练")
    parser.add_argument("--overwrite", action="store_true", help="显式覆盖配置中的原目录（禁用默认独立运行子目录）")
    locations=parser.add_mutually_exclusive_group()
    locations.add_argument('--output-dir',type=Path,help='指定新运行的父目录，仍自动创建独立子目录')
    locations.add_argument('--resume',type=Path,help='指定已有运行目录；校验后复用模型、结果并继续')
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args(argv)
    if args.list_methods:
        print("\n".join(sorted(ADAPTERS)))
        return
    selected = load_config(args.config.resolve())
    if args.case:
        selected = [spec for spec in selected if spec.get("name") == args.case]
        if not selected:
            raise KeyError(f"配置中没有case={args.case!r}")
    if args.resume and (len(selected)!=1 or args.overwrite):
        parser.error('--resume 必须通过 --case 选择一个实验，且不能与 --overwrite 同用')
    for spec in selected:
        if args.resume:
            spec={**spec,'output_dir':args.resume.resolve()}
        elif args.output_dir:
            spec={**spec,'output_dir':args.output_dir.resolve()/spec['name']}
        run_case(spec, dry_run=args.dry_run, overwrite=args.overwrite, resume=bool(args.resume))


if __name__ == "__main__":
    main()
