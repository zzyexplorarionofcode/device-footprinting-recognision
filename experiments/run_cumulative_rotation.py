"""Cumulative labeled history, frozen paired evaluation, no benchmark auto-launch.

python -m experiments.run_cumulative_rotation --case unsw_8class_main --dry-run
"""

# =============================================================================
# 中文阅读导图：持续攻击者（cumulative history）的轮换实验
# =============================================================================
# 与 run_new_attacker_rotation 的唯一区别在“攻击者能看多少历史”：
#   本文件：训练集 = 干净训练流 + D1..D_stage 每个版本的受保护训练流（累积），
#           即攻击者见过全部历史映射，是最不利的长期对手；
#   新攻击者：训练集只含当前版本的受保护流，模拟“攻击者刚出现”。
# 两者共用同一套映射序列，因此可比。
#
# 一轮 stage 的流程（_execute）：
#   apply_view(train, version=stage)   用第 stage 版映射变换训练流，追加进 history；
#   apply_view(test,  version=stage)   → before（当前版本）
#   apply_view(test,  version=stage+1) → after（下一版本）
#   fit 一次分类器（训练集 = clean + 全部历史），同一个对象分别预测 before/after，
#   从而得到“配对”的 BA 下降量 drop_pp，避免两次独立训练的模型噪声。
#
# 其他工程点：
#   * apply_view 按 application_batch_size 分块调用 method33 核心，控制 trace 内存；
#   * 测试集的随机量按 (repeat, block, version=0) 固定，使 before/after 共用同一组
#     抖动分位数，差异只来自映射；训练集则按版本区分；
#   * 结果先写 observations.partial.csv，最后写 manifest.json（status=complete）；
#   * manifest 里记录 mapping_schedule 与“历史类别-码字对是否复用”的断言结果。
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np

from experiments import run_experiment as runner
from experiments.adapters import build_adapters, overhead_arrays
from experiment_support.checkpoints import Checkpoints, run_guard
from experiment_support.provenance import atomic_json, array_digest, runtime_versions
from experiment_support.rotation_history import PROTOCOL, mapping_schedule
from methods.method33 import apply_method33
from methods.method33_tools import validate_application as audit_method33

ROOT = Path(__file__).resolve().parents[1]


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def prepare(spec):
    context = runner.prepare_context(spec)
    if spec['split'].get('unit') != 'capture':
        raise ValueError('cumulative rotation requires grouped source partitions')
    partitions = [context['train_indices'], context['validation_indices'], context['test_indices']]
    for i, left in enumerate(partitions):
        for right in partitions[i+1:]:
            if set(left) & set(right) or set(context['groups'][left]) & set(context['groups'][right]):
                raise ValueError('source groups overlap between train/validation/test')
    stages = _positive(spec['rotation']['stages'], 'stages')
    _positive(spec['rotation']['application_batch_size'], 'application_batch_size')
    _positive(spec['evaluation']['repeat_count'], 'repeat_count')
    models = runner.resolve_attacker_models(spec['evaluation']['attacker_models'])
    if 'LSTM-full' in models:
        runner.lstm_configuration(spec['evaluation'].get('lstm')).resolved_device()
    methods = spec['methods']
    if len(methods) != 1 or methods[0]['key'] != 'method33_shared_budget_v1':
        raise ValueError('one shared-budget Method33 is required')
    if not all(methods[0]['params'].get(key, True) for key in ('include_trigger', 'include_weakener')):
        raise ValueError('this protocol evaluates the full method, not component ablations')
    adapter = build_adapters(deepcopy(methods), shared_budget=spec['shared_budget'])[0]
    adapter.check(context)
    mappings = mapping_schedule(adapter.project.method33, stages)
    return context, adapter, mappings, models


def apply_view(context, adapter, mapping, part, repeat, version, spec):
    """Bound event-trace memory; record chunking as part of the immutable protocol.

    Test RNG is paired across ALL versions. Training RNG differs by version.
    The original source dataset is always used, never the previous output.
    """
    indices = context[f'{part}_indices']
    size = spec['rotation']['application_batch_size']
    blocks, costs = [], {}
    base = int(spec['evaluation']['apply_seed'])
    for block, start in enumerate(range(0, len(indices), size)):
        subset = indices[start:start+size]
        domain = (0, version) if part == 'train' else (1, 0)
        seed = int(np.random.SeedSequence([base, repeat, *domain, block]).generate_state(1)[0])
        # Full dataset/class alignment was checked in prepare(). Individual chunks
        # need not contain all classes, so use the core's range-checked batch API.
        app = apply_method33(*(array[subset] for array in context['arrays']),
                             context['labels'][subset], adapter.project.method33,
                             seed=seed, mapping=mapping, jitter_config=adapter.project.trigger_jitter)
        if not audit_method33(app, config=adapter.project.method33)['passed']:
            raise AssertionError('native Method33 budget audit failed')
        adapter.audit_application(app)
        runner.validate_application(app, (len(subset), context['arrays'][0].shape[1]))
        blocks.append(runner.dataset_pcap.make_features(*app[:4], **context['feature']))
        for key, values in overhead_arrays(app).items():
            costs.setdefault(key, []).append(values)
        if block % 16 == 0 or start+size >= len(indices):
            print(f"  view repeat={repeat+1} {part} version={version}: "
                  f"{min(start+size, len(indices))}/{len(indices)} flows", flush=True)
    features = np.concatenate(blocks)
    summary = {key: {'mean': float(np.mean(values := np.concatenate(chunks))),
                     'max': float(np.max(values))} for key, chunks in costs.items()}
    return {'features': features, 'costs': summary, 'version': version,
            'mapping': list(mapping), 'source_indices_sha256': array_digest(indices)}


def history_arrays(clean, views, labels):
    if not views or any(v.shape != clean.shape for v in views) or len(labels) != len(clean):
        raise ValueError('history views must align to the same source training rows')
    return np.concatenate([clean, *views]), np.tile(labels, len(views)+1)


def summarize(rows, stages, models, repeats):
    """Equal mean across repeats/classifiers; paired delta in percentage points."""
    expected = {(r, s, m, p) for r in range(repeats) for s in range(1, stages+1)
                for m in models for p in ('before', 'after')}
    actual = [(x['repeat'], x['stage'], x['model'], x['phase']) for x in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('incomplete or duplicated rotation results')
    per_model, overall = [], []
    for stage in range(1, stages+1):
        for name in models:
            pair = {p: [x['balanced_accuracy'] for x in rows
                        if x['stage'] == stage and x['model'] == name and x['phase'] == p]
                    for p in ('before', 'after')}
            per_model.append({'stage': stage, 'model': name, 'repeats': repeats,
                              **{f'{p}_ba_percent': 100*float(np.mean(v)) for p, v in pair.items()},
                              **{f'{p}_ba_std_pp': 100*float(np.std(v, ddof=1)) if repeats > 1 else 0.
                                 for p, v in pair.items()},
                              'drop_pp': 100*float(np.mean(pair['before'])-np.mean(pair['after']))})
        chosen = [x for x in per_model if x['stage'] == stage]
        overall.append({'stage': stage, 'history_versions': stage+1,
                        **{key: float(np.mean([x[key] for x in chosen]))
                           for key in ('before_ba_percent', 'after_ba_percent', 'drop_pp')}})
    return per_model, overall


def _execute(spec, output, context, adapter, mappings, models, contract, resume):
    cp = Checkpoints(output, contract, resume=resume)
    stages = spec['rotation']['stages']
    repeats = spec['evaluation']['repeat_count']
    reused_pairs_per_version = [sum(any(previous[c] == mapping[c]
        for previous in mappings[:i]) for c in range(len(mapping)))
        for i, mapping in enumerate(mappings)]
    if any(reused_pairs_per_version):
        raise AssertionError('rotation schedule reuses a historical class-code pair')
    adapter.fit(context, output)
    atomic_json(output/'mapping_schedule.json', {
        'protocol': PROTOCOL, 'mappings': mappings, 'first_pair': 'configured old/next',
        'selection': 'exact length/gap distance from previous; forbid historical class-code pairs; lexicographic tie',
        'class_code_reuse_allowed': False, 'position_distance_used': False,
        'reused_class_code_pairs_per_version': reused_pairs_per_version,
        'source_config_sha256': runner.sha256(Path(spec['methods'][0]['params']['config']))})
    atomic_json(output/'method_config.json', json.loads(
        Path(spec['methods'][0]['params']['config']).read_text(encoding='utf-8-sig')))
    np.savez_compressed(output/'split_indices.npz', train=context['train_indices'],
                        validation=context['validation_indices'], test=context['test_indices'])
    rows, overhead = [], []
    for repeat in range(repeats):
        history = []
        for stage in range(1, stages+1):
            view = cp.get(f'view/{repeat}/train/{stage}', lambda: apply_view(
                context, adapter, mappings[stage-1], 'train', repeat, stage, spec))
            history.append(view['features'])
            mixed, labels = history_arrays(context['clean_train'], history, context['train_labels'])
            atomic_json(output/f'history-r{repeat}-s{stage}.json', {
                'source_train_flows': len(context['train_labels']), 'training_rows': len(labels),
                'versions': list(range(stage+1)), 'train_indices_sha256': array_digest(context['train_indices']),
                'features_sha256': array_digest(mixed), 'labels_sha256': array_digest(labels),
                'independent_new_captures': False})
            tests = {}
            for phase, version in (('before', stage), ('after', stage+1)):
                tests[phase] = cp.get(f'view/{repeat}/test/{version}', lambda v=version: apply_view(
                    context, adapter, mappings[v-1], 'test', repeat, v, spec))
                overhead.append({'repeat': repeat, 'stage': stage, 'phase': phase,
                                 'mapping_version': version, **{f'{k}_{stat}': value
                                  for k, stats in tests[phase]['costs'].items() for stat, value in stats.items()}})
            model_seed = int(spec['evaluation']['model_seed']) + repeat*int(spec['evaluation']['seed_step'])
            for name in models:
                key = f'model/{repeat}/{stage}/{name}'
                pair_key = f'paired-result/{repeat}/{stage}/{name}'
                def compute_pair():
                    scaler, model = cp.get(key, lambda: runner.fit_one_model(
                        name, mixed, labels, model_seed, lstm_params=spec['evaluation'].get('lstm')))
                    # Exactly one fitted object and scaler serve BOTH sides of the boundary.
                    result = {phase: runner.evaluate(None, None, tests[phase]['features'],
                              context['test_labels'], model_seed,
                              fitted_bundle=(scaler, {name: model}))[name]
                              for phase in ('before', 'after')}
                    del scaler, model
                    gc.collect()
                    return result
                paired = cp.get(pair_key, compute_pair)
                for phase, metrics in paired.items():
                    rows.append({'case': spec['name'], 'repeat': repeat, 'stage': stage,
                                 'model': name, 'phase': phase, **metrics,
                                 'mapping_version': stage if phase == 'before' else stage+1,
                                 'training_rows': len(labels), 'source_train_flows': len(context['train_labels']),
                                 'test_flows': len(context['test_labels']), 'model_seed': model_seed,
                                 'model_checkpoint_sha256': cp.index['entries'][key]['sha256']})
                runner.write_csv(output/'observations.partial.csv', rows)
                print(f"[{spec['name']}] repeat={repeat+1} stage={stage} {name}: "
                      f"BA {paired['before']['balanced_accuracy']:.4f} -> {paired['after']['balanced_accuracy']:.4f}", flush=True)
            del mixed, labels, tests, view
    per_model, overall = summarize(rows, stages, models, repeats)
    for filename, content in [('observations.csv', rows), ('per_model_summary.csv', per_model),
                              ('stage_summary.csv', overall), ('overhead.csv', overhead)]:
        runner.write_csv(output/filename, content)
    atomic_json(output/'stage_summary.json', overall)
    files = {p.name: runner.sha256(p) for p in output.iterdir()
             if p.is_file() and (p.suffix in ('.csv', '.npz', '.json')) and p.name != 'manifest.json'}
    manifest = {'status': 'complete', 'protocol': PROTOCOL, 'experiment': runner.jsonable(spec),
                'mappings': mappings, 'dataset_sha256': runner.sha256(context['dataset_path']),
                'software': runtime_versions(), 'file_sha256': files,
                'aggregation': 'mean repeats then classifiers; CIC Active/Idle equal mean only during report',
                'training': 'from scratch, full cumulative history including clean, no new independent source flows',
                'test_randomness': 'paired by repeat/source block across all mappings',
                'cost_scope': 'all transformed events of extracted source prefix; first gap invisible',
                'checkpoint_contract_sha256': runner.sha256(cp.index_path)}
    atomic_json(output/'manifest.json', manifest)
    return manifest


def run_case(spec, *, dry_run=False, resume=None):
    spec = deepcopy(spec)
    context, adapter, mappings, models = prepare(spec)
    print(f"[{spec['name']}] {context['class_count']}-class; "
          f"train={len(context['train_indices'])}, test={len(context['test_indices'])}; "
          f"models={models}; mappings={mappings}", flush=True)
    if dry_run:
        print('PASS: data, partitions, budgets, models and history-constrained mappings; no training', flush=True)
        return mappings
    base = Path(resume or spec['output_dir']).resolve()
    if base.is_relative_to(ROOT/'artifacts/legacy'):
        raise ValueError('legacy results are read-only')
    if resume:
        if not base.is_dir():
            raise FileNotFoundError(base)
        output = base
    else:
        base.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_'), dir=base))
    spec['output_dir'] = output
    contract = runner.checkpoint_contract(spec)
    contract.update(rotation_protocol=PROTOCOL, frozen_mappings=runner.jsonable(mappings))
    print(f'RUN_DIR={output}', flush=True)
    with run_guard(output):
        # Validate resume before touching an existing manifest.
        if resume:
            Checkpoints(output, contract, resume=True)
        old = output/'manifest.json'
        if old.exists():
            atomic_json(output/f'manifest.previous.{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json',
                        json.loads(old.read_text(encoding='utf-8')))
        atomic_json(old, {'status': 'running', 'pid': os.getpid(), 'protocol': PROTOCOL,
                          'experiment': runner.jsonable(spec)})
        try:
            return _execute(spec, output, context, adapter, mappings, models, contract, bool(resume))
        except BaseException as exc:
            atomic_json(old, {'status': 'failed', 'protocol': PROTOCOL, 'experiment': runner.jsonable(spec),
                              'error': str(exc), 'traceback': traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/cumulative_rotation.py')
    parser.add_argument('--case', help='omit to run CIC Active, CIC Idle and UNSW sequentially')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='parent directory, per-case isolated run folders')
    parser.add_argument('--resume', type=Path, help='trusted existing run directory; requires --case')
    args = parser.parse_args()
    cases = runner.load_config(args.config.resolve())
    if args.case:
        cases = [s for s in cases if s['name'] == args.case]
        if not cases:
            parser.error('unknown case')
    if args.resume and (not args.case or args.output_dir):
        parser.error('--resume requires --case and cannot combine with --output-dir')
    for spec in cases:
        if args.output_dir:
            spec = {**spec, 'output_dir': args.output_dir.resolve()/spec['name']}
        run_case(spec, dry_run=args.dry_run, resume=args.resume)


if __name__ == '__main__':
    main()
