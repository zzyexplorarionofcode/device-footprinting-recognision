"""Evaluate each rotation against a newly arrived, single-version attacker.

For transition D_r -> D_(r+1), a fresh classifier sees labeled defended
training flows from D_r only. It sees neither clean D0 nor D1..D_(r-1).
The same fitted classifier is tested on held-out D_r and D_(r+1) flows.
Transition 1 is D1 -> D2: it is the first rotation, into defense version 2.

Examples:
    python -u -B -m experiments.run_new_attacker_rotation --case unsw_8class_main
    python -u -B -m experiments.run_new_attacker_rotation --case cic_active_session_v2
"""

# =============================================================================
# 中文阅读导图：新攻击者（single-version attacker）的轮换实验
# =============================================================================
# 攻击者假设与 cumulative 版本互补：
#   第 r 次轮换（transition r：D_r → D_(r+1)）来一个全新的分类器，只在
#   “第 r 版映射的受保护训练流”上训练；它既没见过干净 D0，也没见过更早的版本。
#   同一个分类器分别测试 held-out 的 D_r（before）与 D_(r+1)（after），
#   BA 的下降就是“轮换对只看当前版本攻击者的即时效果”。
#
# 函数分工：
#   prepare()    复用 run_cumulative_rotation.prepare 做数据/分区/适配器/映射校验；
#   apply_view() 同样复用，按版本与分块生成特征；
#   _execute()   主循环：每个 repeat × transition × model 训练一次，产出配对结果；
#   summarize()  生成 per_model 与 per_transition 两张表；
#   run_case()   目录隔离、manifest 生命周期（running→complete/failed）。
#
# 结果表里的 observed_version / unseen_version 只是 transition 的两种说法：
# 训练用 D_r，测试用 D_r 与 D_(r+1)，不要误读成“攻击者训练集里混了下一版”。
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np

from configs.experiments.cumulative_rotation import EXPERIMENTS
from experiments import run_experiment as runner
from experiments.run_cumulative_rotation import apply_view, prepare
from experiment_support.checkpoints import Checkpoints, run_guard
from experiment_support.provenance import atomic_json, array_digest, runtime_versions

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'single-defended-version-new-attacker-v1'


def summarize(rows, transitions, models, repeats):
    expected = {(repeat, transition, model, phase)
                for repeat in range(repeats)
                for transition in range(1, transitions + 1)
                for model in models for phase in ('before', 'after')}
    actual = [(row['repeat'], row['transition'], row['model'], row['phase'])
              for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('new-attacker rotation results are incomplete or duplicated')

    per_model, per_transition = [], []
    for transition in range(1, transitions + 1):
        for model in models:
            phases = {phase: [row['balanced_accuracy'] for row in rows
                              if row['transition'] == transition
                              and row['model'] == model and row['phase'] == phase]
                      for phase in ('before', 'after')}
            per_model.append({
                'transition': transition, 'observed_version': transition,
                'unseen_version': transition + 1, 'model': model, 'repeats': repeats,
                'before_ba_percent': 100 * float(np.mean(phases['before'])),
                'after_ba_percent': 100 * float(np.mean(phases['after'])),
                'before_ba_std_pp': 100 * float(np.std(phases['before'], ddof=1)) if repeats > 1 else 0.0,
                'after_ba_std_pp': 100 * float(np.std(phases['after'], ddof=1)) if repeats > 1 else 0.0,
                'drop_pp': 100 * float(np.mean(phases['before']) - np.mean(phases['after'])),
            })
        selected = [row for row in per_model if row['transition'] == transition]
        per_transition.append({
            'transition': transition, 'observed_version': transition,
            'unseen_version': transition + 1, 'classifier_count': len(models),
            'repeats': repeats,
            **{key: float(np.mean([row[key] for row in selected]))
               for key in ('before_ba_percent', 'after_ba_percent', 'drop_pp')},
        })
    return per_model, per_transition


def _execute(spec, output, context, adapter, mappings, models, contract, resume):
    checkpoints = Checkpoints(output, contract, resume=resume)
    transitions = int(spec['rotation']['stages'])
    repeats = int(spec['evaluation']['repeat_count'])
    adapter.fit(context, output)
    atomic_json(output/'mapping_schedule.json', {
        'protocol': PROTOCOL, 'mappings': mappings,
        'transition_numbering': 'transition r uses D_r for training and tests D_r -> D_(r+1)',
        'class_code_reuse_allowed': False,
        'source_config_sha256': runner.sha256(Path(spec['methods'][0]['params']['config'])),
    })
    np.savez_compressed(output/'split_indices.npz', train=context['train_indices'],
                        validation=context['validation_indices'], test=context['test_indices'])

    observations, overhead = [], []
    for repeat in range(repeats):
        for transition in range(1, transitions + 1):
            # Each transition trains an independent attacker from only D_r.
            train_view = checkpoints.get(
                f'view/{repeat}/train/{transition}',
                lambda r=transition: apply_view(context, adapter, mappings[r-1],
                                                 'train', repeat, r, spec),
            )
            tests = {}
            for phase, version in (('before', transition), ('after', transition+1)):
                tests[phase] = checkpoints.get(
                    f'view/{repeat}/test/{version}',
                    lambda v=version: apply_view(context, adapter, mappings[v-1],
                                                 'test', repeat, v, spec),
                )
                overhead.append({
                    'repeat': repeat, 'transition': transition, 'phase': phase,
                    'mapping_version': version,
                    **{f'{key}_{stat}': value for key, values in tests[phase]['costs'].items()
                       for stat, value in values.items()},
                })

            model_seed = (int(spec['evaluation']['model_seed'])
                          + repeat * int(spec['evaluation']['seed_step']))
            for name in models:
                model_key = f'model/{repeat}/{transition}/{name}'
                pair_key = f'paired-result/{repeat}/{transition}/{name}'

                def compute_pair():
                    scaler, model = checkpoints.get(
                        model_key,
                        lambda: runner.fit_one_model(
                            name, train_view['features'], context['train_labels'],
                            model_seed, lstm_params=spec['evaluation'].get('lstm')),
                    )
                    fitted = (scaler, {name: model})
                    result = {
                        phase: runner.evaluate(
                            None, None, tests[phase]['features'], context['test_labels'],
                            model_seed, fitted_bundle=fitted)[name]
                        for phase in ('before', 'after')
                    }
                    del scaler, model, fitted
                    gc.collect()
                    return result

                paired = checkpoints.get(pair_key, compute_pair)
                for phase, metrics in paired.items():
                    observations.append({
                        'case': spec['name'], 'repeat': repeat,
                        'transition': transition, 'model': name, 'phase': phase,
                        **metrics,
                        'training_version': transition,
                        'mapping_version': transition if phase == 'before' else transition + 1,
                        'training_rows': len(context['train_labels']),
                        'clean_training_rows': 0,
                        'source_train_flows': len(context['train_labels']),
                        'test_flows': len(context['test_labels']),
                        'model_seed': model_seed,
                        'model_checkpoint_sha256': checkpoints.index['entries'][model_key]['sha256'],
                    })
                runner.write_csv(output/'observations.partial.csv', observations)
                print(f"[{spec['name']}] repeat={repeat+1} transition={transition} "
                      f"D{transition}->D{transition+1} {name}: "
                      f"BA {paired['before']['balanced_accuracy']:.4f} -> "
                      f"{paired['after']['balanced_accuracy']:.4f}", flush=True)
            del train_view, tests
            gc.collect()

    per_model, per_transition = summarize(observations, transitions, models, repeats)
    for filename, data in (
        ('observations.csv', observations),
        ('per_model_summary.csv', per_model),
        ('transition_summary.csv', per_transition),
        ('overhead.csv', overhead),
    ):
        runner.write_csv(output/filename, data)
    atomic_json(output/'transition_summary.json', per_transition)
    manifest = {
        'status': 'complete', 'protocol': PROTOCOL,
        'experiment': runner.jsonable(spec), 'mappings': mappings,
        'training': 'fresh model at every transition, D_r only, no D0 or older versions',
        'transition_numbering': 'transition 1 is D1 -> D2',
        'dataset_sha256': runner.sha256(context['dataset_path']),
        'train_indices_sha256': array_digest(context['train_indices']),
        'test_indices_sha256': array_digest(context['test_indices']),
        'software': runtime_versions(),
        'checkpoint_contract_sha256': runner.sha256(checkpoints.index_path),
        'summary': per_transition,
    }
    atomic_json(output/'manifest.json', manifest)
    return manifest


def run_case(spec, *, transitions, repeats, dry_run=False, resume=None, output_parent=None):
    spec = deepcopy(spec)
    spec['rotation']['stages'] = transitions
    spec['evaluation']['repeat_count'] = repeats
    context, adapter, mappings, models = prepare(spec)
    print(f"[{spec['name']}] train={len(context['train_labels'])}; "
          f"test={len(context['test_labels'])}; models={models}; "
          f"transitions={[(r, r+1) for r in range(1, transitions+1)]}", flush=True)
    if dry_run:
        print('PASS: partitions, budgets, models, and non-reusing mapping schedule', flush=True)
        return mappings

    if resume:
        output = Path(resume).resolve()
        if not output.is_dir():
            raise FileNotFoundError(output)
    else:
        parent = Path(output_parent or ROOT/'artifacts/runs/new_attacker_rotation'/spec['name']).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(
            prefix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_'), dir=parent))
    spec['output_dir'] = output
    contract = runner.checkpoint_contract(spec)
    contract.update(rotation_protocol=PROTOCOL, frozen_mappings=runner.jsonable(mappings))
    print(f'RUN_DIR={output}', flush=True)
    with run_guard(output):
        if resume:
            Checkpoints(output, contract, resume=True)
        atomic_json(output/'manifest.json', {
            'status': 'running', 'pid': os.getpid(), 'protocol': PROTOCOL,
            'experiment': runner.jsonable(spec),
        })
        try:
            return _execute(spec, output, context, adapter, mappings, models,
                            contract, bool(resume))
        except BaseException as error:
            atomic_json(output/'manifest.json', {
                'status': 'failed', 'protocol': PROTOCOL, 'error': str(error),
                'traceback': traceback.format_exc(),
            })
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[spec['name'] for spec in EXPERIMENTS],
                        help='omit to run CIC Active, CIC Idle, and UNSW sequentially')
    parser.add_argument('--rounds', type=int, default=3,
                        help='number of transitions; 3 means D1->D2, D2->D3, D3->D4')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-dir', type=Path,
                        help='parent directory for new run folders')
    parser.add_argument('--resume', type=Path,
                        help='resume one trusted existing run; requires --case')
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error('--rounds and --repeats must be positive')
    if args.resume and (not args.case or args.output_dir or args.dry_run):
        parser.error('--resume requires --case and cannot combine with --output-dir/--dry-run')
    for source in EXPERIMENTS:
        if args.case and source['name'] != args.case:
            continue
        parent = args.output_dir.resolve()/source['name'] if args.output_dir else None
        run_case(source, transitions=args.rounds, repeats=args.repeats,
                 dry_run=args.dry_run, resume=args.resume, output_parent=parent)


if __name__ == '__main__':
    main()
