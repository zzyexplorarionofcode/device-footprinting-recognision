"""Frozen 48%/56B synthetic ICMP replay on Deakin50 and UNSW P8-48."""
import argparse
import json
from pathlib import Path

import numpy as np

from methods.method33_tools import load_config
from scripts.deakin_external_codebook import fit_mlp, fit_rf
from scripts.generated_policy_transfer import generated_project
from scripts.icmp_feasibility import insert_replies
from scripts.online_nonstationary import evaluate, fit_lr, summarize
from scripts.residual_feature_views import views
from scripts.streaming_arms import make_mappings
from scripts.weakener_padding_pair import render_trace

ROOT = Path(__file__).resolve().parents[1]


def run(family, seed):
    if family == 'unsw8':
        manifest = json.loads((ROOT / 'artifacts/es/online_nonstationary_late_s0.json').read_text())
        dataset = ROOT / 'data/processed/unsw_main8_v1/flows_unsw_8class.npz'
        project = load_config(ROOT / 'configs/methods/fade_unsw8_pos_v2.json')
        horizon = 48
    else:
        manifest = json.loads((ROOT / 'artifacts/splits/deakin_external_short_time.json').read_text())
        dataset = ROOT / manifest['dataset']
        project = load_config(ROOT / 'configs/methods/fade_deakin.json')
        horizon = 50
    with np.load(dataset, allow_pickle=False) as src:
        data = {k: src[k] for k in ('lengths', 'gaps', 'directions', 'mask', 'labels')}
        dates = (src['groups'].astype('datetime64[D]') if family == 'unsw8' else
                 src['timestamps'].max(axis=1).astype('datetime64[s]').astype('datetime64[D]'))
    warm = np.asarray(manifest['warmup_indices'], int)
    daily = manifest['sample_days'][:horizon]
    days = np.asarray([d['day'] for d in daily], dtype='datetime64[D]')
    ids, local = [], []
    for day in daily:
        cal, blind = day['cal_indices'], day['blind_indices']
        start = len(ids)
        ids.extend(cal + blind)
        local.append((np.arange(start, start + len(cal)),
                      np.arange(start + len(cal), start + len(cal) + len(blind))))
    ids = np.asarray(ids)
    assert len(days) == horizon and np.all(dates[warm] < days[0])
    mappings = [[int(c) for c in m] for m in make_mappings(project, 0, count=6)]
    random = (np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1]).random(len(data['labels']))
              if family == 'unsw8' else
              np.random.default_rng(np.random.SeedSequence([7400, seed]).spawn(2)[1]).random(len(data['labels'])))
    states = np.minimum((random * 6).astype(int), 5)
    first = generated_project(project, 5100 + 6 * seed) if family == 'unsw8' else project
    warm_x, _ = render_trace(data, warm, states[warm], mappings, first)
    parts = []
    probes = []
    for stage in range((horizon + 4) // 5):
        subset = np.concatenate([np.r_[local[d][0], local[d][1]]
                                 for d in range(stage * 5, min((stage + 1) * 5, horizon))])
        candidate_seed = (5100 + 6 * seed + stage if stage < 6 else
                          9000 + 100 * seed + stage - 6) if family == 'unsw8' else 9500 + 10 * seed + stage
        base, traces = render_trace(data, ids[subset], states[ids[subset]], mappings,
                                    generated_project(project, candidate_seed))
        parts.append(base)
        probes.extend([[int(e['event_position']) for e in trace
                        if e['role'] == 'trigger_dummy' and e['in_observation_window']]
                       for trace in traces])
    base = np.concatenate(parts)
    assert len(base) == len(ids) == len(probes)
    draws = np.random.default_rng(np.random.SeedSequence([94017, seed])).random((len(ids), 2))
    changed = np.empty_like(base)
    visible = np.zeros(len(ids), dtype=int)
    for i in range(len(ids)):
        changed[i], visible[i], _ = insert_replies(base[i], probes[i], .48, 56, .05, draws[i])
    output = ROOT / f'artifacts/es/icmp_model_matrix_{family}_s{seed}.json'
    result = {'dataset': family, 'protocol': 'P8-48' if family == 'unsw8' else 'Deakin50',
              'seed': seed, 'days': horizon, 'return_probability': .48, 'reply_bytes': 56,
              'delay_seconds': .05, 'mean_visible_replies': float(visible.mean()),
              'mean_visible_downlink_bytes': float((56 * visible).mean()),
              'n_blind': int(sum(len(b) for _, b in local)), 'results': {}}
    models = [('MLP', fit_mlp)] if family == 'deakin10' else [('LR', fit_lr), ('RF', fit_rf), ('MLP', fit_mlp)]
    for view in ('C', 'D'):
        result['results'][view] = {}
        warm_view = views(warm_x)['C'] if view == 'C' else warm_x
        for name, fit in models:
            result['results'][view][name] = {}
            for arm, x in [('FullNew5', base), ('ICMP48', changed)]:
                feature = views(x)['C'] if view == 'C' else x
                rows = evaluate(feature, data['labels'][ids], days, local, dates[warm],
                                data['labels'][warm], warm_view, {}, regime_days=5, fit_model=fit)
                result['results'][view][name][arm] = {
                    'summary': summarize(rows),
                    'daily_post': {mode: [r['attacks'][mode]['post'] for r in rows]
                                   for mode in ('cumulative', 'sliding')}}
                output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                print(family, seed, view, name, arm,
                      {k: v['mean_post_ba'] for k, v in summarize(rows).items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('unsw8', 'deakin10'), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    a = p.parse_args()
    run(a.family, a.seed)
