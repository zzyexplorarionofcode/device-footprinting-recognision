"""Short online A/B/C comparison with paired added bytes on two Trigger dummies."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from methods.method33_tools import load_config
from scripts.deakin_external_codebook import fit_mlp, fit_rf
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import evaluate, fit_lr, summarize
from scripts.residual_feature_views import views
from scripts.attacker_matrix import fit_model as fit_frozen_reference, render_clean, score as frozen_score
from scripts.streaming_arms import make_mappings
from scripts.weakener_padding_pair import render_trace

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A-FullNew5", "B-Dispersed", "C-Concentrated")


def transform(data, ids, states, mappings, candidate, seed, extra, ratio=None, weakener=None):
    base, traces = render_trace(data, ids, states, mappings, candidate)
    arms = {name: base.copy() for name in ARMS}
    audit = []
    for row, (flow_id, trace) in enumerate(zip(ids, traces, strict=True)):
        trigger = [e for e in trace if e["role"] == "trigger_dummy"]
        assert len(trigger) == 2
        visible = [e for e in trigger if e["in_observation_window"]]
        assert visible, "A flow has no visible Trigger"
        chosen = visible[0]
        original = np.asarray([e["source_length_bytes"] for e in trace if e["role"] == "real" and e["direction"] == 1 and e["in_observation_window"]], float)
        real_headroom = float(np.minimum(weakener.max_padding_per_packet, np.maximum(0., 1514. - original)).sum()) if ratio is not None else 0.
        allowed = min(extra, ratio * real_headroom) if ratio is not None else extra
        capacity = max(0., min(allowed, 1514. - float(chosen["length_bytes"])))
        applied = np.random.default_rng(np.random.SeedSequence([seed, int(flow_id), 811731])).random() < .5
        budget = float(capacity if applied else 0.)
        if budget:
            position = int(chosen["event_position"]) - 1
            arms[ARMS[2]][row, position] = (float(chosen["length_bytes"]) + budget) / 1514.
            free = np.asarray([1514. - float(e["length_bytes"]) for e in trigger])
            allocation = np.zeros(2)
            order = np.random.default_rng(np.random.SeedSequence([seed, int(flow_id), 90113])).permutation(2)
            remaining = budget
            for j in order:
                take = min(remaining / 2, free[j])
                allocation[j] += take
                remaining -= take
            for j in order:
                take = min(remaining, free[j] - allocation[j])
                allocation[j] += take
                remaining -= take
            assert remaining <= 1e-8
            for e, amount in zip(trigger, allocation, strict=True):
                if e["in_observation_window"]:
                    pos = int(e["event_position"]) - 1
                    arms[ARMS[1]][row, pos] = (float(e["length_bytes"]) + amount) / 1514.
        audit.append((budget, float((arms[ARMS[1]][row, :32] - base[row, :32]).sum() * 1514),
                      float((arms[ARMS[2]][row, :32] - base[row, :32]).sum() * 1514), real_headroom, allowed, 1514. - float(chosen["length_bytes"])))
    audit = np.asarray(audit)
    assert np.allclose(audit[:, 0], audit[:, 1], atol=1e-7)
    assert np.allclose(audit[:, 1], audit[:, 2], atol=1e-7)
    return arms, audit


def run(days_to_score, seed, extra, proportional=False, family="deakin10", frozen_only=False):
    if family == "unsw8":
        manifest = json.loads((ROOT / "artifacts/es/online_nonstationary_late_s0.json").read_text(encoding="utf-8"))
        dataset = ROOT / "data/processed/unsw_main8_v1/flows_unsw_8class.npz"
        project = load_config(ROOT / "configs/methods/fade_unsw8_pos_v2.json")
    else:
        manifest = json.loads((ROOT / "artifacts/splits/deakin_external_short_time.json").read_text(encoding="utf-8"))
        dataset = ROOT / manifest["dataset"]
        project = load_config(ROOT / "configs/methods/fade_deakin.json")
    with np.load(dataset, allow_pickle=False) as src:
        data = {k: src[k] for k in ("lengths", "gaps", "directions", "mask", "labels")}
        dates_by_row = (src["groups"].astype("datetime64[D]") if family == "unsw8" else
                        src["timestamps"].max(axis=1).astype("datetime64[s]").astype("datetime64[D]"))
    warm = np.asarray(manifest["warmup_indices"], int)
    daily = manifest["sample_days"][:days_to_score]
    days = np.asarray([d["day"] for d in daily], dtype="datetime64[D]")
    ids, local = [], []
    for day in daily:
        cal, blind = day["cal_indices"], day["blind_indices"]
        start = len(ids)
        ids.extend(cal + blind)
        local.append((np.arange(start, start + len(cal)), np.arange(start + len(cal), start + len(cal) + len(blind))))
    ids = np.asarray(ids)
    assert np.all(dates_by_row[warm] < days[0])
    mappings = [[int(c) for c in m] for m in make_mappings(project, 0, count=6)]
    random = (np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1]).random(len(data["labels"]))
              if family == "unsw8" else
              np.random.default_rng(np.random.SeedSequence([7400, seed]).spawn(2)[1]).random(len(data["labels"])))
    states = np.minimum((random * 6).astype(int), 5)
    # All policies use the same initial defended warmup. Only subsequent labels reflect each arm.
    first_candidate = generated_project(project, 5100 + 6 * seed) if family == "unsw8" else project
    warm_x, _ = render_trace(data, warm, states[warm], mappings, first_candidate)
    ratio = None
    if proportional:
        warm_length = np.minimum(data["lengths"][warm], 1514.)
        eligible = data["mask"][warm].astype(bool) & (data["directions"][warm] == 1)
        headroom = (np.minimum(project.method33.weakener.max_padding_per_packet,
                               np.maximum(0., 1514. - warm_length)) * eligible).sum(axis=1)
        median = float(np.median(headroom[headroom > 0]))
        ratio = min(.25, extra / median)
        print("frozen warmup ratio", ratio, "median headroom", median, flush=True)
    pieces = {arm: [] for arm in ARMS}
    audits = []
    for stage in range((len(daily) + 4) // 5):
        subset = np.concatenate([np.r_[local[d][0], local[d][1]] for d in range(stage * 5, min((stage + 1) * 5, len(daily)))])
        candidate_seed = (5100 + 6 * seed + stage if stage < 6 else 9000 + 100 * seed + stage - 6) if family == "unsw8" else 9500 + 10 * seed + stage
        candidate = generated_project(project, candidate_seed)
        arms, audit = transform(data, ids[subset], states[ids[subset]], mappings, candidate, seed, extra, ratio, project.method33.weakener)
        for arm in ARMS:
            pieces[arm].append(arms[arm])
        audits.append(audit)
    audit = np.concatenate(audits)
    x = {arm: np.concatenate(pieces[arm]) for arm in ARMS}
    output = ROOT / f"artifacts/es/trigger_long_online_{family}_{days_to_score}d_s{seed}{'_proportional' if proportional else ''}.json"
    result = {"dataset": family, "protocol": "P8-48" if family == "unsw8" else "Deakin50",
              "seed": seed, "days": days_to_score, "extra_bytes_cap": extra,
              "warmup_ratio": ratio, "warmup_headroom_median": median if proportional else None,
              "budget_definition": "min(r_b * visible real-uplink paddable headroom, extra cap, first visible Trigger frame headroom)",
              "application_probability": .5, "paired_visible_bytes_exact": bool(np.allclose(audit[:, 1], audit[:, 2])),
              "n_flow": len(ids), "n_blind": int(sum(len(b) for _, b in local)),
              "mean_extra_bytes_B_C": [float(audit[:, 1].mean()), float(audit[:, 2].mean())],
              "mean_real_headroom": float(audit[:, 3].mean()),
              "mean_allowed_extra": float(audit[:, 4].mean()),
              "mean_trigger_headroom": float(audit[:, 5].mean()),
              "fraction_zero_extra": float((audit[:, 0] == 0).mean()), "results": {}}
    if frozen_only:
        assert family == "unsw8" and output.exists()
        previous = json.loads(output.read_text(encoding="utf-8"))
        assert previous["seed"] == seed and previous["days"] == days_to_score
        result = previous
    warm_views = {"C": views(warm_x)["C"], "D": warm_x}
    if family == "unsw8":
        clean_data = {**data, "project": project}
        clean_warm = render_clean(clean_data, warm)
        result["frozen"] = {}
    for view in ("C", "D"):
        if not frozen_only:
            result["results"][view] = {}
        if family == "unsw8":
            clean_view = views(clean_warm)["C"] if view == "C" else clean_warm
            result["frozen"][view] = {}
        for model, fit in (("LR", fit_lr), ("RF", fit_rf), ("MLP", fit_mlp)):
            if family == "unsw8":
                # LR/RF match the earlier P8-48 frozen fitting protocol. MLP
                # receives a source-domain scaler fitted only on its warmup.
                frozen = {}
                for source, train_x in (("clean", clean_view), ("initial", warm_views[view])):
                    if model == "MLP":
                        source_scaler = StandardScaler().fit(train_x)
                        frozen[source] = (fit_mlp(source_scaler.transform(train_x), data["labels"][warm]), source_scaler)
                    else:
                        frozen[source] = (fit_frozen_reference(train_x, data["labels"][warm], model.lower()), None)
                result["frozen"][view][model] = {}
            if not frozen_only:
                result["results"][view][model] = {}
            for arm in ARMS:
                features = views(x[arm])["C"] if view == "C" else x[arm]
                if family == "unsw8":
                    by_day = {name: [frozen_score(est, features[blind] if source_scaler is None else
                                                  source_scaler.transform(features[blind]), data["labels"][ids[blind]])
                                     for _, blind in local] for name, (est, source_scaler) in frozen.items()}
                    result["frozen"][view][model][arm] = {name: {"mean_ba": float(np.nanmean(day_ba)),
                                                                   "daily_ba": [None if np.isnan(v) else v for v in day_ba],
                                                                   "evaluated_days": int(np.isfinite(day_ba).sum())}
                                                             for name, day_ba in by_day.items()}
                if frozen_only:
                    continue
                rows = evaluate(features, data["labels"][ids], days, local, dates_by_row[warm],
                                data["labels"][warm], warm_views[view], {}, regime_days=5, fit_model=fit)
                result["results"][view][model][arm] = {"summary": summarize(rows),
                                                         "daily_post": {mode: [r["attacks"][mode]["post"] for r in rows]
                                                                        for mode in ("cumulative", "sliding")}}
                print(view, model, arm, {mode: result["results"][view][model][arm]["summary"][mode]["mean_post_ba"]
                                        for mode in ("cumulative", "sliding")}, flush=True)
            for mode in (() if frozen_only else ("cumulative", "sliding")):
                scores = [result["results"][view][model][arm]["summary"][mode]["mean_post_ba"] for arm in ARMS]
                result["results"][view][model][mode + "_contrasts"] = {"B_minus_A": scores[1] - scores[0],
                                                                          "C_minus_B": scores[2] - scores[1],
                                                                          "C_minus_A": scores[2] - scores[0]}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("output:", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-bytes", type=float, default=256.)
    p.add_argument("--proportional", action="store_true")
    p.add_argument("--family", choices=("deakin10", "unsw8"), default="deakin10")
    p.add_argument("--frozen-only", action="store_true")
    a = p.parse_args()
    run(a.days, a.seed, a.extra_bytes, a.proportional, a.family, a.frozen_only)
