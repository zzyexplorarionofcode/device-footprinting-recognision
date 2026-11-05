"""Continuous online replay with six independently generated Trigger codebooks."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from methods.method33_tools import load_config, validate_config
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import (
    ATTACKS, LABELS_PER_DAY, REGIME_DAYS, REGIMES, SLIDING_DAYS, TARGET_BA,
    balanced_accuracy, draw_states, evaluate, fit_lr, summarize,
)
from scripts.paired_mapping_policies import render, support


def freeze_components(base, candidate, component):
    """Combine Trigger components while preserving each packet pair's budget."""
    if component == "fixed":
        return base
    initial, updated = base.method33, candidate.method33
    components = component.split("+")
    positions = updated.trigger_positions if "position" in components else initial.trigger_positions
    lengths = updated.trigger_codes if "length" in components else initial.trigger_codes
    gaps = updated.trigger_codes if "gap" in components else initial.trigger_codes
    codes = tuple(tuple((l[0], g[1]) for l, g in zip(lpair, gpair, strict=True))
                  for lpair, gpair in zip(lengths, gaps, strict=True))
    mixed = replace(base, method33=replace(initial, trigger_positions=positions, trigger_codes=codes))
    validate_config(mixed)
    return mixed


def add_history_comparison(rows, x, y, days, local, warm_x):
    """Compare each history model with one trained only on labels from this regime."""
    x = StandardScaler().fit(warm_x).transform(x)
    for i, (row, (_, blind)) in enumerate(zip(rows, local, strict=True)):
        first = (i // REGIME_DAYS) * REGIME_DAYS
        for attack in ATTACKS:
            eligible = [local[j][0] for j in range(first, i)
                        if attack == "cumulative" or days[j] >= days[i] - np.timedelta64(SLIDING_DAYS, "D")]
            cal = np.concatenate(eligible) if eligible else np.empty(0, dtype=int)
            scratch = None
            if len(np.unique(y[cal])) >= 2:
                scratch = balanced_accuracy(fit_lr(x[cal], y[cal]), x[blind], y[blind])
            arm = row["attacks"][attack]
            arm["current_only_labels"] = len(cal)
            arm["current_only_ba"] = scratch
            arm["history_value"] = arm["post"] - scratch if scratch is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output-log", required=True, type=Path)
    parser.add_argument("--per-flow", action="store_true",
                        help="Run only the generated-codebook arm with IID mappings within each regime")
    parser.add_argument("--period-sweep", type=int, choices=(5, 10, 50),
                        help="Evaluate a new-codebook update period")
    parser.add_argument("--reuse-library", action="store_true",
                        help="At a five-day period, cycle through the first six codebooks")
    parser.add_argument("--component", choices=("fixed", "position", "length", "gap"),
                        help="At five days, freeze all other Trigger components at the first codebook")
    parser.add_argument("--control-arm", choices=("fast", "slow", "joint"),
                        help="Run one preregistered arm of the two-timescale factorial")
    args = parser.parse_args()
    if args.period_sweep and not args.per_flow:
        parser.error("--period-sweep requires --per-flow")
    if args.reuse_library and args.period_sweep != 5:
        parser.error("--reuse-library requires --period-sweep 5")
    if args.component and (args.period_sweep != 5 or args.reuse_library):
        parser.error("--component requires --period-sweep 5 and a fresh codebook stream")
    if args.control_arm and (args.period_sweep != 5 or args.reuse_library or args.component):
        parser.error("--control-arm requires a fresh, full --period-sweep 5 stream")
    pilot_bytes = args.pilot.read_bytes()
    pilot = json.loads(pilot_bytes)
    baseline = json.loads(args.baseline.read_bytes())
    seed = baseline["seed"]
    assert seed in (0, 1, 2)
    for key in ("sample_days", "warmup_indices", "mappings", "nonstationary_mixes",
                "dataset_sha256", "config_sha256", "settings"):
        assert baseline[key] == pilot[key], key
    with args.dataset.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == pilot["dataset_sha256"]
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == pilot["config_sha256"]
    assert pilot["settings"] == {
        "regime_observed_days": REGIME_DAYS, "labels_per_day": LABELS_PER_DAY,
        "sliding_calendar_days": SLIDING_DAYS, "threshold_ba": TARGET_BA,
    }
    project = load_config(args.config)
    period = args.period_sweep or REGIME_DAYS
    count = REGIMES * REGIME_DAYS // period
    horizon = REGIMES * REGIME_DAYS
    durations = [period] * count
    if args.control_arm in ("slow", "joint"):
        durations = [5, 10] * (horizon // 15)
        count = len(durations)
    assert sum(durations) == horizon
    original_seeds = tuple(range(5100 + seed * REGIMES, 5100 + (seed + 1) * REGIMES))
    extra_seeds = tuple(range(9000 + 100 * seed, 9000 + 100 * seed + max(0, count - REGIMES)))
    generator_seeds = (original_seeds + extra_seeds)[:count]
    if args.reuse_library:
        generator_seeds = original_seeds * (count // len(original_seeds))
    codebooks = {s: generated_project(project, s) for s in dict.fromkeys(generator_seeds)}
    candidates = [codebooks[s] for s in generator_seeds]
    if args.component:
        candidates = [freeze_components(candidates[0], p, args.component) for p in candidates]
    with np.load(args.dataset, allow_pickle=False) as source:
        data = {key: source[key] for key in ("lengths", "gaps", "directions", "mask", "labels", "groups")}
        stamps = source["timestamps"]
    groups = data["groups"].astype("datetime64[D]")
    assert np.array_equal(stamps[:, 0].astype("datetime64[s]").astype("datetime64[D]"), groups)
    assert np.array_equal(stamps.max(axis=1).astype("datetime64[s]").astype("datetime64[D]"), groups)
    del stamps
    y = data["labels"]
    warm = np.asarray(pilot["warmup_indices"])
    mappings = pilot["mappings"]
    mix = np.asarray(pilot["nonstationary_mixes"])
    iid = np.full(len(mappings), 1 / len(mappings))

    def uniforms(seed):
        return np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1]).random(len(y))

    warm_x, _ = render(data, warm, draw_states(uniforms(pilot["seed"])[warm], iid), mappings, project)
    days = np.array([row["day"] for row in pilot["sample_days"]], dtype="datetime64[D]")
    assert len(days) == REGIMES * REGIME_DAYS and np.all(groups[warm] < days[0])
    raw_indices, local = [], []
    offset = 0
    for day, row in zip(days, pilot["sample_days"], strict=True):
        cal, blind = row["cal_indices"], row["blind_indices"]
        assert len(cal) == LABELS_PER_DAY and np.all(groups[cal + blind] == day)
        split, end = offset + len(cal), offset + len(cal) + len(blind)
        raw_indices.extend(cal + blind)
        local.append((np.arange(offset, split), np.arange(split, end)))
        offset = end
    ids = np.asarray(raw_indices)
    assert len(np.unique(ids)) == len(ids) and not np.intersect1d(warm, ids).size
    u = uniforms(seed)[ids]
    day_index = np.repeat(np.arange(len(days)), [len(c) + len(b) for c, b in local])
    regime = np.repeat(np.arange(count), durations)[day_index]
    results = {}
    baseline_cost = None
    policies = ("Continual-New-Trigger",) if args.per_flow else ("IID", "Nonstationary", "Continual-New-Trigger")
    flow_states = draw_states(u, iid)
    probabilities = None
    if args.control_arm in ("fast", "joint"):
        base_weights = np.array([.28, .22, .17, .14, .11, .08])
        probabilities = np.stack([np.roll(base_weights, i % len(mappings)) for i in range(len(days))])
        assert np.allclose(probabilities.mean(axis=0), iid)
        cumulative_weights = probabilities.cumsum(axis=1)[:, :-1]
        policy_states = np.sum(u[:, None] >= cumulative_weights[day_index], axis=1)
    else:
        policy_states = flow_states
    for policy in policies:
        if policy == "Continual-New-Trigger":
            parts = []
            controls = {}
            for r, candidate in enumerate(candidates):
                mask = regime == r
                states = policy_states[mask] if args.per_flow else np.zeros(mask.sum(), dtype=int)
                parts.append(render(data, ids[mask], states, mappings, candidate))
                if r and not args.period_sweep:
                    day_index = r * period
                    blind_rows = local[day_index][1]
                    blind = ids[blind_rows]
                    previous_states = flow_states[blind_rows] if args.per_flow else np.zeros(len(blind), dtype=int)
                    controls[day_index], _ = render(
                        data, blind, previous_states, mappings, candidates[r - 1])
            x = np.concatenate([p[0] for p in parts])
            cost = np.concatenate([p[1] for p in parts])
            rows = evaluate(x, y[ids], days, local, groups[warm], y[warm], warm_x, controls,
                            regime_days=period)
        else:
            states = flow_states if policy == "IID" else np.concatenate([
                draw_states(u[regime == r], weights) for r, weights in enumerate(mix)])
            x, cost = render(data, ids, states, mappings, project)
            rows = baseline["results"][policy]["days"]
            for row, (_, blind) in zip(rows, local, strict=True):
                assert support(states[blind]) == row["blind_state_support"]
        if baseline_cost is None:
            if args.per_flow:
                _, baseline_cost = render(data, ids, flow_states, mappings, project)
            else:
                baseline_cost = cost
        assert np.allclose(cost, baseline_cost, rtol=0, atol=1e-8)
        for i, (row, (_, blind)) in enumerate(zip(rows, local, strict=True)):
            assert row["day"] == str(days[i]) and row["blind_support"] == support(y[ids[blind]])
            assert row["labels_available"] == len(warm) + LABELS_PER_DAY * i
            if args.per_flow:
                row["blind_state_support"] = support(policy_states[blind])
                if probabilities is None:
                    assert row["blind_state_support"] == baseline["results"]["IID"]["days"][i]["blind_state_support"]
        if not args.period_sweep:
            add_history_comparison(rows, x, y[ids], days, local, warm_x)
        results[policy] = {"days": rows, "summary": summarize(rows), "cost_max": cost.max(axis=0).tolist()}
        print(seed, policy, {a: r["mean_post_ba"] for a, r in results[policy]["summary"].items()}, flush=True)
    report = {
        "seed": seed, "pilot_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "settings": pilot["settings"],
        "generated_mapping": mappings if args.per_flow else mappings[0],
        "within_regime_per_flow": args.per_flow,
        "codebook_period_observed_days": period,
        "reuse_library": args.reuse_library,
        "updated_component": args.component,
        "candidate_parameters": [{"seed": s, "positions": p.method33.trigger_positions,
                                  "codes": p.method33.trigger_codes}
                                 for s, p in zip(generator_seeds, candidates, strict=True)],
        "results": results,
    }
    if args.control_arm:
        report["control"] = {"arm": args.control_arm, "codebook_durations": durations,
                             "day_state_probabilities": probabilities.tolist() if probabilities is not None else None}
    args.output_log.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
