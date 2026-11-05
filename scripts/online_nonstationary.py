"""Chronological replay of stationary and nonstationary FADE schedules."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from methods.method33_tools import load_config
from scripts.paired_mapping_policies import render, support
from scripts.streaming_arms import make_mappings


POLICIES = ("Static", "IID", "Periodic", "Nonstationary")
ATTACKS = ("cumulative", "sliding")
WARMUP_DAYS = 20
REGIME_DAYS = 25
REGIMES = 6
LABELS_PER_DAY = 8
BLIND_CAP = 128
SLIDING_DAYS = 14
TARGET_BA = 0.70


def draw_states(uniform, probabilities):
    return np.searchsorted(np.cumsum(probabilities)[:-1], uniform, side="right")


def daily_samples(groups, days, rng):
    samples = []
    for day in days:
        pool = np.flatnonzero(groups == day)
        ids = rng.choice(pool, min(len(pool), LABELS_PER_DAY + BLIND_CAP), replace=False)
        samples.append((ids[:LABELS_PER_DAY], ids[LABELS_PER_DAY:]))
    return samples


def fit_lr(x, y):
    return LogisticRegression(max_iter=500, random_state=0).fit(x, y)


def balanced_accuracy(model, x, y):
    pred = model.predict(x)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def class_conditional_shift(before, after, y):
    """RMS change in conditional feature means on paired original flows."""
    return {
        str(c): float(np.sqrt(np.mean((after[y == c].mean(0) - before[y == c].mean(0))**2)))
        for c in np.unique(y)
    }


def evaluate(x, y, days, samples, warm_days, warm_y, warm_x, switch_control,
             regime_days=REGIME_DAYS, fit_model=fit_lr, scale_reference=None):
    scaler = StandardScaler().fit(warm_x if scale_reference is None else scale_reference)
    warm_x = scaler.transform(warm_x)
    x = scaler.transform(x)
    switch_control = {i: scaler.transform(v) for i, v in switch_control.items()}
    initial = fit_model(warm_x, warm_y)
    snapshots = {a: initial for a in ATTACKS}
    past_x, past_y, past_days = warm_x, warm_y, warm_days
    rows = []
    for i, (cal, blind) in enumerate(samples):
        day = days[i]
        regime = i // regime_days
        if i % regime_days == 0:
            origin = day
            labels_since_switch = 0
            pre = snapshots.copy()
        xb, yb = x[blind], y[blind]
        row = {
            "day": str(day), "regime": regime, "elapsed_days": int((day - origin).astype(int)),
            "labels_since_switch": labels_since_switch, "labels_available": len(past_y),
            "blind_support": support(yb), "attacks": {},
        }
        for attack in ATTACKS:
            eligible = past_days >= day - np.timedelta64(SLIDING_DAYS, "D")
            if attack == "cumulative":
                eligible = np.ones(len(past_y), dtype=bool)
            assert np.all(past_days[eligible] < day)
            used = len(np.unique(past_y[eligible])) >= 2
            snapshots[attack] = fit_model(past_x[eligible], past_y[eligible]) if used else initial
            model = snapshots[attack]
            if i % regime_days == 0:
                pre[attack] = model
            scores = {
                "pre": balanced_accuracy(pre[attack], xb, yb),
                "post": balanced_accuracy(model, xb, yb),
                "labels_used": int(eligible.sum()) if used else len(warm_y),
                "known_classes": model.classes_.tolist(),
                "warmup_fallback": not used,
            }
            if i in switch_control:
                scores["pre_under_previous_mix"] = balanced_accuracy(pre[attack], switch_control[i], yb)
            row["attacks"][attack] = scores
        if i in switch_control:
            row["conditional_feature_mean_shift"] = class_conditional_shift(switch_control[i], xb, yb)
        rows.append(row)
        # Same-day labels are released only after prediction. Blind labels never enter this buffer.
        past_x = np.concatenate((past_x, x[cal]))
        past_y = np.concatenate((past_y, y[cal]))
        past_days = np.concatenate((past_days, np.repeat(day, len(cal))))
        labels_since_switch += len(cal)
    return rows


def summarize(rows):
    result = {}
    for attack in ATTACKS:
        stages = []
        for regime in range(rows[-1]["regime"] + 1):
            window = [r for r in rows if r["regime"] == regime]
            sustained = next((a for a, b in zip(window, window[1:]) if
                              a["attacks"][attack]["post"] >= TARGET_BA and
                              b["attacks"][attack]["post"] >= TARGET_BA), None)
            stages.append({
                "regime": regime,
                "first_sustained_day": sustained["elapsed_days"] if sustained else None,
                "first_sustained_labels": sustained["labels_since_switch"] if sustained else None,
                "horizon_days": window[-1]["elapsed_days"],
                "horizon_labels": window[-1]["labels_since_switch"],
            })
        result[attack] = {
            "mean_pre_ba": float(np.mean([r["attacks"][attack]["pre"] for r in rows])),
            "mean_post_ba": float(np.mean([r["attacks"][attack]["post"] for r in rows])),
            "threshold": stages,
        }
    return result


def sweep_schedules(mixes):
    mixes = np.asarray(mixes)
    order = np.argsort(mixes.argmax(axis=1), kind="stable")
    return {
        "Short": (5, np.tile(mixes, (5, 1))),
        "Diffuse": (REGIME_DAYS, (mixes + 1 / 6) / 2),
        "Ordered": (REGIME_DAYS, mixes[order]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-log", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", type=Path, help="Freeze pilot samples, mappings, schedule and warmup")
    parser.add_argument("--schedule-sweep", action="store_true",
                        help="Run only the three prespecified schedule variants")
    args = parser.parse_args()
    if args.schedule_sweep and not args.replay:
        parser.error("--schedule-sweep requires --replay")
    with np.load(args.dataset, allow_pickle=False) as source:
        data = {key: source[key] for key in ("lengths", "gaps", "directions", "mask", "labels", "groups")}
        timestamps = source["timestamps"]
    groups = data["groups"].astype("datetime64[D]")
    assert np.array_equal(timestamps[:, 0].astype("datetime64[s]").astype("datetime64[D]"), groups)
    assert np.array_equal(timestamps.max(axis=1).astype("datetime64[s]").astype("datetime64[D]"), groups)
    del timestamps
    labels = data["labels"]
    dates = np.unique(groups)
    days = dates[WARMUP_DAYS:WARMUP_DAYS + REGIMES * REGIME_DAYS]
    if len(days) != REGIMES * REGIME_DAYS:
        raise ValueError("Dataset has insufficient days for the declared horizon")
    project = load_config(args.config)
    with args.dataset.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    config_digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
    mappings = make_mappings(project, args.seed)
    rng_sample, rng_state, rng_schedule = [
        np.random.default_rng(seed) for seed in np.random.SeedSequence(args.seed).spawn(3)
    ]
    # One fixed uniform per original row, shared across policies and unaffected by sampling.
    uniform = rng_state.random(len(labels))
    stationary = np.full(6, 1 / 6)
    base = np.array([0.31, 0.25, 0.20, 0.11, 0.08, 0.05])
    mixes = []
    for _ in range(REGIMES // 2):
        mix = rng_schedule.permutation(base)
        mixes.extend((mix, 2 * stationary - mix))
    assert np.allclose(np.mean(mixes, axis=0), stationary)
    fixed = int(rng_schedule.integers(6))
    phase = int(rng_schedule.integers(6))
    warm_pool = np.flatnonzero(groups < days[0])
    warm = np.sort(rng_sample.choice(warm_pool, min(4000, len(warm_pool)), replace=False))
    daily = daily_samples(groups, days, rng_sample)
    warm_uniform = uniform[warm]
    policies = POLICIES
    if args.replay:
        pilot = json.loads(args.replay.read_text(encoding="utf-8"))
        if (pilot["dataset_sha256"], pilot["config_sha256"]) != (digest, config_digest):
            raise ValueError("Replay requires the pilot dataset and configuration")
        if pilot["settings"] != {
            "regime_observed_days": REGIME_DAYS, "labels_per_day": LABELS_PER_DAY,
            "sliding_calendar_days": SLIDING_DAYS, "threshold_ba": TARGET_BA,
        }:
            raise ValueError("Replay requires the pilot timing and evaluation settings")
        if [row["day"] for row in pilot["sample_days"]] != days.astype(str).tolist():
            raise ValueError("Replay dates differ from the declared horizon")
        mappings = pilot["mappings"]
        mixes = np.asarray(pilot["nonstationary_mixes"])
        fixed, phase = pilot["fixed_state"], pilot["periodic_phase"]
        warm = np.asarray(pilot["warmup_indices"])
        daily = [(np.asarray(row["cal_indices"]), np.asarray(row["blind_indices"]))
                 for row in pilot["sample_days"]]
        historical_rng = np.random.default_rng(np.random.SeedSequence(pilot["seed"]).spawn(3)[1])
        warm_uniform = historical_rng.random(len(labels))[warm]
        policies = ("IID", "Nonstationary")
    assert np.all(groups[warm] < days[0])
    for day, (cal, blind) in zip(days, daily):
        assert len(cal) == LABELS_PER_DAY and np.all(groups[np.r_[cal, blind]] == day)
    warm_x, _ = render(data, warm, draw_states(warm_uniform, stationary), mappings, project)
    ids = np.concatenate([np.r_[cal, blind] for cal, blind in daily])
    assert len(np.unique(ids)) == len(ids) and not np.intersect1d(warm, ids).size
    local, offset = [], 0
    for cal, blind in daily:
        split, end = offset + len(cal), offset + len(cal) + len(blind)
        local.append((np.arange(offset, split), np.arange(split, end)))
        offset = end
    day_index = np.repeat(np.arange(len(days)), [len(c) + len(b) for c, b in daily])
    regime = day_index // REGIME_DAYS
    states = {
        "Static": np.full(len(ids), fixed),
        "IID": draw_states(uniform[ids], stationary),
        "Periodic": (regime + phase) % 6,
        "Nonstationary": np.concatenate([
            draw_states(uniform[ids[regime == i]], mix) for i, mix in enumerate(mixes)
        ]),
    }
    schedules = {"Nonstationary": (REGIME_DAYS, np.asarray(mixes))}
    if args.schedule_sweep:
        schedules = sweep_schedules(mixes)
        policies = tuple(schedules)
        states = {}
        for policy, (period, weights) in schedules.items():
            assert period * len(weights) == len(days)
            assert np.allclose(weights.mean(axis=0), stationary)
            policy_regime = day_index // period
            states[policy] = np.concatenate([
                draw_states(uniform[ids[policy_regime == i]], mix)
                for i, mix in enumerate(weights)
            ])
    results = {}
    baseline_cost = None
    for policy in policies:
        x, cost = render(data, ids, states[policy], mappings, project)
        if baseline_cost is None:
            baseline_cost = cost
        assert np.allclose(cost, baseline_cost, rtol=0, atol=1e-8)
        control = {}
        period, weights = schedules.get(policy, (REGIME_DAYS, None))
        if weights is not None:
            for i in range(period, len(days), period):
                blind = ids[local[i][1]]
                counterfactual = draw_states(uniform[blind], weights[i // period - 1])
                control[i], _ = render(data, blind, counterfactual, mappings, project)
        rows = evaluate(x, labels[ids], days, local, groups[warm], labels[warm], warm_x, control, period)
        for row, (_, blind) in zip(rows, local):
            row["blind_state_support"] = support(states[policy][blind])
        results[policy] = {
            "summary": summarize(rows), "days": rows,
            "cost_mean": cost.mean(axis=0).tolist(), "cost_max": cost.max(axis=0).tolist(),
        }
        print(policy, {a: r["mean_post_ba"] for a, r in results[policy]["summary"].items()}, flush=True)
    report = {
        "seed": args.seed, "dataset_sha256": digest,
        "config_sha256": config_digest,
        "replay_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest() if args.replay else None,
        "settings": {"regime_observed_days": REGIME_DAYS, "labels_per_day": LABELS_PER_DAY,
                     "sliding_calendar_days": SLIDING_DAYS, "threshold_ba": TARGET_BA},
        "mappings": np.asarray(mappings).tolist(), "fixed_state": fixed, "periodic_phase": phase,
        "nonstationary_mixes": np.asarray(mixes).tolist(), "warmup_indices": warm.tolist(),
        "warmup_support": support(labels[warm]), "cost_order": ["bytes", "seconds", "dummies"],
        "sample_days": [{"day": str(day), "cal_indices": c.tolist(), "blind_indices": b.tolist(),
                         "cal_support": support(labels[c])} for day, (c, b) in zip(days, daily)],
        "results": results,
    }
    if args.schedule_sweep:
        report["sweep_schedules"] = {
            policy: {"regime_observed_days": period, "mixes": weights.tolist()}
            for policy, (period, weights) in schedules.items()
        }
    args.output_log.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
