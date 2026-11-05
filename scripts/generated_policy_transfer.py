"""Paired transfer pilot for budget-matched generated Trigger codebooks."""

import argparse
import hashlib
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from methods.method33 import resolve_trigger_positions
from methods.method33_tools import load_config, validate_config
from scripts.online_nonstationary import balanced_accuracy, draw_states
from scripts.paired_mapping_policies import render, support


BUDGETS = (0, 8, 32, 128, 512)
GENERATOR_SEEDS = (3101, 3102, 3103)
WINDOWS = (1, 2)
LIBRARY = (0, 2, 4)


def generated_project(project, seed):
    rng = np.random.default_rng(seed)
    config = project.method33
    old_pairs = set(resolve_trigger_positions(config))
    slots = [pair for pair in combinations(range(1, 17), 2) if pair not in old_pairs]
    positions = tuple(slots[i] for i in rng.choice(len(slots), config.class_count, replace=False))
    first_lengths = rng.choice(np.arange(90, 194), config.class_count, replace=False)
    first_gaps = rng.integers(40, 411, size=config.class_count) / 1000
    codes = tuple(((float(length), float(gap)),
                   (config.trigger_budget.bytes - float(length), config.trigger_budget.delay_seconds - float(gap)))
                  for length, gap in zip(first_lengths, first_gaps, strict=True))
    candidate = replace(project, method33=replace(config, trigger_codes=codes, trigger_positions=positions))
    validate_config(candidate)
    return candidate


def train(x, y):
    if len(np.unique(y)) < 2:
        return None
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=0)).fit(x, y)


def sample_windows(groups, completed, pilot):
    dates = np.array([row["day"] for row in pilot["sample_days"]], dtype="datetime64[D]")
    for window in WINDOWS:
        rng = np.random.default_rng(np.random.SeedSequence([4100, window]))
        days = dates[25 * window:25 * (window + 1)]
        updates, remainder = [], []
        for day in days[:12]:
            pool = rng.permutation(np.flatnonzero(groups == day))
            updates.extend(pool[:64])
            remainder.extend(pool[64:])
        update = np.asarray(updates)
        update = update[np.lexsort((update, completed[update]))][:BUDGETS[-1]]
        reference_pool = np.union1d(remainder, np.setdiff1d(updates, update))
        reference = rng.choice(reference_pool, min(1500, len(reference_pool)), replace=False)
        blind_pool = np.flatnonzero(np.isin(groups, days[12:]))
        blind = rng.choice(blind_pool, min(1000, len(blind_pool)), replace=False)
        assert len(update) == BUDGETS[-1] and len(reference) > len(update)
        assert len(np.unique(np.r_[update, reference, blind])) == len(update) + len(reference) + len(blind)
        assert completed[np.r_[update, reference]].max() < completed[blind].min()
        yield window, days, update, reference, blind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output-log", required=True, type=Path)
    args = parser.parse_args()
    pilot_bytes = args.pilot.read_bytes()
    pilot = json.loads(pilot_bytes)
    with args.dataset.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == pilot["dataset_sha256"]
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == pilot["config_sha256"]
    project = load_config(args.config)
    candidates = {f"Generated-{seed}": generated_project(project, seed) for seed in GENERATOR_SEEDS}
    with np.load(args.dataset, allow_pickle=False) as source:
        data = {key: source[key] for key in ("lengths", "gaps", "directions", "mask", "labels", "groups")}
        completed = source["timestamps"].max(axis=1)
    labels = data["labels"]
    groups = data["groups"].astype("datetime64[D]")
    assert np.array_equal(completed.astype("datetime64[s]").astype("datetime64[D]"), groups)
    mappings = pilot["mappings"]
    warm = np.asarray(pilot["warmup_indices"])
    rng = np.random.default_rng(np.random.SeedSequence(pilot["seed"]).spawn(3)[1])
    uniform = rng.random(len(labels))
    iid = draw_states(uniform, np.full(len(mappings), 1 / len(mappings)))
    hx, _ = render(data, warm, iid[warm], mappings, project)
    hy = labels[warm]
    old = train(hx, hy)
    results = []
    for window, days, update, reference, blind in sample_windows(groups, completed, pilot):
        ids = np.r_[update, reference, blind]
        assert completed[warm].max() < completed[ids].min()
        reference_start = len(update)
        blind_start = len(update) + len(reference)
        yb = labels[blind]
        row = {
            "window": window, "cal_days": days[:12].astype(str).tolist(), "blind_days": days[12:].astype(str).tolist(),
            "update_indices": update.tolist(), "reference_indices": reference.tolist(), "blind_indices": blind.tolist(),
            "blind_support": support(yb), "reference_support": support(labels[reference]), "arms": {},
        }
        arms = {"IID": (project, iid[ids])}
        arms.update({f"Library-{s}": (project, np.full(len(ids), s)) for s in LIBRARY})
        arms.update({name: (candidate, np.zeros(len(ids), dtype=int)) for name, candidate in candidates.items()})
        for name, (arm, states) in arms.items():
            x, cost = render(data, ids, states, mappings, arm)
            if name == "IID":
                baseline_cost = cost
            assert np.allclose(cost, baseline_cost, rtol=0, atol=1e-8)
            xb = x[blind_start:]
            zero = balanced_accuracy(old, xb, yb)
            if name == "IID":
                baseline_ba = zero
            ref = train(x[reference_start:blind_start], labels[reference])
            checkpoints = []
            for n in BUDGETS:
                y = labels[update[:n]]
                updated = train(np.concatenate((hx, x[:n])), np.concatenate((hy, y))) if n else old
                scratch = train(x[:n], y)
                updated_ba = balanced_accuracy(updated, xb, yb)
                scratch_ba = balanced_accuracy(scratch, xb, yb) if scratch is not None else None
                checkpoints.append({
                    "n": n, "support": support(y), "history_plus_ba": updated_ba, "scratch_ba": scratch_ba,
                    "history_value": updated_ba - scratch_ba if scratch_ba is not None else None,
                    "last_label_utc": str(np.datetime64(int(completed[update[n - 1]]), "s")) if n else None,
                })
            row["arms"][name] = {
                "transfer_delta_vs_iid": zero - baseline_ba,
                "reference_ba": balanced_accuracy(ref, xb, yb) if ref is not None else None,
                "checkpoints": checkpoints, "cost_mean": cost.mean(axis=0).tolist(),
                "cost_max": cost.max(axis=0).tolist(),
            }
            print(window, name, round(zero, 4), [round(r["history_plus_ba"], 4) for r in checkpoints], flush=True)
        results.append(row)
    report = {
        "pilot_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
        "dataset_sha256": pilot["dataset_sha256"], "config_sha256": pilot["config_sha256"],
        "warmup_indices": warm.tolist(), "warmup_support": support(hy),
        "candidates": {name: {"positions": p.method33.trigger_positions, "codes": p.method33.trigger_codes}
                       for name, p in candidates.items()},
        "library_mappings": {str(s): mappings[s] for s in LIBRARY},
        "generated_mapping": mappings[0], "budget_grid": BUDGETS,
        "cost_order": ["bytes", "seconds", "dummies"], "windows": results,
    }
    args.output_log.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
