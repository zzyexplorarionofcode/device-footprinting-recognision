"""Leave-one-change-out (LOCO) ablation for the searched position layout.

Configurations:

    Base            geometry codebook
    Full            searched position assignment (5 classes changed)
    Full-c{i}       Full, but class i's position is reverted to Base

For every configuration the script first checks that reverting a single class
does not collide with another class's current position.  If it would, the
configuration is marked as a conflict and is NOT silently repaired.

All configurations share the same date folds, train/validation samples,
surrogates, hyper-parameters and bootstrap day clusters.  Deltas are reported
relative to Full:

    I_i = BA(Full-c_i) - BA(Full)

so I_i > 0 means the original change contributed, I_i ~ 0 means it is redundant
in this combination, and I_i < 0 means it may be harmful.  This is a conditional
contribution: LOCO cannot prove a change is necessary in every combination.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from scripts.position_ablation import cap_with_min, predict  # noqa: E402
from scripts.es_codebook_search import defended, features  # noqa: E402
from scripts.fold_review import assign_dates, take  # noqa: E402


def positions_of(config) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in pair) for pair in config.method33.trigger_positions]


def build_configs(base_positions, searched_positions):
    changed = [i for i, (a, b) in enumerate(zip(base_positions, searched_positions)) if a != b]
    variants = {"Base": list(base_positions), "Full": list(searched_positions)}
    conflicts = {}
    for label in changed:
        trial = list(searched_positions)
        trial[label] = base_positions[label]
        if len(set(trial)) != len(trial):
            conflicts[f"Full-c{label}"] = {
                "class": label,
                "base_position": list(base_positions[label]),
                "duplicate_of": [
                    index for index, position in enumerate(trial) if position == trial[label] and index != label
                ],
            }
            continue
        variants[f"Full-c{label}"] = trial
    return changed, variants, conflicts


def run_config(dataset, project, fold_data, codes, positions, surrogate, class_count):
    per_fold = []
    for train_indices, validation_indices in fold_data:
        train = take(dataset, train_indices)
        validation = take(dataset, validation_indices)
        config = dataclasses.replace(
            project.method33, trigger_codes=codes, trigger_positions=tuple(positions)
        )
        train_defended = defended(train, "old", config, 731800, project.trigger_jitter)
        validation_defended = defended(validation, "next", config, 731800, project.trigger_jitter)
        x_train = features(train_defended)
        x_validation = features(validation_defended)
        prediction = predict(x_train, train[4], x_validation, surrogate)
        y_true = validation[4]
        days = np.asarray(dataset["groups"])[validation_indices].astype(str)
        per_fold.append(
            {
                "y_true": y_true,
                "prediction": prediction,
                "days": days,
                "ba": float(balanced_accuracy_score(y_true, prediction)),
                "recall": recall_score(
                    y_true, prediction, labels=np.arange(class_count), average=None, zero_division=0
                ),
            }
        )
    return per_fold


def paired_bootstrap(full_folds, candidate_folds, class_count, rng, repeats):
    deltas_ba = []
    deltas_recall = []
    for _ in range(repeats):
        fold_ba = []
        fold_recall = []
        for full, candidate in zip(full_folds, candidate_folds):
            days = np.unique(full["days"])
            draw = rng.choice(days, size=len(days), replace=True)
            indices = np.concatenate([np.where(full["days"] == day)[0] for day in draw])
            fold_ba.append(
                balanced_accuracy_score(candidate["y_true"][indices], candidate["prediction"][indices])
                - balanced_accuracy_score(full["y_true"][indices], full["prediction"][indices])
            )
            fold_recall.append(
                recall_score(
                    candidate["y_true"][indices],
                    candidate["prediction"][indices],
                    labels=np.arange(class_count),
                    average=None,
                    zero_division=0,
                )
                - recall_score(
                    full["y_true"][indices],
                    full["prediction"][indices],
                    labels=np.arange(class_count),
                    average=None,
                    zero_division=0,
                )
            )
        deltas_ba.append(float(np.mean(fold_ba)))
        deltas_recall.append(np.mean(fold_recall, axis=0))
    low_ba, high_ba = np.percentile(deltas_ba, [2.5, 97.5])
    recall_array = np.asarray(deltas_recall)
    low_recall, high_recall = np.percentile(recall_array, [2.5, 97.5], axis=0)
    return (
        float(np.mean(deltas_ba)),
        float(low_ba),
        float(high_ba),
        np.mean(recall_array, axis=0).tolist(),
        low_recall.tolist(),
        high_recall.tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--validation-budget", type=int, default=6000)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    pool = np.asarray(split["train"])
    if "validation" in split and len(split["validation"]) > 0:
        pool = np.concatenate([pool, np.asarray(split["validation"])])

    days = np.sort(np.unique(groups[pool]))
    class_count = int(labels.max()) + 1
    matrix = np.zeros((len(days), class_count), dtype=np.int64)
    for index, day in enumerate(days):
        matrix[index] = np.bincount(labels[pool][groups[pool] == day], minlength=class_count)
    folds = int(min(args.folds, (matrix > 0).sum(axis=0).min()))
    assignment = assign_dates(matrix, folds, args.seed)
    fold_data = []
    for fold, day_indices in enumerate(assignment):
        fold_days = set(days[day_indices].tolist())
        validation_mask = np.asarray([group in fold_days for group in groups[pool]])
        rng = np.random.default_rng(args.seed + fold)
        train_idx = cap_with_min(pool[~validation_mask], labels, args.max_train, 1, rng)
        validation_idx = cap_with_min(pool[validation_mask], labels, args.validation_budget, args.min_per_class, rng)
        fold_data.append((train_idx, validation_idx))
        print(
            f"[fold {fold}] days={len(fold_days)} train={len(train_idx)} validation={len(validation_idx)} "
            f"n_per_class={np.bincount(labels[validation_idx], minlength=class_count).tolist()}",
            flush=True,
        )

    base = method33_tools.load_config(args.base_config)
    full = method33_tools.load_config(args.full_config)
    base_positions = positions_of(base)
    full_positions = positions_of(full)
    changed, variants, conflicts = build_configs(base_positions, full_positions)
    codes = base.method33.trigger_codes
    print(f"changed classes={changed} conflicts={conflicts}", flush=True)
    print(f"positions: {base_positions} -> {full_positions}", flush=True)

    started = time.time()
    predictions = {}
    for surrogate in ("lr", "rf", "mlp"):
        for name in ["Base", "Full"] + [f"Full-c{label}" for label in changed]:
            if name not in variants:
                continue
            predictions.setdefault(name, {})[surrogate] = run_config(
                dataset, base, fold_data, codes, variants[name], surrogate, class_count
            )
        print(f"[{surrogate}] predictions done", flush=True)

    summary = {}
    for surrogate in ("lr", "rf", "mlp"):
        full_folds = predictions["Full"][surrogate]
        full_ba = float(np.mean([fold["ba"] for fold in full_folds]))
        summary[surrogate] = {"Full": {"ba": full_ba, "delta_ba": 0.0}}
        for name in ["Base"] + [f"Full-c{label}" for label in changed]:
            if name not in variants:
                continue
            candidate_folds = predictions[name][surrogate]
            rng = np.random.default_rng(
                args.seed + sum(ord(character) for character in name) + len(surrogate)
            )
            delta, low, high, recall_delta, recall_low, recall_high = paired_bootstrap(
                full_folds, candidate_folds, class_count, rng, args.bootstrap
            )
            candidate_ba = float(np.mean([fold["ba"] for fold in candidate_folds]))
            summary[surrogate][name] = {
                "ba": candidate_ba,
                "delta_ba": float(candidate_ba - full_ba),
                "delta_ba_bootstrap": delta,
                "delta_ba_ci_low": low,
                "delta_ba_ci_high": high,
                "delta_per_class_recall": recall_delta,
                "delta_per_class_ci_low": recall_low,
                "delta_per_class_ci_high": recall_high,
            }
        print(
            f"[{surrogate}] Full={full_ba:.4f} "
            + " ".join(
                f"{name}={summary[surrogate][name]['delta_ba']:+.4f}"
                for name in summary[surrogate]
                if name != "Full"
            ),
            flush=True,
        )

    log = {
        "dataset": str(args.dataset),
        "base_config": str(args.base_config),
        "full_config": str(args.full_config),
        "changed_classes": changed,
        "conflicts": conflicts,
        "positions": {"base": [list(p) for p in base_positions], "full": [list(p) for p in full_positions]},
        "folds": folds,
        "fold_dates": [[str(days[index]) for index in block] for block in assignment],
        "fold_samples": [
            {
                "train": len(train_idx),
                "validation": len(validation_idx),
                "n_per_class": np.bincount(labels[validation_idx], minlength=class_count).tolist(),
            }
            for train_idx, validation_idx in fold_data
        ],
        "summary": summary,
        "seconds": round(time.time() - started, 1),
        "note": "LOCO conditional contributions; holdout untouched",
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                surrogate: {
                    name: round(value["delta_ba"], 4) for name, value in summary[surrogate].items()
                }
                for surrogate in ("lr", "rf", "mlp")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
