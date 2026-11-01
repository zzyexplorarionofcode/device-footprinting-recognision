"""2x2 factorial experiment for the c5 and c6 position changes.

Four configurations (1 = new/searched position, 0 = base position):

    11  Full                 c5 new, c6 new
    10  Full-c6              c5 new, c6 base
    01  Full-c5              c5 base, c6 new
    00  Full-c5-c6           c5 base, c6 base

For class 5 recall R (lower is better defence), the interaction is

    I = R11 - R10 - R01 + R00

I < 0 means the two changes reduce class-5 recall more than additively
(synergy); I > 0 means the combined effect is smaller than the sum of the
individual effects.  The same interaction is reported for balanced accuracy.

All four configurations share folds, samples, surrogates, seeds and the
day-cluster bootstrap.  Position uniqueness is checked before evaluation.
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
from scripts.es_codebook_search import defended, features  # noqa: E402
from scripts.fold_review import assign_dates, take  # noqa: E402
from scripts.position_ablation import cap_with_min, predict  # noqa: E402


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


def bootstrap_interaction(folds_by_config, class_count, rng, repeats):
    interactions = []
    for _ in range(repeats):
        sampled = []
        for fold in range(len(next(iter(folds_by_config.values())))):
            days = np.unique(folds_by_config["11"][fold]["days"])
            draw = rng.choice(days, size=len(days), replace=True)
            indices = np.concatenate(
                [np.where(folds_by_config["11"][fold]["days"] == day)[0] for day in draw]
            )
            row = {}
            valid = True
            for name, folds in folds_by_config.items():
                truth = folds[fold]["y_true"][indices]
                if not np.any(truth == 5):
                    valid = False
                    break
                row[name] = float(
                    recall_score(truth, folds[fold]["prediction"][indices], labels=[5], average=None)[0]
                )
            if not valid:
                continue
            sampled.append(row)
        if not sampled:
            continue
        means = {
            name: float(np.mean([row[name] for row in sampled])) for name in folds_by_config
        }
        interactions.append(means["11"] - means["10"] - means["01"] + means["00"])
    interactions = np.asarray(interactions)
    return (
        float(interactions.mean()),
        float(np.percentile(interactions, 2.5)),
        float(np.percentile(interactions, 97.5)),
        int(len(interactions)),
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
            f"[fold {fold}] validation={len(validation_idx)} "
            f"n_per_class={np.bincount(labels[validation_idx], minlength=class_count).tolist()}",
            flush=True,
        )

    base = method33_tools.load_config(args.base_config)
    full = method33_tools.load_config(args.full_config)
    base_positions = [tuple(int(v) for v in pair) for pair in base.method33.trigger_positions]
    full_positions = [tuple(int(v) for v in pair) for pair in full.method33.trigger_positions]
    codes = base.method33.trigger_codes

    variants = {"11": list(full_positions)}
    variants["10"] = list(full_positions)
    variants["10"][6] = base_positions[6]
    variants["01"] = list(full_positions)
    variants["01"][5] = base_positions[5]
    variants["00"] = list(full_positions)
    variants["00"][5] = base_positions[5]
    variants["00"][6] = base_positions[6]

    conflicts = {}
    for name, positions in variants.items():
        if len(set(positions)) != len(positions):
            conflicts[name] = [
                index for index, position in enumerate(positions) if positions.count(position) > 1
            ]
    print(f"conflicts={conflicts}", flush=True)
    print(f"positions 11={variants['11']} 00={variants['00']}", flush=True)

    started = time.time()
    folds_by_config = {}
    for surrogate in ("lr", "rf", "mlp"):
        for name in ("11", "10", "01", "00"):
            folds_by_config.setdefault(surrogate, {})[name] = run_config(
                dataset, base, fold_data, codes, variants[name], surrogate, class_count
            )
        print(f"[{surrogate}] predictions done", flush=True)

    summary = {}
    for surrogate in ("lr", "rf", "mlp"):
        folds = folds_by_config[surrogate]
        r = {
            name: float(np.mean([fold["recall"][5] for fold in folds[name]]))
            for name in ("11", "10", "01", "00")
        }
        ba = {
            name: float(np.mean([fold["ba"] for fold in folds[name]]))
            for name in ("11", "10", "01", "00")
        }
        interaction = r["11"] - r["10"] - r["01"] + r["00"]
        rng = np.random.default_rng(args.seed + sum(ord(character) for character in surrogate))
        mean_i, low_i, high_i, effective = bootstrap_interaction(
            folds, class_count, rng, args.bootstrap
        )
        summary[surrogate] = {
            "class5_recall": r,
            "ba": ba,
            "class5_interaction": float(interaction),
            "class5_interaction_bootstrap_mean": mean_i,
            "class5_interaction_ci_low": low_i,
            "class5_interaction_ci_high": high_i,
            "bootstrap_replicates": effective,
            "marginal_c5_effect_when_c6_new": r["11"] - r["01"],
            "marginal_c5_effect_when_c6_base": r["10"] - r["00"],
            "marginal_c6_effect_when_c5_new": r["11"] - r["10"],
            "marginal_c6_effect_when_c5_base": r["01"] - r["00"],
            "ba_interaction": ba["11"] - ba["10"] - ba["01"] + ba["00"],
        }
        print(
            f"[{surrogate}] R11={r['11']:.3f} R10={r['10']:.3f} R01={r['01']:.3f} R00={r['00']:.3f} "
            f"I={interaction:+.3f} CI=[{low_i:+.3f},{high_i:+.3f}]",
            flush=True,
        )

    log = {
        "dataset": str(args.dataset),
        "base_config": str(args.base_config),
        "full_config": str(args.full_config),
        "conflicts": conflicts,
        "positions": {name: [list(position) for position in positions] for name, positions in variants.items()},
        "fold_dates": [[str(days[index]) for index in block] for block in assignment],
        "summary": summary,
        "seconds": round(time.time() - started, 1),
        "note": "2x2 factorial on validation folds; holdout untouched",
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                surrogate: {
                    "I_class5": round(summary[surrogate]["class5_interaction"], 3),
                    "ci": [
                        round(summary[surrogate]["class5_interaction_ci_low"], 3),
                        round(summary[surrogate]["class5_interaction_ci_high"], 3),
                    ],
                }
                for surrogate in ("lr", "rf", "mlp")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
