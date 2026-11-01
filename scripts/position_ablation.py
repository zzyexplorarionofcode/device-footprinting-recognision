"""Position-layout ablation with corrected date folds and rare-class oversampling.

Five configurations are compared under identical folds:

    base          geometry codebook
    class5        base, but only class 5's position moved to the searched pair
    class0        base, but only class 0's position moved
    both          base, with class 0 and class 5 moved together
    searched      the full searched position assignment

Each fold is a multi-label stratified date block (whole dates only), and the
validation cap enforces a minimum number of samples per class (default 200), so
per-class recall is not dominated by 8-13 examples.  Recall uncertainty uses a
cluster bootstrap over validation *days*.

This script is diagnostic: it does not search and does not touch the holdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from scripts.es_codebook_search import defended, features  # noqa: E402
from scripts.fold_review import assign_dates, take  # noqa: E402


def predict(x_train, y_train, x_test, surrogate: str):
    if surrogate == "lr":
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=300).fit(scaler.transform(x_train), y_train)
        return model.predict(scaler.transform(x_test))
    if surrogate == "rf":
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)
        model.fit(x_train, y_train)
        return model.predict(x_test)
    scaler = StandardScaler().fit(x_train)
    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
    model.fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def cap_with_min(indices, labels, total_limit: int, min_per_class: int, rng) -> np.ndarray:
    chosen: list[int] = []
    for label in np.unique(labels[indices]):
        pool = indices[labels[indices] == label]
        share = max(min_per_class, int(round(total_limit * len(pool) / len(indices))))
        share = min(share, len(pool))
        chosen.extend(rng.choice(pool, size=share, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=int)


def cluster_bootstrap(y_true, y_prediction, days, class_count, rng, repeats=200):
    unique_days = np.unique(days)
    samples = []
    for _ in range(repeats):
        draw = rng.choice(unique_days, size=len(unique_days), replace=True)
        indices = np.concatenate([np.where(days == day)[0] for day in draw])
        samples.append(
            recall_score(
                y_true[indices],
                y_prediction[indices],
                labels=np.arange(class_count),
                average=None,
                zero_division=0,
            )
        )
    samples = np.asarray(samples)
    return np.percentile(samples, [2.5, 97.5], axis=0)


def evaluate(
    dataset,
    project,
    train_indices,
    validation_indices,
    codes,
    positions,
    surrogate,
    class_count,
    rng,
    bootstrap_repeats,
):
    train = take(dataset, train_indices)
    validation = take(dataset, validation_indices)
    config = dataclasses.replace(project.method33, trigger_codes=codes, trigger_positions=positions)
    train_defended = defended(train, "old", config, 731800, project.trigger_jitter)
    validation_defended = defended(validation, "next", config, 731800, project.trigger_jitter)
    x_train = features(train_defended)
    x_validation = features(validation_defended)
    prediction = predict(x_train, train[4], x_validation, surrogate)
    y_true = validation[4]
    ba = float(balanced_accuracy_score(y_true, prediction))
    recall = recall_score(y_true, prediction, labels=np.arange(class_count), average=None, zero_division=0)
    days = np.asarray(dataset["groups"])[validation_indices].astype(str)
    low, high = cluster_bootstrap(y_true, prediction, days, class_count, rng, bootstrap_repeats)
    return {
        "ba": ba,
        "per_class_recall": [float(value) for value in recall],
        "recall_ci_low": [float(value) for value in low],
        "recall_ci_high": [float(value) for value in high],
        "n_per_class": np.bincount(y_true, minlength=class_count).tolist(),
        "ba_minus_mean_recall": float(ba - np.mean(recall)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--validation-budget", type=int, default=6000)
    parser.add_argument("--min-per-class", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=200)
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
    candidate = method33_tools.load_config(args.candidate_config)
    base_positions = [tuple(int(v) for v in pair) for pair in base.method33.trigger_positions]
    searched_positions = [tuple(int(v) for v in pair) for pair in candidate.method33.trigger_positions]
    codes = base.method33.trigger_codes

    variants = {"base": base_positions, "searched": searched_positions}
    for label in (0, 5):
        changed = list(base_positions)
        changed[label] = searched_positions[label]
        variants[f"class{label}"] = changed
    both = list(base_positions)
    both[0] = searched_positions[0]
    both[5] = searched_positions[5]
    variants["both"] = both

    order = ["base", "class0", "class5", "both", "searched"]
    started = time.time()
    results = {}
    for surrogate in ("lr", "rf", "mlp"):
        for name in order:
            per_fold = []
            for fold, (train_idx, validation_idx) in enumerate(fold_data):
                rng = np.random.default_rng(args.seed + 100 * fold + len(name))
                per_fold.append(
                    evaluate(
                        dataset,
                        base,
                        train_idx,
                        validation_idx,
                        codes,
                        tuple(variants[name]),
                        surrogate,
                        class_count,
                        rng,
                        args.bootstrap,
                    )
                )
            results.setdefault(name, {})[surrogate] = per_fold
            mean_ba = np.mean([item["ba"] for item in per_fold])
            rec = np.mean([item["per_class_recall"] for item in per_fold], axis=0)
            print(
                f"[{surrogate}] {name:<8} BA={mean_ba:.4f} per_class={[round(float(v), 2) for v in rec]}",
                flush=True,
            )

    summary = {}
    for surrogate in ("lr", "rf", "mlp"):
        base_recall = np.mean([item["per_class_recall"] for item in results["base"][surrogate]], axis=0)
        base_ba = np.mean([item["ba"] for item in results["base"][surrogate]])
        summary[surrogate] = {}
        for name in order:
            mean_recall = np.mean([item["per_class_recall"] for item in results[name][surrogate]], axis=0)
            mean_ba = np.mean([item["ba"] for item in results[name][surrogate]])
            summary[surrogate][name] = {
                "ba": float(mean_ba),
                "delta_ba": float(base_ba - mean_ba),
                "per_class_recall": mean_recall.tolist(),
                "per_class_delta": (base_recall - mean_recall).tolist(),
                "max_recall": float(mean_recall.max()),
            }
        deltas = np.asarray(summary[surrogate]["searched"]["per_class_delta"])
        worst = int(np.argmax(-deltas))
        summary[surrogate]["worst_class"] = worst
        print(
            f"[{surrogate}] searched delta_ba={summary[surrogate]['searched']['delta_ba']:.4f} "
            f"worst_class={worst} delta={summary[surrogate]['searched']['per_class_delta'][worst]:.3f}",
            flush=True,
        )

    consistency = []
    for surrogate in ("lr", "rf", "mlp"):
        for name in order:
            consistency.extend(
                abs(item["ba_minus_mean_recall"]) for item in results[name][surrogate]
            )
    log = {
        "dataset": str(args.dataset),
        "base_config": str(args.base_config),
        "candidate_config": str(args.candidate_config),
        "folds": folds,
        "min_per_class": args.min_per_class,
        "validation_budget": args.validation_budget,
        "fold_dates": [[str(days[index]) for index in block] for block in assignment],
        "results": results,
        "summary": summary,
        "consistency_max_abs_ba_minus_mean_recall": float(max(consistency)),
        "seconds": round(time.time() - started, 1),
        "note": "diagnostic ablation on validation folds; holdout untouched",
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                surrogate: {
                    name: round(summary[surrogate][name]["delta_ba"], 4) for name in order
                }
                for surrogate in ("lr", "rf", "mlp")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"consistency max |BA - mean recall| = {max(consistency):.2e}")


if __name__ == "__main__":
    main()
