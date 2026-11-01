"""Multi-label stratified date folds + frozen-config re-evaluation.

Purpose (no new search):

  1. Build a date x class count matrix over the search pool (train + validation).
  2. Assign *whole dates* to K folds so that every fold covers every class and
     per-class / total volumes are as balanced as possible.
  3. Re-evaluate the already-searched codebook against the geometry baseline on
     these corrected folds, reporting per-fold BA, mean paired delta, and
     per-class recall.

The previous holdout result is deliberately *not* recomputed here; it stays as
preliminary external evidence and must not be reused for selection.
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
from scripts.codebook_experiments import stratified_cap, take  # noqa: E402
from scripts.es_codebook_search import defended, features  # noqa: E402

SURROGATES = ("lr", "rf", "mlp")


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


def assign_dates(matrix: np.ndarray, folds: int, seed: int) -> list[list[int]]:
    """Greedy multi-label assignment with a repair pass; whole dates only."""

    rng = np.random.default_rng(seed)
    day_count, class_count = matrix.shape
    target = matrix.sum(axis=0) / folds
    total_target = matrix.sum() / folds

    rarity = np.zeros(day_count)
    for index in range(day_count):
        present = matrix[index] > 0
        rarity[index] = float((1.0 / np.maximum(matrix.sum(axis=0), 1))[present].sum())
    order = np.argsort(-rarity + rng.normal(scale=1e-9, size=day_count))

    assigned: list[list[int]] = [[] for _ in range(folds)]
    counts = np.zeros((folds, class_count))
    totals = np.zeros(folds)

    def cost(c: np.ndarray, total: float) -> float:
        relative = ((c - target) / np.maximum(target, 1.0)) ** 2
        return float(relative.sum() + ((total - total_target) / total_target) ** 2)

    for index in order:
        best_fold = 0
        best_cost = None
        for fold in range(folds):
            candidate = cost(counts[fold] + matrix[index], totals[fold] + matrix[index].sum())
            if best_cost is None or candidate < best_cost:
                best_cost = candidate
                best_fold = fold
        assigned[best_fold].append(int(index))
        counts[best_fold] += matrix[index]
        totals[best_fold] += matrix[index].sum()

    # Repair: move one date containing a missing class into a deficient fold,
    # but never remove the last date of another class from the donor fold.
    for label in range(class_count):
        for fold in range(folds):
            if counts[fold, label] > 0:
                continue
            donors = sorted(
                (other for other in range(folds) if other != fold),
                key=lambda other: -counts[other, label],
            )
            for donor in donors:
                candidates = [index for index in assigned[donor] if matrix[index, label] > 0]
                candidates.sort(key=lambda index: -matrix[index, label])
                for index in candidates:
                    after = counts[donor] - matrix[index]
                    if np.any((counts[donor] > 0) & (after == 0)):
                        continue
                    assigned[donor].remove(index)
                    assigned[fold].append(index)
                    counts[donor] -= matrix[index]
                    counts[fold] += matrix[index]
                    totals[donor] -= matrix[index].sum()
                    totals[fold] += matrix[index].sum()
                    break
                else:
                    continue
                break

    # Local swap search to balance totals while keeping all-class coverage.
    def objective() -> float:
        value = 0.0
        for fold in range(folds):
            value += cost(counts[fold], totals[fold])
        return value

    for _ in range(1500):
        first, second = rng.choice(folds, size=2, replace=False)
        if not assigned[first] or not assigned[second]:
            continue
        left = int(rng.choice(assigned[first]))
        right = int(rng.choice(assigned[second]))
        new_first = counts[first] - matrix[left] + matrix[right]
        new_second = counts[second] - matrix[right] + matrix[left]
        if np.any(new_first <= 0) or np.any(new_second <= 0):
            continue
        old = cost(counts[first], totals[first]) + cost(counts[second], totals[second])
        new = cost(new_first, totals[first] - matrix[left].sum() + matrix[right].sum()) + cost(
            new_second, totals[second] - matrix[right].sum() + matrix[left].sum()
        )
        if new < old - 1e-12:
            assigned[first].remove(left)
            assigned[second].remove(right)
            assigned[first].append(right)
            assigned[second].append(left)
            counts[first] = new_first
            counts[second] = new_second
            totals[first] = totals[first] - matrix[left].sum() + matrix[right].sum()
            totals[second] = totals[second] - matrix[right].sum() + matrix[left].sum()

    return assigned


def evaluate_fold(dataset, project, train_indices, validation_indices, codes, positions, surrogate):
    train = take(dataset, train_indices)
    validation = take(dataset, validation_indices)
    config = dataclasses.replace(project.method33, trigger_codes=codes, trigger_positions=positions)
    train_defended = defended(train, "old", config, 731800, project.trigger_jitter)
    validation_defended = defended(validation, "next", config, 731800, project.trigger_jitter)
    x_train = features(train_defended)
    x_validation = features(validation_defended)
    prediction = predict(x_train, train[4], x_validation, surrogate)
    class_count = project.method33.class_count
    recall = recall_score(
        validation[4], prediction, labels=np.arange(class_count), average=None, zero_division=0
    )
    return {
        "ba": float(balanced_accuracy_score(validation[4], prediction)),
        "per_class_recall": [float(value) for value in recall],
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
    parser.add_argument("--max-validation", type=int, default=2500)
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
        mask = groups[pool] == day
        matrix[index] = np.bincount(labels[pool][mask], minlength=class_count)

    per_class_days = (matrix > 0).sum(axis=0)
    feasible_folds = int(min(args.folds, per_class_days.min()))
    if feasible_folds < args.folds:
        print(
            f"[warn] class with fewest dates has {per_class_days.min()} days; "
            f"using {feasible_folds} folds instead of {args.folds}",
            flush=True,
        )
    folds = feasible_folds
    assignment = assign_dates(matrix, folds, args.seed)

    fold_indices = []
    for fold, day_indices in enumerate(assignment):
        fold_days = set(days[day_indices].tolist())
        validation_mask = np.asarray([group in fold_days for group in groups[pool]])
        validation_pool = pool[validation_mask]
        train_pool = pool[~validation_mask]
        rng = np.random.default_rng(args.seed + fold)
        train_indices = stratified_cap(train_pool, labels, args.max_train, rng)
        validation_indices = stratified_cap(validation_pool, labels, args.max_validation, rng)
        fold_indices.append((train_indices, validation_indices))
        print(
            f"[fold {fold}] days={len(fold_days)} train={len(train_indices)} "
            f"validation={len(validation_indices)} "
            f"val_per_class={np.bincount(labels[validation_indices], minlength=class_count).tolist()}",
            flush=True,
        )

    base_project = method33_tools.load_config(args.base_config)
    candidate_project = method33_tools.load_config(args.candidate_config)
    base_codes, base_positions = base_project.method33.trigger_codes, base_project.method33.trigger_positions
    cand_codes, cand_positions = candidate_project.method33.trigger_codes, candidate_project.method33.trigger_positions

    started = time.time()
    results = {"base": {}, "candidate": {}}
    for surrogate in SURROGATES:
        for name, codes, positions in (
            ("base", base_codes, base_positions),
            ("candidate", cand_codes, cand_positions),
        ):
            per_fold = [
                evaluate_fold(dataset, base_project, train_idx, val_idx, codes, positions, surrogate)
                for train_idx, val_idx in fold_indices
            ]
            results[name][surrogate] = per_fold
        print(f"[{surrogate}] done", flush=True)

    summary = {}
    for surrogate in SURROGATES:
        base_scores = [item["ba"] for item in results["base"][surrogate]]
        cand_scores = [item["ba"] for item in results["candidate"][surrogate]]
        delta = np.asarray(base_scores) - np.asarray(cand_scores)
        base_recall = np.mean([item["per_class_recall"] for item in results["base"][surrogate]], axis=0)
        cand_recall = np.mean([item["per_class_recall"] for item in results["candidate"][surrogate]], axis=0)
        summary[surrogate] = {
            "base_folds": base_scores,
            "candidate_folds": cand_scores,
            "delta_per_fold": delta.tolist(),
            "delta_mean": float(delta.mean()),
            "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0,
            "folds_improved": int((delta > 0).sum()),
            "base_per_class_recall": base_recall.tolist(),
            "candidate_per_class_recall": cand_recall.tolist(),
            "per_class_delta": (base_recall - cand_recall).tolist(),
        }
        print(
            f"[{surrogate}] base={np.mean(base_scores):.4f} candidate={np.mean(cand_scores):.4f} "
            f"delta={delta.mean():.4f}±{summary[surrogate]['delta_se']:.4f} "
            f"folds_improved={summary[surrogate]['folds_improved']}",
            flush=True,
        )

    log = {
        "dataset": str(args.dataset),
        "base_config": str(args.base_config),
        "candidate_config": str(args.candidate_config),
        "folds": folds,
        "fold_dates": [[str(days[index]) for index in day_indices] for day_indices in assignment],
        "per_class_dates": per_class_days.tolist(),
        "fold_class_flows": [
            matrix[assignment[fold]].sum(axis=0).tolist() for fold in range(folds)
        ],
        "results": results,
        "summary": summary,
        "seconds": round(time.time() - started, 1),
        "note": "frozen-config review only; the earlier holdout evaluation is kept as preliminary evidence",
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                surrogate: {
                    "delta_mean": round(summary[surrogate]["delta_mean"], 4),
                    "delta_se": round(summary[surrogate]["delta_se"], 4),
                    "folds_improved": summary[surrogate]["folds_improved"],
                }
                for surrogate in SURROGATES
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
