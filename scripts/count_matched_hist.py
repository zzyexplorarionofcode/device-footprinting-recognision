"""Event-count-matched length-histogram evaluation.

Flows are binned by their *pre-defense* packet count.  Inside each bin every
class contributes the same number of train and test flows, so classes have an
identical event-count distribution.  Three definitions are then evaluated:

    C->C  clean train  -> clean test
    C->D  clean train  -> defended test
    D->D  defended train (old mapping) -> defended test (next mapping)

Per-bin (stratified) BAs are reported separately, which shows whether a
difference survives when flow length is held fixed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402


def hist_from_arrays(lengths, mask, indices, bins):
    output = np.zeros((len(indices), len(bins) - 1))
    for row, index in enumerate(indices):
        values = lengths[index][mask[index].astype(bool)]
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bins)
        counts = counts.astype(float)
        total = counts.sum()
        output[row] = counts / total if total > 0 else 0.0
    return output


def hist_from_app(app, bins):
    lengths = np.asarray(app[0])
    mask = np.asarray(app[3]).astype(bool)
    output = np.zeros((len(lengths), len(bins) - 1))
    for row in range(len(lengths)):
        values = lengths[row][mask[row]]
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bins)
        counts = counts.astype(float)
        total = counts.sum()
        output[row] = counts / total if total > 0 else 0.0
    return output


def fit_score(x_train, y_train, x_test, y_test):
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
    prediction = model.predict(scaler.transform(x_test))
    return float(balanced_accuracy_score(y_test, prediction))


def match_indices(train_idx, test_idx, labels, counts, edges, cap, min_per_bin, rng):
    train_bins = np.digitize(counts[train_idx], edges) - 1
    test_bins = np.digitize(counts[test_idx], edges) - 1
    matched_train, matched_test, per_bin = [], [], []
    for b in range(len(edges) - 1):
        available_train = {}
        available_test = {}
        for label in np.unique(labels):
            available_train[label] = len(train_idx[(train_bins == b) & (labels[train_idx] == label)])
            available_test[label] = len(test_idx[(test_bins == b) & (labels[test_idx] == label)])
        n = min(min(available.values()), cap)
        eligible = [label for label in available if available[label] >= min_per_bin]
        if len(eligible) < 2:
            continue
        n = min(min(available[label] for label in eligible), cap)
        if n < min_per_bin:
            continue
        per_bin.append(
            {
                "bin": b,
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "n_per_class": int(n),
                "classes": [int(label) for label in eligible],
            }
        )
        for label in eligible:
            pool_train = train_idx[(train_bins == b) & (labels[train_idx] == label)]
            pool_test = test_idx[(test_bins == b) & (labels[test_idx] == label)]
            matched_train.extend(rng.choice(pool_train, size=n, replace=False).tolist())
            matched_test.extend(rng.choice(pool_test, size=n, replace=False).tolist())
    return np.asarray(sorted(matched_train)), np.asarray(sorted(matched_test)), per_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--count-edges", default="4,7,10,15,20,32,33")
    parser.add_argument("--cap", type=int, default=300)
    parser.add_argument("--min-per-bin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    counts = mask.sum(axis=1)
    split = np.load(args.split_indices)
    train_idx = np.asarray(split["train"])
    test_idx = np.asarray(split["test"])
    edges = np.asarray([float(value) for value in args.count_edges.split(",")])
    hist_bins = np.linspace(0.0, 1514.0, args.hist_bins + 1)
    rng = np.random.default_rng(args.seed)
    project = method33_tools.load_config(args.config)
    config = project.method33

    matched_train, matched_test, per_bin = match_indices(
        train_idx, test_idx, labels, counts, edges, args.cap, args.min_per_bin, rng
    )
    results = {"matched": {"train": len(matched_train), "test": len(matched_test)}, "per_bin": per_bin}
    if len(matched_train) < 100 or len(matched_test) < 100:
        results["error"] = "not enough matched samples"
        args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(results, indent=2))
        return

    clean_train = hist_from_arrays(lengths, mask, matched_train, hist_bins)
    clean_test = hist_from_arrays(lengths, mask, matched_test, hist_bins)
    train_app = apply_method33(
        *take(dataset, matched_train), config, 731800, mapping="old", jitter_config=project.trigger_jitter
    )
    test_app = apply_method33(
        *take(dataset, matched_test), config, 731800, mapping="next", jitter_config=project.trigger_jitter
    )
    defended_train = hist_from_app(train_app, hist_bins)
    defended_test = hist_from_app(test_app, hist_bins)
    y_train, y_test = labels[matched_train], labels[matched_test]

    results["C_to_C"] = fit_score(clean_train, y_train, clean_test, y_test)
    results["C_to_D"] = fit_score(clean_train, y_train, defended_test, y_test)
    results["D_to_D"] = fit_score(defended_train, y_train, defended_test, y_test)
    results["loss"] = results["C_to_C"] - results["C_to_D"]
    results["recovery"] = results["D_to_D"] - results["C_to_D"]

    # Stratified per-bin evaluation (same matched samples, split by count bin).
    train_bins = np.digitize(counts[matched_train], edges) - 1
    test_bins = np.digitize(counts[matched_test], edges) - 1
    per_bin_scores = []
    for item in per_bin:
        b = item["bin"]
        mask_tr = train_bins == b
        mask_te = test_bins == b
        if mask_tr.sum() < 20 or mask_te.sum() < 20:
            continue
        entry = {"bin": b, "low": item["low"], "high": item["high"]}
        entry["C_to_C"] = fit_score(clean_train[mask_tr], y_train[mask_tr], clean_test[mask_te], y_test[mask_te])
        entry["C_to_D"] = fit_score(clean_train[mask_tr], y_train[mask_tr], defended_test[mask_te], y_test[mask_te])
        entry["D_to_D"] = fit_score(defended_train[mask_tr], y_train[mask_tr], defended_test[mask_te], y_test[mask_te])
        per_bin_scores.append(entry)
    results["per_bin_scores"] = per_bin_scores

    print(
        f"matched train={len(matched_train)} test={len(matched_test)} "
        f"C->C={results['C_to_C']:.4f} C->D={results['C_to_D']:.4f} "
        f"D->D={results['D_to_D']:.4f} loss={results['loss']:.4f}",
        flush=True,
    )
    for entry in per_bin_scores:
        print(
            f"  bin {entry['low']:.0f}-{entry['high']:.0f}: "
            f"C->C={entry['C_to_C']:.4f} C->D={entry['C_to_D']:.4f} D->D={entry['D_to_D']:.4f}",
            flush=True,
        )
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
