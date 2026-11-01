"""Paired bootstrap for C->C / C->D / D->D length-histogram attackers.

Matching modes:
    none    proportional 8k/8k cap (baseline)
    class   equal number of flows per class (P1-A)
    bin     equal number per class *and* pre-defense event-count bin (P1-B)

All three BAs are computed from predictions on the *same* test flows, and the
bootstrap resamples test days, so loss = C->C - C->D and recovery = D->D - C->D
get paired confidence intervals.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402


def hist_clean(lengths, mask, indices, bins):
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


def hist_app(app, bins):
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


def match_indices(mode, train_idx, test_idx, labels, counts, edges, cap, min_per_bin, rng):
    if mode == "none":
        def cap_prop(indices):
            chosen = []
            for label in np.unique(labels):
                pool = indices[labels[indices] == label]
                take_n = max(1, int(round(cap * len(pool) / len(indices))))
                chosen.extend(rng.choice(pool, size=min(take_n, len(pool)), replace=False).tolist())
            return np.asarray(sorted(chosen))
        return cap_prop(train_idx), cap_prop(test_idx), []
    if mode == "class":
        n = min(
            min(np.sum(labels[train_idx] == label) for label in np.unique(labels)),
            min(np.sum(labels[test_idx] == label) for label in np.unique(labels)),
            cap,
        )
        if n < 5:
            return np.asarray([], dtype=int), np.asarray([], dtype=int), []
        train = np.concatenate([
            rng.choice(train_idx[labels[train_idx] == label], size=n, replace=False)
            for label in np.unique(labels)
        ])
        test = np.concatenate([
            rng.choice(test_idx[labels[test_idx] == label], size=n, replace=False)
            for label in np.unique(labels)
        ])
        return np.asarray(sorted(train)), np.asarray(sorted(test)), [{"n_per_class": int(n)}]
    # bin mode
    train_bins = np.digitize(counts[train_idx], edges) - 1
    test_bins = np.digitize(counts[test_idx], edges) - 1
    train_out, test_out, per_bin = [], [], []
    for b in range(len(edges) - 1):
        available = {}
        for label in np.unique(labels):
            available[label] = min(
                int(np.sum((train_bins == b) & (labels[train_idx] == label))),
                int(np.sum((test_bins == b) & (labels[test_idx] == label))),
            )
        eligible = [label for label in available if available[label] >= min_per_bin]
        if len(eligible) < 2:
            continue
        n = min(min(available[label] for label in eligible), cap)
        per_bin.append(
            {
                "bin": b,
                "low": float(edges[b]),
                "high": float(edges[b + 1]),
                "n": int(n),
                "classes": [int(label) for label in eligible],
            }
        )
        for label in eligible:
            train_out.extend(
                rng.choice(train_idx[(train_bins == b) & (labels[train_idx] == label)], size=n, replace=False).tolist()
            )
            test_out.extend(
                rng.choice(test_idx[(test_bins == b) & (labels[test_idx] == label)], size=n, replace=False).tolist()
            )
    return np.asarray(sorted(train_out)), np.asarray(sorted(test_out)), per_bin


def fit_predict(x_train, y_train, x_test, surrogate="lr"):
    if surrogate == "lr":
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
        return model.predict(scaler.transform(x_test))
    if surrogate == "rf":
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)
        model.fit(x_train, y_train)
        return model.predict(x_test)
    if surrogate == "mlp":
        scaler = StandardScaler().fit(x_train)
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
        model.fit(scaler.transform(x_train), y_train)
        return model.predict(scaler.transform(x_test))
    raise ValueError(surrogate)


def macro_ba(y_true, prediction, classes):
    recalls = []
    for label in classes:
        mask = y_true == label
        if mask.sum() == 0:
            recalls.append(0.0)
        else:
            recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--mode", choices=("none", "class", "bin"), default="none")
    parser.add_argument("--surrogates", default="lr")
    parser.add_argument("--cap", type=int, default=300)
    parser.add_argument("--count-edges", default="4,8,12,16,20,24,32,33")
    parser.add_argument("--min-per-bin", type=int, default=5)
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    counts = mask.sum(axis=1)
    split = np.load(args.split_indices)
    edges = np.asarray([float(value) for value in args.count_edges.split(",")])
    hist_bins = np.linspace(0.0, 1514.0, args.hist_bins + 1)
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx, matched = match_indices(
        args.mode, np.asarray(split["train"]), np.asarray(split["test"]), labels, counts,
        edges, args.cap, args.min_per_bin, rng,
    )
    if len(train_idx) < 100 or len(test_idx) < 100:
        result = {"error": "not enough matched samples", "matched": matched}
        args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result))
        return

    project = method33_tools.load_config(args.config)
    config = project.method33
    clean_train = hist_clean(lengths, mask, train_idx, hist_bins)
    clean_test = hist_clean(lengths, mask, test_idx, hist_bins)
    train_app = apply_method33(*take(dataset, train_idx), config, 731800, mapping="old", jitter_config=project.trigger_jitter)
    test_app = apply_method33(*take(dataset, test_idx), config, 731800, mapping="next", jitter_config=project.trigger_jitter)
    defended_train = hist_app(train_app, hist_bins)
    defended_test = hist_app(test_app, hist_bins)
    y_train, y_test = labels[train_idx], labels[test_idx]
    classes = [int(value) for value in np.unique(y_test)]

    days = groups[test_idx]
    unique_days = np.unique(days)
    per_surrogate = {}
    for surrogate in args.surrogates.split(","):
        pred_cc = fit_predict(clean_train, y_train, clean_test, surrogate)
        pred_cd = fit_predict(clean_train, y_train, defended_test, surrogate)
        pred_dd = fit_predict(defended_train, y_train, defended_test, surrogate)
        boot = {"CC": [], "CD": [], "DD": [], "loss": [], "recovery": [], "adaptation": []}
        for _ in range(args.bootstrap):
            draw = rng.choice(unique_days, size=len(unique_days), replace=True)
            idx = np.concatenate([np.where(days == day)[0] for day in draw])
            cc = macro_ba(y_test[idx], pred_cc[idx], classes)
            cd = macro_ba(y_test[idx], pred_cd[idx], classes)
            dd = macro_ba(y_test[idx], pred_dd[idx], classes)
            boot["CC"].append(cc)
            boot["CD"].append(cd)
            boot["DD"].append(dd)
            boot["loss"].append(cc - cd)
            boot["recovery"].append(dd - cd)
            boot["adaptation"].append(dd - cc)

        def interval(values):
            return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

        per_surrogate[surrogate] = {
            "C_to_C": {"ba": macro_ba(y_test, pred_cc, classes), "ci": interval(boot["CC"])},
            "C_to_D": {"ba": macro_ba(y_test, pred_cd, classes), "ci": interval(boot["CD"])},
            "D_to_D": {"ba": macro_ba(y_test, pred_dd, classes), "ci": interval(boot["DD"])},
            "loss": {
                "value": macro_ba(y_test, pred_cc, classes) - macro_ba(y_test, pred_cd, classes),
                "ci": interval(boot["loss"]),
            },
            "recovery": {
                "value": macro_ba(y_test, pred_dd, classes) - macro_ba(y_test, pred_cd, classes),
                "ci": interval(boot["recovery"]),
            },
            "adaptation": {
                "value": macro_ba(y_test, pred_dd, classes) - macro_ba(y_test, pred_cc, classes),
                "ci": interval(boot["adaptation"]),
            },
        }
        print(
            f"{args.mode}/{surrogate}: n={len(train_idx)}/{len(test_idx)} "
            f"C->C={per_surrogate[surrogate]['C_to_C']['ba']:.4f} "
            f"C->D={per_surrogate[surrogate]['C_to_D']['ba']:.4f} "
            f"D->D={per_surrogate[surrogate]['D_to_D']['ba']:.4f} "
            f"loss={per_surrogate[surrogate]['loss']['value']:.4f} "
            f"recovery={per_surrogate[surrogate]['recovery']['value']:.4f}",
            flush=True,
        )
    result = {
        "mode": args.mode,
        "matched_train": int(len(train_idx)),
        "matched_test": int(len(test_idx)),
        "matched_detail": matched,
        "surrogates": per_surrogate,
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
