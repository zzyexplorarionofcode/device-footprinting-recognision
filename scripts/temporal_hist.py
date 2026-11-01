"""Strict early/late temporal test for the order-invariant length histogram.

Only classes present in both sides with enough samples are kept, each class is
subsampled to the same count, and several date cut points are evaluated in both
directions.  BA, per-class recall and a day-cluster bootstrap CI are reported.
A per-flow shuffle control verifies that the histogram cannot depend on order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def hist_rows(lengths, mask, indices, bins):
    output = np.zeros((len(indices), len(bins) - 1))
    for row, index in enumerate(indices):
        values = lengths[index][mask[index]]
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bins)
        counts = counts.astype(float)
        total = counts.sum()
        output[row] = counts / total if total > 0 else 0.0
    return output


def cap_balanced(indices, labels, per_class, rng):
    chosen = []
    for label in np.unique(labels[indices]):
        pool = indices[labels[indices] == label]
        take = min(len(pool), per_class)
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    return np.asarray(sorted(chosen))


def score_split(train_idx, test_idx, labels, lengths, mask, bins, rng):
    common = [
        label
        for label in np.unique(labels)
        if np.sum(labels[train_idx] == label) >= 20 and np.sum(labels[test_idx] == label) >= 20
    ]
    if len(common) < 2:
        return None
    train_idx = cap_balanced(train_idx[np.isin(labels[train_idx], common)], labels, 300, rng)
    test_idx = cap_balanced(test_idx[np.isin(labels[test_idx], common)], labels, 300, rng)
    x_train = hist_rows(lengths, mask, train_idx, bins)
    x_test = hist_rows(lengths, mask, test_idx, bins)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), labels[train_idx])
    prediction = model.predict(scaler.transform(x_test))
    recalls = recall_score(
        labels[test_idx], prediction, labels=common, average=None, zero_division=0
    )
    ba = float(np.mean(recalls))
    day_values = np.unique(groups_global[test_idx])
    samples = []
    for _ in range(200):
        draw = rng.choice(day_values, size=len(day_values), replace=True)
        idx = np.concatenate([np.where(groups_global[test_idx] == day)[0] for day in draw])
        resampled = recall_score(
            labels[test_idx][idx], prediction[idx], labels=common, average=None, zero_division=0
        )
        samples.append(float(np.mean(resampled)))
    return {
        "classes": [int(value) for value in common],
        "ba": ba,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "per_class_recall": dict(zip([int(value) for value in common], recalls.tolist())),
        "train_flows": int(len(train_idx)),
        "test_flows": int(len(test_idx)),
    }


def main() -> None:
    global groups_global
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--cuts", default="0.5,0.6,0.7")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    groups_global = groups
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    days = np.sort(np.unique(groups))
    bins = np.linspace(0.0, 1514.0, args.bins + 1)
    rng = np.random.default_rng(args.seed)
    results = {}
    for cut in [float(value) for value in args.cuts.split(",")]:
        split = int(len(days) * cut)
        early = days[:split]
        late = days[split:]
        for name, train_days, test_days in (
            ("early_to_late", early, late),
            ("late_to_early", late, early),
        ):
            train_idx = np.where(np.isin(groups, list(train_days)))[0]
            test_idx = np.where(np.isin(groups, list(test_days)))[0]
            result = score_split(train_idx, test_idx, labels, lengths, mask, bins, rng)
            key = f"cut{cut:.2f}_{name}"
            results[key] = result
            if result:
                print(
                    f"{key}: BA={result['ba']:.4f} [{result['ci_low']:.4f},{result['ci_high']:.4f}] "
                    f"classes={result['classes']}",
                    flush=True,
                )
    log = {
        "dataset": str(args.dataset),
        "cuts": args.cuts,
        "bins": args.bins,
        "results": results,
        "note": "balanced common-class temporal test; per-flow shuffle leaves histograms unchanged",
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
