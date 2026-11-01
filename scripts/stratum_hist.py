"""Stratified length-histogram evaluation by pre-defense event count.

For each count stratum (e.g. 4-8, 9-15, 16-32) train and test only on flows in
that stratum, with a per-class cap.  Reports C->C, C->D and D->D BA together
with the number of flows/classes available, which shows whether the defense
transfer loss is concentrated in short or long flows.
"""

from __future__ import annotations

import argparse
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


def hist_rows(lengths, mask, indices, bins):
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
    return hist_rows(np.asarray(app[0]), np.asarray(app[3]).astype(bool), np.arange(len(app[0])), bins)


def fit_score(x_train, y_train, x_test, y_test):
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
    prediction = model.predict(scaler.transform(x_test))
    return float(balanced_accuracy_score(y_test, prediction))


def cap_per_class(indices, labels, cap, rng):
    chosen = []
    for label in np.unique(labels[indices]):
        pool = indices[labels[indices] == label]
        chosen.extend(rng.choice(pool, size=min(cap, len(pool)), replace=False).tolist())
    return np.asarray(sorted(chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--edges", default="4,9,16,33")
    parser.add_argument("--cap", type=int, default=300)
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--min-test", type=int, default=30)
    parser.add_argument("--common-classes", action="store_true")
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    counts = mask.sum(axis=1)
    split = np.load(args.split_indices)
    train_all = np.asarray(split["train"])
    test_all = np.asarray(split["test"])
    edges = np.asarray([float(value) for value in args.edges.split(",")])
    bins = np.linspace(0.0, 1514.0, args.hist_bins + 1)
    rng = np.random.default_rng(args.seed)
    project = method33_tools.load_config(args.config)
    config = project.method33

    strata = []
    for b in range(len(edges) - 1):
        train_idx = train_all[(counts[train_all] >= edges[b]) & (counts[train_all] < edges[b + 1])]
        test_idx = test_all[(counts[test_all] >= edges[b]) & (counts[test_all] < edges[b + 1])]
        if len(test_idx) < args.min_test:
            continue
        classes = [label for label in np.unique(labels[test_idx]) if np.sum(labels[test_idx] == label) >= 3]
        if len(classes) < 2:
            continue
        strata.append({"b": b, "train": train_idx, "test": test_idx, "classes": classes})
    if args.common_classes and strata:
        common = set(strata[0]["classes"])
        for item in strata[1:]:
            common &= set(item["classes"])
        strata = [
            {**item, "classes": [value for value in item["classes"] if value in common]}
            for item in strata
        ]
        strata = [item for item in strata if len(item["classes"]) >= 2]

    results = []
    for item in strata:
        b = item["b"]
        train_idx = item["train"]
        test_idx = item["test"]
        classes = item["classes"]
        train_idx = train_idx[np.isin(labels[train_idx], classes)]
        test_idx = test_idx[np.isin(labels[test_idx], classes)]
        train_idx = cap_per_class(train_idx, labels, args.cap, rng)
        test_idx = cap_per_class(test_idx, labels, args.cap, rng)
        clean_train = hist_rows(lengths, mask, train_idx, bins)
        clean_test = hist_rows(lengths, mask, test_idx, bins)
        defended_train = hist_app(
            apply_method33(*take(dataset, train_idx), config, 731800, mapping="old", jitter_config=project.trigger_jitter),
            bins,
        )
        defended_test = hist_app(
            apply_method33(*take(dataset, test_idx), config, 731800, mapping="next", jitter_config=project.trigger_jitter),
            bins,
        )
        y_train, y_test = labels[train_idx], labels[test_idx]
        entry = {
            "low": float(edges[b]),
            "high": float(edges[b + 1]),
            "train_flows": int(len(train_idx)),
            "test_flows": int(len(test_idx)),
            "classes": [int(value) for value in classes],
            "C_to_C": fit_score(clean_train, y_train, clean_test, y_test),
            "C_to_D": fit_score(clean_train, y_train, defended_test, y_test),
            "D_to_D": fit_score(defended_train, y_train, defended_test, y_test),
        }
        entry["loss"] = entry["C_to_C"] - entry["C_to_D"]
        entry["recovery"] = entry["D_to_D"] - entry["C_to_D"]
        results.append(entry)
        print(
            f"count {entry['low']:.0f}-{entry['high']:.0f}: n={entry['train_flows']}/{entry['test_flows']} "
            f"classes={len(classes)} C->C={entry['C_to_C']:.4f} C->D={entry['C_to_D']:.4f} "
            f"D->D={entry['D_to_D']:.4f} loss={entry['loss']:.4f}",
            flush=True,
        )
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
