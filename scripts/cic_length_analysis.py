"""CIC length-distribution and per-class analysis.

Outputs:
  * pairwise Jensen-Shannon / Wasserstein distances between class mean
    real-packet length histograms;
  * confusion matrices and per-class recall for C->C, C->D and D->D attackers;
  * the largest confusion pairs, which shows which devices lose separability.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
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
    lengths = np.asarray(app[0])
    mask = np.asarray(app[3]).astype(bool)
    return hist_rows(lengths, mask, np.arange(len(lengths)), bins)


def fit_predict(x_train, y_train, x_test):
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def cap_prop(indices, labels, cap, rng):
    chosen = []
    for label in np.unique(labels):
        pool = indices[labels[indices] == label]
        take_n = max(1, int(round(cap * len(pool) / len(indices))))
        chosen.extend(rng.choice(pool, size=min(take_n, len(pool)), replace=False).tolist())
    return np.asarray(sorted(chosen))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--per-class-sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    names = [str(value) for value in dataset["device_names"]]
    split = np.load(args.split_indices)
    bins = np.linspace(0.0, 1514.0, args.bins + 1)
    rng = np.random.default_rng(args.seed)
    class_count = len(names)

    # Mean clean histogram per class.
    class_hist = []
    for label in range(class_count):
        pool = np.where(labels == label)[0]
        if len(pool) > args.per_class_sample:
            pool = rng.choice(pool, size=args.per_class_sample, replace=False)
        hist = hist_rows(lengths, mask, pool, bins).mean(axis=0)
        class_hist.append(hist)
    class_hist = np.asarray(class_hist)
    centers = (bins[:-1] + bins[1:]) / 2

    distances = []
    for first in range(class_count):
        for second in range(first + 1, class_count):
            distances.append(
                {
                    "pair": [names[first], names[second]],
                    "js": float(jensenshannon(class_hist[first] + 1e-12, class_hist[second] + 1e-12)),
                    "wasserstein": float(wasserstein_distance(centers, centers, class_hist[first], class_hist[second])),
                }
            )
    distances.sort(key=lambda item: item["js"])

    train_idx = cap_prop(np.asarray(split["train"]), labels, args.cap, rng)
    test_idx = cap_prop(np.asarray(split["test"]), labels, args.cap, rng)
    project = method33_tools.load_config(args.config)
    config = project.method33
    clean_train = hist_rows(lengths, mask, train_idx, bins)
    clean_test = hist_rows(lengths, mask, test_idx, bins)
    defended_train = hist_app(
        apply_method33(*take(dataset, train_idx), config, 731800, mapping="old", jitter_config=project.trigger_jitter), bins
    )
    defended_test = hist_app(
        apply_method33(*take(dataset, test_idx), config, 731800, mapping="next", jitter_config=project.trigger_jitter), bins
    )
    y_train, y_test = labels[train_idx], labels[test_idx]
    labels_all = list(range(class_count))

    predictions = {
        "C_to_C": fit_predict(clean_train, y_train, clean_test),
        "C_to_D": fit_predict(clean_train, y_train, defended_test),
        "D_to_D": fit_predict(defended_train, y_train, defended_test),
    }
    results = {"class_names": names, "distances": distances, "recall": {}, "confusion": {}}
    for key, prediction in predictions.items():
        recall = recall_score(y_test, prediction, labels=labels_all, average=None, zero_division=0)
        matrix = confusion_matrix(y_test, prediction, labels=labels_all)
        results["recall"][key] = dict(zip(names, recall.tolist()))
        results["confusion"][key] = matrix.tolist()
        pairs = []
        for true in range(class_count):
            for predicted in range(class_count):
                if true != predicted and matrix[true, predicted] > 0:
                    pairs.append((int(matrix[true, predicted]), names[true], names[predicted]))
        pairs.sort(reverse=True)
        results.setdefault("top_confusions", {})[key] = [
            {"count": count, "true": true_name, "predicted": predicted_name}
            for count, true_name, predicted_name in pairs[:8]
        ]

    print("closest class pairs by JS:")
    for item in distances[:8]:
        print(f"  {item['pair'][0]} <-> {item['pair'][1]}  JS={item['js']:.3f} W={item['wasserstein']:.0f}")
    print("\nper-class recall:")
    header = f"{'class':<38} {'C->C':>7} {'C->D':>7} {'D->D':>7}"
    print(header)
    for index, name in enumerate(names):
        print(
            f"{name:<38} {results['recall']['C_to_C'][name]:7.3f} "
            f"{results['recall']['C_to_D'][name]:7.3f} {results['recall']['D_to_D'][name]:7.3f}"
        )
    for key in ("C_to_C", "C_to_D", "D_to_D"):
        print(f"\ntop confusions {key}:")
        for item in results["top_confusions"][key]:
            print(f"  {item['count']:4d}  {item['true']} -> {item['predicted']}")

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
