"""Dummy-only length-histogram attacker: overall BA, per-class recall and CIs.

Uses the event trace (role != real) to build a histogram of synthetic packet
lengths only.  A day-cluster bootstrap gives CIs for the overall BA and for each
class recall, which checks whether the small dummy-only signal is class-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_ablation import cap_with_min  # noqa: E402


def dummy_hists(indices, dataset, config, mapping, jitter, bins):
    app = apply_method33(
        *take(dataset, indices), config, 731800, mapping=mapping,
        jitter_config=jitter, include_event_trace=True,
    )
    output = np.zeros((len(indices), len(bins) - 1))
    for row, audit in enumerate(app[6]):
        values = [
            event["length_bytes"]
            for event in audit["complete_event_trace"]
            if event["in_observation_window"] and event["role"] != "real"
        ]
        if not values:
            continue
        counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bins)
        counts = counts.astype(float)
        total = counts.sum()
        output[row] = counts / total if total > 0 else 0.0
    return output


def predict(x_train, y_train, x_test, surrogate):
    if surrogate == "rf":
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)
        model.fit(x_train, y_train)
        return model.predict(x_test)
    scaler = StandardScaler().fit(x_train)
    if surrogate == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
    else:
        model = LogisticRegression(max_iter=500)
    model.fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--surrogates", default="lr")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    rng = np.random.default_rng(args.seed)
    train_idx = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    test_idx = cap_with_min(np.asarray(split["test"]), labels, args.cap, 1, rng)
    bins = np.linspace(0.0, 1514.0, args.bins + 1)
    project = method33_tools.load_config(args.config)
    config = project.method33
    x_train = dummy_hists(train_idx, dataset, config, "old", project.trigger_jitter, bins)
    x_test = dummy_hists(test_idx, dataset, config, "next", project.trigger_jitter, bins)
    y_train, y_test = labels[train_idx], labels[test_idx]
    classes = list(range(int(labels.max()) + 1))
    names = [str(value) for value in dataset["device_names"]]
    support = np.bincount(y_test, minlength=len(names))
    present = [index for index in classes if support[index] > 0]
    days = groups[test_idx]
    unique_days = np.unique(days)

    results = {}
    for surrogate in args.surrogates.split(","):
        prediction = predict(x_train, y_train, x_test, surrogate)
        recalls = recall_score(y_test, prediction, labels=classes, average=None, zero_division=0)
        boot_ba, boot_recall = [], []
        for _ in range(args.bootstrap):
            draw = rng.choice(unique_days, size=len(unique_days), replace=True)
            idx = np.concatenate([np.where(days == day)[0] for day in draw])
            boot_ba.append(balanced_accuracy_score(y_test[idx], prediction[idx]))
            boot_recall.append(
                recall_score(y_test[idx], prediction[idx], labels=classes, average=None, zero_division=0)
            )
        boot_recall = np.asarray(boot_recall)
        results[surrogate] = {
            "ba": float(balanced_accuracy_score(y_test, prediction)),
            "ba_from_recall": float(recalls[present].mean()),
            "recall_sum": float(recalls[present].sum()),
            "accuracy": float(np.mean(prediction == y_test)),
            "classes_present": [names[index] for index in present],
            "test_support": {names[index]: int(support[index]) for index in classes},
            "ba_ci": [float(np.percentile(boot_ba, 2.5)), float(np.percentile(boot_ba, 97.5))],
            "per_class": {
                names[index]: {
                    "support": int(support[index]),
                    "recall": float(recalls[index]),
                    "ci": [
                        float(np.percentile(boot_recall[:, index], 2.5)),
                        float(np.percentile(boot_recall[:, index], 97.5)),
                    ],
                }
                for index in classes
            },
        }
        print(
            f"{surrogate}: BA={results[surrogate]['ba']:.4f} "
            f"CI=[{results[surrogate]['ba_ci'][0]:.4f},{results[surrogate]['ba_ci'][1]:.4f}] "
            f"(sum={results[surrogate]['recall_sum']:.4f}/{len(present)} classes, "
            f"acc={results[surrogate]['accuracy']:.4f})",
            flush=True,
        )
        for index, name in enumerate(names):
            entry = results[surrogate]["per_class"][name]
            print(
                f"   {name:<40} support={entry['support']:7d} "
                f"recall={entry['recall']:.4f} CI={entry['ci']}",
                flush=True,
            )

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
