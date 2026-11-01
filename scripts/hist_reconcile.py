"""Reconcile C->C / C->D / D->D with one sample, one split and explicit sources.

Every run gets a run_id describing:
  dataset_version / split_id / sample_protocol / feature_source / binning /
  classifier / regime.

The same train/test indices are used for clean and defended views.  Defended
views are built from the event trace so real-only and merged histograms are
exact, not inferred from lengths.
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
from scripts.position_ablation import cap_with_min  # noqa: E402


def hist(values, bins):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def clean_hists(dataset, indices, bins):
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    output = np.zeros((len(indices), len(bins) - 1))
    for row, index in enumerate(indices):
        output[row] = hist(lengths[index][mask[index]], bins)
    return output


def defended_hists(dataset, indices, config, mapping, jitter, bins):
    app = apply_method33(
        *take(dataset, indices), config, 731800, mapping=mapping,
        jitter_config=jitter, include_event_trace=True,
    )
    real = np.zeros((len(indices), len(bins) - 1))
    merged = np.zeros((len(indices), len(bins) - 1))
    for row, audit in enumerate(app[6]):
        trace = audit["complete_event_trace"]
        visible = [event for event in trace if event["in_observation_window"]]
        real[row] = hist([event["length_bytes"] for event in visible if event["role"] == "real"], bins)
        merged[row] = hist([event["length_bytes"] for event in visible], bins)
    return real, merged


def fit_pred(x_train, y_train, x_test):
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def macro_ba(y_true, prediction, classes):
    recalls = []
    for label in classes:
        mask = y_true == label
        recalls.append(float(np.mean(prediction[mask] == label)) if mask.any() else 0.0)
    return float(np.mean(recalls))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    rng = np.random.default_rng(args.seed)
    train_idx = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    test_idx = cap_with_min(np.asarray(split["test"]), labels, args.cap, 1, rng)
    y_train, y_test = labels[train_idx], labels[test_idx]
    bins = np.linspace(0.0, 1514.0, args.bins + 1)
    project = method33_tools.load_config(args.config)
    config = project.method33

    clean_train = clean_hists(dataset, train_idx, bins)
    clean_test = clean_hists(dataset, test_idx, bins)
    real_train, merged_train = defended_hists(dataset, train_idx, config, "old", project.trigger_jitter, bins)
    real_test, merged_test = defended_hists(dataset, test_idx, config, "next", project.trigger_jitter, bins)

    variants = {
        "C_to_C": (clean_train, clean_test),
        "C_to_D_merged": (clean_train, merged_test),
        "C_to_D_real": (clean_train, real_test),
        "D_to_D_merged": (merged_train, merged_test),
        "D_to_D_real": (real_train, real_test),
    }
    predictions = {name: fit_pred(xtr, y_train, xte) for name, (xtr, xte) in variants.items()}
    classes = [int(value) for value in np.unique(y_test)]
    days = groups[test_idx]
    unique_days = np.unique(days)

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    boot = {name: [] for name in variants}
    for _ in range(args.bootstrap):
        draw = rng.choice(unique_days, size=len(unique_days), replace=True)
        idx = np.concatenate([np.where(days == day)[0] for day in draw])
        for name, prediction in predictions.items():
            boot[name].append(macro_ba(y_test[idx], prediction[idx], classes))

    def ba(name):
        return macro_ba(y_test, predictions[name], classes)

    result = {
        "run_id": {
            "dataset_version": args.dataset_version,
            "split_id": args.split_id,
            "sample_protocol": f"cap_with_min:{args.cap}:seed{args.seed}",
            "binning": f"length:{args.bins}:0-1514",
            "classifier": "lr",
            "regime": "clean+defended_old_to_next",
        },
        "matched_train": int(len(train_idx)),
        "matched_test": int(len(test_idx)),
        "classes_present": len(classes),
        "variants": {
            name: {"ba": ba(name), "ci": interval(boot[name])}
            for name in variants
        },
        "loss": {
            "merged": ba("C_to_C") - ba("C_to_D_merged"),
            "real": ba("C_to_C") - ba("C_to_D_real"),
        },
        "recovery": {
            "merged": ba("D_to_D_merged") - ba("C_to_D_merged"),
            "real": ba("D_to_D_real") - ba("C_to_D_real"),
        },
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.dataset_version}: CC={ba('C_to_C'):.4f} "
        f"CDm={ba('C_to_D_merged'):.4f} Dm={ba('D_to_D_merged'):.4f} "
        f"CDr={ba('C_to_D_real'):.4f} Dr={ba('D_to_D_real'):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
