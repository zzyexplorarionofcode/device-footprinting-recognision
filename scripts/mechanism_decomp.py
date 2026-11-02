"""Decompose clean->defended loss into representation, displacement and dummy mixing.

Counterfactual chain (all with the same train/test sample and one batch):

    CC             clean real hist -> clean real hist
    CD_complete    clean -> all real packets in the emitted sequence (no window cut)
    CD_visible     clean -> real packets still visible in the 32-event window
    CD_merged      clean -> all visible packets (real + dummy)

Components (BA scale, positive = defense reduces attacker BA):
    representation = CC - CD_complete
    displacement   = CD_complete - CD_visible
    dummy_mixing   = CD_visible - CD_merged
    total          = CC - CD_merged = sum of the three
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
    return counts / counts.sum() if counts.sum() else counts


def clean_hist(dataset, indices, bins):
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    out = np.zeros((len(indices), len(bins) - 1))
    for row, index in enumerate(indices):
        out[row] = hist(lengths[index][mask[index]], bins)
    return out


def defended_hists(dataset, indices, config, mapping, jitter, bins):
    app = apply_method33(
        *take(dataset, indices), config, 731800, mapping=mapping,
        jitter_config=jitter, include_event_trace=True,
    )
    complete = np.zeros((len(indices), len(bins) - 1))
    visible = np.zeros_like(complete)
    merged = np.zeros_like(complete)
    for row, audit in enumerate(app[6]):
        trace = audit["complete_event_trace"]
        complete[row] = hist([e["length_bytes"] for e in trace if e["role"] == "real"], bins)
        visible[row] = hist(
            [e["length_bytes"] for e in trace if e["in_observation_window"] and e["role"] == "real"], bins
        )
        merged[row] = hist([e["length_bytes"] for e in trace if e["in_observation_window"]], bins)
    return complete, visible, merged


def fit_pred(x_train, y_train, x_test):
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test))


def macro_ba(y_true, prediction, classes):
    return float(np.mean([
        np.mean(prediction[y_true == c] == c) if np.any(y_true == c) else 0.0 for c in classes
    ]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--bins", type=int, default=16)
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
    y_train, y_test = labels[train_idx], labels[test_idx]
    bins = np.linspace(0.0, 1514.0, args.bins + 1)
    project = method33_tools.load_config(args.config)
    config = project.method33

    clean_train = clean_hist(dataset, train_idx, bins)
    clean_test = clean_hist(dataset, test_idx, bins)
    full_train, vis_train, merged_train = defended_hists(
        dataset, train_idx, config, "old", project.trigger_jitter, bins
    )
    full_test, vis_test, merged_test = defended_hists(
        dataset, test_idx, config, "next", project.trigger_jitter, bins
    )

    variants = {
        "CC": (clean_train, clean_test),
        "CD_complete": (clean_train, full_test),
        "CD_visible": (clean_train, vis_test),
        "CD_merged": (clean_train, merged_test),
        "DD_complete": (full_train, full_test),
        "DD_visible": (vis_train, vis_test),
        "DD_merged": (merged_train, merged_test),
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

    components = {
        "representation": [cc - cd for cc, cd in zip(boot["CC"], boot["CD_complete"])],
        "displacement": [complete - visible for complete, visible in zip(boot["CD_complete"], boot["CD_visible"])],
        "dummy_mixing": [visible - merged for visible, merged in zip(boot["CD_visible"], boot["CD_merged"])],
        "total": [cc - merged for cc, merged in zip(boot["CC"], boot["CD_merged"])],
    }
    result = {
        "run_id": {
            "dataset_version": args.dataset_version,
            "sample": f"cap_with_min:{args.cap}",
            "bins": f"length:{args.bins}:0-1514",
            "classifier": "lr",
        },
        "variants": {name: {"ba": ba(name), "ci": interval(boot[name])} for name in variants},
        "components": {
            name: {"value": float(np.mean(values)), "ci": interval(values)}
            for name, values in components.items()
        },
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        args.dataset_version,
        "CC", round(ba("CC"), 4),
        "CD_complete", round(ba("CD_complete"), 4),
        "CD_visible", round(ba("CD_visible"), 4),
        "CD_merged", round(ba("CD_merged"), 4),
        "components", {k: round(v["value"], 4) for k, v in result["components"].items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
