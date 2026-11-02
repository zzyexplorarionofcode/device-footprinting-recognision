"""Stage A: trigger-pair on/off x weakener-count dose, equal 630 B / 1.5 s cap.

Cell budgets:
  T=1: trigger 283 B / .45 s / 2 events; weakener 347 B / 1.05 s / W
  T=0: trigger disabled; trigger budget kept structurally, weakener 630 B / 1.5 s / W

Actual spend is recorded per cell because a cap is not a guarantee that the
weakener spends all bytes (especially W=0).
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
from scripts.position_ablation import cap_with_min  # noqa: E402

CELLS = [
    ("T0_W0", 0, 0), ("T0_W1", 0, 1), ("T0_W2", 0, 2),
    ("T0_W3", 0, 3), ("T0_W4", 0, 4), ("T0_W5", 0, 5),
    ("T1_W0", 1, 0), ("T1_W1", 1, 1), ("T1_W2", 1, 2), ("T1_W3", 1, 3),
]


def make_config(project, trigger_on: int, weakener_count: int):
    method = project.method33
    trigger_bytes, trigger_delay = 283.0, 0.45
    if trigger_on:
        weakener_bytes, weakener_delay = 347.0, 1.05
    else:
        weakener_bytes, weakener_delay = 630.0, 1.50
    hard = dataclasses.replace(
        method.hard_budget,
        bytes=trigger_bytes + weakener_bytes,
        delay_seconds=trigger_delay + weakener_delay,
        dummy_packets=trigger_on * 2 + weakener_count + 1,
    )
    weakener_budget = dataclasses.replace(
        method.weakener_budget,
        bytes=weakener_bytes,
        delay_seconds=weakener_delay,
        dummy_packets=weakener_count,
    )
    slots = method.weakener.dummy_insertion_outbound_slots
    if weakener_count > len(slots):
        slots = tuple(range(1, weakener_count + 2))
    weakener = dataclasses.replace(
        method.weakener,
        dummy_insertion_outbound_slots=slots,
    )
    return dataclasses.replace(
        method,
        hard_budget=hard,
        weakener_budget=weakener_budget,
        weakener_dummy_count=weakener_count,
        weakener=weakener,
    )


def hist_rows(lengths, mask, indices, bins):
    out = np.zeros((len(indices), len(bins) - 1))
    for row, index in enumerate(indices):
        values = lengths[index][mask[index].astype(bool)]
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bins)
        counts = counts.astype(float)
        out[row] = counts / counts.sum() if counts.sum() else counts
    return out


def clean_features(dataset, indices, bins, mode):
    lengths = np.asarray(dataset["lengths"]).astype(np.float64)
    gaps = np.asarray(dataset["gaps"]).astype(np.float64)
    directions = np.asarray(dataset["directions"]).astype(np.float64)
    mask = np.asarray(dataset["mask"]).astype(bool)
    if mode == "hist":
        return hist_rows(lengths, mask, indices, bins)
    first = directions * lengths / 1514.0
    last = np.log1p(np.minimum(gaps, 10.0) * 1000.0)
    return np.concatenate([first[indices], last[indices]], axis=1)


def app_features(app, bins, mode):
    lengths = np.asarray(app[0], dtype=np.float64)
    gaps = np.asarray(app[1], dtype=np.float64)
    directions = np.asarray(app[2], dtype=np.float64)
    mask = np.asarray(app[3]).astype(bool)
    if mode == "hist":
        return hist_rows(lengths, mask, np.arange(len(lengths)), bins)
    first = directions * lengths / 1514.0
    last = np.log1p(np.minimum(gaps, 10.0) * 1000.0)
    return np.concatenate([first, last], axis=1)


def fit_ba(x_train, y_train, x_test, y_test, surrogate):
    if surrogate == "rf":
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
    else:
        scaler = StandardScaler().fit(x_train)
        if surrogate == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
        else:
            model = LogisticRegression(max_iter=500)
        model.fit(scaler.transform(x_train), y_train)
        pred = model.predict(scaler.transform(x_test))
    return float(balanced_accuracy_score(y_test, pred))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--feature-mode", choices=("hist", "combined"), default="combined")
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    rng = np.random.default_rng(20261030)
    train_idx = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    test_idx = cap_with_min(np.asarray(split["test"]), labels, args.cap, 1, rng)
    y_train, y_test = labels[train_idx], labels[test_idx]
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    bins = np.linspace(0, 1514, args.bins + 1)
    clean_train = clean_features(dataset, train_idx, bins, args.feature_mode)
    clean_test = clean_features(dataset, test_idx, bins, args.feature_mode)
    project = method33_tools.load_config(args.config)
    classes = [int(v) for v in np.unique(y_test)]
    days = groups[test_idx]
    unique_days = np.unique(days)

    results = {}
    for name, trigger_on, weakener_count in CELLS:
        config = make_config(project, trigger_on, weakener_count)
        train_app = apply_method33(
            *take(dataset, train_idx), config, 731800, mapping="old",
            include_trigger=bool(trigger_on), jitter_config=project.trigger_jitter,
        )
        test_app = apply_method33(
            *take(dataset, test_idx), config, 731800, mapping="next",
            include_trigger=bool(trigger_on), jitter_config=project.trigger_jitter,
        )
        defended_train = app_features(train_app, bins, args.feature_mode)
        defended_test = app_features(test_app, bins, args.feature_mode)
        stats = test_app[4]
        cell = {
            "trigger_on": trigger_on,
            "weakener_count": weakener_count,
            "ba": {},
            "cost": {
                "real_added_bytes": float(np.mean(stats["real_added_bytes"])),
                "dummy_bytes": float(np.mean(stats["dummy_bytes"])),
                "dummy_count": float(np.mean(stats["dummy_count"])),
                "retained_dummy_count": float(np.mean(stats["retained_dummy_count"])),
                "displaced_real_packets": float(np.mean(stats["displaced_real_packets"])),
                "total_added_bytes": float(np.mean(np.asarray(stats["real_added_bytes"]) + np.asarray(stats["dummy_bytes"]))),
            },
        }
        for surrogate in ("lr", "rf", "mlp"):
            clean_ba = fit_ba(clean_train, y_train, clean_test, y_test, surrogate)
            retrained_ba = fit_ba(defended_train, y_train, defended_test, y_test, surrogate)
            cross_ba = fit_ba(clean_train, y_train, defended_test, y_test, surrogate)
            cell["ba"][surrogate] = {
                "clean": clean_ba,
                "retrained": retrained_ba,
                "clean_to_defended": cross_ba,
                "drop_clean": clean_ba - cross_ba,
                "recovery": retrained_ba - cross_ba,
                "adaptation": retrained_ba - clean_ba,
            }
        results[name] = cell
        print(name, "cost", {k: round(v, 3) for k, v in cell["cost"].items()},
              "retrained", {s: round(cell["ba"][s]["retrained"], 4) for s in ("lr", "rf", "mlp")},
              flush=True)

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(
        {"dataset_version": args.dataset_version, "cap": args.cap, "cells": results},
        ensure_ascii=False, indent=2,
    ) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
