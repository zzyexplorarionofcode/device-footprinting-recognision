"""Experiment A: does the length-histogram leak come from real packets or dummies?

The event trace returned by ``apply_method33(..., include_event_trace=True)``
labels every emitted event as ``real``, ``trigger_dummy`` or ``weakener_dummy``.
For each flow we build three normalized length histograms:

    real      only real packets in the 32-event observation window
    dummy     only synthetic packets in the window
    combined  all events in the window (what an attacker can see)

Attackers are LR on defended train (old mapping) -> defended test (next mapping),
plus a clean reference trained/tested on undefended real histograms.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_ablation import cap_with_min  # noqa: E402

BINS = np.linspace(0.0, 1514.0, 17)


def _hist(values, bins=BINS) -> np.ndarray:
    if not values:
        return np.zeros(len(bins) - 1, dtype=float)
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bins)
    counts = counts.astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def source_views(indices, dataset, config, mapping, seed, chunk):
    real_h, dummy_h, combined_h = [], [], []
    real_counts, dummy_counts = [], []
    for start in range(0, len(indices), chunk):
        subset = indices[start : start + chunk]
        application = take(dataset, subset)
        output = apply_method33(
            *application,
            config,
            seed,
            mapping=mapping,
            jitter_config=config_jitter[0],
            include_event_trace=True,
        )
        for row in output[6]:
            trace = row["complete_event_trace"]
            real = [
                event["length_bytes"]
                for event in trace
                if event["in_observation_window"] and event["role"] == "real"
            ]
            dummy = [
                event["length_bytes"]
                for event in trace
                if event["in_observation_window"] and event["role"] != "real"
            ]
            real_h.append(_hist(real))
            dummy_h.append(_hist(dummy))
            combined_h.append(_hist(real + dummy))
            real_counts.append(len(real))
            dummy_counts.append(len(dummy))
    return (
        np.asarray(real_h),
        np.asarray(dummy_h),
        np.asarray(combined_h),
        np.asarray(real_counts),
        np.asarray(dummy_counts),
    )


def clean_real_hist(indices, dataset):
    lengths = np.asarray(dataset["lengths"])[indices]
    mask = np.asarray(dataset["mask"])[indices].astype(bool)
    output = np.zeros((len(indices), len(BINS) - 1), dtype=float)
    for index in range(len(indices)):
        output[index] = _hist(lengths[index][mask[index]].tolist())
    return output


def fit_score(x_train, y_train, x_test, y_test) -> dict:
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
    prediction = model.predict(scaler.transform(x_test))
    return {
        "ba": float(balanced_accuracy_score(y_test, prediction)),
        "recall": recall_score(
            y_test, prediction, labels=np.arange(int(y_test.max()) + 1), average=None, zero_division=0
        ).tolist(),
    }


def main() -> None:
    global config_jitter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=8000)
    parser.add_argument("--test-cap", type=int, default=8000)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    split = np.load(args.split_indices)
    rng = np.random.default_rng(args.seed)
    train_indices = cap_with_min(np.asarray(split["train"]), labels, args.train_cap, 1, rng)
    test_indices = cap_with_min(np.asarray(split["test"]), labels, args.test_cap, 1, rng)
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    project = method33_tools.load_config(args.config)
    config_jitter = (project.trigger_jitter,)
    config = project.method33
    started = time.time()

    clean_train = clean_real_hist(train_indices, dataset)
    clean_test = clean_real_hist(test_indices, dataset)
    results = {
        "clean_real": fit_score(clean_train, train_labels, clean_test, test_labels)
    }
    print(f"[clean real] BA={results['clean_real']['ba']:.4f}", flush=True)

    for mapping, train_map, test_map in (("old", "old", "next"),):
        real_train, dummy_train, combined_train, rc_tr, dc_tr = source_views(
            train_indices, dataset, config, train_map, args.seed, args.chunk
        )
        real_test, dummy_test, combined_test, rc_te, dc_te = source_views(
            test_indices, dataset, config, test_map, args.seed, args.chunk
        )
        for name, xtr, xte in (
            ("real", real_train, real_test),
            ("dummy", dummy_train, dummy_test),
            ("combined", combined_train, combined_test),
        ):
            results[name] = fit_score(xtr, train_labels, xte, test_labels)
            print(f"[{name}] BA={results[name]['ba']:.4f}", flush=True)
        results["clean_to_defended"] = fit_score(
            clean_train, train_labels, combined_test, test_labels
        )
        print(f"[clean_to_defended] BA={results['clean_to_defended']['ba']:.4f}", flush=True)

    results["counts"] = {
        "real_train_mean": float(np.mean(rc_tr)),
        "dummy_train_mean": float(np.mean(dc_tr)),
        "real_test_mean": float(np.mean(rc_te)),
        "dummy_test_mean": float(np.mean(dc_te)),
    }
    log = {
        "dataset": str(args.dataset),
        "config": str(args.config),
        "train_flows": len(train_indices),
        "test_flows": len(test_indices),
        "results": results,
        "seconds": round(time.time() - started, 1),
        "note": "oracle source ablation; real/dummy labels come from the event trace",
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: round(v["ba"], 4) if isinstance(v, dict) and "ba" in v else v for k, v in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
