"""Stage A.1/A.2: real vs placebo trigger, padding-matched T0, fixed 5-event.

Arms (all W fixed except A.2):
  real_T1_W3       class-specific trigger code+positions, weakener 347/1.05/W3
  placebo_T1_W3    all classes share one code+position (no class code/rotation)
  T0_matched_W3    trigger off, weakener 347/1.05/W3 (padding matched to T1)
  T0_full_W3       trigger off, weakener 630/1.5/W3 (full reallocation)
  T0_W5            trigger off, weakener 630/1.5/W5
  T1_W3            same as real_T1_W3 (A.2)

Contrasts are paired by test day on the same flows.
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

ARMS = [
    ("real_T1_W3", "real", True, 3, 347.0, 1.05),
    ("placebo_T1_W3", "placebo", True, 3, 347.0, 1.05),
    ("T0_matched_W3", "off", False, 3, 347.0, 1.05),
    ("T0_full_W3", "off", False, 3, 630.0, 1.50),
    ("T0_W5", "off", False, 5, 630.0, 1.50),
    ("T1_W3", "real", True, 3, 347.0, 1.05),
]


def make_config(project, mode: str, weakener_count: int, weakener_bytes: float, weakener_delay: float):
    method = project.method33
    if mode == "real":
        codes, positions = method.trigger_codes, method.trigger_positions
    elif mode == "placebo":
        # Same 2-event structure, same positions and budget; only the
        # class-specific code (and therefore rotation) is removed.
        codes = tuple(method.trigger_codes[0] for _ in range(method.class_count))
        positions = method.trigger_positions
    else:
        codes, positions = method.trigger_codes, method.trigger_positions
    hard = dataclasses.replace(
        method.hard_budget,
        bytes=283.0 + weakener_bytes,
        delay_seconds=0.45 + weakener_delay,
        dummy_packets=2 + weakener_count + 1,
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
    weakener = dataclasses.replace(method.weakener, dummy_insertion_outbound_slots=slots)
    return dataclasses.replace(
        method,
        hard_budget=hard,
        weakener_budget=weakener_budget,
        weakener_dummy_count=weakener_count,
        weakener=weakener,
        trigger_codes=codes,
        trigger_positions=positions,
    )


def features(app):
    lengths = np.asarray(app[0], dtype=np.float64)
    gaps = np.asarray(app[1], dtype=np.float64)
    directions = np.asarray(app[2], dtype=np.float64)
    return np.concatenate([directions * lengths / 1514.0, np.log1p(np.minimum(gaps, 10.0) * 1000.0)], axis=1)


def clean_features(dataset, indices):
    lengths = np.asarray(dataset["lengths"]).astype(np.float64)
    gaps = np.asarray(dataset["gaps"]).astype(np.float64)
    directions = np.asarray(dataset["directions"]).astype(np.float64)
    return np.concatenate(
        [directions[indices] * lengths[indices] / 1514.0,
         np.log1p(np.minimum(gaps[indices], 10.0) * 1000.0)],
        axis=1,
    )


def fit_predict(x_train, y_train, x_test, surrogate):
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


def macro_ba(y_true, prediction, classes):
    return float(np.mean([np.mean(prediction[y_true == c] == c) if np.any(y_true == c) else 0.0 for c in classes]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    rng = np.random.default_rng(20261030)
    train_idx = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    test_idx = cap_with_min(np.asarray(split["test"]), labels, args.cap, 1, rng)
    y_train, y_test = labels[train_idx], labels[test_idx]
    clean_train = clean_features(dataset, train_idx)
    clean_test = clean_features(dataset, test_idx)
    project = method33_tools.load_config(args.config)
    classes = [int(v) for v in np.unique(y_test)]
    days = groups[test_idx]
    unique_days = np.unique(days)

    predictions = {}
    costs = {}
    for name, mode, trigger, weakener_count, weakener_bytes, weakener_delay in ARMS:
        config = make_config(project, mode, weakener_count, weakener_bytes, weakener_delay)
        train_app = apply_method33(
            *take(dataset, train_idx), config, 731800, mapping="old",
            include_trigger=trigger, jitter_config=project.trigger_jitter,
        )
        test_app = apply_method33(
            *take(dataset, test_idx), config, 731800, mapping="next",
            include_trigger=trigger, jitter_config=project.trigger_jitter,
        )
        defended_train, defended_test = features(train_app), features(test_app)
        stats = test_app[4]
        costs[name] = {
            "trigger": bool(trigger),
            "weakener_count": weakener_count,
            "weakener_bytes_cap": weakener_bytes,
            "total_added_bytes": float(np.mean(np.asarray(stats["real_added_bytes"]) + np.asarray(stats["dummy_bytes"]))),
            "real_added_bytes": float(np.mean(stats["real_added_bytes"])),
            "dummy_bytes": float(np.mean(stats["dummy_bytes"])),
            "dummy_count": float(np.mean(stats["dummy_count"])),
            "displaced_real_packets": float(np.mean(stats["displaced_real_packets"])),
        }
        predictions[name] = {}
        for surrogate in ("lr", "rf", "mlp"):
            pred_clean = fit_predict(clean_train, y_train, clean_test, surrogate)
            pred_retrained = fit_predict(defended_train, y_train, defended_test, surrogate)
            pred_cross = fit_predict(clean_train, y_train, defended_test, surrogate)
            predictions[name][surrogate] = {
                "clean": pred_clean,
                "retrained": pred_retrained,
                "cross": pred_cross,
            }
        print(name, "cost", {k: round(v, 3) for k, v in costs[name].items()}, flush=True)

    def ba(pred):
        return macro_ba(y_test, pred, classes)

    def ba_idx(pred, idx):
        return macro_ba(y_test[idx], pred[idx], classes)

    boot = {name: {s: {"clean": [], "retrained": [], "cross": []} for s in ("lr", "rf", "mlp")} for name in costs}
    boot_contrast = {}
    contrasts = {
        "real_minus_placebo": ("real_T1_W3", "placebo_T1_W3"),
        "real_minus_T0matched": ("real_T1_W3", "T0_matched_W3"),
        "placebo_minus_T0matched": ("placebo_T1_W3", "T0_matched_W3"),
        "T1W3_minus_T0W5": ("T1_W3", "T0_W5"),
    }
    for key in contrasts:
        boot_contrast[key] = {
            s: {scenario: [] for scenario in ("retrained", "cross")}
            for s in ("lr", "rf", "mlp")
        }
    for _ in range(args.bootstrap):
        draw = rng.choice(unique_days, size=len(unique_days), replace=True)
        idx = np.concatenate([np.where(days == day)[0] for day in draw])
        for name in costs:
            for surrogate in ("lr", "rf", "mlp"):
                for scenario in ("clean", "retrained", "cross"):
                    boot[name][surrogate][scenario].append(ba_idx(predictions[name][surrogate][scenario], idx))
        for key, (left, right) in contrasts.items():
            for surrogate in ("lr", "rf", "mlp"):
                for scenario in ("retrained", "cross"):
                    value = ba_idx(predictions[left][surrogate][scenario], idx) - ba_idx(predictions[right][surrogate][scenario], idx)
                    boot_contrast[key][surrogate][scenario].append(value)

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    result = {"dataset_version": args.dataset_version, "arms": {}, "contrasts": {}}
    for name in costs:
        result["arms"][name] = {
            "cost": costs[name],
            "ba": {
                surrogate: {
                    scenario: {
                        "value": ba(predictions[name][surrogate][scenario]),
                        "ci": interval(boot[name][surrogate][scenario]),
                    }
                    for scenario in ("clean", "retrained", "cross")
                }
                for surrogate in ("lr", "rf", "mlp")
            },
        }
    for key, (left, right) in contrasts.items():
        result["contrasts"][key] = {
            surrogate: {
                scenario: {
                    "value": ba(predictions[left][surrogate][scenario]) - ba(predictions[right][surrogate][scenario]),
                    "ci": interval(boot_contrast[key][surrogate][scenario]),
                }
                for scenario in ("retrained", "cross")
            }
            for surrogate in ("lr", "rf", "mlp")
        }

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key in contrasts:
        print(key, {s: round(result["contrasts"][key][s]["retrained"]["value"], 4) for s in ("lr", "rf", "mlp")}, flush=True)


if __name__ == "__main__":
    main()
