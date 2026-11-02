"""Position placebo: remove class-position association while keeping 2 events.

Arms (W=3, weakener 347 B / 1.05 s):
  A placebo_code      all classes share one code; original class positions
  B placebo_position  all classes share one code; per-flow pooled random positions
  C T0_matched        trigger off; weakener 347/W3
  D real              class-specific code and positions

B samples positions per flow from the pooled original pairs, independent of
labels, preserving the marginal pair/spacing distribution.  All arms use the
same train/test flows and paired day bootstrap.
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


def make_config(project, mode: str, weakener_count: int = 3):
    method = project.method33
    if mode in ("placebo_code", "placebo_position"):
        codes = tuple(method.trigger_codes[0] for _ in range(method.class_count))
        positions = method.trigger_positions
    elif mode == "real":
        codes, positions = method.trigger_codes, method.trigger_positions
    else:
        codes, positions = method.trigger_codes, method.trigger_positions
    weakener_budget = dataclasses.replace(
        method.weakener_budget, bytes=347.0, delay_seconds=1.05, dummy_packets=weakener_count
    )
    hard = dataclasses.replace(
        method.hard_budget, bytes=630.0, delay_seconds=1.50, dummy_packets=2 + weakener_count + 1
    )
    return dataclasses.replace(
        method,
        hard_budget=hard,
        weakener_budget=weakener_budget,
        weakener_dummy_count=weakener_count,
        trigger_codes=codes,
        trigger_positions=positions,
    )


def random_positions(pool, n, rng):
    repeats = (n + len(pool) - 1) // len(pool)
    pairs = list(pool) * repeats
    pairs = pairs[:n]
    rng.shuffle(pairs)
    return tuple((int(a), int(b)) for a, b in pairs)


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


def position_stats(app):
    trigger_counts, weakener_counts, real_counts = [], [], []
    anchors = []
    for audit in app[6]:
        trace = audit["complete_event_trace"]
        visible = [e for e in trace if e["in_observation_window"]]
        trigger = [e for e in visible if e["role"] == "trigger_dummy"]
        weakener = [e for e in visible if e["role"] == "weakener_dummy"]
        real = [e for e in visible if e["role"] == "real"]
        trigger_counts.append(len(trigger))
        weakener_counts.append(len(weakener))
        real_counts.append(len(real))
        for event in trigger:
            before = sum(1 for e in trace[: event["event_position"]] if e["role"] == "real")
            anchors.append(before)
    return {
        "mean_trigger_in_window": float(np.mean(trigger_counts)),
        "mean_weakener_in_window": float(np.mean(weakener_counts)),
        "mean_real_in_window": float(np.mean(real_counts)),
        "mean_dummy_fraction": float(np.mean([
            (t + w) / max(1, t + w + r)
            for t, w, r in zip(trigger_counts, weakener_counts, real_counts)
        ])),
        "anchor_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(anchors, return_counts=True))} if anchors else {},
    }


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
    pool = list(project.method33.trigger_positions)
    train_positions = random_positions(pool, len(train_idx), np.random.default_rng(20261031))
    test_positions = random_positions(pool, len(test_idx), np.random.default_rng(20261032))
    classes = [int(v) for v in np.unique(y_test)]
    days = groups[test_idx]
    unique_days = np.unique(days)

    arms = ["placebo_code", "placebo_position", "T0_matched", "real"]
    predictions, costs, positions = {}, {}, {}
    for arm in arms:
        mode = "off" if arm == "T0_matched" else arm
        config = make_config(project, mode)
        override = train_positions if arm == "placebo_position" else None
        test_override = test_positions if arm == "placebo_position" else None
        train_app = apply_method33(
            *take(dataset, train_idx), config, 731800, mapping="old",
            include_trigger=(arm != "T0_matched"), include_event_trace=True,
            jitter_config=project.trigger_jitter, trigger_position_override=override,
        )
        test_app = apply_method33(
            *take(dataset, test_idx), config, 731800, mapping="next",
            include_trigger=(arm != "T0_matched"), include_event_trace=True,
            jitter_config=project.trigger_jitter, trigger_position_override=test_override,
        )
        defended_train, defended_test = features(train_app), features(test_app)
        stats = test_app[4]
        costs[arm] = {
            "total_added_bytes": float(np.mean(np.asarray(stats["real_added_bytes"]) + np.asarray(stats["dummy_bytes"]))),
            "real_added_bytes": float(np.mean(stats["real_added_bytes"])),
            "dummy_bytes": float(np.mean(stats["dummy_bytes"])),
            "dummy_count": float(np.mean(stats["dummy_count"])),
            "displaced_mean": float(np.mean(stats["displaced_real_packets"])),
            "displaced_median": float(np.median(stats["displaced_real_packets"])),
            "displaced_p95": float(np.percentile(stats["displaced_real_packets"], 95)),
            "displaced_any": float(np.mean(np.asarray(stats["displaced_real_packets"]) > 0)),
        }
        positions[arm] = position_stats(test_app)
        predictions[arm] = {}
        for surrogate in ("lr", "rf", "mlp"):
            predictions[arm][surrogate] = {
                "clean": fit_predict(clean_train, y_train, clean_test, surrogate),
                "retrained": fit_predict(defended_train, y_train, defended_test, surrogate),
                "cross": fit_predict(clean_train, y_train, defended_test, surrogate),
            }
        print(arm, "cost", {k: round(v, 3) for k, v in costs[arm].items()}, flush=True)

    def ba_idx(pred, idx):
        return macro_ba(y_test[idx], pred[idx], classes)

    contrasts = {"A_minus_B": ("placebo_code", "placebo_position"),
                 "B_minus_C": ("placebo_position", "T0_matched"),
                 "A_minus_D": ("placebo_code", "real")}
    boot = {arm: {s: {sc: [] for sc in ("clean", "retrained", "cross")} for s in ("lr", "rf", "mlp")} for arm in arms}
    boot_contrast = {k: {s: {sc: [] for sc in ("retrained", "cross")} for s in ("lr", "rf", "mlp")} for k in contrasts}
    for _ in range(args.bootstrap):
        draw = rng.choice(unique_days, size=len(unique_days), replace=True)
        idx = np.concatenate([np.where(days == day)[0] for day in draw])
        for arm in arms:
            for s in ("lr", "rf", "mlp"):
                for sc in ("clean", "retrained", "cross"):
                    boot[arm][s][sc].append(ba_idx(predictions[arm][s][sc], idx))
        for key, (left, right) in contrasts.items():
            for s in ("lr", "rf", "mlp"):
                for sc in ("retrained", "cross"):
                    boot_contrast[key][s][sc].append(ba_idx(predictions[left][s][sc], idx) - ba_idx(predictions[right][s][sc], idx))

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    result = {"dataset_version": args.dataset_version, "arms": {}, "contrasts": {}, "positions": positions}
    for arm in arms:
        result["arms"][arm] = {
            "cost": costs[arm],
            "ba": {
                s: {
                    sc: {"value": macro_ba(y_test, predictions[arm][s][sc], classes), "ci": interval(boot[arm][s][sc])}
                    for sc in ("clean", "retrained", "cross")
                }
                for s in ("lr", "rf", "mlp")
            },
        }
    for key, (left, right) in contrasts.items():
        result["contrasts"][key] = {
            s: {
                sc: {"value": macro_ba(y_test, predictions[left][s][sc], classes) - macro_ba(y_test, predictions[right][s][sc], classes),
                     "ci": interval(boot_contrast[key][s][sc])}
                for sc in ("retrained", "cross")
            }
            for s in ("lr", "rf", "mlp")
        }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key in contrasts:
        print(key, {s: round(result["contrasts"][key][s]["retrained"]["value"], 4) for s in ("lr", "rf", "mlp")}, flush=True)


if __name__ == "__main__":
    main()
