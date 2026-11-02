"""Diagnose classifier failure: per-class recall, confusion, permutation alignment.

The permutation is learned on a held-out validation split (predictions under the
next mapping), then applied to the test set.  The test labels are never used to
search the permutation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_ablation import cap_with_min  # noqa: E402
from scripts.position_placebo import (  # noqa: E402
    clean_features,
    features,
    fit_predict,
    make_config,
    random_positions,
)


def fit_model(x_train, y_train, surrogate):
    if surrogate == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)
    elif surrogate == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train)
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
        model.fit(scaler.transform(x_train), y_train)
        return lambda x: model.predict(scaler.transform(x))
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=500)
        model.fit(scaler.transform(x_train), y_train)
        return lambda x: model.predict(scaler.transform(x))
    model.fit(x_train, y_train)
    return lambda x: model.predict(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    rng = np.random.default_rng(20261030)
    train_idx = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    test_idx = cap_with_min(np.asarray(split["test"]), labels, args.cap, 1, rng)
    project = method33_tools.load_config(args.config)
    pool = list(project.method33.trigger_positions)

    # Stratified 80/20 inner validation, never touching test labels.
    fit_parts, val_parts = [], []
    for label in np.unique(labels[train_idx]):
        pool_idx = train_idx[labels[train_idx] == label]
        order = rng.permutation(len(pool_idx))
        cut = max(1, int(round(0.2 * len(pool_idx))))
        val_parts.extend(pool_idx[order[:cut]].tolist())
        fit_parts.extend(pool_idx[order[cut:]].tolist())
    fit_idx = np.asarray(sorted(fit_parts))
    val_idx = np.asarray(sorted(val_parts))
    y_fit, y_val, y_test = labels[fit_idx], labels[val_idx], labels[test_idx]
    clean_fit = clean_features(dataset, fit_idx)
    clean_val = clean_features(dataset, val_idx)
    clean_test = clean_features(dataset, test_idx)

    arms = ["placebo_code", "placebo_position", "T0_matched", "real"]
    results = {"dataset_version": args.dataset_version, "arms": {}}
    for arm in arms:
        mode = "off" if arm == "T0_matched" else arm
        config = make_config(project, mode)
        fit_override = random_positions(pool, len(fit_idx), np.random.default_rng(1)) if arm == "placebo_position" else None
        val_override = random_positions(pool, len(val_idx), np.random.default_rng(2)) if arm == "placebo_position" else None
        test_override = random_positions(pool, len(test_idx), np.random.default_rng(3)) if arm == "placebo_position" else None
        fit_app = apply_method33(
            *take(dataset, fit_idx), config, 731800, mapping="old",
            include_trigger=(arm != "T0_matched"), jitter_config=project.trigger_jitter,
            trigger_position_override=fit_override,
        )
        val_app = apply_method33(
            *take(dataset, val_idx), config, 731800, mapping="next",
            include_trigger=(arm != "T0_matched"), jitter_config=project.trigger_jitter,
            trigger_position_override=val_override,
        )
        test_app = apply_method33(
            *take(dataset, test_idx), config, 731800, mapping="next",
            include_trigger=(arm != "T0_matched"), jitter_config=project.trigger_jitter,
            trigger_position_override=test_override,
        )
        x_fit, x_val, x_test = features(fit_app), features(val_app), features(test_app)
        arm_result = {"cost": {
            "total_added_bytes": float(np.mean(np.asarray(test_app[4]["real_added_bytes"]) + np.asarray(test_app[4]["dummy_bytes"]))),
            "displaced_mean": float(np.mean(test_app[4]["displaced_real_packets"])),
        }, "surrogates": {}}
        for surrogate in ("lr", "rf", "mlp"):
            predict = fit_model(x_fit, y_fit, surrogate)
            pred_val = predict(x_val)
            pred_test = predict(x_test)
            classes = sorted(set(y_val.tolist()) | set(y_test.tolist()))
            cm_val = confusion_matrix(y_val, pred_val, labels=classes)
            row, col = linear_sum_assignment(-cm_val)
            mapping = {int(pred): int(true) for pred, true in zip(col, row)}
            aligned = np.asarray([mapping.get(int(p), int(p)) for p in pred_test])
            recall = recall_score(y_test, pred_test, labels=classes, average=None, zero_division=0)
            cm_test = confusion_matrix(y_test, pred_test, labels=classes)
            pairs = []
            for true in range(len(classes)):
                for pred in range(len(classes)):
                    if true != pred and cm_test[true, pred] > 0:
                        pairs.append((int(cm_test[true, pred]), classes[true], classes[pred]))
            pairs.sort(reverse=True)
            arm_result["surrogates"][surrogate] = {
                "unaligned_ba": float(balanced_accuracy_score(y_test, pred_test)),
                "aligned_ba": float(balanced_accuracy_score(y_test, aligned)),
                "permutation": {str(k): v for k, v in mapping.items()},
                "per_class_recall": {str(c): float(r) for c, r in zip(classes, recall)},
                "top_confusions": pairs[:8],
                "confusion": cm_test.tolist(),
                "classes": classes,
            }
        results["arms"][arm] = arm_result
        print(arm, {s: (round(arm_result["surrogates"][s]["unaligned_ba"], 3), round(arm_result["surrogates"][s]["aligned_ba"], 3)) for s in ("lr", "rf", "mlp")}, flush=True)

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
