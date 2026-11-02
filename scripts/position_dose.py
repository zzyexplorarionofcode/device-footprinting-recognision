"""Position-association dose-response: alpha in [0,1].

alpha=1: each flow uses its class-specific pair.
alpha=0: each flow samples the global pool, weighted to match the class-pair
         marginal (so the marginal position distribution is comparable).
0<alpha<1: mixture; only the class-position association changes.

Trigger code is class-independent (placebo code), weakener W=3, 630 B cap.
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
from scripts.position_placebo import clean_features, features  # noqa: E402


def make_config(project):
    method = project.method33
    codes = tuple(method.trigger_codes[0] for _ in range(method.class_count))
    weakener_budget = dataclasses.replace(
        method.weakener_budget, bytes=347.0, delay_seconds=1.05, dummy_packets=3
    )
    hard = dataclasses.replace(method.hard_budget, bytes=630.0, delay_seconds=1.50, dummy_packets=6)
    return dataclasses.replace(
        method,
        trigger_codes=codes,
        weakener_budget=weakener_budget,
        weakener_dummy_count=3,
        hard_budget=hard,
    )


def sample_positions(indices, labels, config, alpha, class_weights, rng, mapping=None):
    pool = list(config.trigger_positions)
    weights = np.asarray([class_weights.get(i, 1.0) for i in range(len(pool))], dtype=float)
    weights = weights / weights.sum()
    result = []
    for index in indices:
        if rng.random() < alpha:
            code_index = int(mapping[int(labels[index])]) if mapping is not None else int(labels[index])
            result.append(pool[code_index])
        else:
            result.append(pool[int(rng.choice(len(pool), p=weights))])
    return tuple(result)


def position_features(pairs, pair_to_index):
    x = np.zeros((len(pairs), len(pair_to_index)))
    for row, pair in enumerate(pairs):
        x[row, pair_to_index[tuple(pair)]] = 1.0
    return x


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


def macro_ba(y_true, pred, classes):
    return float(np.mean([np.mean(pred[y_true == c] == c) if np.any(y_true == c) else 0.0 for c in classes]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=300)
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
    config = make_config(project)
    pool = list(config.trigger_positions)
    pair_to_index = {tuple(pair): i for i, pair in enumerate(pool)}
    class_weights = {int(c): int(np.sum(y_train == c)) for c in np.unique(y_train)}
    classes = [int(c) for c in np.unique(y_test)]
    days = groups[test_idx]
    unique_days = np.unique(days)
    alphas = [float(v) for v in args.alphas.split(",")]

    results = {}
    for alpha in alphas:
        train_pos = sample_positions(
            train_idx, labels, config, alpha, class_weights, np.random.default_rng(11),
            mapping=config.old_mapping if args.rotation else None,
        )
        test_pos = sample_positions(
            test_idx, labels, config, alpha, class_weights, np.random.default_rng(12),
            mapping=config.next_mapping if args.rotation else None,
        )
        train_app = apply_method33(
            *take(dataset, train_idx), config, 731800, mapping="old",
            include_trigger=True, jitter_config=project.trigger_jitter,
            trigger_position_override=train_pos,
        )
        test_app = apply_method33(
            *take(dataset, test_idx), config, 731800, mapping="next",
            include_trigger=True, jitter_config=project.trigger_jitter,
            trigger_position_override=test_pos,
        )
        x_train, x_test = features(train_app), features(test_app)
        # Position-only predictability: train on train positions, test on test positions.
        px_train = position_features(train_pos, pair_to_index)
        px_test = position_features(test_pos, pair_to_index)
        position_pred = fit_predict(px_train, y_train, px_test, "lr")
        position_ba = macro_ba(y_test, position_pred, classes)
        entry = {
            "alpha": alpha,
            "position_only_ba": position_ba,
            "cost": {
                "total_added_bytes": float(np.mean(np.asarray(test_app[4]["real_added_bytes"]) + np.asarray(test_app[4]["dummy_bytes"]))),
                "displaced_mean": float(np.mean(test_app[4]["displaced_real_packets"])),
                "displaced_median": float(np.median(test_app[4]["displaced_real_packets"])),
                "displaced_p95": float(np.percentile(test_app[4]["displaced_real_packets"], 95)),
                "displaced_any": float(np.mean(np.asarray(test_app[4]["displaced_real_packets"]) > 0)),
            },
            "ba": {},
        }
        predictions = {}
        for surrogate in ("lr", "rf", "mlp"):
            pred_retr = fit_predict(x_train, y_train, x_test, surrogate)
            pred_cross = fit_predict(clean_train, y_train, x_test, surrogate)
            predictions[surrogate] = {"retrained": pred_retr, "cross": pred_cross}
            entry["ba"][surrogate] = {
                "retrained": macro_ba(y_test, pred_retr, classes),
                "cross": macro_ba(y_test, pred_cross, classes),
            }
        boot = {s: [] for s in ("lr", "rf", "mlp")}
        for _ in range(args.bootstrap):
            draw = rng.choice(unique_days, size=len(unique_days), replace=True)
            idx = np.concatenate([np.where(days == day)[0] for day in draw])
            for surrogate in ("lr", "rf", "mlp"):
                boot[surrogate].append(macro_ba(y_test[idx], predictions[surrogate]["retrained"][idx], classes))
        for surrogate in ("lr", "rf", "mlp"):
            entry["ba"][surrogate]["ci"] = [
                float(np.percentile(boot[surrogate], 2.5)),
                float(np.percentile(boot[surrogate], 97.5)),
            ]
        results[str(alpha)] = entry
        print(
            alpha,
            "position_only", round(position_ba, 3),
            "retrained", {s: round(entry["ba"][s]["retrained"], 3) for s in ("lr", "rf", "mlp")},
            "displaced", round(entry["cost"]["displaced_mean"], 2),
            flush=True,
        )

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps({"dataset_version": args.dataset_version, "rotation": bool(args.rotation), "alphas": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
