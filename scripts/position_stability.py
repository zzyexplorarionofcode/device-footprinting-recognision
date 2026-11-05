"""Within-class position stability: K pairs per class under mapping rotation.

K=1 uses the original v2 class position.  K=2/4/8 add random legal pairs to the
same class pool (nested), sampled per flow after applying the mapping.  The
trigger length code is held class-independent (placebo code), W=3, 630 B cap.
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


def build_pools(project, k_max: int, rng):
    pairs = [tuple(pair) for pair in project.method33.trigger_positions]
    master = [
        (a, b)
        for a in range(1, 17)
        for b in range(a + 1, 21)
        if (a, b) not in pairs
    ]
    pools = []
    for pair in pairs:
        chosen = [pair]
        candidates = list(master)
        rng.shuffle(candidates)
        for candidate in candidates:
            if len(chosen) >= k_max:
                break
            if candidate not in chosen:
                chosen.append(candidate)
        pools.append(tuple(chosen))
    return pools


def sample_positions(indices, labels, pools, mapping, k, rng):
    result = []
    for index in indices:
        code_index = int(mapping[int(labels[index])])
        pool = pools[code_index][:k]
        result.append(tuple(pool[int(rng.integers(0, len(pool)))]))
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
    parser.add_argument("--ks", default="1,2,4,8")
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
    pools = build_pools(project, max(int(k) for k in args.ks.split(",")), np.random.default_rng(7))
    all_pairs = [pair for pool in pools for pair in pool]
    pair_to_index = {pair: i for i, pair in enumerate(dict.fromkeys(all_pairs))}
    classes = [int(c) for c in np.unique(y_test)]
    days = groups[test_idx]
    unique_days = np.unique(days)

    results = {}
    for k in [int(v) for v in args.ks.split(",")]:
        train_pos = sample_positions(train_idx, labels, pools, config.old_mapping, k, np.random.default_rng(21))
        test_pos = sample_positions(test_idx, labels, pools, config.next_mapping, k, np.random.default_rng(22))
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
        position_pred = fit_predict(
            position_features(train_pos, pair_to_index), y_train,
            position_features(test_pos, pair_to_index), "lr",
        )
        entry = {
            "k": k,
            "position_only_ba": macro_ba(y_test, position_pred, classes),
            "cost": {
                "total_added_bytes": float(np.mean(np.asarray(test_app[4]["real_added_bytes"]) + np.asarray(test_app[4]["dummy_bytes"]))),
                "displaced_mean": float(np.mean(test_app[4]["displaced_real_packets"])),
                "displaced_median": float(np.median(test_app[4]["displaced_real_packets"])),
                "displaced_p95": float(np.percentile(test_app[4]["displaced_real_packets"], 95)),
                "displaced_any": float(np.mean(np.asarray(test_app[4]["displaced_real_packets"]) > 0)),
            },
            "ba": {},
            "pair_counts": {},
        }
        for pair, count in zip(*np.unique(test_pos, axis=0, return_counts=True)):
            entry["pair_counts"][f"{int(pair[0])}-{int(pair[1])}"] = int(count)
        predictions = {}
        for surrogate in ("lr", "rf", "mlp"):
            pred_retr = fit_predict(x_train, y_train, x_test, surrogate)
            predictions[surrogate] = pred_retr
            entry["ba"][surrogate] = {"retrained": macro_ba(y_test, pred_retr, classes)}
        displacement = np.asarray(test_app[4]["displaced_real_packets"])
        edges = np.asarray([0.0, 0.5, 1.5, 2.5, 3.5, 10.0])
        bins = []
        for b in range(len(edges) - 1):
            idx = np.where((displacement >= edges[b]) & (displacement < edges[b + 1]))[0]
            if len(idx) == 0:
                continue
            row = {"low": float(edges[b]), "high": float(edges[b + 1]), "n": int(len(idx)), "ba": {}}
            for s in ("lr", "rf", "mlp"):
                row["ba"][s] = macro_ba(y_test[idx], predictions[s][idx], classes)
            bins.append(row)
        entry["displacement_bins"] = bins
        boot = {s: [] for s in ("lr", "rf", "mlp")}
        for _ in range(args.bootstrap):
            draw = rng.choice(unique_days, size=len(unique_days), replace=True)
            idx = np.concatenate([np.where(days == day)[0] for day in draw])
            for s in ("lr", "rf", "mlp"):
                boot[s].append(macro_ba(y_test[idx], predictions[s][idx], classes))
        for s in ("lr", "rf", "mlp"):
            entry["ba"][s]["ci"] = [float(np.percentile(boot[s], 2.5)), float(np.percentile(boot[s], 97.5))]
        results[str(k)] = entry
        print(k, "position_only", round(entry["position_only_ba"], 3),
              "retr", {s: round(entry["ba"][s]["retrained"], 3) for s in ("lr", "rf", "mlp")},
              "disp", round(entry["cost"]["displaced_mean"], 2), flush=True)

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    # Standardize each K to the K=1 displacement-bin distribution.
    reference = results.get("1", {}).get("displacement_bins", [])
    total_ref = sum(item["n"] for item in reference)
    if total_ref:
        for k, entry in results.items():
            weights = {(item["low"], item["high"]): item["n"] / total_ref for item in reference}
            entry["displacement_standardized"] = {}
            for s in ("lr", "rf", "mlp"):
                value = 0.0
                used = 0.0
                for item in entry["displacement_bins"]:
                    weight = weights.get((item["low"], item["high"]), 0.0)
                    value += weight * item["ba"][s]
                    used += weight
                entry["displacement_standardized"][s] = value / used if used else float("nan")
    args.output_log.write_text(json.dumps({"dataset_version": args.dataset_version, "ks": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
