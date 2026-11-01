"""Feature-group attackers: what does the attacker actually rely on?

Four observable views are derived from the same defended flows:

    combined   64-d signed-normalized length + log-gap (current feature)
    length     32-d signed-normalized length only
    gap        32-d log-gap only
    sequence   32-d occupancy mask + 32-d direction (no magnitudes)

Base and Full(position) are evaluated under two attacker-training scenarios,
clean and retrained, with day-cluster bootstrap CIs for balanced accuracy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
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

from data_pipeline import pcap  # noqa: E402
from methods import method33_tools  # noqa: E402
from scripts.es_codebook_search import defended, features  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_ablation import cap_with_min, predict  # noqa: E402


def _hist(
    values: np.ndarray,
    mask: np.ndarray,
    bins: np.ndarray,
    *,
    normalize: bool = True,
    alpha: float = 0.0,
) -> np.ndarray:
    output = np.zeros((values.shape[0], len(bins) - 1), dtype=float)
    for index in range(values.shape[0]):
        valid = values[index][mask[index]]
        if valid.size == 0:
            continue
        counts, _ = np.histogram(valid, bins=bins)
        counts = counts.astype(float)
        if alpha > 0:
            counts = counts + alpha
        if normalize:
            total = counts.sum()
            output[index] = counts / total if total > 0 else 0.0
        else:
            output[index] = counts
    return output


def build_views(application) -> dict[str, np.ndarray]:
    combined = features(application)
    length = combined[:, :32]
    gap = combined[:, 32:]
    mask = application[3].astype(float)
    direction = application[2].astype(float)
    sequence = np.concatenate([mask, direction], axis=1)
    lengths = application[0].astype(float)
    gaps = application[1].astype(float)
    valid = application[3].astype(bool)
    count = valid.sum(axis=1, keepdims=True).astype(float)
    length_bins = np.linspace(0.0, 1514.0, 17)
    gap_values = np.log1p(np.minimum(gaps, 10.0) * 1000.0)
    gap_bins = np.linspace(0.0, 10.0, 17)
    length_hist = _hist(lengths, valid, length_bins)
    return {
        "combined": combined,
        "length": length,
        "gap": gap,
        "sequence": sequence,
        "direction": direction,
        "mask": mask,
        "count": count,
        "length_hist": length_hist,
        "length_hist_raw": _hist(lengths, valid, length_bins, normalize=False),
        "length_hist_smooth": _hist(lengths, valid, length_bins, alpha=1.0),
        "length_hist8": _hist(lengths, valid, np.linspace(0.0, 1514.0, 9)),
        "length_hist32": _hist(lengths, valid, np.linspace(0.0, 1514.0, 33)),
        "length_hist64": _hist(lengths, valid, np.linspace(0.0, 1514.0, 65)),
        "count_plus_hist": np.concatenate([count, length_hist], axis=1),
        "gap_hist": _hist(gap_values, valid, gap_bins),
        "gap_hist_smooth": _hist(gap_values, valid, gap_bins, alpha=1.0),
        "gap_hist8": _hist(gap_values, valid, np.linspace(0.0, 10.0, 9)),
        "gap_hist32": _hist(gap_values, valid, np.linspace(0.0, 10.0, 33)),
        "gap_hist64": _hist(gap_values, valid, np.linspace(0.0, 10.0, 65)),
    }


COMBOS = {
    "L": ("length",),
    "G": ("gap",),
    "S": ("sequence",),
    "LG": ("length", "gap"),
    "LS": ("length", "sequence"),
    "GS": ("gap", "sequence"),
    "LGS": ("length", "gap", "sequence"),
}


def feature_set(name: str, views: dict[str, np.ndarray]) -> np.ndarray:
    if name in views:
        return views[name]
    if name in COMBOS:
        return np.concatenate([views[item] for item in COMBOS[name]], axis=1)
    raise ValueError(f"unknown feature set: {name}")


def fit_and_score(view, x_train, y_train, x_test, y_test, surrogate):
    if surrogate == "lr":
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=300).fit(scaler.transform(x_train), y_train)
        prediction = model.predict(scaler.transform(x_test))
    elif surrogate == "rf":
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
    elif surrogate == "mlp":
        scaler = StandardScaler().fit(x_train)
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
        model.fit(scaler.transform(x_train), y_train)
        prediction = model.predict(scaler.transform(x_test))
    else:
        raise ValueError(surrogate)
    return {
        "ba": float(balanced_accuracy_score(y_test, prediction)),
        "recall": recall_score(
            y_test, prediction, labels=np.arange(int(y_test.max()) + 1), average=None, zero_division=0
        ).tolist(),
        "prediction": prediction,
        "y_true": y_test,
        "days": None,
    }


def bootstrap_ba(y_true, prediction, days, rng, repeats=200):
    unique = np.unique(days)
    values = []
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(days == day)[0] for day in draw])
        values.append(balanced_accuracy_score(y_true[idx], prediction[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--surrogates", default="lr", help="comma separated: lr,rf,mlp")
    parser.add_argument(
        "--feature-sets",
        default="combined,length,gap,sequence",
        help="comma separated view names or combos L,G,S,LG,LS,GS,LGS",
    )
    parser.add_argument("--scenarios", default="clean,retrained")
    parser.add_argument("--test-cap", type=int, default=40000)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    split = np.load(args.split_indices)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    train_indices = np.asarray(split["train"])
    test_indices = np.asarray(split["test"])
    rng = np.random.default_rng(args.seed)
    test_indices = cap_with_min(test_indices, labels, args.test_cap, 500, rng)

    base = method33_tools.load_config(args.base_config)
    full = method33_tools.load_config(args.full_config)
    configs = {
        "base": (base.method33.trigger_codes, base.method33.trigger_positions),
        "full": (full.method33.trigger_codes, full.method33.trigger_positions),
    }
    train = take(dataset, train_indices)
    test = take(dataset, test_indices)
    test_days = groups[test_indices]
    train_labels = np.asarray(train[4])
    test_labels = np.asarray(test[4])
    class_count = int(labels.max()) + 1

    started = time.time()
    results = {}
    for scenario in args.scenarios.split(","):
        for name, (codes, positions) in configs.items():
            config = dataclasses.replace(
                base.method33, trigger_codes=codes, trigger_positions=tuple(positions)
            )
            if scenario == "clean":
                train_app = train
            else:
                train_app = defended(train, "old", config, 731800, base.trigger_jitter)
            test_app = defended(test, "next", config, 731800, base.trigger_jitter)
            train_views = build_views(train_app)
            test_views = build_views(test_app)
            for feature_name in args.feature_sets.split(","):
                for surrogate in args.surrogates.split(","):
                    scored = fit_and_score(
                        feature_name,
                        feature_set(feature_name, train_views),
                        train_labels,
                        feature_set(feature_name, test_views),
                        test_labels,
                        surrogate,
                    )
                    low, high = bootstrap_ba(
                        scored["y_true"], scored["prediction"], test_days,
                        np.random.default_rng(args.seed + len(feature_name) + len(name) + len(scenario)),
                        args.bootstrap,
                    )
                    results.setdefault(scenario, {}).setdefault(feature_name, {}).setdefault(surrogate, {})[name] = {
                        "ba": scored["ba"],
                        "ci_low": low,
                        "ci_high": high,
                        "recall": scored["recall"],
                    }
                    print(
                        f"[{scenario} {feature_name} {surrogate}] {name}={scored['ba']:.4f} "
                        f"[{low:.4f},{high:.4f}]",
                        flush=True,
                    )

    log = {
        "dataset": str(args.dataset),
        "base_config": str(args.base_config),
        "full_config": str(args.full_config),
        "test_flows": len(test_indices),
        "test_days": int(len(np.unique(test_days))),
        "results": results,
        "seconds": round(time.time() - started, 1),
        "note": "feature-group diagnostic; Base vs Full(position); sequence view is a proxy",
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nview        scenario   base    full    delta")
    for scenario in args.scenarios.split(","):
        for view in args.feature_sets.split(","):
            for surrogate in args.surrogates.split(","):
                cell = results[scenario][view][surrogate]
                delta = cell["base"]["ba"] - cell["full"]["ba"]
                print(
                    f"{view:<10} {scenario:<10} {cell['base']['ba']:.4f}  "
                    f"{cell['full']['ba']:.4f}  {delta:+.4f}  ({surrogate})"
                )


if __name__ == "__main__":
    main()
