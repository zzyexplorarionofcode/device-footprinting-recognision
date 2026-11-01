"""Frozen evaluation on the capture-disjoint held-out set (A) and the late subset (B).

A: all 43 pure-test days (8 classes) -> capture-disjoint held-out evaluation.
   This split never entered the position search, LOCO, 2x2 experiment or model
   selection, but it was already reported as the older holdout.  It is therefore
   NOT an untouched or independent future test.

B: the 11 pure-test days inside the last 20% date block (classes 0,2,3,4,5 only)
   -> date-isolated five-class supplementary analysis.  B is a subset of A and
   must not be counted as a second independent piece of evidence.

Two attacker scenarios, both trained on pool data only:

  clean      train on clean pool traffic, test on defended (next) test traffic
  retrained  train on defended (old) pool traffic, test on defended (next) traffic

Base and Full differ only in trigger positions, so the communication cost is
identical; it is still reported explicitly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from scripts.es_codebook_search import defended, features  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_ablation import cap_with_min, predict  # noqa: E402

B_CLASSES = (0, 2, 3, 4, 5)


def macro_ba(y_true, prediction, classes) -> float:
    """Macro balanced accuracy over a *fixed* class list.

    Classes absent from the resample still count as recall 0, so B is exactly
    the mean of the five class recalls (not sklearn's present-class average).
    """

    recalls = recall_score(
        y_true, prediction, labels=np.asarray(classes), average=None, zero_division=0
    )
    return float(np.mean(recalls))


def evaluate(dataset, project, train_indices, test_indices, codes, positions, scenario, surrogate, class_count, classes):
    train = take(dataset, train_indices)
    test = take(dataset, test_indices)
    config = dataclasses.replace(
        project.method33, trigger_codes=codes, trigger_positions=tuple(positions)
    )
    if scenario == "clean":
        x_train = features(train)
    else:
        x_train = features(defended(train, "old", config, 731800, project.trigger_jitter))
    x_test = features(defended(test, "next", config, 731800, project.trigger_jitter))
    prediction = predict(x_train, train[4], x_test, surrogate)
    truth = test[4]
    return {
        "y_true": truth,
        "prediction": prediction,
        "days": np.asarray(dataset["groups"])[test_indices].astype(str),
        "ba": macro_ba(truth, prediction, classes),
        "recall": recall_score(truth, prediction, labels=np.arange(class_count), average=None, zero_division=0),
    }


def bootstrap_delta(base_result, full_result, class_count, rng, repeats, classes=None):
    deltas = []
    per_class = []
    days = np.unique(base_result["days"])
    for _ in range(repeats):
        draw = rng.choice(days, size=len(days), replace=True)
        indices = np.concatenate([np.where(base_result["days"] == day)[0] for day in draw])
        truth = base_result["y_true"][indices]
        if classes is None:
            classes = tuple(range(class_count))
        ba_base = macro_ba(truth, base_result["prediction"][indices], classes)
        ba_full = macro_ba(truth, full_result["prediction"][indices], classes)
        deltas.append(ba_base - ba_full)
        per_class.append(
            recall_score(truth, base_result["prediction"][indices], labels=np.arange(class_count), average=None, zero_division=0)
            - recall_score(truth, full_result["prediction"][indices], labels=np.arange(class_count), average=None, zero_division=0)
        )
    deltas = np.asarray(deltas)
    per_class = np.asarray(per_class)
    return {
        "delta_mean": float(deltas.mean()),
        "delta_ci_low": float(np.percentile(deltas, 2.5)),
        "delta_ci_high": float(np.percentile(deltas, 97.5)),
        "replicates": int(len(deltas)),
        "per_class_delta": per_class.mean(axis=0).tolist(),
        "per_class_ci_low": np.percentile(per_class, 2.5, axis=0).tolist(),
        "per_class_ci_high": np.percentile(per_class, 97.5, axis=0).tolist(),
    }


def cost_table(dataset, test_indices, class_count):
    lengths = np.asarray(dataset["lengths"])[test_indices].astype(float)
    mask = np.asarray(dataset["mask"])[test_indices].astype(bool)
    events = mask.sum(axis=1)
    window_bytes = (lengths * mask).sum(axis=1)
    added_bytes = 630.0
    added_events = 5
    return {
        "added_bytes_per_flow": added_bytes,
        "added_events_per_flow": added_events,
        "window_bytes_mean": float(window_bytes.mean()),
        "window_bytes_median": float(np.median(window_bytes)),
        "window_bytes_p95": float(np.percentile(window_bytes, 95)),
        "window_overhead_mean": float(added_bytes / window_bytes.mean()),
        "window_overhead_median": float(added_bytes / np.median(window_bytes)),
        "window_overhead_p95": float(added_bytes / np.percentile(window_bytes, 95)),
        "dummy_event_fraction": float(added_events / (events.mean() + added_events)),
        "identical_for_base_and_full": True,
        "note": "window bytes are truncated at 32 events; full-flow overhead was measured separately",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=15000)
    parser.add_argument("--train-only", action="store_true", help="train on split['train'] instead of train+validation")
    parser.add_argument("--test-cap", type=int, default=40000)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    pool = np.asarray(split["train"])
    if "validation" in split and len(split["validation"]) > 0:
        pool = np.concatenate([pool, np.asarray(split["validation"])])
    test_a = np.asarray(split["test"])
    days = np.sort(np.unique(groups))
    last_days = set(days[-int(np.ceil(0.2 * len(days))):].tolist())
    test_b_mask = np.isin(groups[test_a], list(last_days)) & np.isin(labels[test_a], B_CLASSES)
    test_b = test_a[test_b_mask]

    class_count = int(labels.max()) + 1
    rng = np.random.default_rng(args.seed)
    train_source = np.asarray(split["train"]) if args.train_only else pool
    train_indices = cap_with_min(train_source, labels, args.train_cap, 1, rng)
    test_a_indices = cap_with_min(test_a, labels, args.test_cap, 500, rng)
    # B is small enough to use in full; it is a subset of A.  (cap_with_min has
    # no "unlimited" mode, so it must not be called here.)
    test_b = test_b

    print(
        f"train={len(train_indices)} A_days={len(np.unique(groups[test_a_indices]))} "
        f"A_flows={len(test_a_indices)} B_days={len(np.unique(groups[test_b]))} B_flows={len(test_b)}",
        flush=True,
    )
    print(f"A per_class={np.bincount(labels[test_a_indices], minlength=class_count).tolist()}", flush=True)
    print(f"B per_class={np.bincount(labels[test_b], minlength=class_count).tolist()}", flush=True)

    base = method33_tools.load_config(args.base_config)
    full = method33_tools.load_config(args.full_config)
    base_codes = base.method33.trigger_codes
    full_codes = full.method33.trigger_codes
    base_positions = tuple(tuple(int(v) for v in pair) for pair in base.method33.trigger_positions)
    full_positions = tuple(tuple(int(v) for v in pair) for pair in full.method33.trigger_positions)

    started = time.time()
    results = {}
    for split_name, test_indices, classes in (
        ("A", test_a_indices, None),
        ("B", test_b, B_CLASSES),
    ):
        eval_classes = tuple(range(class_count)) if classes is None else classes
        results[split_name] = {}
        for scenario in ("clean", "retrained"):
            for surrogate in ("lr", "rf", "mlp"):
                for name, codes_for_config, positions in (
                    ("base", base_codes, base_positions),
                    ("full", full_codes, full_positions),
                ):
                    results[split_name].setdefault(scenario, {}).setdefault(surrogate, {})[name] = evaluate(
                        dataset,
                        base,
                        train_indices,
                        test_indices,
                        codes_for_config,
                        positions,
                        scenario,
                        surrogate,
                        class_count,
                        eval_classes,
                    )
                base_result = results[split_name][scenario][surrogate]["base"]
                full_result = results[split_name][scenario][surrogate]["full"]
                rng_boot = np.random.default_rng(
                    args.seed + 97 * len(split_name) + 13 * len(scenario) + sum(ord(c) for c in surrogate)
                )
                results[split_name][scenario][surrogate]["delta"] = bootstrap_delta(
                    base_result, full_result, class_count, rng_boot, args.bootstrap, classes
                )
                print(
                    f"[{split_name} {scenario} {surrogate}] base={base_result['ba']:.4f} "
                    f"full={full_result['ba']:.4f} delta={results[split_name][scenario][surrogate]['delta']['delta_mean']:+.4f}",
                    flush=True,
                )

    log = {
        "dataset": str(args.dataset),
        "base_config": str(args.base_config),
        "full_config": str(args.full_config),
        "train_flows": len(train_indices),
        "A_flows": len(test_a_indices),
        "A_days": [str(day) for day in sorted(np.unique(groups[test_a_indices]).tolist())],
        "B_flows": len(test_b),
        "B_days": [str(day) for day in sorted(np.unique(groups[test_b]).tolist())],
        "B_classes": list(B_CLASSES),
        "results": {
            split_name: {
                scenario: {
                    surrogate: {
                        name: {
                            "ba": result["ba"],
                            "recall": np.asarray(result["recall"]).tolist(),
                        }
                        for name, result in values.items()
                        if name in ("base", "full")
                    }
                    | {"delta": values["delta"]}
                    for surrogate, values in scenario_values.items()
                }
                for scenario, scenario_values in split_values.items()
            }
            for split_name, split_values in results.items()
        },
        "cost": {
            "A": cost_table(dataset, test_a_indices, class_count),
            "B": cost_table(dataset, test_b, class_count),
        },
        "seconds": round(time.time() - started, 1),
        "labels": {
            "A": "capture-disjoint held-out evaluation (8 classes); previously reported as old holdout, not an untouched test",
            "B": "date-isolated five-class supplement (classes 0,2,3,4,5); subset of A, not independent evidence",
        },
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", args.output_log)


if __name__ == "__main__":
    main()
