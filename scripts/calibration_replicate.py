"""Calibration replication: permutation, linear probe, fine-tune.

Splits: day-disjoint or random flow split (same days).
Arms: placebo_code (A) or real (D).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_ablation import cap_with_min  # noqa: E402
from scripts.position_placebo import features, make_config  # noqa: E402


def split_cal_eval(test_idx, labels, groups, mode, rng):
    if mode == "days":
        days = np.unique(groups[test_idx])
        rng.shuffle(days)
        for cut in range(max(1, int(0.2 * len(days))), max(2, int(0.5 * len(days)))):
            cal_days = set(days[:cut])
            cal = test_idx[np.isin(groups[test_idx], list(cal_days))]
            ev = test_idx[~np.isin(groups[test_idx], list(cal_days))]
            if len(cal) >= 200 and len(ev) >= 500:
                return cal, ev
        raise RuntimeError("no day split")
    cal_parts, ev_parts = [], []
    for label in np.unique(labels[test_idx]):
        pool = test_idx[labels[test_idx] == label]
        order = rng.permutation(len(pool))
        cut = max(1, int(round(0.3 * len(pool))))
        cal_parts.extend(pool[order[:cut]].tolist())
        ev_parts.extend(pool[order[cut:]].tolist())
    return np.asarray(sorted(cal_parts)), np.asarray(sorted(ev_parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--arm", choices=("placebo_code", "real"), default="placebo_code")
    parser.add_argument("--cal-split", choices=("days", "random"), default="days")
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--ns", default="0,4,16,64,256,1024")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    rng = np.random.default_rng(20261030)
    train_idx = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    test_idx = cap_with_min(np.asarray(split["test"]), labels, args.cap, 1, rng)
    project = method33_tools.load_config(args.config)
    config = make_config(project, args.arm)
    cal_idx, eval_idx = split_cal_eval(test_idx, labels, groups, args.cal_split, rng)

    def apply(indices, mapping):
        app = apply_method33(
            *take(dataset, indices), config, 731800, mapping=mapping,
            include_trigger=True, jitter_config=project.trigger_jitter,
        )
        return features(app)

    x_train = apply(train_idx, "old")
    x_cal = apply(cal_idx, "next")
    x_eval = apply(eval_idx, "next")
    y_train, y_cal, y_eval = labels[train_idx], labels[cal_idx], labels[eval_idx]
    scaler = StandardScaler().fit(x_train)
    z_train, z_cal, z_eval = scaler.transform(x_train), scaler.transform(x_cal), scaler.transform(x_eval)
    base = LogisticRegression(max_iter=500).fit(z_train, y_train)
    pred_cal = base.predict(z_cal)
    pred_eval = base.predict(z_eval)
    classes = sorted(set(y_eval.tolist()))
    unaligned = float(balanced_accuracy_score(y_eval, pred_eval))

    results = {"dataset_version": args.dataset_version, "arm": args.arm, "cal_split": args.cal_split,
               "unaligned": unaligned, "cal_flows": int(len(cal_idx)), "eval_flows": int(len(eval_idx)),
               "methods": {}}
    for method in ("permutation", "probe", "finetune"):
        curves = {}
        for n in [int(v) for v in args.ns.split(",")]:
            values = []
            for _ in range(args.repeats):
                if n == 0:
                    draw = np.asarray([], dtype=int)
                else:
                    draw = rng.choice(len(cal_idx), size=min(n, len(cal_idx)), replace=False)
                if method == "permutation":
                    if n == 0:
                        mapping = {}
                    else:
                        cm = confusion_matrix(y_cal[draw], pred_cal[draw], labels=classes)
                        row, col = linear_sum_assignment(-cm)
                        mapping = {int(p): int(t) for p, t in zip(col, row)}
                    aligned = np.asarray([mapping.get(int(p), int(p)) for p in pred_eval])
                    values.append(float(balanced_accuracy_score(y_eval, aligned)))
                elif method == "probe":
                    if n == 0 or len(np.unique(y_cal[draw])) < 2:
                        values.append(unaligned)
                    else:
                        model = LogisticRegression(max_iter=500).fit(z_cal[draw], y_cal[draw])
                        values.append(float(balanced_accuracy_score(y_eval, model.predict(z_eval))))
                else:
                    if n == 0 or len(np.unique(y_cal[draw])) < 2:
                        values.append(unaligned)
                    else:
                        weight_cal = len(train_idx) / max(1, len(draw))
                        weights = np.concatenate([np.ones(len(train_idx)), np.full(len(draw), weight_cal)])
                        model = LogisticRegression(max_iter=500).fit(
                            np.concatenate([z_train, z_cal[draw]]),
                            np.concatenate([y_train, y_cal[draw]]),
                            sample_weight=weights,
                        )
                        values.append(float(balanced_accuracy_score(y_eval, model.predict(z_eval))))
            curves[str(n)] = {
                "mean": float(np.mean(values)),
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
            }
            print(method, n, round(curves[str(n)]["mean"], 3), flush=True)
        results["methods"][method] = curves

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
