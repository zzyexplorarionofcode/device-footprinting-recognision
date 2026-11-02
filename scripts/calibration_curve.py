"""How many current-mapping labels are needed to recover the rotation permutation?

Train on old mapping; split test days into disjoint calibration and evaluation
sets; draw n labeled calibration samples, estimate pred->true permutation with
Hungarian, apply it to evaluation predictions.  Test labels are never used for
the permutation.
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
from scripts.position_placebo import clean_features, features, make_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--arm", choices=("placebo_code", "real"), default="placebo_code")
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--ns", default="0,2,4,8,16,32,64,128,256,512,1024")
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

    # Day-disjoint calibration/evaluation split inside test.
    test_days = np.sort(np.unique(groups[test_idx]))
    rng.shuffle(test_days)
    best = None
    for cut in range(max(1, int(0.2 * len(test_days))), max(2, int(0.5 * len(test_days)))):
        cal_days = set(test_days[:cut])
        cal_idx = test_idx[np.isin(groups[test_idx], list(cal_days))]
        eval_idx = test_idx[~np.isin(groups[test_idx], list(cal_days))]
        if len(cal_idx) < 200 or len(eval_idx) < 500:
            continue
        common = [c for c in np.unique(labels) if np.sum(labels[cal_idx] == c) >= 2 and np.sum(labels[eval_idx] == c) >= 2]
        if len(common) >= 0.8 * len(np.unique(labels)):
            best = (cal_idx, eval_idx, common)
            break
    if best is None:
        raise RuntimeError("could not build calibration/evaluation split")
    cal_idx, eval_idx, classes = best

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
    model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
    pred_cal = model.predict(scaler.transform(x_cal))
    pred_eval = model.predict(scaler.transform(x_eval))
    unaligned = float(balanced_accuracy_score(y_eval, pred_eval))

    results = {"dataset_version": args.dataset_version, "arm": args.arm, "unaligned": unaligned,
               "cal_flows": int(len(cal_idx)), "eval_flows": int(len(eval_idx)), "curves": {}}
    for n in [int(v) for v in args.ns.split(",")]:
        values = []
        for _ in range(args.repeats):
            if n == 0:
                mapping = {}
            else:
                take_n = min(n, len(cal_idx))
                draw = rng.choice(len(cal_idx), size=take_n, replace=False)
                cm = confusion_matrix(y_cal[draw], pred_cal[draw], labels=classes)
                row, col = linear_sum_assignment(-cm)
                mapping = {int(p): int(t) for p, t in zip(col, row)}
            aligned = np.asarray([mapping.get(int(p), int(p)) for p in pred_eval])
            values.append(float(balanced_accuracy_score(y_eval, aligned)))
        results["curves"][str(n)] = {
            "mean": float(np.mean(values)),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        if n in (64, 1024):
            print("debug n", n, "mapping", mapping,
                  "cal_confusion_diag", np.diag(confusion_matrix(y_cal, pred_cal, labels=classes)).tolist(),
                  "eval_confusion_diag", np.diag(confusion_matrix(y_eval, pred_eval, labels=classes)).tolist(),
                  flush=True)
        print(n, round(results["curves"][str(n)]["mean"], 3), flush=True)

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
