"""Position-only classifier: unaligned vs validation-learned permutation, K=1..8."""

from __future__ import annotations

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
from scripts.position_ablation import cap_with_min  # noqa: E402
from scripts.position_stability import build_pools, sample_positions  # noqa: E402


def one_hot(pairs, index):
    x = np.zeros((len(pairs), len(index)))
    for row, pair in enumerate(pairs):
        x[row, index[tuple(pair)]] = 1.0
    return x


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--ks", default="1,2,4,8")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    split = np.load(args.split_indices)
    rng = np.random.default_rng(20261030)
    train_idx = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    test_idx = cap_with_min(np.asarray(split["test"]), labels, args.cap, 1, rng)
    fit_parts, val_parts = [], []
    for label in np.unique(labels[train_idx]):
        pool = train_idx[labels[train_idx] == label]
        order = rng.permutation(len(pool))
        cut = max(1, int(round(0.2 * len(pool))))
        val_parts.extend(pool[order[:cut]].tolist())
        fit_parts.extend(pool[order[cut:]].tolist())
    fit_idx = np.asarray(sorted(fit_parts))
    val_idx = np.asarray(sorted(val_parts))
    project = method33_tools.load_config(args.config)
    pools = build_pools(project, max(int(v) for v in args.ks.split(",")), np.random.default_rng(7))
    all_pairs = list(dict.fromkeys([pair for pool in pools for pair in pool]))
    index = {pair: i for i, pair in enumerate(all_pairs)}
    classes = sorted(set(labels[test_idx].tolist()))

    result = {"dataset_version": args.dataset_version, "ks": {}}
    for k in [int(v) for v in args.ks.split(",")]:
        fit_pos = sample_positions(fit_idx, labels, pools, project.method33.old_mapping, k, np.random.default_rng(31))
        val_pos = sample_positions(val_idx, labels, pools, project.method33.next_mapping, k, np.random.default_rng(32))
        test_pos = sample_positions(test_idx, labels, pools, project.method33.next_mapping, k, np.random.default_rng(33))
        x_fit, x_val, x_test = one_hot(fit_pos, index), one_hot(val_pos, index), one_hot(test_pos, index)
        scaler = StandardScaler().fit(x_fit)
        model = LogisticRegression(max_iter=500).fit(scaler.transform(x_fit), labels[fit_idx])
        pred_val = model.predict(scaler.transform(x_val))
        pred_test = model.predict(scaler.transform(x_test))
        cm = confusion_matrix(labels[val_idx], pred_val, labels=classes)
        row, col = linear_sum_assignment(-cm)
        mapping = {int(p): int(t) for p, t in zip(col, row)}
        aligned = np.asarray([mapping.get(int(p), int(p)) for p in pred_test])
        result["ks"][str(k)] = {
            "unaligned_ba": float(balanced_accuracy_score(labels[test_idx], pred_test)),
            "aligned_ba": float(balanced_accuracy_score(labels[test_idx], aligned)),
            "permutation": {str(kk): vv for kk, vv in mapping.items()},
        }
        print(k, "unaligned", round(result["ks"][str(k)]["unaligned_ba"], 3),
              "aligned", round(result["ks"][str(k)]["aligned_ba"], 3), flush=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
