"""Rotation time-series pilot: Fixed vs Periodic vs Per-flow mappings.

Blocks are consecutive day groups.  The attacker trains on all past defended
blocks (cumulative, from scratch) and is tested on the current block.
Pilot uses LR only; sliding-window and warm-start are left for the full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_placebo import features  # noqa: E402


def make_mappings(project, count=8):
    base = list(project.method33.old_mapping)
    nxt = list(project.method33.next_mapping)
    mappings = [base, nxt]
    rng = np.random.default_rng(123)
    while len(mappings) < count:
        perm = list(rng.permutation(len(base)))
        if perm != base and perm not in mappings:
            mappings.append(perm)
    return mappings


def block_features(dataset, indices, project, mapping_for_flow, mappings):
    groups = {}
    for row, index in enumerate(indices):
        mapping_id = int(mapping_for_flow[row])
        groups.setdefault(mapping_id, []).append(row)
    xs, ys = [], []
    for mapping_id, rows in groups.items():
        subset = indices[np.asarray(rows)]
        application = list(take(dataset, subset))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mappings[mapping_id], jitter_config=project.trigger_jitter,
        )
        xs.append(features(app))
        ys.append(np.asarray(dataset["labels"])[subset])
    return np.concatenate(xs), np.concatenate(ys)


def ttau(ba, threshold, consecutive=2):
    run = 0
    for index, value in enumerate(ba):
        if value >= threshold:
            run += 1
            if run >= consecutive:
                return index - consecutive + 1
        else:
            run = 0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--block-days", type=int, default=5)
    parser.add_argument("--period", type=int, default=2, help="blocks per mapping for Periodic")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project)
    days = np.sort(np.unique(groups))
    blocks = []
    for start in range(0, len(days), args.block_days):
        chosen = set(days[start : start + args.block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 50:
            blocks.append(idx)
    rng = np.random.default_rng(20261030)

    policies = ("Fixed", "Periodic", "Per_flow")
    result = {"dataset_version": args.dataset_version, "block_days": args.block_days, "blocks": len(blocks), "policies": {}}
    for policy in policies:
        mapping_per_block = []
        for b in range(len(blocks)):
            if policy == "Fixed":
                mapping_per_block.append(np.zeros(len(blocks[b]), dtype=int))
            elif policy == "Periodic":
                mapping_per_block.append(np.full(len(blocks[b]), (b // args.period) % len(mappings), dtype=int))
            else:
                mapping_per_block.append(rng.integers(0, len(mappings), size=len(blocks[b])))
        block_x, block_y = [], []
        for b, idx in enumerate(blocks):
            x, y = block_features(dataset, idx, project, mapping_per_block[b], mappings)
            # Reorder back to the original block order for stable bookkeeping.
            block_x.append(x)
            block_y.append(y)
        ba_curve = []
        for b in range(1, len(blocks)):
            x_train = np.concatenate(block_x[:b])
            y_train = np.concatenate(block_y[:b])
            x_test, y_test = block_x[b], block_y[b]
            scaler = StandardScaler().fit(x_train)
            model = LogisticRegression(max_iter=500).fit(scaler.transform(x_train), y_train)
            pred = model.predict(scaler.transform(x_test))
            ba_curve.append(float(balanced_accuracy_score(y_test, pred)))
        result["policies"][policy] = {
            "ba_curve": ba_curve,
            "J": float(np.mean(ba_curve)) if ba_curve else None,
            "T_0.5": ttau(ba_curve, 0.5),
            "T_0.8": ttau(ba_curve, 0.8),
            "censored_0.5": ttau(ba_curve, 0.5) is None,
            "censored_0.8": ttau(ba_curve, 0.8) is None,
        }
        print(policy, "J", round(result["policies"][policy]["J"], 3),
              "T0.5", result["policies"][policy]["T_0.5"],
              "T0.8", result["policies"][policy]["T_0.8"], flush=True)

    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
