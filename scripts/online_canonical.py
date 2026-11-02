"""Canonicalized shared model: train on code indices, invert known mapping at test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from scripts.online_adaptation_v2 import build_blocks, make_mappings  # noqa: E402


def fit_model(x, y):
    return SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(x, y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--policies", default="Periodic")
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--period", type=int, default=2)
    parser.add_argument("--phase", type=int, default=0)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project)
    rng = np.random.default_rng(20261030)

    output = {"policies": {}}
    for policy in args.policies.split(","):
        blocks = build_blocks(dataset, labels, groups, project, mappings, policy, args.block_days, args.period, args.phase, rng)
        ba_curve = []
        for b in range(1, len(blocks)):
            hist_x = np.concatenate([blocks[i]["x"] for i in range(b)])
            hist_y = np.concatenate([blocks[i]["y"] for i in range(b)])
            hist_state = np.concatenate([blocks[i]["state"] for i in range(b)])
            if len(hist_x) > 6000:
                keep = rng.choice(len(hist_x), size=6000, replace=False)
                hist_x, hist_y, hist_state = hist_x[keep], hist_y[keep], hist_state[keep]
            # Canonical target: code index under each historical mapping.
            code_target = np.asarray([mappings[int(s)][int(y)] for s, y in zip(hist_state, hist_y)])
            scaler = StandardScaler().fit(hist_x)
            model = fit_model(scaler.transform(hist_x), code_target)
            cur = blocks[b]
            half = len(cur["y"]) // 2
            blind = np.arange(half, len(cur["y"]))
            pred_code = model.predict(scaler.transform(cur["x"][blind]))
            pred_class = []
            for row, code in zip(blind, pred_code):
                state = int(cur["state"][row])
                inv = {int(mappings[state][c]): c for c in range(len(mappings[state]))}
                pred_class.append(inv.get(int(code), 0))
            ba_curve.append(float(balanced_accuracy_score(cur["y"][blind], pred_class)))
        output["policies"][policy] = {
            "curves": ba_curve,
            "J": float(np.mean(ba_curve)) if ba_curve else None,
        }
        print(policy, "J", round(output["policies"][policy]["J"], 3), flush=True)

    args.output_log.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
