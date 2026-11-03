"""Arm 3: oracle code + detected trigger positions."""

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
from scripts.observed_inverse import build  # noqa: E402
from scripts.reconstruct_audit import reconstruct  # noqa: E402


def detect_rows(x, code, project):
    packets = project.method33.trigger_codes[code]
    lengths = np.abs(x[:32]) * 1514.0
    gaps = np.expm1(x[32:]) / 1000.0
    err0 = np.abs(lengths - packets[0][0]) + np.abs(gaps - packets[0][1]) * 100
    err1 = np.abs(lengths - packets[1][0]) + np.abs(gaps - packets[1][1]) * 100
    candidates = np.argsort(err0)[:4]
    best = (np.inf, 0, 1)
    for i in candidates:
        for j in range(32):
            if i == j:
                continue
            value = err0[i] + err1[j]
            if value < best[0]:
                best = (value, int(i), int(j))
    return [best[1], best[2]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=10)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = [list(project.method33.old_mapping), list(project.method33.next_mapping)]
    rng = np.random.default_rng(123)
    while len(mappings) < 6:
        perm = list(rng.permutation(len(mappings[0])))
        if perm not in mappings:
            mappings.append(perm)
    blocks = build(dataset, labels, groups, project, mappings, args.block_days, 2, 0, "Periodic")

    curves = []
    for b in range(1, len(blocks)):
        hist = [blocks[i] for i in range(b)]
        hist_x = np.concatenate([h["x"] for h in hist])
        hist_y = np.concatenate([h["y"] for h in hist])
        hist_state = np.concatenate([h["state"] for h in hist])
        hist_trace = sum([h["trace"] for h in hist], [])
        hist_codes = [int(mappings[int(s)][int(y)]) for s, y in zip(hist_state, hist_y)]
        canon = np.stack([
            reconstruct(hist_trace[j], project, int(project.method33.old_mapping[int(hist_y[j])]), int(hist_codes[j]), hist_x[j], shortcut=False, actual_rows=detect_rows(hist_x[j], int(hist_codes[j]), project))
            for j in range(len(hist_x))
        ])
        cur = blocks[b]
        cur_codes = [int(mappings[int(s)][int(y)]) for s, y in zip(cur["state"], cur["y"])]
        canon_cur = np.stack([
            reconstruct(cur["trace"][j], project, int(project.method33.old_mapping[int(cur["y"][j])]), int(cur_codes[j]), cur["x"][j], shortcut=False, actual_rows=detect_rows(cur["x"][j], int(cur_codes[j]), project))
            for j in range(len(cur["x"]))
        ])
        half = len(cur["y"]) // 2
        blind = np.arange(half, len(cur["y"]))
        scaler = StandardScaler().fit(canon)
        model = SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(scaler.transform(canon), hist_y)
        pred = model.predict(scaler.transform(canon_cur[blind]))
        curves.append(float(balanced_accuracy_score(cur["y"][blind], pred)))
        print(b, round(curves[-1], 3), flush=True)

    result = {"J": float(np.mean(curves)), "curves": curves}
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("J", result["J"], flush=True)


if __name__ == "__main__":
    main()
