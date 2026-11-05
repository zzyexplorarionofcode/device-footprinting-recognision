"""Transform ablations: code-only, position-only, combined (oracle code/rows)."""

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


def code_only(x, rows, ref_code, project):
    out = x.copy()
    packets = project.method33.trigger_codes[ref_code]
    for j, row in enumerate(rows[:2]):
        length, gap = packets[j]
        sign = 1.0 if out[row] >= 0 else -1.0
        out[row] = sign * length / 1514.0
        out[32 + row] = np.log1p(min(gap, 10.0) * 1000.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--variant", choices=("code", "position", "combined"), required=True)
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

    def transform(x, trace, code, state, y):
        cls = int(y)
        ref_code = int(project.method33.old_mapping[cls])
        rows = [e["event_position"] - 1 for e in trace["complete_event_trace"] if e["role"] == "trigger_dummy"]
        if args.variant == "code":
            return code_only(x, rows, ref_code, project)
        return reconstruct(trace, project, ref_code, int(code), x, shortcut=False, actual_rows=rows, replace_code=(args.variant == "combined"))

    curves = []
    for b in range(1, len(blocks)):
        hist = [blocks[i] for i in range(b)]
        hist_x = np.concatenate([h["x"] for h in hist])
        hist_y = np.concatenate([h["y"] for h in hist])
        hist_state = np.concatenate([h["state"] for h in hist])
        hist_trace = sum([h["trace"] for h in hist], [])
        hist_codes = [int(mappings[int(s)][int(y)]) for s, y in zip(hist_state, hist_y)]
        canon = np.stack([
            transform(hist_x[j], hist_trace[j], hist_codes[j], int(hist_state[j]), hist_y[j])
            for j in range(len(hist_x))
        ])
        cur = blocks[b]
        cur_codes = [int(mappings[int(s)][int(y)]) for s, y in zip(cur["state"], cur["y"])]
        canon_cur = np.stack([
            transform(cur["x"][j], cur["trace"][j], cur_codes[j], int(cur["state"][j]), cur["y"][j])
            for j in range(len(cur["x"]))
        ])
        half = len(cur["y"]) // 2
        blind = np.arange(half, len(cur["y"]))
        scaler = StandardScaler().fit(canon)
        model = SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(scaler.transform(canon), hist_y)
        pred = model.predict(scaler.transform(canon_cur[blind]))
        curves.append(float(balanced_accuracy_score(cur["y"][blind], pred)))

    result = {"variant": args.variant, "J": float(np.mean(curves)), "curves": curves}
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.variant, "J", round(result["J"], 3), flush=True)


if __name__ == "__main__":
    main()
