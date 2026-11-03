"""Streaming transform arms: one block at a time, trace released immediately."""

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
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_placebo import features  # noqa: E402
from scripts.reconstruct_audit import reconstruct  # noqa: E402


def make_mappings(project, seed=0):
    out = [list(project.method33.old_mapping), list(project.method33.next_mapping)]
    rng = np.random.default_rng(seed)
    while len(out) < 6:
        perm = list(rng.permutation(len(out[0])))
        if perm not in out:
            out.append(perm)
    return out


def detect_rows(x, code, project):
    packets = project.method33.trigger_codes[code]
    lengths = np.abs(x[:32]) * 1514.0
    gaps = np.expm1(x[32:]) / 1000.0
    err0 = np.abs(lengths - packets[0][0]) + np.abs(gaps - packets[0][1]) * 100
    err1 = np.abs(lengths - packets[1][0]) + np.abs(gaps - packets[1][1]) * 100
    best = (np.inf, 0, 1)
    for i in np.argsort(err0)[:4]:
        for j in range(32):
            if i == j:
                continue
            value = err0[i] + err1[j]
            if value < best[0]:
                best = (value, int(i), int(j))
    return [best[1], best[2]]


def detect_code_rows(x, state, mappings, project):
    best = (np.inf, 0, [0, 1])
    for code in range(len(mappings[state])):
        rows = detect_rows(x, code, project)
        value = 0.0
        for j, row in enumerate(rows):
            length, gap = project.method33.trigger_codes[code][j]
            value += abs(abs(x[row]) * 1514.0 - length) + abs(np.expm1(x[32 + row]) / 1000.0 - gap) * 100
        if value < best[0]:
            best = (value, code, rows)
    return best[1], best[2]


def code_only(x, rows, ref_code, project):
    out = x.copy()
    for j, row in enumerate(rows[:2]):
        length, gap = project.method33.trigger_codes[ref_code][j]
        sign = 1.0 if out[row] >= 0 else -1.0
        out[row] = sign * length / 1514.0
        out[32 + row] = np.log1p(min(gap, 10.0) * 1000.0)
    return out


def build_canonical(dataset, labels, groups, project, mappings, block_days, variant, phase=0):
    days = np.sort(np.unique(groups))
    raw_blocks = []
    for start in range(0, len(days), block_days):
        chosen = set(days[start : start + block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            raw_blocks.append(idx)
    blocks = []
    for b, idx in enumerate(raw_blocks):
        state = np.full(len(idx), ((b // 2) + phase) % len(mappings), dtype=int)
        mapping = list(project.method33.old_mapping) if variant == "oracle_regen" else mappings[int(state[0])]
        application = list(take(dataset, idx))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mapping, jitter_config=project.trigger_jitter,
            include_event_trace=True,
        )
        x = features(app)
        canonical = x.copy()
        if variant in ("position_only", "combined", "arm4"):
            for i in range(len(idx)):
                trace = app[6][i]
                oracle_code = int(mappings[int(state[i])][int(labels[idx[i]])])
                if variant == "arm4":
                    code, rows = detect_code_rows(x[i], int(state[i]), mappings, project)
                else:
                    code = oracle_code
                    rows = [e["event_position"] - 1 for e in trace["complete_event_trace"] if e["role"] == "trigger_dummy"]
                cls = int(labels[idx[i]])
                ref_code = int(project.method33.old_mapping[cls])
                canonical[i] = reconstruct(trace, project, ref_code, code, x[i], shortcut=False, actual_rows=rows, replace_code=(variant != "position_only"))
        blocks.append({"x": canonical, "y": labels[idx]})
        del app, application, x
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--variant", choices=("arm4", "code_only", "position_only", "combined", "unaware", "oracle_regen"), required=True)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project, args.seed)
    blocks = build_canonical(dataset, labels, groups, project, mappings, args.block_days, args.variant, args.phase)

    curves = []
    for b in range(1, len(blocks)):
        x_train = np.concatenate([blocks[i]["x"] for i in range(b)])
        y_train = np.concatenate([blocks[i]["y"] for i in range(b)])
        scaler = StandardScaler().fit(x_train)
        model = SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(scaler.transform(x_train), y_train)
        half = len(blocks[b]["y"]) // 2
        blind = np.arange(half, len(blocks[b]["y"]))
        pred = model.predict(scaler.transform(blocks[b]["x"][blind]))
        curves.append(float(balanced_accuracy_score(blocks[b]["y"][blind], pred)))
        print(b, round(curves[-1], 3), flush=True)

    result = {"variant": args.variant, "J": float(np.mean(curves)), "curves": curves}
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.variant, "J", round(result["J"], 3), flush=True)


if __name__ == "__main__":
    main()
