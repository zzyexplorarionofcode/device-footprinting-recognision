"""Observed-feature inverse: code-content canonicalization (positions unchanged)."""

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


def fit_model(x, y):
    return SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(x, y)


def code_index(lengths, state, mappings, project):
    best, best_err = 0, np.inf
    for code in range(len(mappings[state])):
        expected = np.asarray([p[0] for p in project.method33.trigger_codes[code]])
        err = float(np.sum(np.abs(np.asarray(lengths) - expected)))
        if err < best_err:
            best, best_err = code, err
    return best


def canonicalize(x, state, mappings, project):
    """Return a copy with trigger code lengths/gaps replaced by reference code."""
    out = x.copy()
    reference = list(project.method33.old_mapping)
    inverse = {code: cls for cls, code in enumerate(mappings[state])}
    # Trigger rows are identified by the two largest code-like length candidates;
    # caller supplies their positions via the second return of build_features.
    return out


def build(dataset, labels, groups, project, mappings, block_days, period, phase, policy="Periodic"):
    days = np.sort(np.unique(groups))
    raw = []
    for start in range(0, len(days), block_days):
        chosen = set(days[start : start + block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            raw.append(idx)
    blocks = []
    for b, idx in enumerate(raw):
        if policy == "Fixed":
            state = np.zeros(len(idx), dtype=int)
        else:
            state = np.full(len(idx), ((b // period) + phase) % len(mappings), dtype=int)
        application = list(take(dataset, idx))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mappings[int(state[0])], jitter_config=project.trigger_jitter,
            include_event_trace=True,
        )
        x = features(app)
        rows = []
        codes = []
        anchors = []
        for i in range(len(idx)):
            trace = app[6][i]["complete_event_trace"]
            trig = [e for e in trace if e["role"] == "trigger_dummy" and e["in_observation_window"]]
            rows.append([e["event_position"] for e in trig][:2])
            lengths = [x[i, e["event_position"]] * 1514.0 for e in trig][:2]
            codes.append(code_index(np.abs(lengths), int(state[i]), mappings, project))
            anchors.append({int(e["source_index"]): int(e["event_position"]) for e in trace if e["role"] == "real"})
        blocks.append({"x": x, "y": labels[idx], "state": state, "rows": rows, "codes": codes, "anchors": anchors})
    return blocks


def inverse_code(x, rows, code, state, anchors, mappings, project, mode="both"):
    """Replace trigger code rows and optionally move them to reference anchors."""
    if mode == "identity":
        return x.copy()
    out = x.copy()
    reference = list(project.method33.old_mapping)
    inverse = {code: cls for cls, code in enumerate(mappings[state])}
    cls = inverse[code]
    ref_code = reference[cls]
    ref_packets = project.method33.trigger_codes[ref_code]
    ref_anchors = project.method33.trigger_positions[ref_code]
    for j, slot in enumerate(rows):
        if slot < 0 or j >= len(ref_packets):
            break
        target_slot = slot
        if mode == "both":
            source_index = int(ref_anchors[j]) - 1
            if source_index in anchors:
                target_slot = anchors[source_index] + 1
        if not (0 <= target_slot < 32):
            target_slot = slot
        if target_slot != slot:
            out[[slot, target_slot]] = out[[target_slot, slot]]
        if ref_code == code and target_slot == slot:
            continue
        length, gap = ref_packets[j]
        sign = 1.0 if out[target_slot] >= 0 else -1.0
        out[target_slot] = sign * length / 1514.0
        out[32 + target_slot] = np.log1p(min(gap, 10.0) * 1000.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--period", type=int, default=2)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--mode", choices=("code", "both", "identity", "oracle"), default="both")
    parser.add_argument("--policy", choices=("Fixed", "Periodic"), default="Periodic")
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
    blocks = build(dataset, labels, groups, project, mappings, args.block_days, args.period, args.phase, args.policy)

    curves = []
    for b in range(1, len(blocks)):
        hist_x = np.concatenate([blocks[i]["x"] for i in range(b)])
        hist_y = np.concatenate([blocks[i]["y"] for i in range(b)])
        hist_state = np.concatenate([blocks[i]["state"] for i in range(b)])
        hist_rows = sum([blocks[i]["rows"] for i in range(b)], [])
        hist_codes = sum([blocks[i]["codes"] for i in range(b)], [])
        hist_anchors = sum([blocks[i]["anchors"] for i in range(b)], [])
        if args.mode == "oracle":
            hist_codes = [int(mappings[int(s)][int(y)]) for s, y in zip(hist_state, hist_y)]
        canon = np.stack([
            inverse_code(hist_x[j], hist_rows[j], hist_codes[j], int(hist_state[j]), hist_anchors[j], mappings, project, args.mode)
            for j in range(len(hist_x))
        ])
        cur = blocks[b]
        cur_codes = cur["codes"]
        if args.mode == "oracle":
            cur_codes = [int(mappings[int(s)][int(y)]) for s, y in zip(cur["state"], cur["y"])]
        canonical_cur = np.stack([
            inverse_code(cur["x"][j], cur["rows"][j], cur_codes[j], int(cur["state"][j]), cur["anchors"][j], mappings, project, args.mode)
            for j in range(len(cur["x"]))
        ])
        half = len(cur["y"]) // 2
        blind = np.arange(half, len(cur["y"]))
        scaler = StandardScaler().fit(canon)
        model = fit_model(scaler.transform(canon), hist_y)
        pred = model.predict(scaler.transform(canonical_cur[blind]))
        curves.append(float(balanced_accuracy_score(cur["y"][blind], pred)))

    result = {"J": float(np.mean(curves)), "curves": curves}
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("J", round(result["J"], 3), flush=True)


if __name__ == "__main__":
    main()
