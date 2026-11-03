"""Trace-free adaptive matrix: state inference, canonicalization, probe/finetune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_placebo import features  # noqa: E402
from scripts.streaming_arms import detect_code_rows, make_mappings, make_model  # noqa: E402


def detection_error(x, code, rows, project):
    value = 0.0
    for j, row in enumerate(rows):
        length, gap = project.method33.trigger_codes[code][j]
        value += abs(abs(x[row]) * 1514.0 - length)
        value += abs(np.expm1(x[32 + row]) / 1000.0 - gap) * 100
    return value


def infer_state(x, mappings, project, sample=100, rng=None):
    rng = rng or np.random.default_rng(0)
    rows = rng.choice(len(x), size=min(sample, len(x)), replace=False)
    best_state, best_score = 0, np.inf
    for state in range(len(mappings)):
        score = 0.0
        for row in rows:
            code, trig_rows = detect_code_rows(x[row], state, mappings, project)
            score += detection_error(x[row], code, trig_rows, project)
        if score < best_score:
            best_state, best_score = state, score
    return best_state


def canonicalize_tracefree(x, state, code, rows, ref_code, project):
    out = x.copy()
    targets = [int(p) - 1 for p in project.method33.trigger_positions[ref_code]]
    for j, row in enumerate(rows[:2]):
        target = targets[j]
        if not (0 <= target < 32):
            target = row
        if target != row:
            out[[row, target]] = out[[target, row]]
        length, gap = project.method33.trigger_codes[ref_code][j]
        sign = 1.0 if out[target] >= 0 else -1.0
        out[target] = sign * length / 1514.0
        out[32 + target] = np.log1p(min(gap, 10.0) * 1000.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--policy", choices=("Fixed", "Periodic", "Per_flow"), default="Periodic")
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ns", default="0,64,256,1024")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project, args.seed)
    ns = [int(v) for v in args.ns.split(",")]
    days = np.sort(np.unique(groups))
    raw_blocks = []
    for start in range(0, len(days), args.block_days):
        chosen = set(days[start : start + args.block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            raw_blocks.append(idx)

    blocks = []
    rng = np.random.default_rng(args.seed)
    for b, idx in enumerate(raw_blocks):
        if args.policy == "Fixed":
            state = np.zeros(len(idx), dtype=int)
        elif args.policy == "Per_flow":
            state = rng.integers(0, len(mappings), size=len(idx))
        else:
            state = np.full(len(idx), ((b // 2) + args.phase) % len(mappings), dtype=int)
        x = np.zeros((len(idx), 64))
        for state_id in np.unique(state):
            rows = np.where(state == state_id)[0]
            subset = idx[rows]
            application = list(take(dataset, subset))
            application[0] = np.minimum(application[0], 1514.0)
            app = apply_method33(
                *application, project.method33, 731800,
                mapping=mappings[int(state_id)], jitter_config=project.trigger_jitter,
            )
            x[rows] = features(app)
            del app, application
        inferred = infer_state(x, mappings, project, sample=100, rng=rng) if len(np.unique(state)) == 1 else 0
        canonical = np.zeros_like(x)
        for i in range(len(idx)):
            flow_state = inferred if len(np.unique(state)) == 1 else infer_state(x[i:i + 1], mappings, project, sample=1, rng=rng)
            code, trig_rows = detect_code_rows(x[i], int(flow_state), mappings, project)
            cls = int(labels[idx[i]])
            ref_code = int(project.method33.old_mapping[cls])
            canonical[i] = canonicalize_tracefree(x[i], int(flow_state), code, trig_rows, ref_code, project)
        blocks.append({"x": canonical, "y": labels[idx]})
        print("block", b, "inferred", inferred, flush=True)

    results = {}
    for attacker in ("probe", "finetune"):
        curves = {n: [] for n in ns}
        for b in range(1, len(blocks)):
            hist_x = np.concatenate([blocks[i]["x"] for i in range(b)])
            hist_y = np.concatenate([blocks[i]["y"] for i in range(b)])
            scaler = StandardScaler().fit(hist_x)
            z_hist = scaler.transform(hist_x)
            base = make_model("lr").fit(z_hist, hist_y)
            cur = blocks[b]
            z_cur = scaler.transform(cur["x"])
            half = len(cur["y"]) // 2
            pool = np.arange(half)
            blind = np.arange(half, len(cur["y"]))
            for n in ns:
                draw = np.asarray([], dtype=int) if n == 0 else rng.choice(pool, size=min(n, len(pool)), replace=False)
                if attacker == "probe" and len(draw) >= 8 and len(np.unique(cur["y"][draw])) >= 2:
                    model = make_model("lr").fit(z_cur[draw], cur["y"][draw])
                    pred = model.predict(z_cur[blind])
                elif attacker == "finetune" and len(draw) >= 8 and len(np.unique(cur["y"][draw])) >= 2:
                    weight = len(hist_x) / len(draw)
                    model = make_model("lr").fit(
                        np.concatenate([z_hist, z_cur[draw]]),
                        np.concatenate([hist_y, cur["y"][draw]]),
                        sample_weight=np.concatenate([np.ones(len(z_hist)), np.full(len(draw), weight)]),
                    )
                    pred = model.predict(z_cur[blind])
                else:
                    pred = base.predict(z_cur[blind])
                curves[n].append(float(balanced_accuracy_score(cur["y"][blind], pred)))
        results[attacker] = {
            "curves": {str(n): curves[n] for n in ns},
            "J": {str(n): float(np.mean(curves[n])) if curves[n] else None for n in ns},
        }
        print(attacker, results[attacker]["J"], flush=True)

    args.output_log.write_text(json.dumps({"policy": args.policy, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
