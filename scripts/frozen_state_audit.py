"""Frozen historical classifier: can it distinguish correct candidate state?"""

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
from scripts.streaming_arms import detect_code_rows, make_mappings  # noqa: E402
from scripts.adaptive_matrix import canonicalize_tracefree  # noqa: E402


def stratified_nested(y, max_n, rng):
    chosen = []
    classes = np.unique(y)
    per = max(1, max_n // len(classes))
    for c in classes:
        idx = np.where(y == c)[0]
        chosen.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
    if len(chosen) < max_n:
        left = np.setdiff1d(np.arange(len(y)), chosen)
        chosen.extend(rng.choice(left, size=min(max_n - len(chosen), len(left)), replace=False).tolist())
    return np.asarray(chosen)


def canonical(x, state, mappings, project):
    code, rows = detect_code_rows(x, state, mappings, project)
    cls = {c: k for k, c in enumerate(mappings[state])}[int(code)]
    ref = int(project.method33.old_mapping[cls])
    return canonicalize_tracefree(x, state, code, rows, ref, project)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ns", default="64,256,1024")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project, args.seed)
    ns = [int(v) for v in args.ns.split(",")]
    max_n = max(ns)
    days = np.sort(np.unique(groups))
    raw = []
    for start in range(0, len(days), args.block_days):
        chosen = set(days[start : start + args.block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            raw.append(idx)
    rng = np.random.default_rng(args.seed)
    history_x, history_y = [], []
    results = {str(n): [] for n in ns}
    oracle = []
    for b, idx in enumerate(raw):
        true_state = int(((b // 2) + args.phase) % len(mappings))
        application = list(take(dataset, idx))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mappings[true_state], jitter_config=project.trigger_jitter,
        )
        x = features(app)
        y = labels[idx]
        half = len(idx) // 2
        pool, blind = np.arange(half), np.arange(half, len(idx))
        if len(np.unique(y[pool])) < 3 or len(history_x) == 0:
            # still build history with true canonicalization
            pass
        true_canon = np.stack([canonical(x[i], true_state, mappings, project) for i in range(len(x))])
        if len(history_x):
            hx = np.concatenate(history_x)
            hy = np.concatenate(history_y)
            scaler = StandardScaler().fit(hx)
            frozen = LogisticRegression(max_iter=300).fit(scaler.transform(hx), hy)
            draw = stratified_nested(y[pool], max_n, rng)
            for n in ns:
                sub = draw[:n]
                true_pred = frozen.predict(scaler.transform(true_canon[sub]))
                oracle.append(float(balanced_accuracy_score(y[sub], true_pred)))
                scores = {}
                for candidate in range(len(mappings)):
                    canon = np.stack([canonical(x[pool[j]], candidate, mappings, project) for j in sub])
                    pred = frozen.predict(scaler.transform(canon))
                    scores[str(candidate)] = float(balanced_accuracy_score(y[sub], pred))
                best = max(scores, key=scores.get)
                results[str(n)].append({
                    "block": b,
                    "true_state": true_state,
                    "best_state": int(best),
                    "true_score": scores[str(true_state)],
                    "best_score": scores[best],
                    "margin": scores[best] - scores[str(true_state)],
                    "rank": 1 + sum(v > scores[str(true_state)] for v in scores.values()),
                })
        history_x.append(true_canon)
        history_y.append(y)
        print("block", b, "history", len(history_x), flush=True)

    out = {"ns": {}, "oracle_ba": float(np.mean(oracle)) if oracle else None}
    for n in ns:
        rows = results[str(n)]
        out["ns"][str(n)] = {
            "top1": sum(r["best_state"] == r["true_state"] for r in rows),
            "mean_rank": float(np.mean([r["rank"] for r in rows])) if rows else None,
            "mean_margin": float(np.mean([r["margin"] for r in rows])) if rows else None,
            "rows": rows,
        }
        print(n, "top1", out["ns"][str(n)]["top1"], "/", len(rows), "rank", out["ns"][str(n)]["mean_rank"], flush=True)
    args.output_log.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
