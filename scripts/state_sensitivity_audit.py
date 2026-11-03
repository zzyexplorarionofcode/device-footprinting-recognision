"""Canonicalization state sensitivity + frozen continuous-score audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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


def canonical(x, state, mappings, project):
    code, rows = detect_code_rows(x, state, mappings, project)
    cls = {c: k for k, c in enumerate(mappings[state])}[int(code)]
    ref = int(project.method33.old_mapping[cls])
    return canonicalize_tracefree(x, state, code, rows, ref, project)


def stratified(y, n, rng):
    chosen = []
    classes = np.unique(y)
    per = max(1, n // len(classes))
    for c in classes:
        idx = np.where(y == c)[0]
        chosen.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
    if len(chosen) < n:
        left = np.setdiff1d(np.arange(len(y)), chosen)
        chosen.extend(rng.choice(left, size=min(n - len(chosen), len(left)), replace=False).tolist())
    return np.asarray(chosen[:n])


def brier(y, proba, classes):
    total = 0.0
    for i, label in enumerate(y):
        for j, c in enumerate(classes):
            total += (proba[i, j] - (1.0 if label == c else 0.0)) ** 2
    return total / len(y)


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
    out = {str(n): [] for n in ns}
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
        pool = np.arange(half)
        true_canon = np.stack([canonical(x[i], true_state, mappings, project) for i in range(len(x))])
        if history_x and len(np.unique(y[pool])) >= 3:
            hx, hy = np.concatenate(history_x), np.concatenate(history_y)
            scaler = StandardScaler().fit(hx)
            frozen = LogisticRegression(max_iter=300).fit(scaler.transform(hx), hy)
            classes = sorted(set(hy.tolist()))
            draw = stratified(y[pool], max_n, rng)
            for n in ns:
                sub = draw[:n]
                rows = []
                for candidate in range(len(mappings)):
                    canon = np.stack([canonical(x[pool[j]], candidate, mappings, project) for j in sub])
                    proba = frozen.predict_proba(scaler.transform(canon))
                    pred = frozen.predict(scaler.transform(canon))
                    loss = float(log_loss(y[sub], proba, labels=classes))
                    bs = float(brier(y[sub], proba, classes))
                    # margin: mean max prob minus true-class prob
                    idx_true = [classes.index(int(v)) for v in y[sub]]
                    margin = float(np.mean(proba[np.arange(len(sub)), idx_true] - np.max(proba, axis=1)))
                    dist = float(np.mean(np.linalg.norm(canon - true_canon[pool[sub]], axis=1)))
                    rows.append({
                        "state": candidate,
                        "log_loss": loss,
                        "brier": bs,
                        "margin": margin,
                        "distance": dist,
                        "pred": pred.tolist(),
                    })
                out[str(n)].append({"block": b, "true_state": true_state, "candidates": rows})
        history_x.append(true_canon)
        history_y.append(y)
        print("block", b, flush=True)

    summary = {}
    for n in ns:
        rows = out[str(n)]
        agg = {}
        for metric in ("log_loss", "brier", "margin"):
            reverse = metric == "margin"
            ranks = []
            gaps = []
            for block in rows:
                vals = {r["state"]: r[metric] for r in block["candidates"]}
                true_val = vals[block["true_state"]]
                best_val = max(vals.values()) if reverse else min(vals.values())
                ranks.append(1 + sum((v > true_val) if reverse else (v < true_val) for v in vals.values()))
                gaps.append((true_val - best_val) if reverse else (best_val - true_val))
            agg[metric] = {"mean_rank": float(np.mean(ranks)), "mean_gap": float(np.mean(gaps)), "n": len(rows)}
        summary[str(n)] = agg
    result = {"summary": summary, "blocks": out}
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
