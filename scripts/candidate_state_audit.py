"""Audit whether labels can distinguish candidate mapping states via CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ns", default="64,256")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project, args.seed)
    days = np.sort(np.unique(groups))
    raw = []
    for start in range(0, len(days), args.block_days):
        chosen = set(days[start : start + args.block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            raw.append(idx)
    rng = np.random.default_rng(args.seed)
    out = {"ns": {}, "blocks": len(raw)}
    for n in [int(v) for v in args.ns.split(",")]:
        rows = []
        for b, idx in enumerate(raw):
            true_state = int(((b // 2) + args.phase) % len(mappings))
            state = np.full(len(idx), true_state, dtype=int)
            application = list(take(dataset, idx))
            application[0] = np.minimum(application[0], 1514.0)
            app = apply_method33(
                *application, project.method33, 731800,
                mapping=mappings[true_state], jitter_config=project.trigger_jitter,
            )
            x = features(app)
            if len(np.unique(labels[idx])) < 3:
                continue
            draw = stratified(labels[idx], n, rng)
            scores = {}
            for candidate in range(len(mappings)):
                inverse = {c: cls for cls, c in enumerate(mappings[candidate])}
                canon = []
                for j in draw:
                    code, trig_rows = detect_code_rows(x[j], candidate, mappings, project)
                    cls = inverse[int(code)]
                    ref_code = int(project.method33.old_mapping[int(cls)])
                    canon.append(canonicalize_tracefree(x[j], candidate, code, trig_rows, ref_code, project))
                canon = np.stack(canon)
                z = StandardScaler().fit_transform(canon)
                model = LogisticRegression(max_iter=300)
                score = cross_val_score(model, z, labels[idx[draw]], cv=3).mean()
                scores[str(candidate)] = float(score)
            best = max(scores, key=scores.get)
            rows.append({
                "block": b,
                "true_state": true_state,
                "best_state": int(best),
                "true_score": scores[str(true_state)],
                "best_score": scores[best],
                "rank": 1 + sum(v > scores[str(true_state)] for v in scores.values()),
                "scores": scores,
            })
        out["ns"][str(n)] = rows
        print("n", n, "top1", sum(r["best_state"] == r["true_state"] for r in rows), "/", len(rows),
              "mean_rank", np.mean([r["rank"] for r in rows]), flush=True)
    args.output_log.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
