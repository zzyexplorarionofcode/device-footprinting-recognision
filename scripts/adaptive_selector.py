"""Label-based state selector with shared canonicalization for probe/finetune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=20)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ns", default="64,256,1024")
    parser.add_argument("--ambiguity", type=float, default=0.01)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project, args.seed)
    ns = [int(v) for v in args.ns.split(",")]
    days = np.sort(np.unique(groups))
    raw = []
    for start in range(0, len(days), args.block_days):
        block_days = days[start : start + args.block_days]
        half = max(1, len(block_days) // 2)
        idx = np.where(np.isin(groups, list(block_days[:half])))[0]
        if len(idx) >= 100:
            raw.append(idx)
    rng = np.random.default_rng(args.seed)
    history_x, history_y = [], []
    results = []
    for b, idx in enumerate(raw):
        true_state = (b // 2 + args.phase) % len(mappings)
        application = list(take(dataset, idx))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mappings[true_state], jitter_config=project.trigger_jitter,
        )
        x = features(app)
        y = labels[idx]
        true_canon = np.stack([canonical(x[i], true_state, mappings, project) for i in range(len(x))])
        entry = {"block": b, "true_state": true_state, "classes": sorted(set(y.tolist()))}
        if history_x and len(np.unique(y)) >= 2:
            hx, hy = np.concatenate(history_x), np.concatenate(history_y)
            scaler = StandardScaler().fit(hx)
            frozen = LogisticRegression(max_iter=300).fit(scaler.transform(hx), hy)
            history_classes = [int(c) for c in frozen.classes_]
            allowed = np.asarray([j for j in range(len(y)) if int(y[j]) in history_classes])
            unknown = int(len(y) - len(allowed))
            entry["history_classes"] = history_classes
            entry["unknown_count"] = unknown
            if len(allowed) >= 50 and len(np.unique(y[allowed])) >= 2:
                draw = stratified(y[allowed], max(ns), rng)
                nout = {}
                for n in ns:
                    sub = draw[:n]
                    scores = {}
                    for s in range(len(mappings)):
                        canon = np.stack([canonical(x[allowed[j]], s, mappings, project) for j in sub])
                        proba = frozen.predict_proba(scaler.transform(canon))
                        scores[str(s)] = float(log_loss(y[allowed][sub], proba, labels=history_classes))
                    ordered = sorted(scores.items(), key=lambda kv: kv[1])
                    best, best_score = int(ordered[0][0]), ordered[0][1]
                    second = ordered[1][1]
                    ambiguous = (second - best_score) < args.ambiguity
                    chosen = best
                    canon_all = np.stack([canonical(x[allowed[j]], chosen, mappings, project) for j in range(len(allowed))])
                    z_all = scaler.transform(canon_all)
                    half = len(allowed) // 2
                    tr, blind = np.arange(half), np.arange(half, len(allowed))
                    # Probe and finetune share chosen canonicalization.
                    probe = LogisticRegression(max_iter=300).fit(z_all[tr], y[allowed][tr])
                    weight = len(hx) / max(1, len(tr))
                    fin = LogisticRegression(max_iter=300).fit(
                        np.concatenate([scaler.transform(hx), z_all[tr]]),
                        np.concatenate([hy, y[allowed][tr]]),
                        sample_weight=np.concatenate([np.ones(len(hx)), np.full(len(tr), weight)]),
                    )
                    nout[str(n)] = {
                        "scores": scores,
                        "selected": chosen,
                        "true_state": true_state,
                        "gap": second - best_score,
                        "ambiguous": bool(ambiguous),
                        "probe_ba": float(balanced_accuracy_score(y[allowed][blind], probe.predict(z_all[blind]))),
                        "finetune_ba": float(balanced_accuracy_score(y[allowed][blind], fin.predict(z_all[blind]))),
                        "blind_classes": sorted(set(y[allowed][blind].tolist())),
                    }
                entry["ns"] = nout
        history_x.append(true_canon)
        history_y.append(y)
        results.append(entry)
        print("block", b, "unknown", entry.get("unknown_count"), flush=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
