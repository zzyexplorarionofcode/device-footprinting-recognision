"""Per-flow canonicalization attack: no-canon, predicted top1, oracle, top2 fusion."""

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
    parser.add_argument("--block-days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flows", type=int, default=1000)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project, args.seed)
    days = np.sort(np.unique(groups))
    raw = []
    for start in range(0, len(days), args.block_days):
        block_days = days[start : start + args.block_days]
        half = max(1, len(block_days) // 2)
        idx = np.where(np.isin(groups, list(block_days[:half])))[0]
        if len(idx) >= 100:
            raw.append(idx)
    rng = np.random.default_rng(args.seed)
    hist_state_x, hist_state, hist_dev_x, hist_dev_y = [], [], [], []
    results = []
    for b, idx in enumerate(raw):
        if len(idx) > args.flows:
            idx = rng.choice(idx, size=args.flows, replace=False)
        state = rng.integers(0, len(mappings), size=len(idx))
        x = np.zeros((len(idx), 64))
        for s in np.unique(state):
            rows = np.where(state == s)[0]
            subset = idx[rows]
            application = list(take(dataset, subset))
            application[0] = np.minimum(application[0], 1514.0)
            app = apply_method33(
                *application, project.method33, 731800,
                mapping=mappings[int(s)], jitter_config=project.trigger_jitter,
            )
            x[rows] = features(app)
            del app, application
        y = labels[idx]
        true_canon = np.stack([canonical(x[i], int(state[i]), mappings, project) for i in range(len(x))])
        entry = {"block": b, "flows": int(len(idx))}
        if hist_state_x:
            hsx, hsy = np.concatenate(hist_state_x), np.concatenate(hist_state)
            state_scaler = StandardScaler().fit(hsx)
            state_clf = LogisticRegression(max_iter=300).fit(state_scaler.transform(hsx), hsy)
            hdx, hdy = np.concatenate(hist_dev_x), np.concatenate(hist_dev_y)
            dev_scaler = StandardScaler().fit(hdx)
            dev_clf = LogisticRegression(max_iter=300).fit(dev_scaler.transform(hdx), hdy)
            sproba = state_clf.predict_proba(state_scaler.transform(x))
            top1 = state_clf.predict(state_scaler.transform(x))
            top2 = np.argsort(-sproba, axis=1)[:, :2]
            # Cache per-state canonicalization.
            cache = {s: np.stack([canonical(x[i], int(s), mappings, project) for i in range(len(x))]) for s in range(len(mappings))}
            no_canon_pred = dev_clf.predict(dev_scaler.transform(x))
            pred_canon = cache
            oracle_canon = cache
            # predicted top1
            pred1 = np.stack([cache[int(top1[i])][i] for i in range(len(x))])
            oracle = np.stack([cache[int(state[i])][i] for i in range(len(x))])
            p1 = dev_clf.predict_proba(dev_scaler.transform(pred1))
            po = dev_clf.predict_proba(dev_scaler.transform(oracle))
            # top2 fusion, normalized state weights
            fusion = np.zeros_like(p1)
            for i in range(len(x)):
                s1, s2 = int(top2[i, 0]), int(top2[i, 1])
                w1, w2 = float(sproba[i, s1]), float(sproba[i, s2])
                total = w1 + w2 + 1e-12
                fusion[i] = (w1 * dev_clf.predict_proba(dev_scaler.transform(cache[s1][i:i + 1]))[0]
                             + w2 * dev_clf.predict_proba(dev_scaler.transform(cache[s2][i:i + 1]))[0]) / total
            classes = dev_clf.classes_
            entry.update({
                "state_top1": float(np.mean(top1 == state)),
                "state_top2": float(np.mean([state[i] in top2[i] for i in range(len(x))])),
                "no_canon_ba": float(balanced_accuracy_score(y, dev_clf.predict(dev_scaler.transform(x)))),
                "predicted_ba": float(balanced_accuracy_score(y, classes[np.argmax(p1, axis=1)])),
                "oracle_ba": float(balanced_accuracy_score(y, classes[np.argmax(po, axis=1)])),
                "top2_ba": float(balanced_accuracy_score(y, classes[np.argmax(fusion, axis=1)])),
            })
        hist_state_x.append(x)
        hist_state.append(state)
        hist_dev_x.append(true_canon)
        hist_dev_y.append(y)
        results.append(entry)
        print("block", b, entry, flush=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
