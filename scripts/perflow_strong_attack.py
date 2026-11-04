"""Per-flow strong attackers: calibrated top-k marginalization and confidence gating."""

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


def split_cal(y, rng):
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(0.2 * len(idx))))
        va.extend(idx[:cut].tolist())
        tr.extend(idx[cut:].tolist())
    return np.asarray(tr), np.asarray(va)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(proba, y, bins=10):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    total = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flows", type=int, default=1000)
    parser.add_argument("--coverage", type=float, default=0.8)
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
            hdx, hdy = np.concatenate(hist_dev_x), np.concatenate(hist_dev_y)
            # Independent calibration split from history only.
            tr, va = split_cal(hsy, rng)
            state_scaler = StandardScaler().fit(hsx[tr])
            state_clf = LogisticRegression(max_iter=300).fit(state_scaler.transform(hsx[tr]), hsy[tr])
            logits_va = state_clf.decision_function(state_scaler.transform(hsx[va]))
            best_t, best_loss = 1.0, np.inf
            for t in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
                loss = log_loss(hsy[va], softmax(logits_va / t), labels=state_clf.classes_)
                if loss < best_loss:
                    best_t, best_loss = t, loss
            dev_scaler = StandardScaler().fit(hdx)
            dev_clf = LogisticRegression(max_iter=300).fit(dev_scaler.transform(hdx), hdy)
            logits = state_clf.decision_function(state_scaler.transform(x))
            raw_proba = softmax(logits)
            cal_proba = softmax(logits / best_t)
            top1 = cal_proba.argmax(axis=1)
            top2 = np.argsort(-cal_proba, axis=1)[:, :2]
            conf = cal_proba.max(axis=1)
            cal_va = softmax(logits_va / best_t)
            gate_thr = float(np.quantile(cal_va.max(axis=1), args.coverage))
            cache = {s: np.stack([canonical(x[i], s, mappings, project) for i in range(len(x))]) for s in range(len(mappings))}
            dev_proba = {s: dev_clf.predict_proba(dev_scaler.transform(cache[s])) for s in range(len(mappings))}
            classes = dev_clf.classes_
            no_proba = dev_clf.predict_proba(dev_scaler.transform(x))
            p1 = dev_proba[0].copy()
            for i in range(len(x)):
                p1[i] = dev_proba[int(top1[i])][i]
            p2 = np.zeros_like(p1)
            for i in range(len(x)):
                ids = top2[i]
                w = cal_proba[i, ids]
                w = w / (w.sum() + 1e-12)
                p2[i] = w[0] * dev_proba[int(ids[0])][i] + w[1] * dev_proba[int(ids[1])][i]
            p6 = np.zeros_like(p1)
            for s in range(len(mappings)):
                p6 += cal_proba[:, s:s + 1] * dev_proba[s]
            gate = np.where(conf[:, None] >= gate_thr, p1, no_proba)
            oracle = np.stack([dev_proba[int(state[i])][i] for i in range(len(x))])

            def ba(proba, truth=y):
                return float(balanced_accuracy_score(truth, classes[np.argmax(proba, axis=1)]))

            correct = top1 == state
            wrong = ~correct
            entry.update({
                "state_top1": float(correct.mean()),
                "state_top2": float(np.mean([state[i] in top2[i] for i in range(len(x))])),
                "state_log_loss": float(log_loss(state, cal_proba, labels=state_clf.classes_)),
                "state_ece": ece(cal_proba, state),
                "temperature": float(best_t),
                "gate_threshold": float(gate_thr),
                "no_canon_ba": ba(no_proba),
                "predicted_ba": ba(p1),
                "top2_ba": ba(p2),
                "top6_ba": ba(p6),
                "gated_ba": ba(gate),
                "oracle_ba": ba(oracle),
                "gate_coverage": float(np.mean(conf >= gate_thr)),
                "correct_state_ba": ba(p1[correct], y[correct]) if len(np.unique(y[correct])) >= 2 else None,
                "wrong_state_ba": ba(p1[wrong], y[wrong]) if len(np.unique(y[wrong])) >= 2 else None,
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
