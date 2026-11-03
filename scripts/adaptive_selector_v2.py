"""Correct adaptive selector: calibration labels for selection, original blind for eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, recall_score
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
        cal = np.where(np.isin(groups, list(block_days[:half])))[0]
        blind = np.where(np.isin(groups, list(block_days[half:])))[0]
        if len(cal) >= 100 and len(blind) >= 100:
            raw.append((cal, blind))
    rng = np.random.default_rng(args.seed)
    history_x, history_y = [], []
    results = []
    for b, (cal, blind) in enumerate(raw):
        true_state = (b // 2 + args.phase) % len(mappings)
        union = np.concatenate([cal, blind])
        application = list(take(dataset, union))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mappings[true_state], jitter_config=project.trigger_jitter,
        )
        x = features(app)
        ncal = len(cal)
        x_cal, y_cal = x[:ncal], labels[cal]
        x_blind, y_blind = x[ncal:], labels[blind]
        true_cal = np.stack([canonical(x_cal[i], true_state, mappings, project) for i in range(len(x_cal))])
        true_blind = np.stack([canonical(x_blind[i], true_state, mappings, project) for i in range(len(x_blind))])
        entry = {"block": b, "true_state": true_state}
        if history_x and len(np.unique(y_cal)) >= 2:
            hx, hy = np.concatenate(history_x), np.concatenate(history_y)
            scaler = StandardScaler().fit(hx)
            frozen = LogisticRegression(max_iter=300).fit(scaler.transform(hx), hy)
            history_classes = [int(c) for c in frozen.classes_]
            cal_allowed = np.asarray([j for j in range(len(y_cal)) if int(y_cal[j]) in history_classes])
            blind_allowed = np.asarray([j for j in range(len(y_blind)) if int(y_blind[j]) in history_classes])
            entry["history_classes"] = history_classes
            entry["unknown_cal"] = int(len(y_cal) - len(cal_allowed))
            entry["unknown_blind"] = int(len(y_blind) - len(blind_allowed))
            entry["unknown_blind_ratio"] = float((len(y_blind) - len(blind_allowed)) / max(1, len(y_blind)))
            if len(cal_allowed) >= 50 and len(blind_allowed) >= 50:
                draw = stratified(y_cal[cal_allowed], max(ns), rng)
                nout = {}
                for n in ns:
                    sub = draw[:n]
                    if len(np.unique(y_cal[cal_allowed][sub])) < 2:
                        continue
                    scores = {}
                    for s in range(len(mappings)):
                        canon = np.stack([canonical(x_cal[cal_allowed[j]], s, mappings, project) for j in sub])
                        proba = frozen.predict_proba(scaler.transform(canon))
                        scores[str(s)] = float(log_loss(y_cal[cal_allowed][sub], proba, labels=history_classes))
                    ordered = sorted(scores.items(), key=lambda kv: kv[1])
                    best, best_score = int(ordered[0][0]), ordered[0][1]
                    gap = ordered[1][1] - best_score
                    selected = best
                    # Shared canonicalization for probe and finetune.
                    train_canon = np.stack([canonical(x_cal[cal_allowed[j]], selected, mappings, project) for j in sub])
                    blind_canon = np.stack([canonical(x_blind[j], selected, mappings, project) for j in blind_allowed])
                    probe = LogisticRegression(max_iter=300).fit(train_canon, y_cal[cal_allowed][sub])
                    weight = len(hx) / max(1, len(sub))
                    fin = LogisticRegression(max_iter=300).fit(
                        np.concatenate([scaler.transform(hx), train_canon]),
                        np.concatenate([hy, y_cal[cal_allowed][sub]]),
                        sample_weight=np.concatenate([np.ones(len(hx)), np.full(len(sub), weight)]),
                    )
                    probe_pred = probe.predict(blind_canon)
                    before_pred = frozen.predict(scaler.transform(blind_canon))
                    fin_pred = fin.predict(scaler.transform(blind_canon))
                    known_pred = frozen.predict(scaler.transform(true_blind[blind_allowed]))
                    labels_eval = y_blind[blind_allowed]
                    classes_eval = sorted(set(labels_eval.tolist()))
                    nout[str(n)] = {
                        "scores": scores,
                        "selected": selected,
                        "gap": gap,
                        "ambiguous": bool(gap < args.ambiguity),
                        "probe_ba": float(balanced_accuracy_score(labels_eval, probe_pred)),
                        "finetune_ba": float(balanced_accuracy_score(labels_eval, fin_pred)),
                        "before_ba": float(balanced_accuracy_score(labels_eval, before_pred)),
                        "known_ba": float(balanced_accuracy_score(labels_eval, known_pred)),
                        "probe_recall": {str(c): float(v) for c, v in zip(classes_eval, recall_score(labels_eval, probe_pred, labels=classes_eval, average=None, zero_division=0))},
                        "finetune_recall": {str(c): float(v) for c, v in zip(classes_eval, recall_score(labels_eval, fin_pred, labels=classes_eval, average=None, zero_division=0))},
                        "before_recall": {str(c): float(v) for c, v in zip(classes_eval, recall_score(labels_eval, before_pred, labels=classes_eval, average=None, zero_division=0))},
                        "known_recall": {str(c): float(v) for c, v in zip(classes_eval, recall_score(labels_eval, known_pred, labels=classes_eval, average=None, zero_division=0))},
                        "probe_log_loss": float(log_loss(y_cal[cal_allowed][sub], probe.predict_proba(train_canon), labels=sorted(set(y_cal[cal_allowed][sub].tolist())))),
                        "finetune_train_log_loss": float(log_loss(y_cal[cal_allowed][sub], fin.predict_proba(train_canon), labels=history_classes)),
                    }
                entry["ns"] = nout
        history_x.append(true_cal)
        history_y.append(y_cal)
        results.append(entry)
        print("block", b, "unknown_blind", entry.get("unknown_blind"), flush=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
