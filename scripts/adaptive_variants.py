"""Common-class BA and finetune variants with frozen selector."""

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


def split_train_val(y, rng):
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(0.2 * len(idx))))
        va.extend(idx[:cut].tolist())
        tr.extend(idx[cut:].tolist())
    return np.asarray(tr), np.asarray(va)


def fit_variant(x, y, hy, hx, variant, rng):
    if variant == "baseline":
        w = len(hx) / max(1, len(x))
        return LogisticRegression(max_iter=300).fit(
            np.concatenate([hx, x]), np.concatenate([hy, y]),
            sample_weight=np.concatenate([np.ones(len(hx)), np.full(len(x), w)]),
        )
    if variant == "l2":
        w = len(hx) / max(1, len(x))
        return LogisticRegression(max_iter=300, C=0.1).fit(
            np.concatenate([hx, x]), np.concatenate([hy, y]),
            sample_weight=np.concatenate([np.ones(len(hx)), np.full(len(x), w)]),
        )
    if variant == "balanced":
        w = len(hx) / max(1, len(x))
        return LogisticRegression(max_iter=300, class_weight="balanced").fit(
            np.concatenate([hx, x]), np.concatenate([hy, y]),
            sample_weight=np.concatenate([np.ones(len(hx)), np.full(len(x), w)]),
        )
    # early stop: choose C on calibration validation only
    tr, va = split_train_val(y, rng)
    best_c, best_loss = 1.0, np.inf
    for c in (0.01, 0.1, 1.0):
        model = LogisticRegression(max_iter=300, C=c).fit(x[tr], y[tr])
        loss = log_loss(y[va], model.predict_proba(x[va]), labels=sorted(set(y.tolist())))
        if loss < best_loss:
            best_c, best_loss = c, loss
    return LogisticRegression(max_iter=300, C=best_c, class_weight="balanced").fit(x, y)


def common_ba(y, pred, classes, base_pred=None):
    mask = np.isin(y, classes)
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return None
    return float(balanced_accuracy_score(y[mask], pred[mask]))


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
                    selected, best = int(ordered[0][0]), ordered[0][1]
                    gap = ordered[1][1] - best
                    train_canon = np.stack([canonical(x_cal[cal_allowed[j]], selected, mappings, project) for j in sub])
                    blind_canon = np.stack([canonical(x_blind[j], selected, mappings, project) for j in blind_allowed])
                    labels_eval = y_blind[blind_allowed]
                    before_pred = frozen.predict(scaler.transform(blind_canon))
                    known_pred = frozen.predict(scaler.transform(true_blind[blind_allowed]))
                    probe = LogisticRegression(max_iter=300).fit(train_canon, y_cal[cal_allowed][sub])
                    variants = {}
                    for variant in ("baseline", "l2", "balanced", "early"):
                        model = fit_variant(train_canon, y_cal[cal_allowed][sub], hy, scaler.transform(hx), variant, rng)
                        pred = model.predict(scaler.transform(blind_canon))
                        train_classes = sorted(set(y_cal[cal_allowed][sub].tolist()))
                        blind_classes = sorted(set(labels_eval.tolist()))
                        common = sorted(set(train_classes) & set(blind_classes))
                        variants[variant] = {
                            "full_ba": float(balanced_accuracy_score(labels_eval, pred)),
                            "common_ba": common_ba(labels_eval, pred, common) if common else None,
                            "common_classes": common,
                            "train_loss": float(log_loss(y_cal[cal_allowed][sub], model.predict_proba(train_canon), labels=sorted(model.classes_))),
                        }
                    common = sorted(set(y_cal[cal_allowed][sub].tolist()) & set(labels_eval.tolist()))
                    nout[str(n)] = {
                        "selected": selected,
                        "gap": gap,
                        "ambiguous": bool(gap < args.ambiguity),
                        "before_full": float(balanced_accuracy_score(labels_eval, before_pred)),
                        "known_full": float(balanced_accuracy_score(labels_eval, known_pred)),
                        "probe_full": float(balanced_accuracy_score(labels_eval, probe.predict(blind_canon))),
                        "probe_common": common_ba(labels_eval, probe.predict(blind_canon), common) if common else None,
                        "before_common": common_ba(labels_eval, before_pred, common) if common else None,
                        "known_common": common_ba(labels_eval, known_pred, common) if common else None,
                        "common_classes": common,
                        "variants": variants,
                    }
                entry["ns"] = nout
        history_x.append(true_cal)
        history_y.append(y_cal)
        results.append(entry)
        print("block", b, flush=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
