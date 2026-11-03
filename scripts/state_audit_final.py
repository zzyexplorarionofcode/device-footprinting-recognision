"""Final state audit: pairwise distances + class-filtered continuous scores."""

from __future__ import annotations

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

SELECTED = [0, 3, 5, 7]


def canonical(x, state, mappings, project):
    code, rows = detect_code_rows(x, state, mappings, project)
    cls = {c: k for k, c in enumerate(mappings[state])}[int(code)]
    ref = int(project.method33.old_mapping[cls])
    return canonicalize_tracefree(x, state, code, rows, ref, project)


def nested_sample(y, n, rng):
    chosen = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        per = max(1, n // len(np.unique(y)))
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
    root = Path(__file__).resolve().parents[1]
    dataset = np.load(root / "data/processed/unsw_main8_v1/flows_unsw_8class.npz", allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(root / "configs/methods/fade_unsw8_pos_v2.json")
    mappings = make_mappings(project, 0)
    days = np.sort(np.unique(groups))
    raw = []
    for start in range(0, len(days), 20):
        block_days = days[start : start + 20]
        half = max(1, len(block_days) // 2)
        cal_days = set(block_days[:half])
        idx = np.where(np.isin(groups, list(cal_days)))[0]
        if len(idx) >= 100:
            raw.append(idx)
    rng = np.random.default_rng(0)
    history_x, history_y = [], []
    out = []
    for b, idx in enumerate(raw):
        true_state = (b // 2) % len(mappings)
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
        if b in SELECTED:
            sample = np.arange(min(50, len(x)))
            cand = {s: np.stack([canonical(x[j], s, mappings, project) for j in sample]) for s in range(len(mappings))}
            D = np.zeros((len(mappings), len(mappings)))
            Dor = {}
            for s in range(len(mappings)):
                Dor[str(s)] = float(np.mean(np.linalg.norm(cand[s] - true_canon[sample], axis=1)))
                for t in range(len(mappings)):
                    D[s, t] = float(np.mean(np.linalg.norm(cand[s] - cand[t], axis=1)))
            entry["pairwise"] = D.tolist()
            entry["oracle_distance"] = Dor
        if history_x and b in SELECTED:
            hx, hy = np.concatenate(history_x), np.concatenate(history_y)
            scaler = StandardScaler().fit(hx)
            frozen = LogisticRegression(max_iter=300).fit(scaler.transform(hx), hy)
            history_classes = [int(c) for c in frozen.classes_]
            allowed = np.asarray([j for j in range(len(y)) if int(y[j]) in history_classes])
            counts = {c: int(np.sum(y[allowed] == c)) for c in history_classes}
            entry["history_classes"] = history_classes
            entry["E_counts"] = counts
            if len(allowed) < 100 or sum(v >= 2 for v in counts.values()) < 2:
                entry["continuous"] = "skipped"
            else:
                draw = nested_sample(y[allowed], 1024, rng)
                ns = {}
                for n in (64, 256, 1024):
                    sub = draw[: min(n, len(draw))]
                    actual = {int(c): int(np.sum(y[allowed][sub] == c)) for c in np.unique(y[allowed][sub])}
                    scores = {}
                    for s in range(len(mappings)):
                        canon = np.stack([canonical(x[allowed[j]], s, mappings, project) for j in sub])
                        proba = frozen.predict_proba(scaler.transform(canon))
                        loss = float(log_loss(y[allowed][sub], proba, labels=history_classes))
                        bs = float(brier(y[allowed][sub], proba, history_classes))
                        idx_true = [history_classes.index(int(v)) for v in y[allowed][sub]]
                        margin = float(np.mean(proba[np.arange(len(sub)), idx_true] - np.max(proba, axis=1)))
                        scores[str(s)] = {"log_loss": loss, "brier": bs, "margin": margin}
                    true_loss = scores[str(true_state)]["log_loss"]
                    true_margin = scores[str(true_state)]["margin"]
                    ns[str(n)] = {
                        "actual_counts": actual,
                        "scores": scores,
                        "G_LL": min(v["log_loss"] for s, v in scores.items() if int(s) != true_state) - true_loss,
                        "G_margin": true_margin - max(v["margin"] for s, v in scores.items() if int(s) != true_state),
                        "rank_LL": 1 + sum(v["log_loss"] < true_loss for v in scores.values()),
                        "rank_margin": 1 + sum(v["margin"] > true_margin for v in scores.values()),
                    }
                known_pred = frozen.predict(scaler.transform(true_canon[allowed][sub]))
                entry["continuous"] = {
                    "known_ba": float(balanced_accuracy_score(y[allowed][sub], known_pred)),
                    "known_log_loss": float(log_loss(y[allowed][sub], frozen.predict_proba(scaler.transform(true_canon[allowed][sub])), labels=history_classes)),
                    "ns": ns,
                }
        history_x.append(true_canon)
        history_y.append(y)
        out.append(entry)
        print("block", b, "classes", len(entry["classes"]), flush=True)
    (root / "artifacts/es/state_audit_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
