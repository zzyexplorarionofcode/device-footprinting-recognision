"""Pairwise state distances + frozen continuous-score audit on selected blocks."""

from __future__ import annotations

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

SELECTED = [0, 3, 5, 7]


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
        application = list(take(dataset, idx))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mappings[(b // 2) % len(mappings)], jitter_config=project.trigger_jitter,
        )
        x = features(app)
        y = labels[idx]
        pool = np.arange(len(idx))
        true_state = (b // 2) % len(mappings)
        true_canon = np.stack([canonical(x[i], true_state, mappings, project) for i in range(len(x))])
        if b in SELECTED and history_x and len(np.unique(y[pool])) >= 3:
            sample = pool[: min(50, len(pool))]
            cand = {s: np.stack([canonical(x[j], s, mappings, project) for j in sample]) for s in range(len(mappings))}
            D = np.zeros((len(mappings), len(mappings)))
            Dor = {}
            for s in range(len(mappings)):
                Dor[str(s)] = float(np.mean(np.linalg.norm(cand[s] - true_canon[sample], axis=1)))
                for t in range(len(mappings)):
                    D[s, t] = float(np.mean(np.linalg.norm(cand[s] - cand[t], axis=1)))
            hx, hy = np.concatenate(history_x), np.concatenate(history_y)
            scaler = StandardScaler().fit(hx)
            frozen = LogisticRegression(max_iter=300).fit(scaler.transform(hx), hy)
            classes = sorted(set(hy.tolist()))
            rows = {}
            for n in (64, 256, 1024):
                draw = stratified(y[pool], n, rng)
                actual = {int(c): int(np.sum(y[pool][draw] == c)) for c in np.unique(y[pool][draw])}
                scores = {}
                for s in range(len(mappings)):
                    canon = np.stack([canonical(x[pool[j]], s, mappings, project) for j in draw])
                    proba = frozen.predict_proba(scaler.transform(canon))
                    loss = float(log_loss(y[pool][draw], proba, labels=classes))
                    bs = float(brier(y[pool][draw], proba, classes))
                    idx_true = [classes.index(int(v)) for v in y[pool][draw]]
                    margin = float(np.mean(proba[np.arange(len(draw)), idx_true] - np.max(proba, axis=1)))
                    scores[str(s)] = {"log_loss": loss, "brier": bs, "margin": margin}
                true_s = true_state
                true_loss = scores[str(true_s)]["log_loss"]
                true_margin = scores[str(true_s)]["margin"]
                best_loss_other = min(v["log_loss"] for s, v in scores.items() if int(s) != true_s)
                best_margin_other = max(v["margin"] for s, v in scores.items() if int(s) != true_s)
                rows[str(n)] = {
                    "actual_counts": actual,
                    "scores": scores,
                    "G_LL": best_loss_other - true_loss,
                    "G_margin": true_margin - best_margin_other,
                    "rank_LL": 1 + sum(v["log_loss"] < true_loss for v in scores.values()),
                    "rank_margin": 1 + sum(v["margin"] > true_margin for v in scores.values()),
                }
            out.append({"block": b, "pairwise": D.tolist(), "oracle_distance": Dor, "ns": rows})
            print("block", b, "pairwise_min_offdiag", round(float(D[~np.eye(len(mappings), dtype=bool)].min()), 4), flush=True)
        history_x.append(true_canon)
        history_y.append(y)
    (root / "artifacts/es/state_pairwise_continuous.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
