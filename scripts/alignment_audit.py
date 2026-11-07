"""Four-layer alignment audit for Frozen-clean C/RF and pseudo-label diagnostics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method33_tools import load_config
from scripts.attacker_matrix import fit_model, render_clean
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import draw_states
from scripts.paired_mapping_policies import render
from scripts.residual_feature_views import views


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def daily_ba(pred, y, local):
    return float(np.nanmean([balanced_accuracy_score(y[b], pred[b]) for _, b in local]))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pilot = json.loads((root / "artifacts/es/online_nonstationary_s0.json").read_text(encoding="utf-8"))
    baseline = json.loads((root / "artifacts/es/new_codebook_period5_s0.json").read_text(encoding="utf-8"))
    project = load_config(root / "configs/methods/fade_unsw8_pos_v2.json")
    data = dict(np.load(root / "data/processed/unsw_main8_v1/flows_unsw_8class.npz", allow_pickle=False))
    data["project"] = project
    labels = data["labels"]
    mappings = pilot["mappings"]
    iid = np.full(len(mappings), 1 / len(mappings))
    uniform = np.random.default_rng(np.random.SeedSequence(0).spawn(3)[1]).random(len(labels))
    candidates = [generated_project(project, row["seed"]) for row in baseline["candidate_parameters"]]
    warm = np.asarray(pilot["warmup_indices"])
    days = np.array([row["day"] for row in pilot["sample_days"]], dtype="datetime64[D]")
    ids, local, offset = [], [], 0
    for row in pilot["sample_days"]:
        cal, blind = np.asarray(row["cal_indices"]), np.asarray(row["blind_indices"])
        ids.extend(np.r_[cal, blind].tolist())
        split = offset + len(cal)
        local.append((np.arange(offset, split), np.arange(split, offset + len(cal) + len(blind))))
        offset += len(cal) + len(blind)
    ids = np.asarray(ids)
    day = np.repeat(np.arange(len(days)), [len(c) + len(b) for c, b in local])
    codebook = day // 5
    y = labels[ids]
    states = draw_states(uniform[ids], iid)
    warm_states = draw_states(uniform[warm], iid)
    x_warm_clean = render_clean(data, warm)
    x_clean = render_clean(data, ids)
    x_new = np.empty((len(ids), 64), dtype=np.float32)
    for r, candidate in enumerate(candidates):
        m = codebook == r
        x, _ = render(data, ids[m], states[m], mappings, candidate)
        x_new[m] = x

    cw, ct = views(x_warm_clean)["C"], views(x_new)["C"]
    print("LAYER data")
    print("warm", len(warm), "ids", len(ids), "days", len(days), "unique_ids", len(np.unique(ids)))
    print("labels", np.bincount(labels[warm], minlength=8).tolist(), np.bincount(y, minlength=8).tolist())
    print("target_days", str(days[0]), str(days[-1]))
    print("LAYER feature")
    print("C shapes", cw.shape, ct.shape, cw.dtype, ct.dtype)
    print("C hashes", digest(cw), digest(ct))
    print("C means", np.round(cw.mean(0)[:8], 4).tolist(), np.round(ct.mean(0)[:8], 4).tolist())
    print("C first", np.round(cw[0, :6], 4).tolist(), np.round(ct[0, :6], 4).tolist())
    model = fit_model(cw, labels[warm], "rf")
    print("LAYER model", type(model).__name__, model.n_estimators, model.random_state)
    pred = model.predict(ct)
    print("LAYER prediction", digest(pred), np.bincount(pred, minlength=8).tolist())
    print("matrix-style daily", round(daily_ba(pred, y, local), 4))
    print("pooled", round(float(balanced_accuracy_score(y, pred)), 4))
    print("stored_matrix", 0.6091, "stored_uda", 0.4956)

    # Pseudo-label diagnostics, C/LR.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cwl, ctl = views(x_warm_clean)["C"], views(x_new)["C"]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=0)).fit(cwl, labels[warm])
    for r in range(4):
        proba = m.predict_proba(ctl)
        pred = proba.argmax(1)
        conf = proba.max(1)
        mask = conf >= 0.9
        print("pseudo round", r, "dist", np.bincount(pred, minlength=8).tolist(), "coverage", round(float(mask.mean()), 3),
              "BA", round(float(balanced_accuracy_score(y, pred)), 4))
        if r == 0:
            print("confusion", confusion_matrix(y, pred, labels=list(range(8))).tolist())
        if mask.sum() < 50:
            break
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=0)).fit(
            np.concatenate([cwl, ctl[mask]]), np.concatenate([labels[warm], pred[mask]]))


if __name__ == "__main__":
    main()
