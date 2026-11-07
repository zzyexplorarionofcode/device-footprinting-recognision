"""Frozen-clean attacker with unsupervised adaptation on defended traffic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method33_tools import load_config
from scripts.attacker_matrix import fit_model, render_clean
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import draw_states
from scripts.paired_mapping_policies import render
from scripts.residual_feature_views import views


def coral(source, target):
    ms, mt = source.mean(0), target.mean(0)
    cs = np.cov(source, rowvar=False) + 1e-6 * np.eye(source.shape[1])
    ct = np.cov(target, rowvar=False) + 1e-6 * np.eye(target.shape[1])
    def sqrtm(a):
        w, v = np.linalg.eigh(a)
        return v @ np.diag(np.sqrt(np.clip(w, 1e-12, None))) @ v.T
    def invsqrt(a):
        w, v = np.linalg.eigh(a)
        return v @ np.diag(1 / np.sqrt(np.clip(w, 1e-12, None))) @ v.T
    return (target - mt) @ invsqrt(ct) @ sqrtm(cs) + ms


def pseudo_train(xs, ys, xt, kind, threshold=0.9, rounds=3):
    model = fit_model(xs, ys, kind)
    for _ in range(rounds):
        proba = model.predict_proba(xt)
        conf = proba.max(1)
        mask = conf >= threshold
        if mask.sum() < 50:
            break
        model = fit_model(
            np.concatenate([xs, xt[mask]]),
            np.concatenate([ys, proba.argmax(1)[mask]]),
            kind,
        )
    return model


def cluster_align(xs, ys, xt):
    classes = np.unique(ys)
    k = len(classes)
    xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
    xt = np.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(xt)
    src = np.stack([xs[ys == c].mean(0) for c in classes])
    dst = km.cluster_centers_
    rows, cols = linear_sum_assignment(cdist(src, dst))
    mapping = {int(c): int(classes[r]) for r, c in zip(rows, cols)}
    pred = np.array([mapping[int(v)] for v in km.labels_])
    return pred


def score(pred, y):
    return float(balanced_accuracy_score(y, pred))


def daily_score(pred, y, local):
    return float(np.nanmean([balanced_accuracy_score(y[b], pred[b]) for _, b in local]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default="lr,rf")
    parser.add_argument("--views", default="D,C")
    args = parser.parse_args()

    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    project = load_config(args.config)
    data = dict(np.load(args.dataset, allow_pickle=False))
    data["project"] = project
    labels = data["labels"]
    mappings = pilot["mappings"]
    iid = np.full(len(mappings), 1 / len(mappings))
    uniform = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(3)[1]).random(len(labels))
    candidates = [generated_project(project, row["seed"]) for row in baseline["candidate_parameters"]]
    warm = np.asarray(pilot["warmup_indices"])
    days = np.array([row["day"] for row in pilot["sample_days"]], dtype="datetime64[D]")

    ids, local = [], []
    offset = 0
    for row in pilot["sample_days"]:
        cal, blind = np.asarray(row["cal_indices"]), np.asarray(row["blind_indices"])
        ids.extend(np.r_[cal, blind].tolist())
        split = offset + len(cal)
        local.append((np.arange(offset, split), np.arange(split, offset + len(cal) + len(blind))))
        offset += len(cal) + len(blind)
    ids = np.asarray(ids)
    y_ids = labels[ids]
    day = np.repeat(np.arange(len(days)), [len(c) + len(b) for c, b in local])
    codebook = day // 5

    states = draw_states(uniform[ids], iid)
    warm_states = draw_states(uniform[warm], iid)
    y_warm = labels[warm]
    x_clean = render_clean(data, ids)
    x_warm_clean = render_clean(data, warm)
    x_new = np.empty((len(ids), 64), dtype=np.float32)
    for r, candidate in enumerate(candidates):
        m = codebook == r
        x, _ = render(data, ids[m], states[m], mappings, candidate)
        x_new[m] = x
    warm_new, _ = render(data, warm, warm_states, mappings, candidates[0])

    report = {"seed": args.seed, "views": {}}
    base = (x_warm_clean, x_clean, warm_new, x_new)
    raw = {"D": base, "C": tuple(views(v)["C"] for v in base)}
    for base_view, (xsr, xst, xwr, xwt) in raw.items():
        for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
            models = {
                "Clean-only": fit_model(xsr, y_warm, kind),
                "CORAL": fit_model(xsr, y_warm, kind),
                "Pseudo-label": pseudo_train(xsr, y_warm, xwt, kind),
            }
            out = {}
            for name, model in models.items():
                xt = xwt if name != "CORAL" else coral(xsr, xwt)
                out[name] = daily_score(model.predict(xt), y_ids, local)
            out["KMeans-Hungarian"] = daily_score(cluster_align(xsr, y_warm, xwt), y_ids, local)
            report["views"][f"{base_view}/{kind}"] = out
            print(base_view, kind, out, flush=True)
    args.output_log.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
