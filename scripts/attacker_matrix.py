"""Attacker label matrix on the frozen Full New5 defence stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method33 import apply_method33
from methods.method33_tools import load_config
from scripts.fold_review import take
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import draw_states
from scripts.paired_mapping_policies import render
from scripts.position_placebo import features
from scripts.residual_feature_views import views


def render_clean(data, ids):
    lengths, gaps, directions, mask, labels = take(data, ids)
    app = apply_method33(
        np.minimum(lengths, 1514), gaps, directions, mask, labels,
        data["project"].method33, 731800,
        include_weakener=False, include_trigger=False,
    )
    return features(app)


def fit_model(x, y, kind):
    if len(np.unique(y)) < 2:
        return None
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=0)).fit(x, y)
    return RandomForestClassifier(
        n_estimators=64, max_depth=12, min_samples_leaf=2,
        random_state=7500, n_jobs=-1,
    ).fit(x, y)


def score(model, x, y):
    if model is None or len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, model.predict(x)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rates", default="1,2,4,8")
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
    seed = args.seed
    pig = json.loads(args.pilot.read_text(encoding="utf-8"))
    rng_state = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1])
    uniform = rng_state.random(len(labels))

    candidate_seeds = [row["seed"] for row in baseline["candidate_parameters"]]
    candidates = [generated_project(project, s) for s in candidate_seeds]
    warm = np.asarray(pilot["warmup_indices"])
    days = np.array([row["day"] for row in pilot["sample_days"]], dtype="datetime64[D]")

    raw_ids, local = [], []
    offset = 0
    for row in pilot["sample_days"]:
        cal = np.asarray(row["cal_indices"])
        blind = np.asarray(row["blind_indices"])
        raw_ids.extend(np.r_[cal, blind].tolist())
        split = offset + len(cal)
        local.append((np.arange(offset, split), np.arange(split, offset + len(cal) + len(blind))))
        offset += len(cal) + len(blind)
    ids = np.asarray(raw_ids)
    y_ids = labels[ids]
    day_of_pos = np.repeat(np.arange(len(days)), [len(c) + len(b) for c, b in local])
    codebook = day_of_pos // 5
    u = uniform[ids]
    states = draw_states(u, iid)

    # Defence features, codebook by codebook.
    xd = np.empty((len(ids), 64), dtype=np.float32)
    for r, candidate in enumerate(candidates):
        m = codebook == r
        x, _ = render(data, ids[m], states[m], mappings, candidate)
        xd[m] = x
    xc = views(xd)["C"]

    # Warm features for initial training.
    warm_states = draw_states(uniform[warm], iid)
    warm_xd, _ = render(data, warm, warm_states, mappings, candidates[0])
    warm_xc = views(warm_xd)["C"]
    warm_clean = render_clean(data, warm)
    warm_clean_c = views(warm_clean)["C"]
    y_warm = labels[warm]
    day = np.repeat(np.arange(len(days)), [len(c) + len(b) for c, b in local])
    cal_pos = [c for c, _ in local]
    blind_pos = [b for _, b in local]

    feature_sets = {"D": (xd, warm_xd, warm_clean), "C": (xc, warm_xc, warm_clean_c)}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rates = [int(v) for v in args.rates.split(",") if v.strip()]
    report = {"seed": seed, "days": [str(d) for d in days], "attacks": {}}

    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        x_all, x_warm, x_clean = feature_sets[view]
        for kind in models:
            frozen_clean = fit_model(x_clean, y_warm, kind)
            frozen_initial = fit_model(x_warm, y_warm, kind)
            out = {name: [] for name in ["Frozen-clean", "Frozen-initial"]}
            out.update({f"Label-{k}": [] for k in rates})
            out.update({f"Adaptive-{k}": [] for k in rates if k == 8})
            seen = {k: [] for k in rates}
            seen_all = []
            for d in range(len(days)):
                bx, by = x_all[blind_pos[d]], y_ids[blind_pos[d]]
                out["Frozen-clean"].append(score(frozen_clean, bx, by))
                out["Frozen-initial"].append(score(frozen_initial, bx, by))
                for k in rates:
                    pools = [(f"Label-{k}", seen[k])]
                    if k == 8:
                        pools.append((f"Adaptive-{k}", seen_all))
                    for name, pool in pools:
                        tx = np.concatenate([x_warm, x_all[pool]]) if len(pool) else x_warm
                        ty = np.concatenate([y_warm, y_ids[pool]]) if len(pool) else y_warm
                        model = fit_model(tx, ty, kind)
                        out[name].append(score(model, bx, by))
                seen_all.extend(cal_pos[d].tolist())
                for k in rates:
                    seen[k].extend(cal_pos[d][:k].tolist())
            key = f"{view}/{kind}"
            report["attacks"][key] = {
                name: {
                    "mean": float(np.nanmean(vals)),
                    "daily": [None if np.isnan(v) else round(v, 6) for v in vals],
                }
                for name, vals in out.items()
            }
            print(key, {name: round(float(np.nanmean(v)), 4) for name, v in out.items()}, flush=True)

    args.output_log.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
