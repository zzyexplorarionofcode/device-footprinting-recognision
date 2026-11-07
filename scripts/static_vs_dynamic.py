"""Same-budget, zero-label comparison: no defence, fixed codebook, Full New5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method33_tools import load_config
from scripts.attacker_matrix import fit_model, render_clean, score
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import draw_states
from scripts.paired_mapping_policies import render
from scripts.residual_feature_views import views


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

    candidate_seeds = [row["seed"] for row in baseline["candidate_parameters"]]
    candidates = [generated_project(project, s) for s in candidate_seeds]
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

    u = uniform[ids]
    states = draw_states(u, iid)
    warm_states = draw_states(uniform[warm], iid)
    y_warm = labels[warm]

    # No defence.
    x_clean = render_clean(data, ids)
    warm_clean = render_clean(data, warm)

    # Fixed codebook: one candidate for the whole horizon.
    fixed0 = candidates[0]
    x_fixed_iid, _ = render(data, ids, states, mappings, fixed0)
    warm_fixed_iid, _ = render(data, warm, warm_states, mappings, fixed0)
    x_fixed_map, _ = render(data, ids, np.zeros(len(ids), dtype=int), mappings, fixed0)
    warm_fixed_map, _ = render(data, warm, np.zeros(len(warm), dtype=int), mappings, fixed0)

    # Dynamic Full New5: candidate changes every 5 observed days.
    x_new = np.empty((len(ids), 64), dtype=np.float32)
    for r, candidate in enumerate(candidates):
        m = codebook == r
        x, _ = render(data, ids[m], states[m], mappings, candidate)
        x_new[m] = x
    warm_new, _ = render(data, warm, warm_states, mappings, fixed0)

    raw = (x_clean, warm_clean, x_fixed_map, warm_fixed_map, x_fixed_iid, warm_fixed_iid, x_new, warm_new)
    dviews = {"D": raw, "C": tuple(views(v)["C"] for v in raw)}
    report = {"seed": args.seed, "days": [str(d) for d in days], "arms": {}}
    for view, (xc, wc, xfm, wfm, xfi, wfi, xn, wn) in dviews.items():
        if view == "C":
            xc, wc = views(xc)["C"], views(wc)["C"]
            xfm, wfm = views(xfm)["C"], views(wfm)["C"]
            xfi, wfi = views(xfi)["C"], views(wfi)["C"]
            xn, wn = views(xn)["C"], views(wn)["C"]
        for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
            arms = {
                "No-defense": (fit_model(wc, y_warm, kind), xc),
                "Fixed-map": (fit_model(wfm, y_warm, kind), xfm),
                "Fixed-codebook-IID": (fit_model(wfi, y_warm, kind), xfi),
                "Full-New5": (fit_model(wn, y_warm, kind), xn),
            }
            for name, (model, x) in arms.items():
                daily = [score(model, x[b], y_ids[b]) for _, b in local]
                report["arms"][f"{view}/{kind}/{name}"] = {
                    "mean": float(np.nanmean(daily)),
                    "blocks": [
                        float(np.nanmean(daily[a:b]))
                        for a, b in ((0, 50), (50, 100), (100, 150))
                    ],
                    "daily": [None if np.isnan(v) else round(v, 6) for v in daily],
                }
            print(view, kind, {name: round(report["arms"][f"{view}/{kind}/{name}"]["mean"], 4) for name in arms}, flush=True)
    args.output_log.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
