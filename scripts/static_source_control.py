"""P8-48 static control with one fixed model per source domain."""

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
    ids, local, offset = [], [], 0
    for row in pilot["sample_days"]:
        cal, blind = np.asarray(row["cal_indices"]), np.asarray(row["blind_indices"])
        ids.extend(np.r_[cal, blind].tolist())
        split = offset + len(cal)
        local.append((np.arange(offset, split), np.arange(split, offset + len(cal) + len(blind))))
        offset += len(cal) + len(blind)
    ids = np.asarray(ids)
    y = labels[ids]
    day = np.repeat(np.arange(len(local)), [len(c) + len(b) for c, b in local])
    codebook = day // 5
    states = draw_states(uniform[ids], iid)
    warm_states = draw_states(uniform[warm], iid)

    x_clean = render_clean(data, ids)
    w_clean = render_clean(data, warm)
    fixed = candidates[0]
    x_fm, _ = render(data, ids, np.zeros(len(ids), dtype=int), mappings, fixed)
    w_fm, _ = render(data, warm, np.zeros(len(warm), dtype=int), mappings, fixed)
    x_fi, _ = render(data, ids, states, mappings, fixed)
    w_fi, _ = render(data, warm, warm_states, mappings, fixed)
    x_new = np.empty((len(ids), 64), dtype=np.float32)
    for r, candidate in enumerate(candidates):
        m = codebook == r
        x, _ = render(data, ids[m], states[m], mappings, candidate)
        x_new[m] = x

    report = {"seed": args.seed, "groups": {}}
    raw = {"D": (x_clean, w_clean, x_fm, w_fm, x_fi, w_fi, x_new), "C": None}
    base = raw["D"]
    raw["C"] = tuple(views(v)["C"] for v in base)
    for view, (xc, wc, xfm, wfm, xfi, wfi, xn) in raw.items():
        for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
            clean_model = fit_model(wc, labels[warm], kind)
            initial_model = fit_model(wfi, labels[warm], kind)
            arms = {
                "Clean-model/Clean": (clean_model, xc),
                "Clean-model/Fixed-map": (clean_model, xfm),
                "Clean-model/Fixed-IID": (clean_model, xfi),
                "Clean-model/Full-New5": (clean_model, xn),
                "Initial-model/Fixed-map": (initial_model, xfm),
                "Initial-model/Fixed-IID": (initial_model, xfi),
                "Initial-model/Full-New5": (initial_model, xn),
            }
            for name, (model, x) in arms.items():
                daily = [score(model, x[b], y[b]) for _, b in local]
                report["groups"][f"{view}/{kind}/{name}"] = {
                    "mean": float(np.nanmean(daily)),
                    "daily": [None if np.isnan(v) else round(v, 6) for v in daily],
                }
            print(view, kind, {name: round(report["groups"][f"{view}/{kind}/{name}"]["mean"], 4) for name in arms}, flush=True)
    args.output_log.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
