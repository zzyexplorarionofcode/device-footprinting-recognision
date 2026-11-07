"""Audit BA aggregation: daily mean vs pooled flows for Frozen-clean."""

from __future__ import annotations

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
    states = draw_states(uniform[ids], iid)
    warm_states = draw_states(uniform[warm], iid)
    x_clean = render_clean(data, ids)
    warm_clean = render_clean(data, warm)
    x_full = np.empty((len(ids), 64), dtype=np.float32)
    for r, candidate in enumerate(candidates):
        m = day // 5 == r
        x, _ = render(data, ids[m], states[m], mappings, candidate)
        x_full[m] = x
    warm_full, _ = render(data, warm, warm_states, mappings, candidates[0])

    for view in ("D", "C"):
        if view == "C":
            src, tgt = views(warm_clean)["C"], views(x_full)["C"]
        else:
            src, tgt = warm_clean, x_full
        model = fit_model(src, labels[warm], "rf")
        daily = np.array([score(model, tgt[b], y[b]) for _, b in local])
        pooled = score(model, tgt, y)
        print(view, "frozen_clean daily_mean", round(float(np.nanmean(daily)), 4), "pooled", round(pooled, 4),
              "n_days", len(daily), flush=True)


if __name__ == "__main__":
    main()
