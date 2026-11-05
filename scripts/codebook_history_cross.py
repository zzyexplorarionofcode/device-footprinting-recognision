"""Cross histories and subsequent codebooks within each development time block."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from methods.method33_tools import load_config
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import ATTACKS, REGIME_DAYS, draw_states, evaluate
from scripts.paired_mapping_policies import render


HISTORIES = ((8200, 8201), (8202, 8203), (8204, 8205))
CANDIDATES = (8300, 8301, 8302, 8303)
STARTS = (0, 50)
HISTORY_DAYS = 25
FIRST_HISTORY_DAYS = 12
HORIZON = 25


def contrasts(matrix):
    result = {}
    for h1, h2 in combinations(range(len(HISTORIES)), 2):
        for c1, c2 in combinations(CANDIDATES, 2):
            a = matrix[h1][str(c1)] - matrix[h1][str(c2)]
            b = matrix[h2][str(c1)] - matrix[h2][str(c2)]
            result[f"{h1}-{h2}:{c1}-{c2}"] = {
                "candidate_gap_h1": a, "candidate_gap_h2": b,
                "interaction": a - b, "rank_flip": a * b < 0,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "pilot", "output-log"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    pilot_bytes = args.pilot.read_bytes()
    pilot = json.loads(pilot_bytes)
    with args.dataset.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == pilot["dataset_sha256"]
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == pilot["config_sha256"]
    project = load_config(args.config)
    seeds = [s for pair in HISTORIES for s in pair] + list(CANDIDATES)
    candidates = {s: generated_project(project, s) for s in seeds}
    with np.load(args.dataset, allow_pickle=False) as source:
        data = {k: source[k] for k in ("lengths", "gaps", "directions", "mask", "labels", "groups")}
        dates = source["timestamps"].max(axis=1).astype("datetime64[s]").astype("datetime64[D]")
    assert np.array_equal(dates, data["groups"].astype("datetime64[D]"))
    y = data["labels"]
    mappings = pilot["mappings"]
    u = np.random.default_rng(np.random.SeedSequence(pilot["seed"]).spawn(3)[1]).random(len(y))
    state = draw_states(u, np.full(len(mappings), 1 / len(mappings)))
    blocks = []
    for start in STARTS:
        previous = pilot["sample_days"][:start]
        daily = pilot["sample_days"][start:start+HISTORY_DAYS+HORIZON]
        assert len(daily) == HISTORY_DAYS+HORIZON == 2 * REGIME_DAYS
        warm = np.asarray(pilot["warmup_indices"] + [j for d in previous for j in d["cal_indices"]])
        days = np.array([d["day"] for d in daily], dtype="datetime64[D]")
        assert np.all(dates[warm] < days[0]) and np.all(days[1:] > days[:-1])
        ids, local = [], []
        for day, row in zip(days, daily, strict=True):
            cal, blind = row["cal_indices"], row["blind_indices"]
            assert len(cal) == 8 and np.all(dates[cal+blind] == day)
            if len(local) < HISTORY_DAYS:
                blind = []
            offset = len(ids)
            ids.extend(cal+blind)
            local.append((np.arange(offset, offset+len(cal)), np.arange(offset+len(cal), len(ids))))
        ids = np.asarray(ids)
        assert len(np.unique(np.r_[warm, ids])) == len(warm)+len(ids)
        warm_x, _ = render(data, warm, state[warm], mappings, project)
        # Scale against shared initial history; all crossed cells retain identical label access.
        h_split = local[FIRST_HISTORY_DAYS][0][0]
        c_split = local[HISTORY_DAYS][0][0]
        costs = render(data, ids, state[ids], mappings, project)[1]

        def render_part(seed, begin, end):
            x, cost = render(data, ids[begin:end], state[ids[begin:end]], mappings, candidates[seed])
            assert np.allclose(cost, costs[begin:end], rtol=0, atol=1e-8)
            return x

        future = {s: render_part(s, c_split, len(ids)) for s in CANDIDATES}
        # No history Blind samples are needed to form attack state. Append only historical Calibration.
        evaluation_local = [(cal-c_split, blind-c_split) for cal, blind in local[HISTORY_DAYS:]]
        history_ids = np.r_[warm, ids[:c_split]]
        # The shared scaler must not depend on which history was assigned. Fit it through
        # evaluate's explicit scaler reference, while models consume each crossed history.
        matrices = {a: [] for a in ATTACKS}
        cells = []
        for h, (first, second) in enumerate(HISTORIES):
            hx = np.concatenate((warm_x, render_part(first, 0, h_split),
                                render_part(second, h_split, c_split)))
            row_scores = {a: {} for a in ATTACKS}
            for s in CANDIDATES:
                rows = evaluate(future[s], y[ids[c_split:]], days[HISTORY_DAYS:], evaluation_local,
                                dates[history_ids], y[history_ids], hx, {}, scale_reference=warm_x)
                assert all(row["labels_available"] == len(history_ids)+8*i for i,row in enumerate(rows))
                for a in ATTACKS:
                    row_scores[a][str(s)] = float(np.mean([r["attacks"][a]["post"] for r in rows]))
                cells.append({"history": h, "candidate": s, "days": rows})
            for a in ATTACKS:
                matrices[a].append(row_scores[a])
            print("block", start, "history", h, row_scores, flush=True)
        blocks.append({"start": start, "history_days": days[:HISTORY_DAYS].astype(str).tolist(),
                       "candidate_days": days[HISTORY_DAYS:].astype(str).tolist(),
                       "matrix": matrices, "contrasts": {a: contrasts(matrices[a]) for a in ATTACKS},
                       "cells": cells})
    report = {"pilot_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
              "dataset_sha256": pilot["dataset_sha256"], "config_sha256": pilot["config_sha256"],
              "histories": HISTORIES, "candidates": CANDIDATES, "first_history_days": FIRST_HISTORY_DAYS,
              "parameters": {s: {"positions": p.method33.trigger_positions, "codes": p.method33.trigger_codes}
                             for s,p in candidates.items()},
              "blocks": blocks}
    args.output_log.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
