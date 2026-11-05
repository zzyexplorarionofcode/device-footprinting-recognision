"""Reuse a crossed continuation matrix to measure two-stage hindsight value."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from methods.method33_tools import load_config
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import ATTACKS, balanced_accuracy, draw_states, evaluate, fit_lr
from scripts.paired_mapping_policies import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "pilot", "cross", "output-log"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    pilot_bytes, cross_bytes = args.pilot.read_bytes(), args.cross.read_bytes()
    pilot, cross = json.loads(pilot_bytes), json.loads(cross_bytes)
    assert hashlib.sha256(pilot_bytes).hexdigest() == cross["pilot_sha256"]
    with args.dataset.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == cross["dataset_sha256"]
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == cross["config_sha256"]
    project = load_config(args.config)
    with np.load(args.dataset, allow_pickle=False) as src:
        data = {k: src[k] for k in ("lengths", "gaps", "directions", "mask", "labels", "groups")}
        dates = src["timestamps"].max(axis=1).astype("datetime64[s]").astype("datetime64[D]")
    assert np.array_equal(dates, data["groups"].astype("datetime64[D]"))
    y = data["labels"]
    maps = pilot["mappings"]
    u = np.random.default_rng(np.random.SeedSequence(pilot["seed"]).spawn(3)[1]).random(len(y))
    states = draw_states(u, np.full(len(maps), 1 / len(maps)))
    seeds = {s for h in cross["histories"] for s in h} | set(cross["candidates"])
    candidates = {s: generated_project(project, s) for s in sorted(seeds)}
    for s,p in candidates.items():
        frozen = cross["parameters"][str(s)]
        assert np.array_equal(frozen["positions"], p.method33.trigger_positions)
        assert np.array_equal(frozen["codes"], p.method33.trigger_codes)
    blocks = []
    for block in cross["blocks"]:
        start = block["start"]
        h_days = block["history_days"]
        future_days = block["candidate_days"]
        assert h_days[-1] < future_days[0]
        warm = np.asarray(pilot["warmup_indices"] +
                          [j for d in pilot["sample_days"][:start] for j in d["cal_indices"]])
        daily = pilot["sample_days"][start:start+len(h_days)]
        ids, local = [], []
        for row, day in zip(daily, h_days, strict=True):
            assert row["day"] == day and len(row["cal_indices"]) == 8
            cal, blind = row["cal_indices"], row["blind_indices"]
            assert np.all(dates[cal+blind] == np.datetime64(day))
            offset = len(ids)
            ids.extend(cal+blind)
            local.append((np.arange(offset, offset+len(cal)), np.arange(offset+len(cal), len(ids))))
        ids = np.asarray(ids)
        assert len(np.unique(np.r_[warm, ids])) == len(warm)+len(ids)
        assert np.all(dates[warm] < np.datetime64(h_days[0]))
        warm_x, _ = render(data, warm, states[warm], maps, project)
        baseline_cost = render(data, ids, states[ids], maps, project)[1]
        offset = local[cross["first_history_days"]][0][0]
        stage_one = {a: [] for a in ATTACKS}
        daily_results = []
        for first, second in cross["histories"]:
            features = []
            for seed, span in ((first, slice(None, offset)), (second, slice(offset, None))):
                part = ids[span]
                x, cost = render(data, part, states[part], maps, candidates[seed])
                assert np.allclose(cost, baseline_cost[span], rtol=0, atol=1e-8)
                features.append(x)
            rows = evaluate(np.concatenate(features), y[ids], np.array(h_days, dtype="datetime64[D]"),
                            local, dates[warm], y[warm], warm_x, {})
            daily_results.append(rows)
            for a in ATTACKS:
                stage_one[a].append(float(np.mean([r["attacks"][a]["post"] for r in rows])))
        summary = {}
        # The crossed runs initialized fallback at the branch boundary. A full
        # 50-day replay instead falls back to its original shared warmup model.
        scaler = StandardScaler().fit(warm_x)
        initial = fit_lr(scaler.transform(warm_x), y[warm])
        fallback_scores = {}
        fallback_counts = {}
        for attack, first_scores in stage_one.items():
            second = [row.copy() for row in block["matrix"][attack]]
            fallback_counts[attack] = 0
            for cell in block["cells"]:
                corrections = []
                seed = cell["candidate"]
                for i, day in enumerate(cell["days"]):
                    score = day["attacks"][attack]
                    if not score["warmup_fallback"]:
                        continue
                    key = (seed, i)
                    if key not in fallback_scores:
                        blind_row = pilot["sample_days"][start+len(h_days)+i]
                        assert blind_row["day"] == day["day"]
                        blind = np.asarray(blind_row["blind_indices"])
                        x, _ = render(data, blind, states[blind], maps, candidates[seed])
                        fallback_scores[key] = balanced_accuracy(initial, scaler.transform(x), y[blind])
                    corrections.append(fallback_scores[key]-score["post"])
                second[cell["history"]][str(seed)] += sum(corrections)/len(future_days)
                fallback_counts[attack] += len(corrections)
            options = []
            for h, first in enumerate(first_scores):
                c, post = min(second[h].items(), key=lambda pair: (pair[1], int(pair[0])))
                value = (len(h_days)*first + len(future_days)*post)/(len(h_days)+len(future_days))
                options.append({"history": h, "candidate": int(c), "first": first, "second": post, "full": value})
            greedy = min(options, key=lambda x: (x["first"], x["full"], x["history"]))
            joint = min(options, key=lambda x: (x["full"], x["first"], x["history"]))
            summary[attack] = {"options": options, "greedy": greedy, "joint": joint,
                               "first_stage_sacrifice": joint["first"]-greedy["first"],
                               "full_ba_gain": greedy["full"]-joint["full"],
                               "same_history": greedy["history"]==joint["history"]}
        blocks.append({"start": start, "history_days": h_days, "candidate_days": future_days,
                       "stage_one_daily": daily_results, "summary": summary,
                       "continuation_fallback_corrections": fallback_counts})
    result = {"cross_sha256": hashlib.sha256(cross_bytes).hexdigest(),
              "histories": cross["histories"], "blocks": blocks}
    args.output_log.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for block in blocks:
        print(block["start"], block["summary"])


if __name__ == "__main__":
    main()
