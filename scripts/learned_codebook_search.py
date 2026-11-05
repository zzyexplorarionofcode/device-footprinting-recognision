"""Train a budget-feasible codebook selector on past online replay scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from methods.method33_tools import load_config
from scripts.deakin_external_codebook import descriptors
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import ATTACKS, REGIME_DAYS, draw_states, evaluate, summarize
from scripts.paired_mapping_policies import render


POOLS = ((6200, 6201, 6202), (6203, 6204, 6205))
RANK_TRAIN = tuple(range(8000, 8016))
RANK_POOLS = (tuple(range(8100, 8104)), tuple(range(8104, 8108)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output-log", required=True, type=Path)
    parser.add_argument("--unseen-ranking", action="store_true")
    args = parser.parse_args()
    pilot = json.loads(args.pilot.read_bytes())
    with args.dataset.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == pilot["dataset_sha256"]
    assert hashlib.sha256(args.config.read_bytes()).hexdigest() == pilot["config_sha256"]
    project = load_config(args.config)
    pools = RANK_POOLS if args.unseen_ranking else POOLS
    pool_seeds = tuple(s for pool in pools for s in pool)
    training_seeds = RANK_TRAIN if args.unseen_ranking else pool_seeds
    candidates = {s: generated_project(project, s) for s in dict.fromkeys((*training_seeds, *pool_seeds))}
    with np.load(args.dataset, allow_pickle=False) as source:
        data = {k: source[k] for k in ("lengths", "gaps", "directions", "mask", "labels", "groups")}
        assert np.array_equal(source["timestamps"].max(axis=1).astype("datetime64[s]").astype("datetime64[D]"),
                              data["groups"].astype("datetime64[D]"))
    y = data["labels"]
    groups = data["groups"].astype("datetime64[D]")
    mappings = pilot["mappings"]
    warm = np.asarray(pilot["warmup_indices"])
    random_rows = np.random.default_rng(np.random.SeedSequence(pilot["seed"]).spawn(3)[1]).random(len(y))
    states = draw_states(random_rows, np.full(len(mappings), 1 / len(mappings)))
    days, local, ids = [], [], []
    for row in pilot["sample_days"][:4 * REGIME_DAYS]:
        day = np.datetime64(row["day"], "D")
        cal, blind = row["cal_indices"], row["blind_indices"]
        split = len(ids) + len(cal)
        local.append((np.arange(len(ids), split), np.arange(split, split + len(blind))))
        days.append(day)
        ids.extend(cal + blind)
        assert np.all(groups[cal + blind] == day)
    days, ids = np.asarray(days), np.asarray(ids)
    assert np.all(groups[warm] < days[0]) and np.all(days[1:] > days[:-1])
    assert len(np.unique(np.r_[warm, ids])) == len(warm) + len(ids)
    cutoff = 2 * REGIME_DAYS
    train_ids = ids[:local[cutoff][0][0]]
    history_cal = np.concatenate([cal for cal, _ in local[:cutoff]])
    validation_warm = np.r_[warm, ids[history_cal]]
    assert groups[validation_warm].max() < days[cutoff]
    warm_x, _ = render(data, warm, states[warm], mappings, project)
    train_control = render(data, train_ids, states[train_ids], mappings, project)[1]
    training = {}
    for seed in training_seeds:
        config = candidates[seed]
        x, cost = render(data, train_ids, states[train_ids], mappings, config)
        assert np.allclose(cost, train_control, rtol=0, atol=1e-8)
        rows = evaluate(x, y[train_ids], days[:cutoff], local[:cutoff], groups[warm], y[warm], warm_x, {})
        training[seed] = {
            "score": float(np.mean([row["attacks"]["cumulative"]["post"] for row in rows])),
            "regime_scores": [float(np.mean([row["attacks"]["cumulative"]["post"] for row in rows
                                             if row["regime"] == regime])) for regime in range(2)],
        }
        print("train", seed, training[seed], flush=True)
    if args.unseen_ranking:
        def features_for(seed):
            config = candidates[seed].method33
            return descriptors(config.trigger_positions, config.trigger_codes)

        fit = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        fit.fit(np.stack([features_for(s) for s in training_seeds]), [training[s]["score"] for s in training_seeds])
        predicted = dict(zip(pool_seeds, fit.predict(np.stack([features_for(s) for s in pool_seeds])), strict=True))
    else:
        predicted = {s: training[s]["score"] for s in pool_seeds}
    selected = tuple(min(pool, key=lambda s: (predicted[s], s)) for pool in pools)
    # Every pool member appears exactly once in its stage of the random reference.
    random_reference = tuple(zip(pools[0], (*pools[1][1:], pools[1][0]), strict=True))
    print("frozen selection", selected, "random reference", random_reference, flush=True)

    validation_warm_x, _ = render(data, validation_warm, states[validation_warm], mappings, project)
    offset = len(train_ids)
    validation_ids = ids[offset:]
    validation_local = [(c-offset, b-offset) for c, b in local[cutoff:]]
    validation_days = days[cutoff:]
    split = validation_local[REGIME_DAYS][0][0]
    parts = (validation_ids[:split], validation_ids[split:])
    features = {}
    ranking = {}
    if args.unseen_ranking:
        control_cost = render(data, validation_ids, states[validation_ids], mappings, project)[1]
        for s in pool_seeds:
            x, cost = render(data, validation_ids, states[validation_ids], mappings, candidates[s])
            assert np.allclose(cost, control_cost, rtol=0, atol=1e-8)
            rows = evaluate(x, y[validation_ids], validation_days, validation_local,
                            groups[validation_warm], y[validation_warm], validation_warm_x, {})
            scores = summarize(rows)
            ranking[s] = {"predicted": float(predicted[s]),
                          "observed": scores["cumulative"]["mean_post_ba"],
                          "stage_ba": [float(np.mean([r["attacks"]["cumulative"]["post"] for r in rows
                                                      if r["regime"] == i])) for i in range(2)]}
            features[s] = x[:split] if s in pools[0] else x[split:]
            print("ranking", s, ranking[s], flush=True)
    else:
        for pool, part in zip(pools, parts, strict=True):
            cost_reference = render(data, part, states[part], mappings, project)[1]
            for s in pool:
                x, cost = render(data, part, states[part], mappings, candidates[s])
                assert np.allclose(cost, cost_reference, rtol=0, atol=1e-8)
                features[s] = x

    validation = {}
    for sequence in dict.fromkeys((selected, *random_reference)):
        rows = evaluate(np.concatenate([features[s] for s in sequence]), y[validation_ids],
                        validation_days, validation_local, groups[validation_warm], y[validation_warm],
                        validation_warm_x, {})
        summary = summarize(rows)
        validation[",".join(map(str, sequence))] = {
            "summary": summary, "days": rows,
        }
        print("validation", sequence, {a: summary[a]["mean_post_ba"] for a in ATTACKS}, flush=True)
    result = {
        "pilot_sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
        "dataset_sha256": pilot["dataset_sha256"], "config_sha256": pilot["config_sha256"],
        "pools": pools, "training_days": days[:cutoff].astype(str).tolist(),
        "validation_days": validation_days.astype(str).tolist(),
        "validation_warm_indices": validation_warm.tolist(),
        "candidate_parameters": {s: {"positions": p.method33.trigger_positions, "codes": p.method33.trigger_codes}
                                 for s,p in candidates.items()},
        "training": training, "selected": selected, "random_reference": random_reference, "validation": validation,
    }
    if args.unseen_ranking:
        predictions = [r["predicted"] for r in ranking.values()]
        rho = spearmanr(predictions, [r["observed"] for r in ranking.values()]).statistic
        result["unseen_ranking"] = {"ridge_alpha": 1.0, "scores": ranking,
                                    "spearman_rho": float(rho) if np.isfinite(rho) else None}
    args.output_log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
