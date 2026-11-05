"""Paired nested Mapping-State count ablation under Full New5."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from methods.method33_tools import load_config
from scripts.deakin_external_codebook import RF_SETTINGS, fit_rf
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import ATTACKS, draw_states, evaluate, fit_lr, summarize
from scripts.paired_mapping_policies import render, support
from scripts.streaming_arms import make_mappings


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("unsw8", "deakin10"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--novelty", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--two-pairs", action="store_true")
    parser.add_argument("--lower-period", type=int, choices=(1, 2),
                        help="Compare faster Full-codebook updates at K=6")
    args = parser.parse_args()
    if args.novelty and args.lower_period:
        parser.error("--novelty and --lower-period are separate experiments")
    seed = args.seed
    artifacts = ROOT / "artifacts"
    if args.family == "unsw8":
        manifest_path = artifacts / "es/online_nonstationary_s0.json"
        manifest = read(manifest_path)
        dataset_path = ROOT / "data/processed/unsw_main8_v1/flows_unsw_8class.npz"
        config_path = ROOT / "configs/methods/fade_unsw8_pos_v2.json"
        daily = manifest["sample_days"]
        horizon = 150
        codebook_seeds = list(range(5100+6*seed, 5106+6*seed)) + list(range(9000+100*seed, 9024+100*seed))
        lr_baseline_path = artifacts / f"es/new_codebook_period5_s{seed}.json"
        lr_baseline = read(lr_baseline_path)
        assert lr_baseline["seed"] == seed
        assert [r["seed"] for r in lr_baseline["candidate_parameters"]] == codebook_seeds
        source = {"lr": lr_baseline["results"]["Continual-New-Trigger"]}
    else:
        manifest_path = artifacts / "splits/deakin_external_short_time.json"
        manifest = read(manifest_path)
        dataset_path = ROOT / manifest["dataset"]
        config_path = ROOT / "configs/methods/fade_deakin.json"
        daily = manifest["sample_days"][:50]
        horizon = 50
        codebook_seeds = list(range(9500+10*seed, 9510+10*seed))
        source = {}
        for model in ("lr", "rf"):
            frozen = read(artifacts / f"es/deakin_position5_{model}_s{seed}.json")
            assert frozen["repeat"] == seed
            assert frozen["results"]["Full-New5"]["sequence"] == codebook_seeds
            source[model] = frozen["results"]["Full-New5"]
    with dataset_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    assert digest == manifest.get("dataset_sha256", manifest.get("sha256"))
    project = load_config(config_path)
    if args.family == "unsw8":
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == manifest["config_sha256"]
    period = args.lower_period or 5
    if args.lower_period:
        needed = horizon // period - len(codebook_seeds)
        codebook_seeds += list(range(200000+1000*seed, 200000+1000*seed+max(0,needed)))
        codebook_seeds = codebook_seeds[:horizon//period]
    candidates = [generated_project(project, s) for s in codebook_seeds]
    with np.load(dataset_path, allow_pickle=False) as src:
        data = {k: src[k] for k in ("lengths", "gaps", "directions", "mask", "labels")}
        ts = src["timestamps"]
    starts = ts[:,0].astype("datetime64[s]").astype("datetime64[D]")
    ends = ts.max(axis=1).astype("datetime64[s]").astype("datetime64[D]")
    del ts
    y = data["labels"]
    warm = np.asarray(manifest["warmup_indices"])
    days, local, ids = [], [], []
    for row in daily:
        day = np.datetime64(row["day"], "D")
        cal, blind = row["cal_indices"], row["blind_indices"]
        assert len(cal) == 8 and np.all(ends[cal+blind] == day)
        split = len(ids)+len(cal)
        local.append((np.arange(len(ids), split), np.arange(split, split+len(blind))))
        ids.extend(cal+blind)
        days.append(day)
    days, ids = np.asarray(days), np.asarray(ids)
    assert len(days)==horizon and np.all(days[1:]>days[:-1]) and np.all(ends[warm]<days[0])
    assert len(np.unique(np.r_[warm,ids]))==len(warm)+len(ids) and np.all(starts[np.r_[warm,ids]]==ends[np.r_[warm,ids]])
    mappings = [[int(c) for c in mapping] for mapping in make_mappings(project, 0, count=12)]
    assert len({tuple(m) for m in mappings})==12
    if args.family == "unsw8":
        assert mappings[:6]==manifest["mappings"]
        uniform = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[1]).random(len(y))
        historical_uniform = np.random.default_rng(np.random.SeedSequence(0).spawn(3)[1]).random(len(y))
    else:
        uniform = np.random.default_rng(np.random.SeedSequence([7400,seed]).spawn(2)[1]).random(len(y))
        historical_uniform = uniform
    # All K start from the same historical six-State training data.
    warm_states = draw_states(historical_uniform[warm], np.full(6,1/6))
    warm_x, _ = render(data,warm,warm_states,mappings,project)
    cost_reference = render(data,ids,draw_states(uniform[ids],np.full(6,1/6)),mappings,project)[1]
    if args.novelty:
        from scripts.novelty_codebook import run_novelty
        run_novelty(args.family, seed, data, project, mappings, warm, warm_x, ids, days, local,
                    ends, uniform, codebook_seeds, source, cost_reference, manifest_path, dataset_path)
        return
    if args.two_pairs:
        from scripts.two_pair_replay import run
        run(args.family,seed,data,project,mappings,warm,warm_x,ids,days,local,ends,uniform,candidates)
        return
    if args.residual:
        from scripts.residual_feature_views import run
        run(args.family, seed, data, project, mappings, warm, warm_x, ids, days, local,
            ends, uniform, candidates, cost_reference, manifest_path, digest)
        return
    results = {}
    output = artifacts / (f"es/full_codebook_period{period}_{args.family}_s{seed}.json" if args.lower_period
                          else f"es/state_count_{args.family}_s{seed}.json")
    for k in ((6,) if args.lower_period else (3,6,12)):
        states = draw_states(uniform[ids],np.full(k,1/k))
        model_results = {}
        models = [m for m in ("lr","rf") if args.lower_period or not (k==6 and m in source)]
        x = None
        if models:
            parts = []
            for stage, candidate in enumerate(candidates):
                lo=local[stage*period][0][0]
                hi=local[(stage+1)*period][0][0] if (stage+1)*period<horizon else len(ids)
                f,cost=render(data,ids[lo:hi],states[lo:hi],mappings,candidate)
                assert np.allclose(cost,cost_reference[lo:hi],rtol=0,atol=1e-8)
                parts.append(f)
            x=np.concatenate(parts)
        for model in ("lr","rf"):
            if not args.lower_period and k==6 and model in source:
                model_results[model]={"reused":True,**source[model]}
            else:
                rows=evaluate(x,y[ids],days,local,ends[warm],y[warm],warm_x,{},
                              regime_days=period,fit_model=fit_lr if model=="lr" else fit_rf)
                for i,(row,(_,blind)) in enumerate(zip(rows,local,strict=True)):
                    row["blind_state_support"]=support(states[blind])
                    row["cal_support"]=support(y[ids[local[i][0]]])
                model_results[model]={"reused":False,"days":rows,"summary":summarize(rows)}
            rows=model_results[model]["days"]
            for i,(row,(_,blind)) in enumerate(zip(rows,local,strict=True)):
                assert row["day"]==str(days[i]) and row["blind_support"]==support(y[ids[blind]])
                assert row["labels_available"]==len(warm)+8*i
                if "blind_state_support" in row:
                    assert row["blind_state_support"]==support(states[blind])
            print(args.family,seed,k,model,{a:model_results[model]["summary"][a]["mean_post_ba"] for a in ATTACKS},flush=True)
        results[str(k)] = model_results
        report={"family":args.family,"seed":seed,"horizon":horizon,"dataset_sha256":digest,
                "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "mappings":mappings,"codebook_seeds":codebook_seeds,
                "warmup_state_count":6,"codebook_period":period,"rf_settings":RF_SETTINGS,"results":results}
        output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
