"""Frozen UNSW-to-Deakin score transfer and 50-day paired external replay."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from methods.method33_tools import load_config
from scripts.continual_new_trigger import freeze_components
from scripts.generated_policy_transfer import generated_project
from scripts.online_nonstationary import ATTACKS, REGIME_DAYS, SLIDING_DAYS, draw_states, evaluate, fit_lr, summarize
from scripts.paired_mapping_policies import render, support
from scripts.streaming_arms import make_mappings


RF_SETTINGS = {"n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2,
               "random_state": 7500, "n_jobs": 1}
HISTORY_CLASS_QUOTA = 4
MLP_SETTINGS = {"hidden_layer_sizes": (64, 32), "max_iter": 30, "random_state": 7900,
                "tol": 0.0, "n_iter_no_change": 31}


def fit_rf(x, y):
    return RandomForestClassifier(**RF_SETTINGS).fit(x, y)


def fit_mlp(x, y):
    with warnings.catch_warnings():
        # The epoch cap is the declared training budget.
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return MLPClassifier(**MLP_SETTINGS).fit(x, y)


def class_matched_history(past_x, past_y, recent, x_blind, y_blind, rng):
    old = np.flatnonzero(~np.isin(np.arange(len(past_y)), recent))
    pools = {"old": old, "recent": recent}
    available = {name: support(past_y[idx]) for name, idx in pools.items()}
    classes = [c for c in np.unique(past_y) if all(
        counts.get(str(c), 0) >= HISTORY_CLASS_QUOTA for counts in available.values())]
    info = {"quota": HISTORY_CLASS_QUOTA, "available": available,
            "eligible_classes": list(map(int, classes)), "train_n": 0,
            "scored_common_classes": [], "old": None, "recent": None}
    if len(classes) < 2:
        return info
    info["train_n"] = len(classes) * HISTORY_CLASS_QUOTA
    common = np.intersect1d(classes, y_blind)
    info["scored_common_classes"] = common.tolist()
    for name, pool in pools.items():
        train = np.concatenate([rng.choice(pool[past_y[pool] == c], HISTORY_CLASS_QUOTA, replace=False)
                                for c in classes])
        prediction = fit_rf(past_x[train], past_y[train]).predict(x_blind)
        recalls = {int(c): float((prediction[y_blind == c] == c).mean()) for c in np.unique(y_blind)}
        info[name] = {"train_support": support(past_y[train]), "train_buffer_indices": train.tolist(),
                      "full_ba": float(np.mean(list(recalls.values()))),
                      "common_ba": float(np.mean([recalls[c] for c in common])) if len(common) else None}
    return info


def marginal_history(past_x, past_y, recent, x_blind, y_blind, rng):
    quota = HISTORY_CLASS_QUOTA
    old = np.setdiff1d(np.arange(len(past_y)), recent)
    available = {"old": support(past_y[old]), "recent": support(past_y[recent])}
    classes = [c for c in np.unique(past_y) if available["old"].get(str(c), 0) >= quota
               and available["recent"].get(str(c), 0) >= 2 * quota]
    info = {"quota": quota, "available": available, "eligible_classes": list(map(int, classes)),
            "common_classes": [], "samples": None, "scores": None}
    if len(classes) < 2:
        return info
    recent_draw = np.stack([rng.choice(recent[past_y[recent] == c], 2 * quota, replace=False)
                            for c in classes])
    base = recent_draw[:, :quota].ravel()
    extra = recent_draw[:, quota:].ravel()
    historical = np.concatenate([rng.choice(old[past_y[old] == c], quota, replace=False)
                                 for c in classes])
    info["samples"] = {name: rows.tolist() for name, rows in
                       (("recent_base", base), ("old_extra", historical), ("recent_extra", extra))}
    common = np.intersect1d(classes, y_blind)
    info["common_classes"] = common.tolist()
    info["scores"] = {}
    for name, sample in (("B0", base), ("B1", np.r_[base, historical]), ("B2", np.r_[base, extra])):
        pred = fit_rf(past_x[sample], past_y[sample]).predict(x_blind)
        recalls = {int(c): float((pred[y_blind == c] == c).mean()) for c in np.unique(y_blind)}
        info["scores"][name] = {"full_ba": float(np.mean(list(recalls.values()))),
                                "common_ba": float(np.mean([recalls[c] for c in common])) if len(common) else None}
    return info


def matched_history(rows, x, y, days, local, warm_x, warm_y, warm_days, repeat):
    """Add size-matched RF comparisons using only labels available before Blind."""
    scaler = StandardScaler().fit(warm_x)
    x = scaler.transform(x)
    past_x, past_y, past_days = scaler.transform(warm_x), warm_y, warm_days
    for i, (row, (cal, blind)) in enumerate(zip(rows, local, strict=True)):
        assert np.all(past_days < days[i])
        recent = np.flatnonzero(past_days >= days[i] - np.timedelta64(SLIDING_DAYS, "D"))
        row["marginal_history"] = marginal_history(
            past_x, past_y, recent, x[blind], y[blind],
            rng=np.random.default_rng(np.random.SeedSequence([7800, repeat, i])))
        if "class_matched_history" in row:
            past_x = np.concatenate((past_x, x[cal]))
            past_y = np.concatenate((past_y, y[cal]))
            past_days = np.concatenate((past_days, np.repeat(days[i], len(cal))))
            continue
        n = min(64, len(recent))
        rng = np.random.default_rng(np.random.SeedSequence([7600, repeat, i]))
        historical = rng.choice(len(past_y), n, replace=False)
        timely = rng.choice(recent, n, replace=False)
        yb = y[blind]
        info = {"n": n, "common_classes": [], "historical": None, "recent": None}
        common = np.intersect1d(np.intersect1d(past_y[historical], past_y[timely]), yb)
        info["common_classes"] = common.tolist()
        for arm, sample in (("historical", historical), ("recent", timely)):
            classes = np.unique(past_y[sample])
            if len(classes) < 2:
                continue
            prediction = fit_rf(past_x[sample], past_y[sample]).predict(x[blind])
            recall = {int(c): float((prediction[yb == c] == c).mean()) for c in np.unique(yb)}
            info[arm] = {
                "train_support": support(past_y[sample]),
                "older_than_14_days": int((~np.isin(sample, recent)).sum()),
                "ba": float(np.mean(list(recall.values()))),
                "common_ba": float(np.mean([recall[c] for c in common])) if len(common) else None,
            }
        row["matched_history"] = info
        row["class_matched_history"] = class_matched_history(
            past_x, past_y, recent, x[blind], yb,
            np.random.default_rng(np.random.SeedSequence([7700, repeat, i])))
        past_x = np.concatenate((past_x, x[cal]))
        past_y = np.concatenate((past_y, y[cal]))
        past_days = np.concatenate((past_days, np.repeat(days[i], len(cal))))


def descriptors(positions, codes):
    """Permutation-invariant statistics, normalized by the Trigger envelope."""
    p = np.asarray(positions)
    c = np.asarray(codes, dtype=float)
    shares = c[:, 0, :] / c.sum(axis=1)
    return np.r_[p.mean() / 16, np.diff(p, axis=1).mean() / 16,
                 np.abs(2 * shares - 1).mean(axis=0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "split", "output-log"):
        parser.add_argument("--" + name, required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-search", type=Path)
    inputs.add_argument("--rf-reference", type=Path, help="Reuse only IID and Random from this frozen LR run")
    parser.add_argument("--history-reference", type=Path,
                        help="Reuse full-history and sliding RF scores from a frozen RF run")
    parser.add_argument("--component-validation", choices=("lr", "rf", "mlp"))
    parser.add_argument("--component-incremental", action="store_true")
    parser.add_argument("--repeat", required=True, type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.component_incremental and not args.component_validation:
        parser.error("--component-incremental requires --component-validation")
    if args.component_validation and (not args.rf_reference or args.history_reference):
        parser.error("--component-validation requires --rf-reference without --history-reference")
    if args.history_reference and not args.rf_reference:
        parser.error("--history-reference requires --rf-reference")
    project = load_config(args.config)
    split = json.loads(args.split.read_bytes())
    with args.dataset.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == split["sha256"]

    split_hash = hashlib.sha256(args.split.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    pools = np.arange(7300 + 6 * args.repeat, 7306 + 6 * args.repeat).reshape(2, 3).tolist()
    rng_choice, rng_mapping = [np.random.default_rng(s) for s in np.random.SeedSequence([7400, args.repeat]).spawn(2)]
    random = [int(rng_choice.choice(pool)) for pool in pools]
    if args.rf_reference:
        reference_bytes = args.rf_reference.read_bytes()
        reference = json.loads(reference_bytes)
        assert (reference["repeat"], reference["dataset_sha256"], reference["split_sha256"], reference["config_sha256"]) == (
            args.repeat, split["sha256"], split_hash, config_hash)
        assert reference["results"]["Random"]["sequence"] == random
        candidates = {s: generated_project(project, s) for s in random}
        for seed, candidate in candidates.items():
            saved = reference["candidate_parameters"][str(seed)]
            assert np.array_equal(saved["positions"], candidate.method33.trigger_positions)
            assert np.array_equal(saved["codes"], candidate.method33.trigger_codes)
        policies = (("IID", None), ("Random", random))
        metadata = {"classifier": "RF", "rf_settings": RF_SETTINGS,
                    "lr_reference_sha256": hashlib.sha256(reference_bytes).hexdigest()}
        if args.history_reference:
            history_bytes = args.history_reference.read_bytes()
            history_reference = json.loads(history_bytes)
            assert all(history_reference[key] == value for key, value in metadata.items())
            assert history_reference["repeat"] == args.repeat
            assert history_reference["split_sha256"] == split_hash
            metadata["history_reference_sha256"] = hashlib.sha256(history_bytes).hexdigest()
        fit_model = fit_rf
    else:
        # Fit solely on source training scores. Source validation scores are unused.
        search_bytes = args.source_search.read_bytes()
        search = json.loads(search_bytes)
        source = search["candidate_parameters"]
        sx = np.stack([descriptors(p["positions"], p["codes"]) for p in source.values()])
        sy = [search["training"][seed]["score"] for seed in source]
        scorer = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(sx, sy)
        candidates = {s: generated_project(project, s) for pool in pools for s in pool}
        scores = {
            s: float(scorer.predict(descriptors(p.method33.trigger_positions, p.method33.trigger_codes)[None])[0])
            for s, p in candidates.items()
        }
        selected = [min(pool, key=lambda s: (scores[s], s)) for pool in pools]
        policies = (("IID", None), ("Random", random), ("Selected", selected))
        metadata = {"source_search_sha256": hashlib.sha256(search_bytes).hexdigest(),
                    "descriptor_order": ["mean_position", "mean_position_span", "mean_length_imbalance", "mean_gap_imbalance"],
                    "ridge_alpha": 1.0, "source_training_rows": len(source), "scores": scores}
        fit_model = fit_lr
    period = REGIME_DAYS
    mixed = {}
    if args.component_validation:
        period = 5
        seeds = list(range(9500 + 10 * args.repeat, 9510 + 10 * args.repeat))
        candidates = {s: generated_project(project, s) for s in seeds}
        base = candidates[seeds[0]]
        if args.component_incremental:
            policies = (("Position-Length5", seeds), ("Position-Gap5", seeds))
            mixed = {policy: {s: freeze_components(base, p, component) for s, p in candidates.items()}
                     for policy, component in (("Position-Length5", "position+length"),
                                               ("Position-Gap5", "position+gap"))}
        else:
            mixed = {"Position5": {s: freeze_components(base, p, "position") for s, p in candidates.items()}}
            policies = (("Fixed", [seeds[0]] * 10), ("Position5", seeds), ("Full-New5", seeds))
        fit_model = {"rf": fit_rf, "lr": fit_lr, "mlp": fit_mlp}[args.component_validation]
        metadata["classifier"] = args.component_validation.upper()
        if args.component_validation != "rf":
            metadata.pop("rf_settings")
        if args.component_validation == "mlp":
            metadata["mlp_settings"] = MLP_SETTINGS
        metadata["update_period"] = period
    print("frozen", args.repeat, policies, flush=True)

    with np.load(args.dataset, allow_pickle=False) as src:
        data = {k: src[k] for k in ("lengths", "gaps", "directions", "mask", "labels")}
        ts = src["timestamps"]
        names = src["device_names"].tolist()
    assert names == split["device_names"] == list(project.dataset.devices)
    starts = ts[:, 0].astype("datetime64[s]").astype("datetime64[D]")
    days_by_row = ts.max(axis=1).astype("datetime64[s]").astype("datetime64[D]")
    del ts
    labels = data["labels"]
    warm = np.asarray(split["warmup_indices"])
    daily = split["sample_days"][:split["primary_online_days"]]
    assert len(daily) == 2 * REGIME_DAYS
    days = np.array([row["day"] for row in daily], dtype="datetime64[D]")
    assert np.all(days[1:] > days[:-1]) and np.all(days_by_row[warm] < days[0])
    ids, local = [], []
    for day, row in zip(days, daily, strict=True):
        cal, blind = row["cal_indices"], row["blind_indices"]
        assert len(cal) == split["labels_per_day"] == 8
        assert np.all(days_by_row[cal+blind] == day)
        assert support(labels[cal]) == row["cal_support"] and support(labels[blind]) == row["blind_support"]
        offset = len(ids)
        ids.extend(cal+blind)
        local.append((np.arange(offset, offset+len(cal)), np.arange(offset+len(cal), len(ids))))
    ids = np.asarray(ids)
    used = np.r_[warm, ids]
    assert len(np.unique(used)) == len(used) and np.all(starts[used] == days_by_row[used])
    mappings = make_mappings(project, 0)
    states = draw_states(rng_mapping.random(len(labels)), np.full(len(mappings), 1 / len(mappings)))
    warm_x, _ = render(data, warm, states[warm], mappings, project)
    iid_x, iid_cost = render(data, ids, states[ids], mappings, project)
    edges = [local[i][0][0] for i in range(0, len(days), period)] + [len(ids)]
    ranges = [slice(a, b) for a, b in zip(edges, edges[1:])]
    cache = {}
    results = {}
    for policy, sequence in policies:
        x = iid_x
        if sequence is not None:
            parts = []
            for stage, (seed, span) in enumerate(zip(sequence, ranges, strict=True)):
                key = (stage, seed, policy if policy in mixed else None)
                if key not in cache:
                    part = ids[span]
                    config = mixed.get(policy, candidates)[seed]
                    f, cost = render(data, part, states[part], mappings, config)
                    assert np.allclose(cost, iid_cost[span], rtol=0, atol=1e-8)
                    cache[key] = f
                parts.append(cache[key])
            x = np.concatenate(parts)
        if args.history_reference:
            frozen = history_reference["results"][policy]
            assert frozen["sequence"] == sequence
            rows = frozen["days"]
            matched_history(rows, x, labels[ids], days, local, warm_x, labels[warm], days_by_row[warm], args.repeat)
        else:
            rows = evaluate(x, labels[ids], days, local, days_by_row[warm], labels[warm], warm_x, {},
                        regime_days=period, fit_model=fit_model)
        for i, (row, source_row) in enumerate(zip(rows, daily, strict=True)):
            assert row["blind_support"] == source_row["blind_support"]
            assert row["labels_available"] == len(warm) + 8 * i
            row["cal_support"] = source_row["cal_support"]
        summary = summarize(rows)
        stage_post = {a: [float(np.mean([r["attacks"][a]["post"] for r in rows[start:start+25]]))
                          for start in (0, 25)] for a in ATTACKS}
        results[policy] = {"sequence": sequence, "summary": summary, "stage_post_ba": stage_post, "days": rows}
        print(args.repeat, policy, {a: summary[a]["mean_post_ba"] for a in ATTACKS}, flush=True)
    result = {
        "repeat": args.repeat, "dataset_sha256": split["sha256"],
        "split_sha256": split_hash, "config_sha256": config_hash,
        "pools": pools, **metadata,
        "candidate_parameters": {s: {"positions": p.method33.trigger_positions, "codes": p.method33.trigger_codes}
                                 for s,p in candidates.items()},
        "mappings": np.asarray(mappings).tolist(), "warmup_support": support(labels[warm]),
        "cost_max": iid_cost.max(axis=0).tolist(), "results": results,
    }
    if args.component_incremental:
        result["mixed_parameters"] = {
            policy: {s: {"positions": p.method33.trigger_positions, "codes": p.method33.trigger_codes}
                     for s, p in configs.items()} for policy, configs in mixed.items()}
    args.output_log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
