"""Random hybrid allocation between all legal and long real-uplink packets."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.deakin_external_codebook import fit_mlp, fit_rf
from scripts.online_nonstationary import draw_states, evaluate, fit_lr, summarize
from scripts.position_placebo import features
from scripts.residual_feature_views import views
from scripts.weakener_feature_audit import length_hist
from scripts.weakener_padding_pair import render_trace, uplink, headroom


ARMS = {"Baseline": 0.0, "Hybrid-25": 0.25, "Hybrid-50": 0.50}
MIN_LONG_HEADROOM = 64.0


def learn_long_threshold(traces, config):
    lengths = []
    for trace in traces:
        original = np.asarray([e["source_length_bytes"] for e in uplink(trace)], float)
        lengths.extend(original[headroom(original, config) >= MIN_LONG_HEADROOM].tolist())
    if not lengths:
        raise ValueError("Warmup has no real-uplink event with the required headroom")
    return float(np.quantile(lengths, 0.75))


def allocate(order, cap, budget):
    result = np.zeros(len(cap), float)
    remaining = float(budget)
    for j in order:
        amount = min(remaining, cap[j] - result[j])
        if amount > 0:
            result[j] += amount
            remaining -= amount
        if remaining <= 1e-9:
            break
    return result, remaining


def redistribute(x, traces, ids, config, threshold, seed):
    transformed = {arm: x.copy() for arm in ARMS}
    real_lengths = {arm: np.zeros((len(ids), 32)) for arm in ARMS}
    audit = {key: np.zeros((len(ids), len(ARMS))) for key in
             ("spent", "visible", "long_spent", "fallback", "headroom", "long_headroom")}
    for row, trace in enumerate(traces):
        events = uplink(trace)
        original = np.asarray([e["source_length_bytes"] for e in events], float)
        observed = np.asarray([e["length_bytes"] for e in events], float)
        visible = np.asarray([e["in_observation_window"] for e in events], bool)
        positions = np.asarray([e["event_position"] - 1 for e in events], int)
        cap = headroom(original, config)
        legal = np.flatnonzero(cap > 0)
        long = np.flatnonzero((original >= threshold) & (cap >= MIN_LONG_HEADROOM))
        requested = float((observed - original).sum())
        assert requested <= cap.sum() + 1e-8
        for col, (arm, fraction) in enumerate(ARMS.items()):
            rng = np.random.default_rng(np.random.SeedSequence([613031, seed, int(ids[row]), col]))
            allocation = np.zeros(len(events), float)
            long_target = requested * fraction
            first, remaining_long = allocate(rng.permutation(long), cap, long_target)
            allocation += first
            long_actual = long_target - remaining_long
            remaining_total = requested - allocation.sum()
            # Frozen fallback: redistribute every unavailable long-byte to all legal
            # packets using their remaining headroom, without crossing packet caps.
            fallback_order = rng.permutation(legal)
            second, remaining = allocate(fallback_order, cap - allocation, remaining_total)
            allocation += second
            if remaining > 1e-8:
                raise AssertionError("Legal headroom cannot reproduce the reference per-flow spend")
            final = original + allocation
            transformed[arm][row, positions[visible]] = final[visible] / 1514
            real_lengths[arm][row, positions[visible]] = final[visible]
            audit["spent"][row, col] = allocation.sum()
            audit["visible"][row, col] = allocation[visible].sum()
            audit["long_spent"][row, col] = allocation[long].sum()
            audit["fallback"][row, col] = max(0.0, long_target - long_actual)
            audit["headroom"][row, col] = cap.sum()
            audit["long_headroom"][row, col] = cap[long].sum()
        assert np.allclose(audit["spent"][row], requested, atol=1e-8, rtol=0)
    inputs = {}
    for arm in ARMS:
        observable = views(transformed[arm])
        inputs[arm] = {"A": observable["A"], "B": observable["B"], "C": observable["C"],
                       "D": transformed[arm], "real_uplink": length_hist(real_lengths[arm])}
    return inputs, audit


def run(family, seed, data, project, mappings, warm, warm_states, ids, days,
        local, dates, uniform, candidates, manifest_path, digest, mlp_only=False):
    root = Path(__file__).resolve().parents[1] / "artifacts/es"
    output = root / f"weakener_hybrid_{'mlp_' if mlp_only else ''}{family}_s{seed}.json"
    y, config = data["labels"], project.method33
    warm_x, warm_traces = render_trace(data, warm, warm_states, mappings, project)
    threshold = learn_long_threshold(warm_traces, config)
    warm_inputs, warm_audit = redistribute(warm_x, warm_traces, warm, config, threshold, seed)
    states = draw_states(uniform[ids], np.full(6, 1 / 6))
    edges = [local[i][0][0] for i in range(0, len(days), 5)] + [len(ids)]
    parts, audits = [], []
    for lo, hi, candidate in zip(edges[:-1], edges[1:], candidates, strict=True):
        x, traces = render_trace(data, ids[lo:hi], states[lo:hi], mappings, candidate)
        part, audit = redistribute(x, traces, ids[lo:hi], config, threshold, seed)
        parts.append(part)
        audits.append(audit)
    audit = {key: np.concatenate([a[key] for a in audits]) for key in audits[0]}
    blind = np.concatenate([b for _, b in local])
    np.savez_compressed(output.with_suffix(".npz"), ids=ids, blind=blind, arms=np.asarray(list(ARMS)), **audit)
    blind_audit = {key: value[blind] for key, value in audit.items()}
    totals = blind_audit["spent"].sum(axis=0)
    report = {
        "family": family, "seed": seed, "dataset_sha256": digest,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "arms": list(ARMS), "long_length_threshold": threshold,
        "minimum_long_headroom": MIN_LONG_HEADROOM,
        "target_long_fraction": list(ARMS.values()),
        "actual_long_fraction": (blind_audit["long_spent"].sum(axis=0) / totals).tolist(),
        "fallback_fraction": (blind_audit["fallback"].sum(axis=0) / totals).tolist(),
        "mean_spent_per_blind_flow": blind_audit["spent"].mean(axis=0).tolist(),
        "mean_visible_per_blind_flow": blind_audit["visible"].mean(axis=0).tolist(),
        "flow_audit": output.with_suffix(".npz").name, "results": {},
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # A is an exact negative control because only real uplink lengths change.
    # Score it on Deakin for the requested observable-C decomposition, then skip
    # the redundant 150-day UNSW fits.
    eval_views = ("C", "D") if mlp_only else (("A", "B", "C", "D", "real_uplink") if family == "deakin10" else ("B", "C", "D", "real_uplink"))
    for view in eval_views:
        report["results"][view] = {}
        models = [] if mlp_only else [("lr", fit_lr), ("rf", fit_rf)]
        if view in ("C", "D"):
            models.append(("mlp", fit_mlp))
        for model, fit in models:
            report["results"][view][model] = {}
            for arm in ARMS:
                x = np.concatenate([part[arm][view] for part in parts])
                rows = evaluate(x, y[ids], days, local, dates[warm], y[warm],
                                warm_inputs[arm][view], {}, regime_days=5, fit_model=fit)
                report["results"][view][model][arm] = {"days": rows, "summary": summarize(rows)}
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(family, seed, view, model, arm,
                      {k: v["mean_post_ba"] for k, v in summarize(rows).items()}, flush=True)


