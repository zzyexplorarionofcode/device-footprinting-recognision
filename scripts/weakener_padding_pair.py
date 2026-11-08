"""Matched real-uplink padding allocations on the same Full New5 event traces."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from methods.method33 import apply_method33
from scripts.deakin_external_codebook import fit_rf
from scripts.online_nonstationary import draw_states, evaluate, fit_lr, summarize
from scripts.position_placebo import features
from scripts.residual_feature_views import views
from scripts.weakener_feature_audit import length_hist


EDGES = np.linspace(0, 1514, 17)
ARMS = ("Random-all", "Random-candidate", "Targeted-candidate")


def render_trace(data, ids, states, mappings, project):
    x = np.empty((len(ids), 64))
    traces = [None] * len(ids)
    for state in np.unique(states):
        rows = np.flatnonzero(states == state)
        selected = ids[rows]
        app = apply_method33(
            np.minimum(data["lengths"][selected], 1514), data["gaps"][selected],
            data["directions"][selected], data["mask"][selected], data["labels"][selected],
            project.method33, 731800, mapping=mappings[state],
            jitter_config=project.trigger_jitter, include_event_trace=True,
        )
        x[rows] = features(app)
        for row, audit in zip(rows, app[-1], strict=True):
            traces[row] = audit["complete_event_trace"]
    return x, traces


def uplink(trace):
    return [e for e in trace if e["role"] == "real" and e["direction"] == 1]


def length_bins(lengths):
    return np.clip(np.searchsorted(EDGES, lengths, side="right") - 1, 0, 15)


def headroom(lengths, config):
    return np.minimum(config.weakener.max_padding_per_packet,
                      np.maximum(0, config.max_frame_length - lengths))


def c_ablation_views(x, real_lengths):
    """Small residual-statistic views. C1 is provenance-assisted for diagnosis."""
    signed = x[:, :32] * 1514
    valid = signed != 0
    up, down = signed > 0, signed < 0
    absolute = np.abs(signed)
    large = absolute > 256
    counts = np.c_[up.sum(1), down.sum(1), valid.sum(1)]
    byte_totals = np.c_[(absolute * up).sum(1), (absolute * down).sum(1),
                        (absolute * valid).sum(1)]
    large_ratio = np.c_[
        (large & up).sum(1) / np.maximum(1, up.sum(1)),
        (large & down).sum(1) / np.maximum(1, down.sum(1)),
        large.sum(1) / np.maximum(1, valid.sum(1)),
    ]
    c1 = length_hist(real_lengths)
    return {"C1_real_uplink": c1, "C2_counts": counts, "C3_bytes": byte_totals,
            "C4_large_ratio": large_ratio,
            "C5_combined": np.c_[c1, counts, byte_totals, large_ratio]}


def learn_priority(trace_sets, labels, config):
    scores = []
    for traces in trace_sets:
        present = np.zeros((len(labels), 16), dtype=bool)
        for row, trace in enumerate(traces):
            original = np.asarray([e["source_length_bytes"] for e in uplink(trace)
                                   if e["in_observation_window"]])
            eligible = original[headroom(original, config) > 0]
            present[row, length_bins(eligible)] = True
        by_class = np.stack([present[labels == cls].mean(axis=0) for cls in np.unique(labels)])
        scores.append(by_class.std(axis=0))
    stable = np.min(scores, axis=0)
    if not np.any(stable > 0):
        raise ValueError("Warmup contains no discriminative paddable length bins")
    threshold = float(np.median(stable[stable > 0]))
    return stable, threshold


def redistribute(x, traces, ids, config, scores, threshold, seed):
    transformed = {arm: x.copy() for arm in ARMS}
    real_lengths = {arm: np.zeros((len(ids), 32)) for arm in ARMS}
    audit = {key: np.zeros(len(ids)) for key in
             ("requested", "spent", "random_all_visible", "random_candidate_visible",
              "targeted_candidate_visible",
              "headroom", "target_headroom", "real_retained")}
    transitions = {arm: np.zeros((len(ids), 16, 16), dtype=np.uint16) for arm in ARMS}
    visible_transitions = {arm: np.zeros((len(ids), 16, 16), dtype=np.uint16) for arm in ARMS}
    for row, trace in enumerate(traces):
        assert sum(e["role"] != "real" for e in trace) == 5
        events = uplink(trace)
        original = np.asarray([e["source_length_bytes"] for e in events])
        observed = np.asarray([e["length_bytes"] for e in events])
        visible = np.asarray([e["in_observation_window"] for e in events], dtype=bool)
        positions = np.asarray([e["event_position"] - 1 for e in events], dtype=int)
        cap = headroom(original, config)
        priority = scores[length_bins(original)]
        preferred = np.flatnonzero(priority >= threshold)
        requested = float((observed - original).sum())
        assert -1e-8 <= requested <= cap.sum() + 1e-8
        target_cap = float(cap[preferred].sum())
        common = min(max(0., requested), target_cap)
        rng = np.random.default_rng(np.random.SeedSequence([286092, seed, int(ids[row])]))
        order = {"Random-all": rng.permutation(len(events)),
                 "Random-candidate": rng.permutation(preferred),
                 "Targeted-candidate": preferred[np.argsort(-priority[preferred], kind="stable")]}
        for arm in ARMS:
            allocation = np.zeros(len(events))
            remaining = common
            for j in order[arm]:
                allocation[j] = min(remaining, cap[j])
                remaining -= allocation[j]
                if remaining <= 1e-9:
                    break
            assert abs(allocation.sum() - common) <= 1e-8
            final = original + allocation
            assert np.all(final <= config.max_frame_length + 1e-8)
            assert np.allclose(x[row, positions[visible]] * 1514, observed[visible])
            transformed[arm][row, positions[visible]] = final[visible] / 1514
            real_lengths[arm][row, positions[visible]] = final[visible]
            audit[f"{arm.lower().replace('-', '_')}_visible"][row] = allocation[visible].sum()
            before_bins, after_bins = length_bins(original), length_bins(final)
            np.add.at(transitions[arm][row], (before_bins, after_bins), 1)
            np.add.at(visible_transitions[arm][row], (before_bins[visible], after_bins[visible]), 1)
        audit["requested"][row] = requested
        audit["spent"][row] = common
        audit["headroom"][row] = cap.sum()
        audit["target_headroom"][row] = target_cap
        audit["real_retained"][row] = sum(e["in_observation_window"] and e["role"] == "real" for e in trace)
    inputs = {}
    for arm in ARMS:
        inputs[arm] = {"D": transformed[arm], "C": views(transformed[arm])["C"],
                       "real_uplink": length_hist(real_lengths[arm])}
        inputs[arm].update(c_ablation_views(transformed[arm], real_lengths[arm]))
    return inputs, audit, transitions, visible_transitions


def cumulative_rf_confusion(x, y, days, samples, warm_days, warm_y, warm_x):
    """Aggregate same-day Blind predictions made before labels from that day arrive."""
    scaler = StandardScaler().fit(warm_x)
    x, warm_x = scaler.transform(x), scaler.transform(warm_x)
    past_x, past_y, past_days = warm_x, warm_y, warm_days
    labels = np.unique(np.r_[warm_y, y])
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for day, (cal, blind) in zip(days, samples, strict=True):
        model = fit_rf(past_x, past_y)
        matrix += confusion_matrix(y[blind], model.predict(x[blind]), labels=labels)
        past_x = np.concatenate((past_x, x[cal]))
        past_y = np.concatenate((past_y, y[cal]))
        past_days = np.concatenate((past_days, np.repeat(day, len(cal))))
    return labels, matrix


def run(family, seed, data, project, mappings, warm, warm_states, ids, days,
        local, dates, uniform, candidates, manifest_path, digest, c_ablation=False):
    root = Path(__file__).resolve().parents[1] / "artifacts/es"
    stem = "weakener_c_ablation" if c_ablation else "weakener_padding"
    output = root / f"{stem}_{family}_s{seed}.json"
    config = project.method33
    y = data["labels"]
    warm_x, warm_trace = render_trace(data, warm, warm_states, mappings, project)
    warm_variants = [render_trace(data, warm, warm_states, mappings, c)[1] for c in candidates[:2]]
    scores, threshold = learn_priority([warm_trace, *warm_variants], data["labels"][warm], config)
    warm_inputs, _, _, _ = redistribute(warm_x, warm_trace, warm, config, scores, threshold, seed)
    states = draw_states(uniform[ids], np.full(6, 1 / 6))
    edges = [local[i][0][0] for i in range(0, len(days), 5)] + [len(ids)]
    parts, audits, transition_parts, visible_transition_parts = [], [], [], []
    for lo, hi, candidate in zip(edges[:-1], edges[1:], candidates, strict=True):
        x, traces = render_trace(data, ids[lo:hi], states[lo:hi], mappings, candidate)
        inputs, audit, transitions, visible_transitions = redistribute(
            x, traces, ids[lo:hi], config, scores, threshold, seed)
        parts.append(inputs)
        audits.append(audit)
        transition_parts.append(transitions)
        visible_transition_parts.append(visible_transitions)
    audit = {k: np.concatenate([a[k] for a in audits]) for k in audits[0]}
    blind = np.concatenate([b for _, b in local])
    audit_path = output.with_suffix(".npz")
    transition_arrays = {
        f"transition_{arm.replace('-', '_')}": np.concatenate([p[arm] for p in transition_parts])
        for arm in ARMS}
    visible_transition_arrays = {
        f"visible_transition_{arm.replace('-', '_')}":
        np.concatenate([p[arm] for p in visible_transition_parts]) for arm in ARMS}
    np.savez_compressed(audit_path, ids=ids, labels=y[ids], blind=blind,
                        **audit, **transition_arrays, **visible_transition_arrays)
    blind_audit = {k: v[blind] for k, v in audit.items()}
    results = {}
    report = dict(family=family, seed=seed, horizon=len(days), dataset_sha256=digest,
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  scores=scores.tolist(), threshold=threshold, dummy_events=5,
                  flow_audit=audit_path.name, blind_means={k: float(v.mean()) for k, v in blind_audit.items()},
                  blind_original_budget_shortfall_fraction=float(np.mean(
                      blind_audit["requested"] - blind_audit["spent"] > 1e-8)),
                  original_budget_shortfall_fraction=float(np.mean(audit["requested"] - audit["spent"] > 1e-8)),
                  arms=list(ARMS), results=results, confusion={})
    print(family, seed, "matched allocation", report["blind_means"], flush=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    eval_arms = (ARMS[0], ARMS[2]) if c_ablation else ARMS
    eval_views = (("C1_real_uplink", "C2_counts", "C3_bytes", "C4_large_ratio", "C5_combined")
                  if c_ablation else ("C", "D", "real_uplink"))
    for view in eval_views:
        results[view] = {}
        for model, fit in (("lr", fit_lr), ("rf", fit_rf)):
            paired = {}
            for arm in eval_arms:
                x = np.concatenate([part[arm][view] for part in parts])
                rows = evaluate(x, y[ids], days, local, dates[warm], y[warm],
                                warm_inputs[arm][view], {}, regime_days=5, fit_model=fit)
                paired[arm] = dict(days=rows, summary=summarize(rows))
                if arm != eval_arms[0]:
                    for a, b in zip(rows, paired[ARMS[0]]["days"], strict=True):
                        assert (a["day"], a["blind_support"], a["labels_available"]) == (
                            b["day"], b["blind_support"], b["labels_available"])
                        for attack in ("cumulative", "sliding"):
                            assert a["attacks"][attack]["labels_used"] == b["attacks"][attack]["labels_used"]
                results[view][model] = paired
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(family, seed, view, model, arm,
                      {a: v["mean_post_ba"] for a, v in paired[arm]["summary"].items()}, flush=True)
    if not c_ablation:
        c_all = {arm: np.concatenate([part[arm]["C"] for part in parts]) for arm in ARMS}
        for arm in ARMS:
            labels, matrix = cumulative_rf_confusion(
                c_all[arm], y[ids], days, local, dates[warm], y[warm], warm_inputs[arm]["C"])
            report["confusion"][arm] = {"labels": labels.tolist(), "cumulative_rf": matrix.tolist()}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
