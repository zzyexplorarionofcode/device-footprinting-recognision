"""Online adaptation v2: stratified labels, shared state-conditioned model,
position-likelihood state inference.

Block protocol: first half of each block is the calibration pool; the remaining
suffix is blind (time-causal).  n labels are sampled class-stratified from the
pool.  Historical data are split by mapping state; the shared model receives a
state one-hot (schedule-known) or the inferred state (state-inferred).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_placebo import features  # noqa: E402


def make_mappings(project, count=6):
    base = list(project.method33.old_mapping)
    nxt = list(project.method33.next_mapping)
    mappings = [base, nxt]
    rng = np.random.default_rng(123)
    while len(mappings) < count:
        perm = list(rng.permutation(len(base)))
        if perm not in mappings:
            mappings.append(perm)
    return mappings


def fit_model(x, y, weights=None):
    if len(np.unique(y)) < 2:
        return None
    return SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(
        x, y, sample_weight=weights
    )


def one_hot_state(state, n_states):
    out = np.zeros((len(state), n_states))
    out[np.arange(len(state)), state] = 1.0
    return out


def build_blocks(dataset, labels, groups, project, mappings, policy, block_days, period, phase, rng):
    days = np.sort(np.unique(groups))
    raw_blocks = []
    for start in range(0, len(days), block_days):
        chosen = set(days[start : start + block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            raw_blocks.append(idx)
    blocks = []
    for b, idx in enumerate(raw_blocks):
        if policy == "Fixed":
            state = np.zeros(len(idx), dtype=int)
        elif policy == "Periodic":
            state = np.full(len(idx), ((b // period) + phase) % len(mappings), dtype=int)
        else:
            state = rng.integers(0, len(mappings), size=len(idx))
        x = np.zeros((len(idx), 64))
        positions = np.full((len(idx), 2), -1, dtype=int)
        for state_id in np.unique(state):
            rows = np.where(state == state_id)[0]
            subset = idx[rows]
            application = list(take(dataset, subset))
            application[0] = np.minimum(application[0], 1514.0)
            app = apply_method33(
                *application, project.method33, 731800,
                mapping=mappings[int(state_id)], jitter_config=project.trigger_jitter,
                include_event_trace=True,
            )
            x[rows] = features(app)
            for local, row in enumerate(rows):
                trace = app[6][local]["complete_event_trace"]
                anchors = []
                for event in trace:
                    if event["role"] == "trigger_dummy" and event["in_observation_window"]:
                        anchors.append(sum(1 for e in trace[: event["event_position"]] if e["role"] == "real"))
                if len(anchors) >= 2:
                    positions[row] = anchors[:2]
        blocks.append({"x": x, "y": labels[idx], "state": state, "pos": positions})
    return blocks


def stratified_sample(pool_idx, y_pool, n, rng):
    if n <= 0:
        return np.asarray([], dtype=int)
    chosen = []
    classes = np.unique(y_pool)
    per = max(1, n // max(1, len(classes)))
    for c in classes:
        idx = np.where(y_pool == c)[0]
        take_n = min(per, len(idx))
        if take_n:
            chosen.extend(rng.choice(idx, size=take_n, replace=False).tolist())
    if len(chosen) < n:
        left = np.setdiff1d(np.arange(len(y_pool)), chosen)
        if len(left):
            chosen.extend(rng.choice(left, size=min(n - len(chosen), len(left)), replace=False).tolist())
    return np.asarray(chosen[:n], dtype=int)


def position_loglik(positions, state, mappings, project, priors):
    positions = positions[positions[:, 0] >= 0]
    if len(positions) == 0:
        return -np.inf
    expected = np.asarray([project.method33.trigger_positions[mappings[state][c]] for c in range(len(priors))])
    total = 0.0
    for pair in positions:
        weights = [priors[c] for c in range(len(priors)) if tuple(expected[c]) == tuple(pair)]
        total += np.log(max(1e-9, float(np.sum(weights))))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--policy", default="Periodic")
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--period", type=int, default=2)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--ns", default="0,64,256,1024")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project)
    ns = [int(v) for v in args.ns.split(",")]
    rng = np.random.default_rng(20261030)
    blocks = build_blocks(dataset, labels, groups, project, mappings, args.policy, args.block_days, args.period, args.phase, rng)
    n_states = len(mappings)

    results = {"policy": args.policy, "ns": ns, "blocks": len(blocks), "attackers": {}}
    for attacker in (("unaware",) if args.policy == "Per_flow" else ("unaware", "schedule_known", "state_inferred")):
        methods = ("permutation", "probe", "finetune")
        curves = {method: {n: [] for n in ns} for method in methods}
        tau = {method: {"values": [], "censored": 0} for method in methods}
        for b in range(1, len(blocks)):
            hist_x = np.concatenate([blocks[i]["x"] for i in range(b)])
            hist_y = np.concatenate([blocks[i]["y"] for i in range(b)])
            hist_state = np.concatenate([blocks[i]["state"] for i in range(b)])
            if len(hist_x) > 5000:
                keep = rng.choice(len(hist_x), size=5000, replace=False)
                hist_x, hist_y, hist_state = hist_x[keep], hist_y[keep], hist_state[keep]
            scaler = StandardScaler().fit(hist_x)
            cur = blocks[b]
            half = len(cur["y"]) // 2
            cal_pool = np.arange(half)
            blind = np.arange(half, len(cur["y"]))
            if len(blind) < 50:
                continue
            # Shared state-conditioned model.
            if attacker == "unaware":
                z_hist = scaler.transform(hist_x)
                z_cal_all = scaler.transform(cur["x"][cal_pool])
                z_blind = scaler.transform(cur["x"][blind])
            else:
                z_hist = np.concatenate([scaler.transform(hist_x), one_hot_state(hist_state, n_states)], axis=1)
                if attacker == "schedule_known":
                    inferred = int(cur["state"][0])
                else:
                    priors = np.asarray([np.mean(hist_y == c) for c in range(len(np.unique(labels)))])
                    scores = [position_loglik(cur["pos"], s, mappings, project, priors) for s in range(n_states)]
                    inferred = int(np.argmax(scores))
                z_cal_all = np.concatenate([scaler.transform(cur["x"][cal_pool]), one_hot_state(np.full(half, inferred), n_states)], axis=1)
                z_blind = np.concatenate([scaler.transform(cur["x"][blind]), one_hot_state(np.full(len(blind), inferred), n_states)], axis=1)
            base = fit_model(z_hist, hist_y)
            for n in ns:
                draw = stratified_sample(cal_pool, cur["y"][cal_pool], n, rng)
                cal_x, cal_y = z_cal_all[draw], cur["y"][draw]
                for method in ("permutation", "probe", "finetune"):
                    if method == "probe" and len(draw) >= 8 and len(np.unique(cal_y)) >= 2:
                        model = fit_model(cal_x, cal_y)
                        pred = model.predict(z_blind)
                    elif method == "finetune" and len(draw) >= 8 and len(np.unique(cal_y)) >= 2:
                        weight = len(z_hist) / len(draw)
                        model = fit_model(
                            np.concatenate([z_hist, cal_x]),
                            np.concatenate([hist_y, cal_y]),
                            np.concatenate([np.ones(len(z_hist)), np.full(len(draw), weight)]),
                        )
                        pred = model.predict(z_blind)
                    else:
                        pred = base.predict(z_blind) if base is not None else np.zeros(len(blind), dtype=int)
                        if method == "permutation" and len(draw) and base is not None:
                            pred_cal = base.predict(cal_x)
                            classes = sorted(set(cur["y"].tolist()))
                            cm = confusion_matrix(cal_y, pred_cal, labels=classes)
                            row, col = linear_sum_assignment(-cm)
                            mapping = {int(p): int(t) for p, t in zip(col, row)}
                            pred = np.asarray([mapping.get(int(p), int(p)) for p in pred])
                    curves[method][n].append(float(balanced_accuracy_score(cur["y"][blind], pred)))
            for method in methods:
                block_ba = {n: curves[method][n][-1] for n in ns if curves[method][n]}
                eligible = [n for n in ns if block_ba.get(n, 0.0) >= 0.5]
                if eligible:
                    tau[method]["values"].append(min(eligible))
                else:
                    tau[method]["censored"] += 1
        results["attackers"][attacker] = {
            method: {
                "curves": {str(n): curves[method][n] for n in ns},
                "J": {str(n): float(np.mean(curves[method][n])) if curves[method][n] else None for n in ns},
                "N_tau": tau[method],
            }
            for method in methods
        }
        print(attacker, "J", {method: {n: round(results["attackers"][attacker][method]["J"][str(n)], 3) for n in ns} for method in methods}, flush=True)

    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
