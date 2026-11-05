"""Paired UNSW mapping-policy evaluation with device-label-only adaptation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from methods.method33 import apply_method33
from methods.method33_tools import load_config
from scripts.fold_review import take
from scripts.perflow_strong_attack import canonical
from scripts.position_placebo import features
from scripts.streaming_arms import make_mappings


def render(data, ids, states, mappings, project):
    x = np.empty((len(ids), 64))
    cost = np.empty((len(ids), 3))
    for state in np.unique(states):
        rows = np.flatnonzero(states == state)
        lengths, gaps, directions, mask, labels = take(data, ids[rows])
        app = apply_method33(
            np.minimum(lengths, 1514), gaps, directions, mask, labels,
            project.method33, 731800, mapping=mappings[state],
            jitter_config=project.trigger_jitter,
        )
        x[rows] = features(app)
        cost[rows] = [
            [row[key] for key in ("total_added_bytes", "total_added_delay", "dummy_packets")]
            for row in app[-1]
        ]
    return x, cost


def candidate_features(x, mappings, project):
    return np.stack([
        np.stack([canonical(row, state, mappings, project) for row in x])
        for state in range(len(mappings))
    ], axis=1)


def latent_gate(history, blind):
    """Fit state mixture weights using device likelihood only."""
    x = np.concatenate([item[0] for item in history])
    likelihood = np.concatenate([item[1] for item in history])
    scaler = StandardScaler().fit(x)
    design = np.column_stack([scaler.transform(x), np.ones(len(x))])
    test = np.column_stack([scaler.transform(blind), np.ones(len(blind))])

    def objective(flat):
        weights = flat.reshape(design.shape[1], likelihood.shape[1])
        prior = softmax(design @ weights, axis=1)
        joint = prior * likelihood
        evidence = joint.sum(axis=1, keepdims=True)
        loss = -np.log(evidence).mean() + 0.05 * np.sum(weights**2)
        gradient = design.T @ (prior - joint / evidence) / len(design) + 0.1 * weights
        return loss, gradient.ravel()

    fit = minimize(objective, np.zeros(design.shape[1] * likelihood.shape[1]),
                   method="L-BFGS-B", jac=True)
    if not fit.success:
        raise RuntimeError(fit.message)
    return softmax(test @ fit.x.reshape(design.shape[1], -1), axis=1)


def summarize(scores):
    no, adaptive, oracle = (scores[key] for key in ("no_canon", "adaptive", "oracle"))
    return dict(scores, gain=adaptive - no, oracle_gap=oracle - adaptive,
                recovery=(adaptive - no) / (oracle - no) if oracle > no else None)


def support(y):
    classes, counts = np.unique(y, return_counts=True)
    return dict(zip(map(str, classes), map(int, counts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with np.load(args.dataset, allow_pickle=False) as source:
        data = {key: source[key] for key in ("lengths", "gaps", "directions", "mask", "labels", "groups")}
    project = load_config(args.config)
    mappings = make_mappings(project, args.seed)
    labels = data["labels"]
    groups = data["groups"].astype(str)
    days = np.unique(groups)
    state_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 1]))
    fixed = int(state_rng.integers(len(mappings)))
    phase = int(state_rng.integers(len(mappings)))
    perflow = state_rng.integers(len(mappings), size=len(labels))

    def sample(window, cap, stream):
        ids = np.flatnonzero(np.isin(groups, window))
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 2, stream]))
        return np.sort(rng.choice(ids, min(cap, len(ids)), replace=False))

    train = sample(days[:20], 12000, 0)
    ref, _ = render(data, train, np.zeros(len(train), dtype=int), mappings, project)
    ref = candidate_features(ref, mappings[:1], project)[:, 0]
    scaler = StandardScaler().fit(ref)
    clf = LogisticRegression(max_iter=500).fit(scaler.transform(ref), labels[train])

    def predict(x):
        shape = x.shape[:-1] + (len(clf.classes_),)
        return clf.predict_proba(scaler.transform(x.reshape(-1, 64))).reshape(shape)

    history = {policy: [] for policy in ("Fixed", "Periodic", "Per-flow")}
    blocks = []
    for block, start in enumerate(range(20, len(days), 20), start=1):
        window = days[start:start + 20]
        if len(window) < 2:
            continue
        half = len(window) // 2
        cal = sample(window[:half], 1024, 2 * block)
        blind = sample(window[half:], 1000, 2 * block + 1)
        ids = np.r_[cal, blind]
        n = len(cal)
        y_cal = labels[cal]
        row = {
            "block": block, "cal_days": window[:half].tolist(), "blind_days": window[half:].tolist(),
            "cal_indices": cal.tolist(), "blind_indices": blind.tolist(),
            "cal_support": support(y_cal), "blind_support": support(labels[blind]), "policies": {},
        }
        state_vectors = {
            "Fixed": np.full(len(ids), fixed),
            "Periodic": np.full(len(ids), (phase + (block - 1) // 2) % len(mappings)),
            "Per-flow": perflow[ids],
        }
        for policy, states in state_vectors.items():
            x, cost = render(data, ids, states, mappings, project)
            candidates = predict(candidate_features(x, mappings, project))
            known = np.flatnonzero(np.isin(y_cal, clf.classes_))
            likelihood = candidates[known, :, np.searchsorted(clf.classes_, y_cal[known])]
            likelihood = np.clip(likelihood, 1e-15, 1.0)
            history[policy].append((x[known], likelihood))
            if policy == "Per-flow":
                weights = latent_gate(history[policy], x[n:])
                budget = sum(len(item[0]) for item in history[policy])
            else:
                count = block if policy == "Fixed" else 1 + (block - 1) % 2
                eligible = history[policy][-count:]
                score = sum(np.log(item[1]).sum(axis=0) for item in eligible)
                weights = np.broadcast_to(np.eye(len(mappings))[np.argmax(score)], (len(blind), len(mappings)))
                budget = sum(len(item[0]) for item in eligible)
            state = states[n:]
            probs = {
                "no_canon": predict(x[n:]),
                "adaptive": (candidates[n:] * weights[:, :, None]).sum(axis=1),
                "oracle": candidates[np.arange(n, len(ids)), state],
            }
            truth = labels[blind]
            recalls = {}
            for arm, proba in probs.items():
                predicted = clf.classes_[proba.argmax(axis=1)]
                recalls[arm] = {str(c): float((predicted[truth == c] == c).mean()) for c in np.unique(truth)}
            scores = {arm: float(np.mean(list(values.values()))) for arm, values in recalls.items()}
            row["policies"][policy] = dict(
                summarize(scores), recalls=recalls, device_label_budget=budget, state_label_budget=0, current_labels_received=n, current_labels_usable=len(known),
                state_top1=float((weights.argmax(axis=1) == state).mean()),
                state_top2=float((np.argsort(-weights, axis=1, kind="stable")[:, :2] == state[:, None]).any(axis=1).mean())
                if policy == "Per-flow" else None,
                state_support=support(state), cost_mean=cost[n:].mean(axis=0).tolist(),
                cost_max=cost[n:].max(axis=0).tolist(),
            )
        blocks.append(row)
        print("Completed block", block, flush=True)
    summary = {
        policy: summarize({
            arm: float(np.mean([row["policies"][policy][arm] for row in blocks]))
            for arm in ("no_canon", "adaptive", "oracle")
        }) for policy in history
    }
    with args.dataset.open("rb") as handle:
        dataset_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    result = dict(seed=args.seed, mappings=[[int(c) for c in mapping] for mapping in mappings],
                  fixed_state=fixed, periodic_phase=phase,
                  dataset_sha256=dataset_sha256,
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  reference_indices=train.tolist(), reference_support=support(labels[train]),
                  cost_order=["bytes", "seconds", "dummy_packets"], blocks=blocks, summary=summary)
    args.output_log.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()

