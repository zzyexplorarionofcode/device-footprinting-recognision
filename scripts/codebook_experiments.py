"""Codebook experiments with day-grouped folds, negative controls and cross-surrogates.

This is the experimental framework requested for dimension 1:

  * ``--folds K`` splits *capture days* into K contiguous blocks.  Every fold uses
    one block as validation and the remaining days as training, so the paired
    delta ``BA_base,k - BA_candidate,k`` is computed without mixing days.
  * coordinate search over ``position`` / ``length`` / ``gap`` / ``joint``,
    scored by the mean validation BA across folds.
  * negative controls: N legal random position assignments (same codes) and N
    legal random codebooks (random lengths, gaps and positions).
  * the base geometry codebook and the top searched candidates are re-scored
    with LR, RF and MLP on validation, and once on the untouched holdout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline import pcap  # noqa: E402
from methods import method33_tools  # noqa: E402
from scripts.es_codebook_search import (  # noqa: E402
    G_BINS,
    L_BINS,
    POS_PAIRS,
    TRIGGER_BYTES,
    TRIGGER_DELAY,
    defended,
    features,
)

LENGTH_DELTAS = (-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32)
GAP_DELTAS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)
GAP_STEP = 0.005
SURROGATES = ("lr", "rf", "mlp")


def fit_score(x_train, y_train, x_test, y_test, surrogate: str) -> float:
    if surrogate == "lr":
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=300).fit(scaler.transform(x_train), y_train)
        prediction = model.predict(scaler.transform(x_test))
    elif surrogate == "rf":
        model = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=0)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
    elif surrogate == "mlp":
        scaler = StandardScaler().fit(x_train)
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=0)
        model.fit(scaler.transform(x_train), y_train)
        prediction = model.predict(scaler.transform(x_test))
    else:
        raise ValueError(f"unknown surrogate: {surrogate}")
    return float(balanced_accuracy_score(y_test, prediction))


def take(dataset, indices: np.ndarray):
    return (
        np.asarray(dataset["lengths"])[indices],
        np.asarray(dataset["gaps"])[indices],
        np.asarray(dataset["directions"])[indices],
        np.asarray(dataset["mask"])[indices],
        np.asarray(dataset["labels"])[indices],
    )


def stratified_cap(indices: np.ndarray, labels: np.ndarray, limit: int, rng) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    chosen: list[int] = []
    for label in np.unique(labels[indices]):
        pool = indices[labels[indices] == label]
        take_count = max(1, int(round(limit * len(pool) / len(indices))))
        chosen.extend(
            rng.choice(pool, size=min(take_count, len(pool)), replace=False).tolist()
        )
    return np.asarray(sorted(chosen), dtype=int)


def build_folds(
    pool: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fold_count: int,
    max_train: int,
    max_validation: int,
    seed: int,
):
    days = np.sort(np.unique(groups[pool]))
    blocks = np.array_split(days, fold_count)
    rng = np.random.default_rng(seed)
    folds = []
    for block in blocks:
        chosen_days = set(block.tolist())
        validation_mask = np.asarray([group in chosen_days for group in groups[pool]])
        train_indices = pool[~validation_mask]
        validation_indices = pool[validation_mask]
        train_indices = stratified_cap(train_indices, labels, max_train, rng)
        validation_indices = stratified_cap(validation_indices, labels, max_validation, rng)
        folds.append((train_indices, validation_indices, [str(day) for day in block]))
    return folds


class FoldEvaluator:
    def __init__(self, dataset, project, folds, seed: int = 731800):
        self.project = project
        self.method = project.method33
        self.jitter = project.trigger_jitter
        self.seed = int(seed)
        self.folds = []
        for train_indices, validation_indices, days in folds:
            train = take(dataset, train_indices)
            validation = take(dataset, validation_indices)
            self.folds.append(
                {
                    "days": days,
                    "train": train,
                    "validation": validation,
                    "clean_train": features(train),
                    "clean_validation": features(validation),
                }
            )
        self.evaluations = 0

    def score(self, codes: tuple, positions: tuple, surrogate: str = "lr") -> dict:
        config = dataclasses.replace(
            self.method, trigger_codes=codes, trigger_positions=positions
        )
        per_fold = []
        for fold in self.folds:
            train = fold["train"]
            validation = fold["validation"]
            train_defended = defended(train, "old", config, self.seed, self.jitter)
            validation_defended = defended(
                validation, "next", config, self.seed, self.jitter
            )
            x_train = features(train_defended)
            x_validation = features(validation_defended)
            ba_clean = fit_score(
                fold["clean_train"],
                train[4],
                x_validation,
                validation[4],
                surrogate,
            )
            ba_retrained = fit_score(x_train, train[4], x_validation, validation[4], surrogate)
            per_fold.append(
                {
                    "fitness": float(max(ba_clean, ba_retrained)),
                    "ba_clean": float(ba_clean),
                    "ba_retrained": float(ba_retrained),
                }
            )
        self.evaluations += 1
        fitness = np.asarray([item["fitness"] for item in per_fold], dtype=float)
        return {
            "folds": per_fold,
            "mean": float(fitness.mean()),
            "se": float(fitness.std(ddof=1) / np.sqrt(len(fitness))) if len(fitness) > 1 else 0.0,
        }


def random_positions(rng, class_count: int) -> tuple:
    chosen = rng.choice(len(POS_PAIRS), size=class_count, replace=False)
    return tuple(POS_PAIRS[int(index)] for index in chosen)


def random_codebook(rng, class_count: int) -> tuple:
    codes = []
    for _ in range(class_count):
        first_length = int(rng.integers(60, 224))
        first_gap = float(G_BINS[int(rng.integers(0, len(G_BINS)))])
        codes.append(
            (
                (float(first_length), first_gap),
                (float(TRIGGER_BYTES - first_length), round(TRIGGER_DELAY - first_gap, 6)),
            )
        )
    return tuple(codes), random_positions(rng, class_count)


def random_length_variant(rng, base_codes, base_positions) -> tuple:
    """Random first-packet lengths, everything else from the base codebook."""

    codes = []
    for code in base_codes:
        first_length = int(rng.integers(60, 224))
        first_gap = float(code[0][1])
        second_gap = float(code[1][1])
        codes.append(
            (
                (float(first_length), first_gap),
                (float(TRIGGER_BYTES - first_length), second_gap),
            )
        )
    return tuple(codes), tuple(base_positions)


def random_gap_variant(rng, base_codes, base_positions) -> tuple:
    """Random first-packet gaps, everything else from the base codebook."""

    codes = []
    for code in base_codes:
        first_length = float(code[0][0])
        second_length = float(code[1][0])
        first_gap = float(G_BINS[int(rng.integers(0, len(G_BINS)))])
        codes.append(
            (
                (first_length, first_gap),
                (second_length, round(TRIGGER_DELAY - first_gap, 6)),
            )
        )
    return tuple(codes), tuple(base_positions)


def coordinate_search(
    evaluator: FoldEvaluator,
    mode: str,
    sweeps: int,
    position_limit: int,
    min_delta: float,
):
    codes = [tuple(tuple(float(v) for v in packet) for packet in code) for code in evaluator.method.trigger_codes]
    positions = [tuple(int(v) for v in pair) for pair in evaluator.method.trigger_positions]
    best = evaluator.score(tuple(codes), tuple(positions))
    accepted_states = []
    history = [{"sweep": -1, "mean": best["mean"], "accepted": 0}]
    dimensions = ["position", "length", "gap"] if mode == "joint" else [mode]
    print(
        f"[base] mean={best['mean']:.4f} se={best['se']:.4f} "
        f"folds={[round(item['fitness'], 4) for item in best['folds']]}",
        flush=True,
    )

    for sweep in range(sweeps):
        accepted = 0
        for dimension in dimensions:
            for label in range(len(codes)):
                if dimension == "position":
                    current = POS_PAIRS.index(positions[label])
                    used = set(positions) - {positions[label]}
                    order = sorted(range(len(POS_PAIRS)), key=lambda index: abs(index - current))
                    if position_limit > 0:
                        order = order[:position_limit]
                    candidates = [
                        POS_PAIRS[index] for index in order if POS_PAIRS[index] not in used
                    ]
                    for pair in candidates:
                        trial_positions = list(positions)
                        trial_positions[label] = pair
                        score = evaluator.score(tuple(codes), tuple(trial_positions))
                        if score["mean"] < best["mean"] - min_delta:
                            positions = trial_positions
                            best = score
                            accepted += 1
                            accepted_states.append(
                                {"codes": tuple(codes), "positions": tuple(positions), "score": best}
                            )
                elif dimension == "length":
                    first_length = int(codes[label][0][0])
                    first_gap = float(codes[label][0][1])
                    second_gap = float(codes[label][1][1])
                    for delta in LENGTH_DELTAS:
                        new_length = int(np.clip(first_length + delta, 60, 223))
                        trial_codes = list(codes)
                        trial_codes[label] = (
                            (float(new_length), first_gap),
                            (float(TRIGGER_BYTES - new_length), second_gap),
                        )
                        score = evaluator.score(tuple(trial_codes), tuple(positions))
                        if score["mean"] < best["mean"] - min_delta:
                            codes = trial_codes
                            best = score
                            accepted += 1
                            accepted_states.append(
                                {"codes": tuple(codes), "positions": tuple(positions), "score": best}
                            )
                else:  # gap
                    first_length = float(codes[label][0][0])
                    second_length = float(codes[label][1][0])
                    first_gap = float(codes[label][0][1])
                    for delta in GAP_DELTAS:
                        new_gap = round(float(np.clip(first_gap + delta * GAP_STEP, 0.004, 0.446)), 6)
                        trial_codes = list(codes)
                        trial_codes[label] = (
                            (first_length, new_gap),
                            (second_length, round(TRIGGER_DELAY - new_gap, 6)),
                        )
                        score = evaluator.score(tuple(trial_codes), tuple(positions))
                        if score["mean"] < best["mean"] - min_delta:
                            codes = trial_codes
                            best = score
                            accepted += 1
                            accepted_states.append(
                                {"codes": tuple(codes), "positions": tuple(positions), "score": best}
                            )
        history.append({"sweep": sweep, "mean": best["mean"], "accepted": accepted})
        print(f"[sweep {sweep:02d}] mean={best['mean']:.4f} accepted={accepted}", flush=True)
        if accepted == 0:
            break
    return tuple(codes), tuple(positions), best, history, accepted_states


def code_diff(base_codes, base_positions, codes, positions) -> dict:
    length_changes = []
    gap_changes = []
    position_changes = []
    for index, (base_code, code) in enumerate(zip(base_codes, codes)):
        if (float(base_code[0][0]), float(base_code[0][1])) != (float(code[0][0]), float(code[0][1])):
            length_changes.append(
                {
                    "class": index,
                    "from": float(base_code[0][0]),
                    "to": float(code[0][0]),
                }
            )
            gap_changes.append(
                {
                    "class": index,
                    "from": float(base_code[0][1]),
                    "to": float(code[0][1]),
                }
            )
    for index, (base_position, position) in enumerate(zip(base_positions, positions)):
        if tuple(int(v) for v in base_position) != tuple(int(v) for v in position):
            position_changes.append(
                {
                    "class": index,
                    "from": [int(v) for v in base_position],
                    "to": [int(v) for v in position],
                }
            )
    return {
        "length_changes": length_changes,
        "gap_changes": gap_changes,
        "position_changes": position_changes,
    }


def write_config(project, codes, positions, output: Path) -> None:
    payload = json.loads(Path(project.source_path).read_text(encoding="utf-8-sig"))
    payload["method33"]["trigger_codes"] = [
        [[float(length), float(gap)] for length, gap in code] for code in codes
    ]
    payload["method33"]["trigger_positions"] = [[int(a), int(b)] for a, b in positions]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--mode", choices=("position", "length", "gap", "joint"), default="position")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--position-limit", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-validation", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--skip-cross", action="store_true")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    pool = np.asarray(split["train"])
    if "validation" in split and len(split["validation"]) > 0:
        pool = np.concatenate([pool, np.asarray(split["validation"])])
    test_indices = np.asarray(split["test"]) if "test" in split else np.asarray([], dtype=int)

    folds = build_folds(
        pool, labels, groups, args.folds, args.max_train, args.max_validation, args.seed
    )
    evaluator = FoldEvaluator(dataset, method33_tools.load_config(args.config), folds)
    print(
        f"mode={args.mode} folds={args.folds} "
        f"days={[len(fold[2]) for fold in folds]} "
        f"train/val={[(len(fold[0]), len(fold[1])) for fold in folds]}",
        flush=True,
    )

    started = time.time()
    base = evaluator.score(
        evaluator.method.trigger_codes, evaluator.method.trigger_positions
    )
    print(f"[base] mean={base['mean']:.4f} se={base['se']:.4f}", flush=True)

    searched_codes, searched_positions, searched, history, accepted_states = coordinate_search(
        evaluator, args.mode, args.sweeps, args.position_limit, args.min_delta
    )

    rng = np.random.default_rng(args.seed + 1)
    control_positions = []
    for _ in range(args.controls):
        positions = random_positions(rng, len(evaluator.method.trigger_codes))
        score = evaluator.score(evaluator.method.trigger_codes, positions)
        control_positions.append({"positions": [list(map(int, p)) for p in positions], "score": score})
    control_codebooks = []
    for _ in range(args.controls):
        codes, positions = random_codebook(rng, len(evaluator.method.trigger_codes))
        score = evaluator.score(codes, positions)
        control_codebooks.append(
            {
                "codes": [[[float(a), float(b)] for a, b in code] for code in codes],
                "positions": [list(map(int, p)) for p in positions],
                "score": score,
            }
        )
    dimension_controls = []
    if args.mode in ("length", "gap"):
        generator = random_length_variant if args.mode == "length" else random_gap_variant
        for _ in range(args.controls):
            codes, positions = generator(
                rng, evaluator.method.trigger_codes, evaluator.method.trigger_positions
            )
            score = evaluator.score(codes, positions)
            dimension_controls.append(
                {
                    "codes": [[[float(a), float(b)] for a, b in code] for code in codes],
                    "positions": [list(map(int, p)) for p in positions],
                    "score": score,
                }
            )

    cross = {}
    if not args.skip_cross:
        candidates = [
            {"name": "base", "codes": evaluator.method.trigger_codes, "positions": evaluator.method.trigger_positions},
            {"name": "searched", "codes": searched_codes, "positions": searched_positions},
        ]
        accepted_sorted = sorted(accepted_states, key=lambda item: item["score"]["mean"])[:3]
        for index, state in enumerate(accepted_sorted):
            candidates.append(
                {"name": f"accepted_{index}", "codes": state["codes"], "positions": state["positions"]}
            )
        for candidate in candidates:
            cross[candidate["name"]] = {
                surrogate: evaluator.score(candidate["codes"], candidate["positions"], surrogate)
                for surrogate in SURROGATES
            }
            print(
                f"[cross {candidate['name']}] "
                + " ".join(
                    f"{surrogate}={cross[candidate['name']][surrogate]['mean']:.4f}"
                    for surrogate in SURROGATES
                ),
                flush=True,
            )

    # One-shot holdout: train on the full pool, test on the untouched split.
    holdout_rng = np.random.default_rng(args.seed + 2)
    holdout_train = stratified_cap(pool, labels, args.max_train * 2, holdout_rng)
    holdout_test = stratified_cap(test_indices, labels, args.max_validation * 2, holdout_rng)
    holdout_evaluator = FoldEvaluator(
        dataset,
        method33_tools.load_config(args.config),
        [(holdout_train, holdout_test, ["holdout"])],
    )
    holdout = {}
    for name, codes, positions in (
        ("base", evaluator.method.trigger_codes, evaluator.method.trigger_positions),
        ("searched", searched_codes, searched_positions),
    ):
        holdout[name] = {
            surrogate: holdout_evaluator.score(codes, positions, surrogate)
            for surrogate in SURROGATES
        }

    delta = np.asarray(
        [base["folds"][index]["fitness"] - searched["folds"][index]["fitness"] for index in range(len(base["folds"]))]
    )
    log = {
        "dataset": str(args.dataset),
        "config": str(args.config),
        "mode": args.mode,
        "folds": args.folds,
        "fold_days": [fold[2] for fold in folds],
        "folds_improved": int((delta > 0).sum()),
        "delta_mean": float(delta.mean()),
        "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0,
        "delta_per_fold": delta.tolist(),
        "base": base,
        "searched": searched,
        "history": history,
        "diff": code_diff(
            evaluator.method.trigger_codes,
            evaluator.method.trigger_positions,
            searched_codes,
            searched_positions,
        ),
        "random_positions": [
            {"mean": item["score"]["mean"], "folds": [fold["fitness"] for fold in item["score"]["folds"]]}
            for item in control_positions
        ],
        "random_codebooks": [
            {"mean": item["score"]["mean"], "folds": [fold["fitness"] for fold in item["score"]["folds"]]}
            for item in control_codebooks
        ],
        "random_dimension": [
            {"mean": item["score"]["mean"], "folds": [fold["fitness"] for fold in item["score"]["folds"]]}
            for item in dimension_controls
        ],
        "cross": cross,
        "holdout": holdout,
        "evaluations": evaluator.evaluations + holdout_evaluator.evaluations,
        "seconds": round(time.time() - started, 1),
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_config(evaluator.project, searched_codes, searched_positions, args.output_config)

    def summary(items):
        values = np.asarray([item["mean"] for item in items])
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    print(
        json.dumps(
            {
                "base": round(base["mean"], 4),
                "searched": round(searched["mean"], 4),
                "delta_mean": round(float(delta.mean()), 4),
                "delta_se": round(log["delta_se"], 4),
                "folds_improved": log["folds_improved"],
                "random_positions": summary(log["random_positions"]),
                "random_codebooks": summary(log["random_codebooks"]),
                "random_dimension": summary(log["random_dimension"]) if log["random_dimension"] else None,
                "evaluations": log["evaluations"],
                "seconds": log["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
