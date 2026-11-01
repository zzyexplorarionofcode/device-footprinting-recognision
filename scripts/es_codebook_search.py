"""Evolution-strategy search for FADE trigger codes and positions (dimension 1).

Action per class (K classes):

    l1  in [60, 223]        integer bytes    -> l2 = 283 - l1
    g1  in [0.004, 0.446]   5 ms grid       -> g2 = 0.45 - g1
    pos in C(1..16, 2)      distinct pairs  (K pairs must be unique)

Equal byte/delay cost is structural: every candidate automatically satisfies the
283 B / 0.45 s trigger partition.  The search only needs to repair position
collisions.

Fitness is the *worst* balanced accuracy among two surrogate attackers, both
evaluated on a validation split that the search never trains on:

  * clean -> defended (next mapping): attacker trained on clean traffic
  * defended(old) -> defended(next): attacker retrained on defended traffic

Only the validation split is used for selection; the untouched test split is
evaluated once at the end.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline import pcap  # noqa: E402
from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402

TRIGGER_BYTES = 283.0
TRIGGER_DELAY = 0.45
L_BINS = np.arange(60, 224)
G_BINS = np.round(np.arange(0.004, 0.4461, 0.005), 6)
POS_PAIRS = [(a, b) for a in range(1, 17) for b in range(a + 1, 17)]
FEATURE = {"max_frame_length": 1514.0, "max_gap_seconds": 10.0}


def encode(config) -> np.ndarray:
    """Map a legal codebook to the continuous search vector."""

    vector: list[float] = []
    for code, position in zip(config.trigger_codes, config.trigger_positions):
        length = float(code[0][0])
        gap = float(code[0][1])
        vector.extend(
            [
                float(np.argmin(np.abs(L_BINS - length))),
                float(np.argmin(np.abs(G_BINS - gap))),
                float(POS_PAIRS.index(tuple(int(v) for v in position))),
            ]
        )
    return np.asarray(vector, dtype=float)


def decode(vector: np.ndarray, class_count: int) -> tuple[tuple, tuple]:
    """Map a search vector back to a legal (codes, positions) pair."""

    codes: list[tuple] = []
    positions: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for index in range(class_count):
        length_index = int(np.clip(np.rint(vector[3 * index]), 0, len(L_BINS) - 1))
        gap_index = int(np.clip(np.rint(vector[3 * index + 1]), 0, len(G_BINS) - 1))
        position_index = int(np.clip(np.rint(vector[3 * index + 2]), 0, len(POS_PAIRS) - 1))

        first_length = int(L_BINS[length_index])
        second_length = int(round(TRIGGER_BYTES - first_length))
        if not (60 <= first_length <= 223 and 60 <= second_length <= 223):
            raise AssertionError("decoded length violates the trigger partition")
        first_gap = float(G_BINS[gap_index])
        second_gap = round(TRIGGER_DELAY - first_gap, 6)
        if first_gap < 0.004 - 1e-9 or second_gap < 0.004 - 1e-9:
            raise AssertionError("decoded gap violates the trigger partition")

        if POS_PAIRS[position_index] in used:
            order = sorted(range(len(POS_PAIRS)), key=lambda i: abs(i - position_index))
            position_index = next(i for i in order if POS_PAIRS[i] not in used)
        pair = POS_PAIRS[position_index]
        used.add(pair)
        positions.append(pair)
        codes.append(((float(first_length), first_gap), (float(second_length), second_gap)))

    return tuple(codes), tuple(positions)


def features(application) -> np.ndarray:
    return pcap.make_features(
        application[0],
        application[1],
        application[2],
        application[3],
        **FEATURE,
    )


def defended(application, mapping: str, config, seed: int, jitter):
    return apply_method33(
        application[0],
        application[1],
        application[2],
        application[3],
        application[4],
        config,
        seed,
        mapping=mapping,
        jitter_config=jitter,
    )


class Context:
    """Fixed search context: clean features, labels, split arrays."""

    def __init__(self, dataset, project, train_indices, validation_indices, seed):
        self.method = project.method33
        self.jitter = project.trigger_jitter
        self.seed = int(seed)
        self.class_count = self.method.class_count

        def take(indices):
            return (
                np.asarray(dataset["lengths"])[indices],
                np.asarray(dataset["gaps"])[indices],
                np.asarray(dataset["directions"])[indices],
                np.asarray(dataset["mask"])[indices],
                np.asarray(dataset["labels"])[indices],
            )

        self.train = take(train_indices)
        self.validation = take(validation_indices)
        self.clean_train = features(self.train)
        self.clean_validation = features(self.validation)
        self.labels_train = self.train[4]
        self.labels_validation = self.validation[4]
        self.evaluations = 0

    def score_config(self, codes: tuple, positions: tuple) -> dict[str, float]:
        """Evaluate one legal codebook on the search validation split."""

        config = dataclasses.replace(
            self.method, trigger_codes=codes, trigger_positions=positions
        )
        train_defended = defended(self.train, "old", config, self.seed, self.jitter)
        validation_defended = defended(
            self.validation, "next", config, self.seed, self.jitter
        )
        x_train = features(train_defended)
        x_validation = features(validation_defended)

        scaler = StandardScaler().fit(self.clean_train)
        clean_model = LogisticRegression(max_iter=500).fit(
            scaler.transform(self.clean_train), self.labels_train
        )
        ba_clean = balanced_accuracy_score(
            self.labels_validation, clean_model.predict(scaler.transform(x_validation))
        )

        scaler_retrained = StandardScaler().fit(x_train)
        retrained_model = LogisticRegression(max_iter=500).fit(
            scaler_retrained.transform(x_train), self.labels_train
        )
        ba_retrained = balanced_accuracy_score(
            self.labels_validation,
            retrained_model.predict(scaler_retrained.transform(x_validation)),
        )
        self.evaluations += 1
        return {
            "fitness": float(max(ba_clean, ba_retrained)),
            "ba_clean": float(ba_clean),
            "ba_retrained": float(ba_retrained),
        }

    def score(self, vector: np.ndarray) -> dict[str, float]:
        codes, positions = decode(vector, self.class_count)
        return self.score_config(codes, positions)


def search(
    context: Context,
    *,
    iterations: int,
    population: int,
    elite: int,
    sigma: float,
    decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    mean = encode(context.method)
    history: list[dict[str, float]] = []

    best_vector = mean.copy()
    best = context.score_config(
        context.method.trigger_codes, context.method.trigger_positions
    )
    best["iteration"] = -1
    history.append(dict(best, global_best=best["fitness"]))
    print(
        f"[base] fitness={best['fitness']:.4f} clean={best['ba_clean']:.4f} "
        f"retrained={best['ba_retrained']:.4f}",
        flush=True,
    )

    for iteration in range(iterations):
        candidates = [
            np.clip(mean + sigma * rng.normal(size=mean.size), 0, None)
            for _ in range(population)
        ]
        scored = [(context.score(vector), vector) for vector in candidates]
        scored.sort(key=lambda item: item[0]["fitness"])
        if scored[0][0]["fitness"] < best["fitness"]:
            best = dict(scored[0][0])
            best_vector = scored[0][1].copy()
            best["iteration"] = iteration
        elite_vectors = [vector for _, vector in scored[:elite]]
        mean = np.mean(elite_vectors, axis=0)
        sigma *= decay
        history.append(
            {
                "iteration": iteration,
                "best_population": float(scored[0][0]["fitness"]),
                "mean_fitness": float(np.mean([item[0]["fitness"] for item in scored])),
                "global_best": float(best["fitness"]),
                "sigma": float(sigma),
            }
        )
        print(
            f"[iter {iteration:02d}] pop_best={scored[0][0]['fitness']:.4f} "
            f"global_best={best['fitness']:.4f} sigma={sigma:.3f}",
            flush=True,
        )
    return best_vector, best, history


def evaluate_holdout(
    dataset,
    project,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    codes: tuple,
    positions: tuple,
    seed: int,
) -> dict[str, float]:
    """One-shot evaluation on the untouched test split (no search)."""

    def take(indices):
        return (
            np.asarray(dataset["lengths"])[indices],
            np.asarray(dataset["gaps"])[indices],
            np.asarray(dataset["directions"])[indices],
            np.asarray(dataset["mask"])[indices],
            np.asarray(dataset["labels"])[indices],
        )

    train = take(train_indices)
    test = take(test_indices)
    config = dataclasses.replace(
        project.method33, trigger_codes=codes, trigger_positions=positions
    )
    train_defended = defended(train, "old", config, seed, project.trigger_jitter)
    test_defended = defended(test, "next", config, seed, project.trigger_jitter)

    clean_train = features(train)
    clean_test = features(test_defended)
    scaler = StandardScaler().fit(clean_train)
    clean_model = LogisticRegression(max_iter=500).fit(
        scaler.transform(clean_train), train[4]
    )
    ba_clean = balanced_accuracy_score(test[4], clean_model.predict(scaler.transform(clean_test)))

    x_train = features(train_defended)
    x_test = features(test_defended)
    scaler_retrained = StandardScaler().fit(x_train)
    retrained_model = LogisticRegression(max_iter=500).fit(
        scaler_retrained.transform(x_train), train[4]
    )
    ba_retrained = balanced_accuracy_score(
        test[4], retrained_model.predict(scaler_retrained.transform(x_test))
    )
    return {
        "fitness": float(max(ba_clean, ba_retrained)),
        "ba_clean": float(ba_clean),
        "ba_retrained": float(ba_retrained),
    }


def write_config(project, vector: np.ndarray, class_count: int, output: Path) -> dict:
    codes, positions = decode(vector, class_count)
    source = Path(project.source_path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    payload["method33"]["trigger_codes"] = [
        [[float(length), float(gap)] for length, gap in code] for code in codes
    ]
    payload["method33"]["trigger_positions"] = [[int(a), int(b)] for a, b in positions]
    payload["method33"]["trigger_codebook_id"] = (
        str(payload["method33"].get("trigger_codebook_id", "codebook")) + "_es_v1"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True, help="split_indices.npz from a run")
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=8.0)
    parser.add_argument("--decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--max-validation", type=int, default=2000)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    train_indices = np.asarray(split["train"])
    test_indices = np.asarray(split["test"]) if "test" in split else np.asarray([], dtype=int)

    # Split the runner's train partition again (by capture) to obtain a search
    # validation set; the runner's test partition stays untouched.
    original_train = np.asarray(split["train"])
    inner_train, inner_validation = pcap.split_by_capture(
        labels[original_train], groups[original_train], train_ratio=0.75, seed=731100
    )
    train_indices = original_train[inner_train]
    validation_indices = original_train[inner_validation]

    def cap(indices: np.ndarray, limit: int) -> np.ndarray:
        if limit <= 0 or len(indices) <= limit:
            return indices
        rng = np.random.default_rng(args.seed)
        chosen: list[int] = []
        for label in np.unique(labels[indices]):
            pool = indices[labels[indices] == label]
            take = max(1, int(round(limit * len(pool) / len(indices))))
            chosen.extend(rng.choice(pool, size=min(take, len(pool)), replace=False).tolist())
        return np.asarray(sorted(chosen), dtype=int)

    train_indices = cap(train_indices, args.max_train)
    validation_indices = cap(validation_indices, args.max_validation)

    project = method33_tools.load_config(args.config)
    context = Context(dataset, project, train_indices, validation_indices, seed=731800)
    print(
        f"search-train={len(train_indices)} search-val={len(validation_indices)} "
        f"classes={context.class_count}",
        flush=True,
    )

    started = time.time()
    best_vector, best_score, history = search(
        context,
        iterations=args.iterations,
        population=args.population,
        elite=args.elite,
        sigma=args.sigma,
        decay=args.decay,
        seed=args.seed,
    )
    elapsed = time.time() - started
    write_config(project, best_vector, context.class_count, args.output_config)

    base_score = context.score_config(
        project.method33.trigger_codes, project.method33.trigger_positions
    )
    best_codes, best_positions = decode(best_vector, context.class_count)
    holdout_base = evaluate_holdout(
        dataset,
        project,
        np.asarray(split["train"]),
        test_indices,
        project.method33.trigger_codes,
        project.method33.trigger_positions,
        seed=731800,
    )
    holdout_best = evaluate_holdout(
        dataset,
        project,
        np.asarray(split["train"]),
        test_indices,
        best_codes,
        best_positions,
        seed=731800,
    )
    log = {
        "dataset": str(args.dataset),
        "config": str(args.config),
        "output_config": str(args.output_config),
        "seed": args.seed,
        "iterations": args.iterations,
        "population": args.population,
        "evaluations": context.evaluations,
        "seconds": round(elapsed, 1),
        "base": base_score,
        "best": {
            "fitness": float(best_score["fitness"]),
            "ba_clean": float(best_score["ba_clean"]),
            "ba_retrained": float(best_score["ba_retrained"]),
            "vector": best_vector.tolist(),
            "codes_positions": [
                {
                    "code": [[float(a), float(b)] for a, b in code],
                    "position": [int(p) for p in position],
                }
                for code, position in zip(*decode(best_vector, context.class_count))
            ],
        },
        "history": history,
        "holdout": {"base": holdout_base, "best": holdout_best},
        "test_indices": int(len(test_indices)),
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: log[k] for k in ("evaluations", "seconds", "base", "best")}, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
