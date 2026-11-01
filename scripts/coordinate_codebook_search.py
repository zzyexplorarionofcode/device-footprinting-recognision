"""Coordinate-descent search over FADE code/position dimensions.

Modes (``--mode``):

    position    fix lengths/gaps, try alternative insertion-position pairs
    length      fix gaps/positions, try alternative first-packet lengths
    gap         fix lengths/positions, try alternative first-packet gaps
    coordinate  position -> length -> gap sweeps until no improvement

Each candidate still satisfies the equal-cost partition (283 B / 0.45 s), so only
positions need a uniqueness repair (candidates exclude pairs already in use).

The search uses the same surrogate protocol as ``es_codebook_search``:
validation-only selection, clean->next and old->next attackers, one holdout
evaluation at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.es_codebook_search import (
    G_BINS,
    L_BINS,
    POS_PAIRS,
    TRIGGER_BYTES,
    TRIGGER_DELAY,
    Context,
    evaluate_holdout,
    write_config,
)

from data_pipeline import pcap
from methods import method33_tools

LENGTH_DELTAS = (-32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32)
GAP_STEP = 0.005
GAP_DELTAS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)


def encode_state(codes, positions) -> np.ndarray:
    vector: list[float] = []
    for code, position in zip(codes, positions):
        vector.extend(
            [
                float(np.argmin(np.abs(L_BINS - float(code[0][0])))),
                float(np.argmin(np.abs(G_BINS - float(code[0][1])))),
                float(POS_PAIRS.index(tuple(int(v) for v in position))),
            ]
        )
    return np.asarray(vector, dtype=float)


def _copy(codes, positions):
    return (
        [tuple(tuple(float(v) for v in packet) for packet in code) for code in codes],
        [tuple(int(v) for v in pair) for pair in positions],
    )


def coordinate_search(context, mode: str, sweeps: int, position_limit: int, min_delta: float):
    codes, positions = _copy(context.method.trigger_codes, context.method.trigger_positions)
    best = context.score_config(tuple(codes), tuple(positions))
    history = [
        {
            "sweep": -1,
            "fitness": best["fitness"],
            "ba_clean": best["ba_clean"],
            "ba_retrained": best["ba_retrained"],
            "accepted": 0,
        }
    ]
    dimensions = ["position", "length", "gap"] if mode == "coordinate" else [mode]
    print(
        f"[base] fitness={best['fitness']:.4f} clean={best['ba_clean']:.4f} "
        f"retrained={best['ba_retrained']:.4f}",
        flush=True,
    )

    for sweep in range(sweeps):
        accepted = 0
        for dimension in dimensions:
            for label in range(context.class_count):
                if dimension == "position":
                    current = POS_PAIRS.index(positions[label])
                    used = set(positions) - {positions[label]}
                    order = sorted(range(len(POS_PAIRS)), key=lambda i: abs(i - current))
                    if position_limit > 0:
                        order = order[:position_limit]
                    candidates = [
                        POS_PAIRS[i] for i in order if POS_PAIRS[i] not in used
                    ]
                    for pair in candidates:
                        new_positions = list(positions)
                        new_positions[label] = pair
                        score = context.score_config(tuple(codes), tuple(new_positions))
                        if score["fitness"] < best["fitness"] - min_delta:
                            positions = new_positions
                            best = score
                            accepted += 1
                elif dimension == "length":
                    first_length = int(codes[label][0][0])
                    first_gap = float(codes[label][0][1])
                    second_gap = float(codes[label][1][1])
                    for delta in LENGTH_DELTAS:
                        new_length = int(np.clip(first_length + delta, 60, 223))
                        new_codes = list(codes)
                        new_codes[label] = (
                            (float(new_length), first_gap),
                            (float(TRIGGER_BYTES - new_length), second_gap),
                        )
                        score = context.score_config(tuple(new_codes), tuple(positions))
                        if score["fitness"] < best["fitness"] - min_delta:
                            codes = new_codes
                            best = score
                            accepted += 1
                else:  # gap
                    first_length = float(codes[label][0][0])
                    second_length = float(codes[label][1][0])
                    first_gap = float(codes[label][0][1])
                    for delta in GAP_DELTAS:
                        new_gap = round(float(np.clip(first_gap + delta * GAP_STEP, 0.004, 0.446)), 6)
                        new_codes = list(codes)
                        new_codes[label] = (
                            (first_length, new_gap),
                            (second_length, round(TRIGGER_DELAY - new_gap, 6)),
                        )
                        score = context.score_config(tuple(new_codes), tuple(positions))
                        if score["fitness"] < best["fitness"] - min_delta:
                            codes = new_codes
                            best = score
                            accepted += 1
        history.append(
            {
                "sweep": sweep,
                "fitness": best["fitness"],
                "ba_clean": best["ba_clean"],
                "ba_retrained": best["ba_retrained"],
                "accepted": accepted,
            }
        )
        print(
            f"[sweep {sweep:02d}] fitness={best['fitness']:.4f} "
            f"clean={best['ba_clean']:.4f} retrained={best['ba_retrained']:.4f} "
            f"accepted={accepted}",
            flush=True,
        )
        if accepted == 0:
            break
    return codes, positions, best, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--mode", choices=("position", "length", "gap", "coordinate"), default="position")
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--position-limit", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--max-validation", type=int, default=2000)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    split = np.load(args.split_indices)
    train_pool = np.asarray(split["train"])
    test_pool = np.asarray(split["test"]) if "test" in split else np.asarray([], dtype=int)

    if "validation" in split and len(split["validation"]) > 0:
        # Frozen datasets already provide a validation partition.
        train_indices = train_pool
        validation_indices = np.asarray(split["validation"])
    else:
        inner_train, inner_validation = pcap.split_by_capture(
            labels[train_pool], groups[train_pool], train_ratio=0.75, seed=731100
        )
        train_indices = train_pool[inner_train]
        validation_indices = train_pool[inner_validation]

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
        f"mode={args.mode} train={len(train_indices)} validation={len(validation_indices)} "
        f"classes={context.class_count}",
        flush=True,
    )

    started = time.time()
    codes, positions, best, history = coordinate_search(
        context, args.mode, args.sweeps, args.position_limit, args.min_delta
    )
    elapsed = time.time() - started
    vector = encode_state(codes, positions)
    write_config(project, vector, context.class_count, args.output_config)

    base = context.score_config(
        project.method33.trigger_codes, project.method33.trigger_positions
    )
    holdout_base = evaluate_holdout(
        dataset, project, np.asarray(split["train"]), test_pool,
        project.method33.trigger_codes, project.method33.trigger_positions, seed=731800,
    )
    holdout_best = evaluate_holdout(
        dataset, project, np.asarray(split["train"]), test_pool,
        tuple(codes), tuple(positions), seed=731800,
    )
    log = {
        "dataset": str(args.dataset),
        "config": str(args.config),
        "mode": args.mode,
        "sweeps": args.sweeps,
        "position_limit": args.position_limit,
        "min_delta": args.min_delta,
        "evaluations": context.evaluations,
        "seconds": round(elapsed, 1),
        "base": base,
        "best": best,
        "holdout": {"base": holdout_base, "best": holdout_best},
        "history": history,
        "codes_positions": [
            {"code": [[float(a), float(b)] for a, b in code], "position": [int(p) for p in position]}
            for code, position in zip(codes, positions)
        ],
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "base_val": round(base["fitness"], 4),
                "best_val": round(best["fitness"], 4),
                "base_test": round(holdout_base["fitness"], 4),
                "best_test": round(holdout_best["fitness"], 4),
                "evaluations": context.evaluations,
                "seconds": round(elapsed, 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
