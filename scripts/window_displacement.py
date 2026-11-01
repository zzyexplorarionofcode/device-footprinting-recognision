"""Diagnose post-insertion 32-event windows: displaced reals and retained dummies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_ablation import cap_with_min  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    split = np.load(args.split_indices)
    rng = np.random.default_rng(args.seed)
    indices = cap_with_min(np.asarray(split["train"]), labels, args.cap, 1, rng)
    project = method33_tools.load_config(args.config)
    application = apply_method33(
        *take(dataset, indices),
        project.method33,
        731800,
        mapping="old",
        jitter_config=project.trigger_jitter,
    )
    stats = application[4]
    source_counts = np.asarray(dataset["mask"])[indices].sum(axis=1)
    displaced = np.asarray(stats["displaced_real_packets"])
    retained_dummy = np.asarray(stats["retained_dummy_count"])
    emitted_dummy = np.asarray(stats["dummy_count"])

    result = {
        "flows": int(len(indices)),
        "source_events_mean": float(source_counts.mean()),
        "displaced_real": {
            "mean": float(displaced.mean()),
            "median": float(np.median(displaced)),
            "p95": float(np.percentile(displaced, 95)),
            "max": int(displaced.max()),
            "fraction_any": float((displaced > 0).mean()),
        },
        "retained_dummy": {
            "mean": float(retained_dummy.mean()),
            "median": float(np.median(retained_dummy)),
            "fraction_5": float((retained_dummy == 5).mean()),
            "fraction_lt5": float((retained_dummy < 5).mean()),
            "histogram": {str(int(value)): int(np.sum(retained_dummy == value)) for value in range(7)},
        },
        "emitted_dummy_mean": float(emitted_dummy.mean()),
        "config": str(args.config),
        "dataset": str(args.dataset),
        "split": str(args.split_indices),
        "cap": args.cap,
    }
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
