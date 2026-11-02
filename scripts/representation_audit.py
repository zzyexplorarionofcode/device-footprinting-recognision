"""Audit what 'representation' changes: real-packet padding vs construction mismatch."""

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


def audit(dataset, indices, config, mapping, jitter):
    application = apply_method33(
        *take(dataset, indices), config, 731800, mapping=mapping,
        jitter_config=jitter, include_event_trace=True,
    )
    lengths = np.asarray(dataset["lengths"])
    mask = np.asarray(dataset["mask"]).astype(bool)
    changed_flows = 0
    changed_events = 0
    total_events = 0
    hist_identical = 0
    count_mismatch = 0
    added_bytes = []
    for row, index in enumerate(indices):
        clean = lengths[index][mask[index]]
        trace = application[6][row]["complete_event_trace"]
        defended = np.asarray([e["length_bytes"] for e in trace if e["role"] == "real"], dtype=float)
        if clean.size != defended.size:
            count_mismatch += 1
            continue
        total_events += clean.size
        diff = defended - clean
        changed = int(np.sum(np.abs(diff) > 1e-9))
        if changed:
            changed_flows += 1
            changed_events += changed
        added_bytes.append(float(np.sum(diff)))
        clean_hist, _ = np.histogram(clean, bins=np.linspace(0, 1514, 17))
        defended_hist, _ = np.histogram(defended, bins=np.linspace(0, 1514, 17))
        if np.array_equal(clean_hist, defended_hist):
            hist_identical += 1
    return {
        "flows": int(len(indices)),
        "count_mismatch": int(count_mismatch),
        "changed_flows": int(changed_flows),
        "changed_flow_fraction": float(changed_flows / len(indices)),
        "changed_events": int(changed_events),
        "total_events": int(total_events),
        "changed_event_fraction": float(changed_events / total_events) if total_events else 0.0,
        "real_added_bytes_mean": float(np.mean(added_bytes)) if added_bytes else 0.0,
        "real_added_bytes_median": float(np.median(added_bytes)) if added_bytes else 0.0,
        "hist_identical_fraction": float(hist_identical / len(indices)),
    }


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
    project = method33_tools.load_config(args.config)
    result = {}
    for name, key in (("train", "train"), ("test", "test")):
        indices = cap_with_min(np.asarray(split[key]), labels, args.cap, 1, rng)
        result[name] = audit(dataset, indices, project.method33, "old" if name == "train" else "next", project.trigger_jitter)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
