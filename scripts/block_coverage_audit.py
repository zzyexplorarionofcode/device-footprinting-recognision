"""Block coverage audit: class support in the first 50% of each time block."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = np.load(root / "data/processed/unsw_main8_v1/flows_unsw_8class.npz", allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    days = np.sort(np.unique(groups))
    classes = sorted(set(labels.tolist()))
    result = {"dataset": "unsw8_v1", "classes": classes, "block_sizes": {}}
    for size in (10, 20, 30):
        blocks = []
        for start in range(0, len(days), size):
            block_days = days[start : start + size]
            half = max(1, len(block_days) // 2)
            cal_days = set(block_days[:half])
            blind_days = set(block_days[half:])
            cal_idx = np.where(np.isin(groups, list(cal_days)))[0]
            blind_idx = np.where(np.isin(groups, list(blind_days)))[0]
            counts = {int(c): int(np.sum(labels[cal_idx] == c)) for c in classes}
            present = [c for c, n in counts.items() if n > 0]
            blocks.append({
                "block": len(blocks),
                "cal_days": [str(d) for d in block_days[:half]],
                "cal_flows": int(len(cal_idx)),
                "blind_flows": int(len(blind_idx)),
                "present_classes": present,
                "counts": counts,
                "min_per_present": int(min(counts[c] for c in present)) if present else 0,
                "all_classes": len(present) == len(classes),
            })
        result["block_sizes"][str(size)] = {
            "n_blocks": len(blocks),
            "blocks_with_3plus": sum(len(b["present_classes"]) >= 3 for b in blocks),
            "blocks_with_5plus": sum(len(b["present_classes"]) >= 5 for b in blocks),
            "blocks_all_classes": sum(b["all_classes"] for b in blocks),
            "blocks": blocks,
        }
    out = root / "artifacts/es/block_coverage_unsw.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for size, info in result["block_sizes"].items():
        print(size, "blocks", info["n_blocks"], ">=3", info["blocks_with_3plus"],
              ">=5", info["blocks_with_5plus"], "all", info["blocks_all_classes"])
        for b in info["blocks"]:
            print(" ", b["block"], "flows", b["cal_flows"], "classes", len(b["present_classes"]),
                  "min", b["min_per_present"], "counts", b["counts"])


if __name__ == "__main__":
    main()
