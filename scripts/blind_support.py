"""Support audit for selected 20-day blocks."""

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
    blocks = []
    for start in range(0, len(days), 20):
        block_days = days[start : start + 20]
        half = max(1, len(block_days) // 2)
        cal = np.where(np.isin(groups, list(block_days[:half])))[0]
        blind = np.where(np.isin(groups, list(block_days[half:])))[0]
        if len(cal) >= 100 and len(blind) >= 100:
            blocks.append((cal, blind))
    history = []
    out = []
    for b, (cal, blind) in enumerate(blocks):
        hist = np.concatenate(history) if history else np.asarray([], dtype=int)
        classes = sorted(set(labels.tolist()))
        row = {"block": b, "classes": {}}
        for c in classes:
            row["classes"][str(c)] = {
                "cal": int(np.sum(labels[cal] == c)),
                "blind": int(np.sum(labels[blind] == c)),
                "history": int(np.sum(labels[hist] == c)) if len(hist) else 0,
            }
        if b in (6, 7):
            out.append(row)
        history.append(cal)
    (root / "artifacts/es/blind_support_blocks67.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in out:
        print("block", row["block"])
        for c, v in row["classes"].items():
            print(" ", c, v)


if __name__ == "__main__":
    main()
