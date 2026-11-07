"""Build a causal 8-class warmup pilot after all classes have appeared."""

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
    old = json.loads((root / "artifacts/es/online_nonstationary_s0.json").read_text(encoding="utf-8"))
    data = np.load(root / "data/processed/unsw_main8_v1/flows_unsw_8class.npz", allow_pickle=False)
    labels = data["labels"]
    groups = data["groups"].astype("datetime64[D]")
    start = np.datetime64("2017-02-23")
    end = np.datetime64("2017-04-13")
    warm_pool = np.flatnonzero(groups < start)
    days = np.unique(groups[(groups >= start) & (groups <= end)])
    rng = np.random.default_rng(20261107)

    # Stratified warmup: 500/class where available, fill remainder from class 0.
    chosen = []
    for c in range(8):
        idx = warm_pool[labels[warm_pool] == c]
        chosen.extend(rng.choice(idx, min(500, len(idx)), replace=False).tolist())
    chosen = np.asarray(chosen)
    if len(chosen) < 4000:
        extra = np.setdiff1d(warm_pool, chosen)
        chosen = np.r_[chosen, rng.choice(extra, 4000 - len(chosen), replace=False)]
    chosen = np.sort(chosen[:4000])
    print("warm", len(chosen), np.bincount(labels[chosen], minlength=8).tolist())

    sample_days = []
    used = set(chosen.tolist())
    for day in days:
        pool = np.asarray([i for i in np.flatnonzero(groups == day) if int(i) not in used])
        if len(pool) < 20:
            continue
        pick = rng.choice(pool, min(len(pool), 136), replace=False)
        used.update(int(v) for v in pick)
        cal, blind = pick[: min(8, len(pick))], pick[8:]
        sample_days.append({
            "day": str(day),
            "cal_indices": cal.tolist(),
            "blind_indices": blind.tolist(),
        })
    print("days", len(sample_days), sample_days[0]["day"], sample_days[-1]["day"])

    pilot = {
        "seed": 0,
        "dataset_sha256": old["dataset_sha256"],
        "config_sha256": old["config_sha256"],
        "settings": old["settings"],
        "mappings": old["mappings"],
        "warmup_indices": chosen.tolist(),
        "sample_days": sample_days,
        "warmup_support": {str(c): int(v) for c, v in enumerate(np.bincount(labels[chosen], minlength=8))},
    }
    (root / "artifacts/es/online_nonstationary_late_s0.json").write_text(
        json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
