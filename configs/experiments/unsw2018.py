"""UNSW 2018 "Your Things" mixed-device traces (9 classes).

Caveats recorded with the dataset: this testbed produces many short flows
(median ~11 events), so five FADE dummies occupy a large fraction of the
32-event observation window; the aggregate communication overhead on full flows
is comparable to the paper (~6-7%), but the per-flow median is much higher.
Report median/p95 and the dummy fraction alongside BA.
"""

from pathlib import Path
from copy import deepcopy

from configs.experiments.common import EVALUATION
from experiment_support.budget import load_budget

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = [
    dict(
        name="unsw2018_9class",
        dataset=ROOT / "data/processed/unsw2018_v1/flows_unsw2018_9class.npz",
        data_protocol="unsw2018-mixed-device-day-v1",
        expected_classes=9,
        shared_budget=load_budget(ROOT / "configs/experiments/fade_budget.json").to_dict(),
        budget_profile="FADE-630B-1500ms-6events",
        output_dir=ROOT / "artifacts/runs/fade_unsw2018" / "unsw2018_9class",
        feature={"max_frame_length": 1514.0, "max_gap_seconds": 10.0},
        split={
            "unit": "capture",
            "group_semantics": "one mixed capture per UTC day",
            "train_ratio": 0.75,
            "seed": 42,
        },
        evaluation=deepcopy(EVALUATION),
        methods=[
            dict(
                key="method33_shared_budget_v1",
                name="FADE-unsw2018-old-to-next",
                enabled=True,
                params={
                    "config": ROOT / "configs/methods/fade_unsw2018.json",
                    "train_mapping": "old",
                    "test_mapping": "next",
                    "include_weakener": True,
                    "include_trigger": True,
                },
            )
        ],
    )
]

# FADE cases use one repeat; keep it consistent with the CIC / UNSW-IoTraffic cases.
EXPERIMENTS[0]["evaluation"]["repeat_count"] = 1
