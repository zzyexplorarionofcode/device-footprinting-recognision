"""Generate the FADE method config and experiment case for the UNSW-2018 dataset.

The dataset build writes ``flows_unsw2018_*.json`` (device order + class count).
This helper turns that manifest into:

* ``configs/methods/fade_unsw2018.json``  — codebook/mapping/budget config
* ``configs/experiments/unsw2018.py``     — experiment case for the runner

The codebook keeps the paper's 10-code layout when the dataset has 10 classes;
for a smaller class count it trims codes/positions and rotates the ``next``
mapping into a valid derangement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_configs(
    dataset_json: Path,
    base_config: Path,
    method_output: Path,
    experiment_output: Path,
    *,
    root: Path,
) -> tuple[int, list[str]]:
    manifest = json.loads(Path(dataset_json).read_text(encoding="utf-8"))
    devices = [str(name) for name in manifest["device_names"]]
    class_count = int(manifest.get("class_count", len(devices)))
    if class_count != len(devices) or class_count < 2:
        raise ValueError("dataset manifest has an inconsistent class count")

    base = json.loads(Path(base_config).read_text(encoding="utf-8-sig"))
    method = base["method33"]
    base_codes = method["trigger_codes"]
    base_positions = method["trigger_positions"]
    if class_count > len(base_codes) or class_count > len(base_positions):
        raise ValueError("base config does not contain enough codes/positions")

    method["trigger_codes"] = base_codes[:class_count]
    method["trigger_positions"] = base_positions[:class_count]
    if class_count == len(base_codes):
        # Keep the paper's original old/next mappings for the 10-class case.
        pass
    else:
        method["old_mapping"] = list(range(class_count))
        method["next_mapping"] = list(range(1, class_count)) + [0]
    base["dataset"]["devices"] = devices
    method_output.parent.mkdir(parents=True, exist_ok=True)
    method_output.write_text(
        json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    protocol = str(manifest["protocol"])
    relative_dataset = (
        Path("data/processed/unsw2018_v1") / f"flows_unsw2018_{class_count}class.npz"
    )
    experiment_output.write_text(
        f'''"""UNSW 2018 "Your Things" mixed-device traces ({class_count} classes).

Caveats recorded with the dataset: this testbed produces many short flows
(median ~11 events), so five FADE dummies occupy a large fraction of the
32-event observation window; the aggregate communication overhead on full flows
is comparable to the paper (~6-7%), but the per-flow median is much higher.
Report median/p95 and the dummy fraction alongside BA.
"""

from copy import deepcopy

from configs.experiments.common import EVALUATION
from experiment_support.budget import load_budget

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = [
    dict(
        name="unsw2018_{class_count}class",
        dataset=ROOT / "{relative_dataset.as_posix()}",
        data_protocol="{protocol}",
        expected_classes={class_count},
        shared_budget=load_budget(ROOT / "configs/experiments/fade_budget.json").to_dict(),
        budget_profile="FADE-630B-1500ms-6events",
        output_dir=ROOT / "artifacts/runs/fade_unsw2018" / "unsw2018_{class_count}class",
        feature={{"max_frame_length": 1514.0, "max_gap_seconds": 10.0}},
        split={{
            "unit": "capture",
            "group_semantics": "one mixed capture per UTC day",
            "train_ratio": 0.75,
            "seed": 42,
        }},
        evaluation=deepcopy(EVALUATION),
        methods=[
            dict(
                key="method33_shared_budget_v1",
                name="FADE-unsw2018-old-to-next",
                enabled=True,
                params={{
                    "config": ROOT / "configs/methods/fade_unsw2018.json",
                    "train_mapping": "old",
                    "test_mapping": "next",
                    "include_weakener": True,
                    "include_trigger": True,
                }},
            )
        ],
    )
]

# FADE cases use one repeat; keep it consistent with the CIC / UNSW-IoTraffic cases.
EXPERIMENTS[0]["evaluation"]["repeat_count"] = 1
''',
        encoding="utf-8",
    )
    # The generated file needs Path imported; prepend it after the docstring.
    text = experiment_output.read_text(encoding="utf-8")
    text = text.replace("from copy import deepcopy", "from pathlib import Path\nfrom copy import deepcopy", 1)
    experiment_output.write_text(text, encoding="utf-8")
    return class_count, devices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_json", type=Path)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/methods/fade_cic_active.json"),
    )
    parser.add_argument(
        "--method-output",
        type=Path,
        default=Path("configs/methods/fade_unsw2018.json"),
    )
    parser.add_argument(
        "--experiment-output",
        type=Path,
        default=Path("configs/experiments/unsw2018.py"),
    )
    args = parser.parse_args()
    class_count, devices = build_configs(
        args.dataset_json,
        args.base_config,
        args.method_output,
        args.experiment_output,
        root=Path("."),
    )
    print(f"class_count={class_count}")
    for index, name in enumerate(devices):
        print(f"  {index:2d} {name}")


if __name__ == "__main__":
    main()
