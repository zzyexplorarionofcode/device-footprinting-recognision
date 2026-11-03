"""Per-flow naive vs joint trigger detector analysis."""

from __future__ import annotations

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
from scripts.observed_inverse import code_index  # noqa: E402
from scripts.position_placebo import features  # noqa: E402
from scripts.streaming_arms import detect_code_rows  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = np.load(root / "data/processed/unsw_main8_v1/flows_unsw_8class.npz", allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    project = method33_tools.load_config(root / "configs/methods/fade_unsw8_pos_v2.json")
    mapping = list(project.method33.next_mapping)
    indices = np.random.default_rng(20261030).choice(len(labels), size=300, replace=False)
    indices = np.sort(indices)
    application = list(take(dataset, indices))
    application[0] = np.minimum(application[0], 1514.0)
    app = apply_method33(
        *application, project.method33, 731800, mapping=mapping,
        jitter_config=project.trigger_jitter, include_event_trace=True,
    )
    x = features(app)
    rows = {"true": [], "naive": [], "joint": []}
    errors = {"naive": 0, "joint": 0}
    per_class = {}
    for i in range(len(indices)):
        true_code = int(mapping[int(labels[indices[i]])])
        trace = app[6][i]["complete_event_trace"]
        trigger_rows = [e["event_position"] - 1 for e in trace if e["role"] == "trigger_dummy"]
        lengths = [x[i, r] * 1514.0 for r in trigger_rows]
        naive = code_index(np.abs(lengths), 0, [mapping], project)
        joint, joint_rows = detect_code_rows(x[i], 0, [mapping], project)
        rows["true"].append(true_code)
        rows["naive"].append(naive)
        rows["joint"].append(joint)
        errors["naive"] += int(naive != true_code)
        errors["joint"] += int(joint != true_code)
        cls = int(labels[indices[i]])
        per_class.setdefault(cls, {"n": 0, "naive": 0, "joint": 0})
        per_class[cls]["n"] += 1
        per_class[cls]["naive"] += int(naive != true_code)
        per_class[cls]["joint"] += int(joint != true_code)
    result = {
        "n": len(indices),
        "naive_errors": errors["naive"],
        "joint_errors": errors["joint"],
        "per_class": per_class,
        "naive_confusions": {},
        "joint_confusions": {},
    }
    for name in ("naive", "joint"):
        for t, p in zip(rows["true"], rows[name]):
            if t != p:
                key = f"{t}->{p}"
                result[f"{name}_confusions"][key] = result[f"{name}_confusions"].get(key, 0) + 1
    out = root / "artifacts/es/detector_flow_unsw.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("naive", errors["naive"], "joint", errors["joint"], "n", len(indices), flush=True)


if __name__ == "__main__":
    main()
