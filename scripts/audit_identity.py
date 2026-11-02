"""Strict Fixed identity audit: T(x) must equal x elementwise."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.observed_inverse import code_index, inverse_code  # noqa: E402
from scripts.position_placebo import features  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = np.load(root / "data/processed/unsw_main8_v1/flows_unsw_8class.npz", allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    project = method33_tools.load_config(root / "configs/methods/fade_unsw8_pos_v2.json")
    indices = np.arange(min(200, len(labels)))
    application = list(take(dataset, indices))
    application[0] = np.minimum(application[0], 1514.0)
    app = apply_method33(
        *application, project.method33, 731800, mapping=project.method33.old_mapping,
        jitter_config=project.trigger_jitter, include_event_trace=True,
    )
    x = features(app)
    mappings = [list(project.method33.old_mapping)]
    rows, codes, anchors = [], [], []
    detected = []
    for i in range(len(indices)):
        trace = app[6][i]["complete_event_trace"]
        trig = [e for e in trace if e["role"] == "trigger_dummy" and e["in_observation_window"]]
        rows.append([e["event_position"] for e in trig][:2])
        lengths = [x[i, e["event_position"]] * 1514.0 for e in trig][:2]
        detected.append(code_index(np.abs(lengths), 0, mappings, project))
        codes.append(int(project.method33.old_mapping[int(labels[indices[i]])]))
        anchors.append({int(e["source_index"]): int(e["event_position"]) for e in trace if e["role"] == "real"})
    detector_errors = int(np.sum(np.asarray(detected) != np.asarray(codes)))
    transformed = np.stack([
        inverse_code(x[i], rows[i], codes[i], 0, anchors[i], mappings, project, "both")
        for i in range(len(x))
    ])
    diff = np.abs(transformed - x)
    print("flows", len(x), "max_abs_diff", float(diff.max()), "nonzero_rows", int(np.sum(np.any(diff > 0, axis=1))), "detector_errors", detector_errors)
    for i in range(len(x)):
        if np.any(diff[i] > 0):
            print("first_mismatch", i, "rows", rows[i], "codes", codes[i], "max", float(diff[i].max()))
            break


if __name__ == "__main__":
    main()
