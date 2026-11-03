"""Canonical sequence reconstruction from observed complete trace."""

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
from scripts.position_placebo import features  # noqa: E402


def reconstruct(row, project, ref_code, actual_code, x):
    if ref_code == actual_code:
        return x.copy()
    trace = row["complete_event_trace"]
    anchors = set(int(v) for v in project.method33.trigger_positions[ref_code])
    packets = project.method33.trigger_codes[ref_code]
    observed_trigger_rows = [e["event_position"] - 1 for e in trace if e["role"] == "trigger_dummy"]
    rows = []
    inserted = 0
    for event in trace:
        if event["role"] == "trigger_dummy":
            continue
        slot = int(event["event_position"]) - 1
        if slot < 32:
            rows.append(slot)
        if event["role"] != "real":
            continue
        anchor = int(event["source_index"]) + 1
        if anchor in anchors and inserted < len(packets):
            if actual_code == ref_code and inserted < len(observed_trigger_rows):
                rows.append(observed_trigger_rows[inserted])
            else:
                # Placeholder: raw code row is appended at the anchor position.
                rows.append(("raw", inserted))
            inserted += 1
    out = np.zeros(64)
    for i, item in enumerate(rows[:32]):
        if isinstance(item, tuple):
            length, gap = packets[item[1]]
            out[i] = length / 1514.0
            out[32 + i] = np.log1p(min(gap, 10.0) * 1000.0)
        else:
            out[i] = x[item]
            out[32 + i] = x[32 + item]
    return out


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
    diffs = []
    for i in range(len(indices)):
        actual = int(project.method33.old_mapping[int(labels[indices[i]])])
        rebuilt = reconstruct(app[6][i], project, actual, actual, x[i])
        diffs.append(float(np.max(np.abs(rebuilt - x[i]))))
    diffs = np.asarray(diffs)
    print("flows", len(diffs), "max_abs_diff", float(diffs.max()), "nonzero", int(np.sum(diffs > 0)))
    print("worst", float(np.percentile(diffs, 95)))


if __name__ == "__main__":
    main()
