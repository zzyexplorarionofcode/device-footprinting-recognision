"""20-day block coverage, state projection equivalence, feature sensitivity."""

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
from scripts.position_placebo import features  # noqa: E402
from scripts.streaming_arms import detect_code_rows, make_mappings  # noqa: E402
from scripts.adaptive_matrix import canonicalize_tracefree  # noqa: E402


def canonical(x, state, mappings, project):
    code, rows = detect_code_rows(x, state, mappings, project)
    cls = {c: k for k, c in enumerate(mappings[state])}[int(code)]
    ref = int(project.method33.old_mapping[cls])
    return canonicalize_tracefree(x, state, code, rows, ref, project)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = np.load(root / "data/processed/unsw_main8_v1/flows_unsw_8class.npz", allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(root / "configs/methods/fade_unsw8_pos_v2.json")
    mappings = make_mappings(project, 0)
    days = np.sort(np.unique(groups))
    blocks = []
    for start in range(0, len(days), 20):
        block_days = days[start : start + 20]
        half = max(1, len(block_days) // 2)
        cal_days = set(block_days[:half])
        idx = np.where(np.isin(groups, list(cal_days)))[0]
        if len(idx) >= 100:
            blocks.append((idx, block_days[:half]))
    out = []
    for b, (idx, cal_days) in enumerate(blocks):
        counts = {int(c): int(np.sum(labels[idx] == c)) for c in np.unique(labels)}
        present = [c for c, n in counts.items() if n > 0]
        signatures = {}
        for state in range(len(mappings)):
            sig = []
            for cls in present:
                code = int(mappings[state][cls])
                sig.append((code, tuple(project.method33.trigger_positions[code])))
            signatures[state] = tuple(sig)
        groups_by_sig = {}
        for state, sig in signatures.items():
            groups_by_sig.setdefault(sig, []).append(state)
        # Feature sensitivity on a sample of calibration flows.
        sample = idx[: min(50, len(idx))]
        application = list(take(dataset, sample))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=mappings[0], jitter_config=project.trigger_jitter,
        )
        x = features(app)
        true_canon = np.stack([canonical(x[i], 0, mappings, project) for i in range(len(x))])
        dists = {}
        for state in range(len(mappings)):
            canon = np.stack([canonical(x[i], state, mappings, project) for i in range(len(x))])
            dists[str(state)] = float(np.mean(np.linalg.norm(canon - true_canon, axis=1)))
        out.append({
            "block": b,
            "cal_days": [str(d) for d in cal_days],
            "cal_flows": int(len(idx)),
            "counts": counts,
            "present": present,
            "equivalence_groups": list(groups_by_sig.values()),
            "feature_distance_to_state0": dists,
        })
        print("block", b, "flows", len(idx), "classes", len(present),
              "groups", [len(v) for v in groups_by_sig.values()], flush=True)
    (root / "artifacts/es/state_projection_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
