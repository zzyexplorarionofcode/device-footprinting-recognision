"""CIC-IoT-2022 Idle profile: same 10 devices as Active, night-time traffic.

This case reuses the FADE codebook, budgets, and split seeds from the Active
configuration; only the dataset (``flows_cic_idle_10class.npz``) and the device
name order change.  It is a *profile* generalization test rather than an
independent dataset: the same lab and devices, but idle traffic instead of
daytime activity.
"""

from copy import deepcopy

from configs.experiments.current import EXPERIMENTS as FADE, ROOT

EXPERIMENTS = []

for original in deepcopy(FADE):
    if original["name"] != "cic_active_session_v2":
        continue
    case = deepcopy(original)
    case["name"] = "cic_idle_10class"
    case["dataset"] = ROOT / "data/processed/cic_v2/flows_cic_idle_10class.npz"
    case["expected_classes"] = 10
    case["output_dir"] = ROOT / "artifacts/runs/fade_cic_idle" / case["name"]
    case["methods"][0]["name"] = "FADE-idle-old-to-next"
    case["methods"][0]["params"]["config"] = ROOT / "configs/methods/fade_cic_idle.json"
    EXPERIMENTS.append(case)
