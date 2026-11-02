"""Online adaptation pilot: prefix labels, blind remainder, BA_t(n), N_tau.

Policies: Fixed, Periodic(P=2), Per_flow.
Attackers: mapping-unaware; Periodic adds schedule-known and state-inferred.
Methods: permutation, probe, finetune.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_placebo import features  # noqa: E402


def make_mappings(project, count=6):
    base = list(project.method33.old_mapping)
    nxt = list(project.method33.next_mapping)
    mappings = [base, nxt]
    rng = np.random.default_rng(123)
    while len(mappings) < count:
        perm = list(rng.permutation(len(base)))
        if perm not in mappings:
            mappings.append(perm)
    return mappings


def build_policy_blocks(dataset, labels, groups, project, mappings, policy, block_days, period, rng):
    days = np.sort(np.unique(groups))
    blocks = []
    for start in range(0, len(days), block_days):
        chosen = set(days[start : start + block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            blocks.append(idx)
    result = []
    for b, idx in enumerate(blocks):
        if policy == "Fixed":
            state = np.zeros(len(idx), dtype=int)
        elif policy == "Periodic":
            state = np.full(len(idx), (b // period) % len(mappings), dtype=int)
        else:
            state = rng.integers(0, len(mappings), size=len(idx))
        x = np.zeros((len(idx), 64), dtype=float)
        for state_id in np.unique(state):
            rows = np.where(state == state_id)[0]
            subset = idx[rows]
            application = list(take(dataset, subset))
            application[0] = np.minimum(application[0], 1514.0)
            app = apply_method33(
                *application, project.method33, 731800,
                mapping=mappings[int(state_id)], jitter_config=project.trigger_jitter,
            )
            x[rows] = features(app)
        result.append({"x": x, "y": labels[idx], "state": state})
    return result


def fit_lr(x, y, weights=None):
    if len(np.unique(y)) < 2:
        return None
    return SGDClassifier(loss="log_loss", max_iter=300, tol=1e-3, random_state=0).fit(x, y, sample_weight=weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--block-days", type=int, default=5)
    parser.add_argument("--period", type=int, default=2)
    parser.add_argument("--ns", default="0,64,256,1024")
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project)
    ns = [int(v) for v in args.ns.split(",")]
    rng = np.random.default_rng(20261030)

    policies = ("Fixed", "Periodic")
    output = {"dataset_version": args.dataset_version, "ns": ns, "policies": {}}
    for policy in policies:
        blocks = build_policy_blocks(dataset, labels, groups, project, mappings, policy, args.block_days, args.period, rng)
        attackers = ("unaware",) if policy != "Periodic" else ("unaware", "schedule_known", "state_inferred")
        policy_out = {"attackers": {}}
        for attacker in attackers:
            methods_out = {}
            for method in ("permutation", "probe", "finetune"):
                curves = {n: [] for n in ns}
                ntau = {"values": [], "censored": 0}
                for b in range(1, len(blocks)):
                    print(policy, attacker, method, "block", b, "/", len(blocks) - 1, flush=True)
                    history_x = np.concatenate([blocks[i]["x"] for i in range(b)])
                    history_y = np.concatenate([blocks[i]["y"] for i in range(b)])
                    history_state = np.concatenate([blocks[i]["state"] for i in range(b)])
                    if len(history_x) > 3000:
                        keep = rng.choice(len(history_x), size=3000, replace=False)
                        history_x, history_y, history_state = history_x[keep], history_y[keep], history_state[keep]
                    scaler = StandardScaler().fit(history_x)
                    z_hist = scaler.transform(history_x)
                    base = fit_lr(z_hist, history_y)
                    # Mapping-aware experts.
                    experts = {}
                    if attacker != "unaware":
                        for state_id in np.unique(history_state):
                            rows = history_state == state_id
                            if rows.sum() >= 20:
                                experts[int(state_id)] = fit_lr(z_hist[rows], history_y[rows])
                    cur = blocks[b]
                    z_cur = scaler.transform(cur["x"])
                    y_cur = cur["y"]
                    cur_state = int(cur["state"][0]) if len(np.unique(cur["state"])) == 1 else None
                    for n in ns:
                        take_n = min(n, max(0, len(y_cur) - 1))
                        blind = np.arange(take_n, len(y_cur))
                        if len(blind) == 0:
                            continue
                        cal_x, cal_y = z_cur[:take_n], y_cur[:take_n]
                        # Choose base model.
                        if attacker == "schedule_known" and cur_state in experts:
                            base_model = experts[cur_state]
                        elif attacker == "state_inferred" and take_n >= 8 and experts:
                            scores = {s: np.mean(m.predict(cal_x) == cal_y) for s, m in experts.items() if m is not None}
                            best = max(scores, key=scores.get) if scores else None
                            base_model = experts.get(best, base)
                        else:
                            base_model = base
                        pred_blind = base_model.predict(z_cur[blind]) if base_model is not None else np.zeros(len(blind), dtype=int)
                        pred_cal = base_model.predict(cal_x) if base_model is not None and take_n else np.asarray([])
                        if method == "probe" and take_n >= 8 and len(np.unique(cal_y)) >= 2:
                            model = fit_lr(cal_x, cal_y)
                            pred_blind = model.predict(z_cur[blind])
                        elif method == "finetune" and take_n >= 8 and len(np.unique(cal_y)) >= 2:
                            weight = len(history_x) / take_n
                            weights = np.concatenate([np.ones(len(history_x)), np.full(take_n, weight)])
                            model = fit_lr(np.concatenate([z_hist, cal_x]), np.concatenate([history_y, cal_y]), weights)
                            pred_blind = model.predict(z_cur[blind])
                        elif method == "permutation" and take_n > 0 and len(pred_cal):
                            classes = sorted(set(y_cur.tolist()))
                            cm = confusion_matrix(cal_y, pred_cal, labels=classes)
                            row, col = linear_sum_assignment(-cm)
                            mapping = {int(p): int(t) for p, t in zip(col, row)}
                            pred_blind = np.asarray([mapping.get(int(p), int(p)) for p in pred_blind])
                        curves[n].append(float(balanced_accuracy_score(y_cur[blind], pred_blind)))
                    # N_tau per block: smallest n reaching 0.5.
                    block_ba = {n: curves[n][-1] for n in ns if len(curves[n]) > b - 1}
                    eligible = [n for n in ns if block_ba.get(n, 0.0) >= 0.5]
                    if eligible:
                        ntau["values"].append(min(eligible))
                    else:
                        ntau["censored"] += 1
                methods_out[method] = {
                    "curves": {str(n): curves[n] for n in ns},
                    "J": {str(n): float(np.mean(curves[n])) if curves[n] else None for n in ns},
                    "N_tau": ntau,
                }
            policy_out["attackers"][attacker] = methods_out
            print(policy, attacker, "done", flush=True)
        output["policies"][policy] = policy_out

    args.output_log.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
