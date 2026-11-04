"""Per-flow state inference audit: oracle, observable likelihood, label-assisted."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_placebo import features  # noqa: E402
from scripts.streaming_arms import detect_code_rows, make_mappings  # noqa: E402


def detection_error(x, code, rows, project):
    value = 0.0
    for j, row in enumerate(rows):
        length, gap = project.method33.trigger_codes[code][j]
        value += abs(abs(x[row]) * 1514.0 - length)
        value += abs(np.expm1(x[32 + row]) / 1000.0 - gap) * 100
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    mappings = make_mappings(project, args.seed)
    days = np.sort(np.unique(groups))
    raw = []
    for start in range(0, len(days), args.block_days):
        block_days = days[start : start + args.block_days]
        half = max(1, len(block_days) // 2)
        idx = np.where(np.isin(groups, list(block_days[:half])))[0]
        if len(idx) >= 100:
            raw.append(idx)
    rng = np.random.default_rng(args.seed)
    history_x, history_state = [], []
    results = []
    for b, idx in enumerate(raw):
        if len(idx) > 2000:
            idx = rng.choice(idx, size=2000, replace=False)
        state = rng.integers(0, len(mappings), size=len(idx))
        x = np.zeros((len(idx), 64))
        for s in np.unique(state):
            rows = np.where(state == s)[0]
            subset = idx[rows]
            application = list(take(dataset, subset))
            application[0] = np.minimum(application[0], 1514.0)
            app = apply_method33(
                *application, project.method33, 731800,
                mapping=mappings[int(s)], jitter_config=project.trigger_jitter,
            )
            x[rows] = features(app)
            del app, application
        # Observable likelihood per flow, no labels.
        obs_pred = []
        obs_scores = []
        for i in range(len(x)):
            vals = []
            for s in range(len(mappings)):
                code, rows = detect_code_rows(x[i], s, mappings, project)
                vals.append(detection_error(x[i], code, rows, project))
            obs_pred.append(int(np.argmin(vals)))
            obs_scores.append(float(np.min(vals)))
        entry = {
            "block": b,
            "flows": int(len(idx)),
            "oracle_acc": 1.0,
            "observable_acc": float(accuracy_score(state, obs_pred)),
            "observable_confusion": {},
        }
        for t, p in zip(state, obs_pred):
            key = f"{int(t)}->{int(p)}"
            entry["observable_confusion"][key] = entry["observable_confusion"].get(key, 0) + 1
        if history_x:
            hx, hs = np.concatenate(history_x), np.concatenate(history_state)
            scaler = StandardScaler().fit(hx)
            clf = LogisticRegression(max_iter=300).fit(scaler.transform(hx), hs)
            pred = clf.predict(scaler.transform(x))
            proba = clf.predict_proba(scaler.transform(x))
            top2 = np.argsort(-proba, axis=1)[:, :2]
            top2_acc = float(np.mean([state[i] in top2[i] for i in range(len(state))]))
            entry["label_assisted_acc"] = float(accuracy_score(state, pred))
            entry["label_assisted_top2"] = top2_acc
        else:
            entry["label_assisted_acc"] = None
            entry["label_assisted_top2"] = None
        history_x.append(x)
        history_state.append(state)
        results.append(entry)
        print("block", b, "obs", round(entry["observable_acc"], 3),
              "label", entry["label_assisted_acc"], flush=True)
    args.output_log.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
