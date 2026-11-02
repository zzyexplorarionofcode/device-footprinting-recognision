"""Oracle feature canonicalization: regenerate all blocks with one reference mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods import method33_tools  # noqa: E402
from methods.method33 import apply_method33  # noqa: E402
from scripts.fold_review import take  # noqa: E402
from scripts.position_placebo import features  # noqa: E402


def fit_model(x, y):
    return SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=0).fit(x, y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--block-days", type=int, default=10)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    labels = np.asarray(dataset["labels"])
    groups = np.asarray(dataset["groups"]).astype(str)
    project = method33_tools.load_config(args.config)
    reference = list(project.method33.old_mapping)
    days = np.sort(np.unique(groups))
    blocks = []
    for start in range(0, len(days), args.block_days):
        chosen = set(days[start : start + args.block_days])
        idx = np.where(np.isin(groups, list(chosen)))[0]
        if len(idx) >= 100:
            blocks.append(idx)

    block_x, block_y = [], []
    for idx in blocks:
        application = list(take(dataset, idx))
        application[0] = np.minimum(application[0], 1514.0)
        app = apply_method33(
            *application, project.method33, 731800,
            mapping=reference, jitter_config=project.trigger_jitter,
        )
        block_x.append(features(app))
        block_y.append(labels[idx])

    curves = []
    for b in range(1, len(blocks)):
        x_train = np.concatenate(block_x[:b])
        y_train = np.concatenate(block_y[:b])
        scaler = StandardScaler().fit(x_train)
        model = fit_model(scaler.transform(x_train), y_train)
        pred = model.predict(scaler.transform(block_x[b]))
        curves.append(float(balanced_accuracy_score(block_y[b], pred)))

    result = {"dataset_version": args.dataset_version, "J": float(np.mean(curves)), "curves": curves}
    args.output_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("J", round(result["J"], 3), "blocks", len(curves), flush=True)


if __name__ == "__main__":
    main()
