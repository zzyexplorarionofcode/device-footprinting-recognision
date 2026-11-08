"""Summarize the MLP-only C/D Hybrid check."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/es"
ARMS = ("Baseline", "Hybrid-25", "Hybrid-50")

def main():
    logs = [json.loads((ARTIFACTS / f"weakener_hybrid_mlp_deakin10_s{s}.json").read_text(encoding="utf-8")) for s in range(3)]
    lines = ["# Weakener MLP 表示稳健性检查", "", "Deakin10，轻量 MLP，沿用相同逐日在线训练和三臂预算协议。", "", "| 视图 | Baseline | Hybrid-25 | Δ25 | Hybrid-50 | Δ50 | 三 seed Δ25 / Δ50 |", "|---|---:|---:|---:|---:|---:|---|"]
    out = {}
    for view in ("C", "D"):
        out[view] = {}
        for mode in ("cumulative", "sliding"):
            values = np.asarray([[log["results"][view]["mlp"][arm]["summary"][mode]["mean_post_ba"] for arm in ARMS] for log in logs])
            delta25, delta50 = values[:,1]-values[:,0], values[:,2]-values[:,0]
            means = values.mean(0)
            lines.append(f"| {view} / {mode} | {means[0]:.4f} | {means[1]:.4f} | {delta25.mean():+.4f} | {means[2]:.4f} | {delta50.mean():+.4f} | {' / '.join(f'{x:+.4f}' for x in delta25)} ; {' / '.join(f'{x:+.4f}' for x in delta50)} |")
            out[view][mode] = {"means":means.tolist(),"delta25":delta25.tolist(),"delta50":delta50.tolist()}
    lines += ["", "解释：负 Δ 表示 Hybrid 比 Baseline 的 BA 更低。C 是45维可观测组合，D 是原始64维序列表示。", "MLP 结果只补充 C/D 两个核心表示，B 及 C1–C4 的多模型消融仍以 LR/RF 全视图结果为准。", ""]
    (ARTIFACTS / "weakener_mlp_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (ROOT / "docs/Weakener_MLP稳健性.md").write_text("\n".join(lines),encoding="utf-8")

if __name__ == "__main__": main()
