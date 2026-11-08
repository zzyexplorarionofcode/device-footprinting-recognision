"""Summarize the paired C1-C5 Weakener feature ablation."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/es"
RANDOM, TARGETED = "Random-all", "Targeted-candidate"
VIEWS = ("C1_real_uplink", "C2_counts", "C3_bytes", "C4_large_ratio", "C5_combined")


def main():
    lines = [
        "# Weakener C1–C5 特征消融", "",
        "本实验保持阶段二相同原始 flow、逐流实际 padding 字节、五个 dummy、Full New5 事件序列、",
        "在线标签到达、Blind flows 和三组配对 seed，只比较 Random-all 与 Targeted-candidate。", "",
        "- C1：最终窗口内真实上行包长的 16 桶归一化直方图。该视图使用 role 审计标记，属于机制诊断，不是普通攻击者可直接获得的输入。",
        "- C2：可观测上行、下行和总事件数。",
        "- C3：可观测上行、下行和总字节数。",
        "- C4：可观测上行、下行和全部事件中大于 256 B 的比例。",
        "- C5：拼接 C1–C4。由于包含 C1，C5 也是机制诊断组合视图。", "",
    ]
    machine = {}
    for family in ("unsw8", "deakin10"):
        logs = [json.loads((ARTIFACTS / f"weakener_c_ablation_{family}_s{s}.json").read_text(encoding="utf-8"))
                for s in range(3)]
        machine[family] = {}
        lines += [f"## {family}", "",
                  "| 视图 | 模型与更新 | Random | Targeted | Δ | 三 seed Δ |",
                  "|---|---|---:|---:|---:|---|"]
        for view in VIEWS:
            machine[family][view] = {}
            for model in ("lr", "rf"):
                for mode in ("cumulative", "sliding"):
                    values = np.asarray([[log["results"][view][model][arm]["summary"][mode]["mean_post_ba"]
                                          for arm in (RANDOM, TARGETED)] for log in logs])
                    delta = values[:, 1] - values[:, 0]
                    means = values.mean(axis=0)
                    key = f"{model}_{mode}"
                    machine[family][view][key] = {"random": values[:, 0].tolist(),
                                                  "targeted": values[:, 1].tolist(),
                                                  "delta": delta.tolist()}
                    lines.append(f"| {view} | {model.upper()} {mode} | {means[0]:.4f} | {means[1]:.4f} | "
                                 f"{delta.mean():+.4f} | {' / '.join(f'{v:+.4f}' for v in delta)} |")
        lines.append("")
        rf = {view: np.mean(machine[family][view]["rf_cumulative"]["delta"]) for view in VIEWS}
        if family == "unsw8":
            lines += [
                "UNSW8 中，C1 累计 RF 明显上升，C2 完全不变，C3 与 C4 没有同方向恶化。",
                "C5 仍然上升，但幅度小于 C1。这支持定向规则在 UNSW8 中直接制造了更强的真实上行长度指纹，组合统计不是唯一来源。", "",
            ]
        else:
            lines += [
                "Deakin10 中，C1、C2、C3、C4 的累计 RF 均未出现有意义的单独恶化，C5 却稳定上升。",
                "因此 Deakin 的 C 视图增益不能归因于包数、总字节、大包比例或真实上行长度中的任一单项，更符合联合特征空间中的非线性交互。", "",
            ]
        machine[family]["rf_cumulative_mean_delta"] = rf
    lines += [
        "## 结论", "",
        "同一套确定性 padding 规则产生了两种不同的失败机制。UNSW8 的失败可由真实上行包长单视图直接观察。",
        "Deakin10 的单项统计都不足以解释组合视图的识别增益，说明 RF 利用了这些弱变化的联合划分。",
        "C2 在两数据集中逐 seed 完全一致，验证了逐流等花费重分配没有改变事件数量。C3 的微小差异来自窗口内可见 padding 字节不同，而不是逐流实际通信开销不同。",
        "本实验不支持用一个新的单特征确定性评分规则替代现有规则。后续设计需要同时约束单项分布与联合分布，且保留随机化，避免产生稳定的设备条件组合模式。", "",
    ]
    (ARTIFACTS / "weakener_c_ablation_summary.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "docs/Weakener_C视图特征消融.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

