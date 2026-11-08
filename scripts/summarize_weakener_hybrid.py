"""Summarize the three-arm random long-packet budget experiment."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/es"
ARMS = ("Baseline", "Hybrid-25", "Hybrid-50")


def main():
    lines = [
        "# Weakener 长包随机预算配对实验", "",
        "实验保持 Full New5、五个 dummy、逐流实际真实包 padding 字节、在线标签协议和 Blind flows 不变。",
        "长包阈值仅使用 warmup 决定：在至少保留 64 B headroom 的真实上行事件中取长度上四分位数。",
        "Hybrid 先在长包候选中随机分配预留预算。长包 headroom 不足时，剩余预算按冻结规则回退到全部合法事件。", "",
    ]
    summary = {}
    for family in ("unsw8", "deakin10"):
        logs = [json.loads((ARTIFACTS / f"weakener_hybrid_{family}_s{s}.json").read_text(encoding="utf-8"))
                for s in range(3)]
        summary[family] = {}
        lines += [f"## {family}", "",
                  f"warmup 长包阈值为 **{logs[0]['long_length_threshold']:.0f} B**，最低候选 headroom 为 64 B。", "",
                  "| seed | 实际长包预算比例 Baseline / H25 / H50 | 回退比例 H25 / H50 | 实际 B/flow |",
                  "|---:|---|---|---|"]
        for seed, log in enumerate(logs):
            actual = " / ".join(f"{v:.1%}" for v in log["actual_long_fraction"])
            fallback = " / ".join(f"{v:.1%}" for v in log["fallback_fraction"][1:])
            spent = " / ".join(f"{v:.2f}" for v in log["mean_spent_per_blind_flow"])
            lines.append(f"| {seed} | {actual} | {fallback} | {spent} |")
        lines += ["", "目标25%/50%是预留比例。全部事件回退仍可能落在长包，因此判断以表中的实际比例为准。", "",
                  "| 视图 | 攻击 | Baseline | Hybrid-25 | Δ25 | Hybrid-50 | Δ50 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        views = [v for v in ("A", "B", "C", "D", "real_uplink") if v in logs[0]["results"]]
        for view in views:
            summary[family][view] = {}
            for model in ("lr", "rf"):
                for mode in ("cumulative", "sliding"):
                    values = np.asarray([[log["results"][view][model][arm]["summary"][mode]["mean_post_ba"]
                                          for arm in ARMS] for log in logs])
                    means = values.mean(axis=0)
                    delta25, delta50 = values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]
                    summary[family][view][f"{model}_{mode}"] = {
                        "means": means.tolist(), "delta25": delta25.tolist(), "delta50": delta50.tolist()}
                    lines.append(f"| {view} | {model.upper()} {mode} | {means[0]:.4f} | {means[1]:.4f} | "
                                 f"{delta25.mean():+.4f} | {means[2]:.4f} | {delta50.mean():+.4f} |")
        lines.append("")
        if family == "unsw8":
            lines += [
                "完整45维可观测 C 的累计 RF 在 H25/H50 下均下降约一个百分点，三 seed 同向。",
                "原64维 D 未得到同样稳定的收益，真实上行单视图在 H50 下反而更容易识别。",
                "因此收益属于表示和模型相关的联合效应，不能解释为长包本身被更好隐藏。", "",
            ]
        else:
            lines += [
                "完整45维可观测 C 的累计 RF 在 H50 下明显下降，但累计 LR 在 H25 下明显上升。",
                "真实上行包长的 LR/RF 均随长包偏置增强，说明预算集中本身制造了更明显的长度规律。",
                "C-RF 的下降来自联合表示中的模型交互，不构成跨攻击者的统一防御改进。", "",
            ]
    lines += [
        "## 结论", "",
        "随机化候选内部顺序避免了上一轮确定性排序的直接失败，但长包预算偏置仍没有形成模型一致的收益。",
        "Hybrid-50 对两数据集的可观测 C-RF 有改善，却增强了真实上行长度攻击，并在 Deakin 的累计 LR 上产生反向效果。",
        "所以当前结果适合作为机制证据，不应替换 Baseline。继续微调比例容易转化为针对特定攻击模型的优化。", "",
        "Deakin 的 A/B/C 是真正从防御后64维可观测输入重建的17/17/45维视图，不读取 role、State 或 Blind 标签。",
        "real_uplink 仅作为使用 role 标记的机制审计。", "",
    ]
    (ARTIFACTS / "weakener_hybrid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "docs/Weakener长包随机预算.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
