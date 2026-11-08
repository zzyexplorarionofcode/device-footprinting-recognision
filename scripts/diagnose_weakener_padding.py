"""Diagnose length-bin migration, candidate selection, and device-pair effects."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/es"
ARMS = ("Random-all", "Random-candidate", "Targeted-candidate")


def js_divergence(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a / a.sum() if a.sum() else np.full_like(a, 1 / len(a))
    b = b / b.sum() if b.sum() else np.full_like(b, 1 / len(b))
    m = (a + b) / 2
    terms = []
    for p in (a, b):
        keep = p > 0
        terms.append(np.sum(p[keep] * np.log2(p[keep] / m[keep])))
    return float(sum(terms) / 2)


def pairwise_js(distributions, labels):
    return {(int(a), int(b)): js_divergence(distributions[a], distributions[b])
            for a, b in combinations(labels, 2)}


def class_distributions(transitions, y, blind, labels):
    result = {}
    for label in labels:
        rows = blind & (y == label)
        result[int(label)] = transitions[rows].sum(axis=(0, 1))
    return result


def class_transition_patterns(transitions, y, blind, labels):
    result = {}
    for label in labels:
        rows = blind & (y == label)
        matrix = transitions[rows].sum(axis=0).astype(float)
        # Keep only changed-bin events. The flattened conditional migration pattern
        # measures whether the intervention rule itself differs by device.
        np.fill_diagonal(matrix, 0)
        result[int(label)] = matrix.ravel()
    return result


def entropy(values):
    values = np.asarray(values, float)
    values = values / values.sum() if values.sum() else values
    keep = values > 0
    return float(-np.sum(values[keep] * np.log2(values[keep])))


def confusion_pairs(matrix, labels):
    matrix = np.asarray(matrix, float)
    den = matrix.sum(axis=1)
    den[den == 0] = 1
    rates = matrix / den[:, None]
    return {(int(labels[i]), int(labels[j])): float((rates[i, j] + rates[j, i]) / 2)
            for i, j in combinations(range(len(labels)), 2)}


def fmt_delta(values):
    return f"{np.mean(values):+.5f} [{min(values):+.5f}, {max(values):+.5f}]"


def main():
    lines = [
        "# Weakener 定向 padding 机制诊断", "",
        "本报告复用阶段二完全相同的 Full New5 在线回放、逐流 padding 花费和三组配对 seed。",
        "Random-all 从全部合法真实上行事件随机分配。Random-candidate 只在 warmup 候选桶内随机分配。",
        "Targeted-candidate 在同一候选集合内按稳定辨识分降序分配。三臂逐 flow 实际花费相同。", "",
        "长度迁移使用 0–1514 B 的 16 个等宽桶。主 JS 口径只统计最终 32-event 窗口内可见的真实上行事件。",
        "每个 seed 对全部无序设备对等权汇总。混淆采用 C 视图累计 RF 的逐日 Post 预测，并在全期聚合。", "",
    ]
    machine = {}
    for family in ("unsw8", "deakin10"):
        seed_rows, pair_rows, logs = [], [], []
        for seed in range(3):
            log = json.loads((ARTIFACTS / f"weakener_padding_{family}_s{seed}.json").read_text(encoding="utf-8"))
            logs.append(log)
            with np.load(ARTIFACTS / log["flow_audit"], allow_pickle=False) as z:
                y, blind_index = z["labels"], z["blind"].astype(int)
                blind = np.zeros(len(y), dtype=bool)
                blind[blind_index] = True
                labels = np.unique(y[blind])
                js, migration_js, migration_rate, destination_entropy = {}, {}, {}, {}
                for arm in ARMS:
                    key = f"visible_transition_{arm.replace('-', '_')}"
                    transitions = z[key]
                    js[arm] = pairwise_js(class_distributions(transitions, y, blind, labels), labels)
                    migration_js[arm] = pairwise_js(
                        class_transition_patterns(transitions, y, blind, labels), labels)
                    matrix = transitions[blind].sum(axis=0).astype(float)
                    changed = matrix.copy()
                    np.fill_diagonal(changed, 0)
                    migration_rate[arm] = float(changed.sum() / matrix.sum())
                    destination_entropy[arm] = entropy(changed.sum(axis=0))
            confusion = {arm: confusion_pairs(log["confusion"][arm]["cumulative_rf"],
                                               log["confusion"][arm]["labels"])
                         for arm in ARMS}
            recall_delta = {}
            base_matrix = np.asarray(log["confusion"][ARMS[0]]["cumulative_rf"], float)
            target_matrix = np.asarray(log["confusion"][ARMS[2]]["cumulative_rf"], float)
            for i, label in enumerate(log["confusion"][ARMS[0]]["labels"]):
                base_recall = base_matrix[i, i] / max(1, base_matrix[i].sum())
                target_recall = target_matrix[i, i] / max(1, target_matrix[i].sum())
                recall_delta[int(label)] = float(target_recall - base_recall)
            pairs = sorted(js[ARMS[0]])
            seed_rows.append({
                "seed": seed,
                "js_random_all": np.mean([js[ARMS[0]][p] for p in pairs]),
                "js_random_candidate": np.mean([js[ARMS[1]][p] for p in pairs]),
                "js_targeted": np.mean([js[ARMS[2]][p] for p in pairs]),
                "conf_random_all": np.mean([confusion[ARMS[0]][p] for p in pairs]),
                "conf_random_candidate": np.mean([confusion[ARMS[1]][p] for p in pairs]),
                "conf_targeted": np.mean([confusion[ARMS[2]][p] for p in pairs]),
                "migration_js_random_all": np.mean([migration_js[ARMS[0]][p] for p in pairs]),
                "migration_js_random_candidate": np.mean([migration_js[ARMS[1]][p] for p in pairs]),
                "migration_js_targeted": np.mean([migration_js[ARMS[2]][p] for p in pairs]),
                "migration_rate": migration_rate,
                "destination_entropy": destination_entropy,
                "recall_delta": recall_delta,
            })
            for p in pairs:
                pair_rows.append({"seed": seed, "pair": p,
                                  "js_rc_minus_ra": js[ARMS[1]][p] - js[ARMS[0]][p],
                                  "js_t_minus_ra": js[ARMS[2]][p] - js[ARMS[0]][p],
                                  "js_t_minus_rc": js[ARMS[2]][p] - js[ARMS[1]][p],
                                  "conf_rc_minus_ra": confusion[ARMS[1]][p] - confusion[ARMS[0]][p],
                                  "conf_t_minus_ra": confusion[ARMS[2]][p] - confusion[ARMS[0]][p]})
        machine[family] = {"seeds": seed_rows, "pairs": pair_rows}
        lines += [f"## {family}", "", "### 三臂识别结果", "",
                  "| 视图与攻击 | Random-all | Random-candidate | Targeted-candidate | T−RA | T−RC |",
                  "|---|---:|---:|---:|---:|---:|"]
        for view, model in (("C", "lr"), ("C", "rf"), ("D", "rf"), ("real_uplink", "rf")):
            values = np.asarray([[log["results"][view][model][arm]["summary"]["cumulative"]["mean_post_ba"]
                                  for arm in ARMS] for log in logs])
            means = values.mean(axis=0)
            lines.append(f"| {view} / 累计 {model.upper()} | {means[0]:.4f} | {means[1]:.4f} | "
                         f"{means[2]:.4f} | {means[2]-means[0]:+.4f} | {means[2]-means[1]:+.4f} |")
        lines += ["", "### 设备对 JS divergence", "",
                  "| seed | Random-all | Random-candidate | Targeted-candidate | RC−RA | T−RA | T−RC |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in seed_rows:
            lines.append(f"| {r['seed']} | {r['js_random_all']:.5f} | {r['js_random_candidate']:.5f} | "
                         f"{r['js_targeted']:.5f} | {r['js_random_candidate']-r['js_random_all']:+.5f} | "
                         f"{r['js_targeted']-r['js_random_all']:+.5f} | "
                         f"{r['js_targeted']-r['js_random_candidate']:+.5f} |")
        grouped = {}
        for r in pair_rows:
            grouped.setdefault(r["pair"], []).append(r)
        ranked = sorted(grouped.items(), key=lambda item: np.mean([r["js_t_minus_ra"] for r in item[1]]), reverse=True)
        lines += ["", "Targeted-candidate 相对 Random-all 的设备对 JS 变化最大者：", "",
                  "| 设备对 | ΔJS 均值 | 三 seed | Δ双向混淆率均值 |",
                  "|---|---:|---|---:|"]
        for pair, rows in ranked[:5] + ranked[-5:]:
            lines.append(f"| {pair[0]}–{pair[1]} | {np.mean([r['js_t_minus_ra'] for r in rows]):+.5f} | "
                         f"{' / '.join(f'{r['js_t_minus_ra']:+.5f}' for r in rows)} | "
                         f"{np.mean([r['conf_t_minus_ra'] for r in rows]):+.5f} |")
        lines += ["", "### 全设备对汇总", "",
                  f"- Random-candidate − Random-all 的 JS：{fmt_delta([r['js_random_candidate']-r['js_random_all'] for r in seed_rows])}。",
                  f"- Targeted-candidate − Random-all 的 JS：{fmt_delta([r['js_targeted']-r['js_random_all'] for r in seed_rows])}。",
                  f"- Targeted-candidate − Random-candidate 的 JS：{fmt_delta([r['js_targeted']-r['js_random_candidate'] for r in seed_rows])}。",
                  f"- Targeted-candidate − Random-all 的平均双向混淆率：{fmt_delta([r['conf_targeted']-r['conf_random_all'] for r in seed_rows])}。", ""]
        lines += ["### 确定性迁移模式", "",
                  "| seed | 迁移模式 JS Random-all | Random-candidate | Targeted-candidate | T−RA | 目标桶熵 RA | RC | T |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in seed_rows:
            ent = r["destination_entropy"]
            lines.append(f"| {r['seed']} | {r['migration_js_random_all']:.5f} | "
                         f"{r['migration_js_random_candidate']:.5f} | {r['migration_js_targeted']:.5f} | "
                         f"{r['migration_js_targeted']-r['migration_js_random_all']:+.5f} | "
                         f"{ent[ARMS[0]]:.3f} | {ent[ARMS[1]]:.3f} | {ent[ARMS[2]]:.3f} |")
        lines.append("")
        recall_labels = sorted(seed_rows[0]["recall_delta"])
        lines += ["### 逐设备 recall 变化", "",
                  "Targeted-candidate − Random-all，C 视图累计 RF：", "",
                  "| 设备 | 平均 ΔRecall | 三 seed |",
                  "|---:|---:|---|"]
        for label in sorted(recall_labels,
                            key=lambda c: np.mean([r["recall_delta"][c] for r in seed_rows]),
                            reverse=True):
            values = [r["recall_delta"][label] for r in seed_rows]
            lines.append(f"| {label} | {np.mean(values):+.4f} | "
                         f"{' / '.join(f'{v:+.4f}' for v in values)} |")
        lines.append("")
        lines += ["### 数据集内判断", ""]
        if family == "unsw8":
            lines += [
                "- 候选内随机没有稳定提高识别率。确定性排序相对候选内随机提高 C 视图累计 RF，三 seed 同向。",
                "- Targeted 同时提高最终包长分布的设备对 JS，并使设备条件迁移模式的 JS 大幅增加。",
                "- 真实上行包长视图累计 RF 增加约 0.0404，支持确定性填充顺序形成新的可识别长度规律。设备对相关仅为弱负且未显著，不能把每个 pair 的识别变化都归结为 JS。",
            ]
        else:
            lines += [
                "- 候选内随机相对全候选随机的效果跨 seed 变号。确定性排序相对候选内随机提高 C 视图累计 RF，三 seed 同向。",
                "- Targeted 提高最终包长分布的设备对 JS，目标桶熵也更低。迁移模式 JS 相对 Random-all 并未三 seed 同向。",
                "- C 视图累计 RF 增加约 0.0241，但真实上行包长视图累计 RF 约下降 0.0014。识别增益主要集中在设备 9、8 和 6，更符合包长变化与 C 中其他统计的模型交互，不支持单独的新长度指纹解释。",
            ]
        lines.append("")
        lines += ["### 分布分离与识别变化的关联", "",
                  "| seed | ρ(设备对 ΔJS, Δ双向混淆) | p | ρ(设备平均 ΔJS, ΔRecall) | p |",
                  "|---:|---:|---:|---:|---:|"]
        association = []
        for seed in range(3):
            rows = [r for r in pair_rows if r["seed"] == seed]
            pair_rho, pair_p = spearmanr([r["js_t_minus_ra"] for r in rows],
                                         [r["conf_t_minus_ra"] for r in rows])
            incident = {label: [] for label in seed_rows[seed]["recall_delta"]}
            for r in rows:
                for label in r["pair"]:
                    incident[label].append(r["js_t_minus_ra"])
            device_js = [np.mean(incident[label]) for label in sorted(incident)]
            device_recall = [seed_rows[seed]["recall_delta"][label] for label in sorted(incident)]
            device_rho, device_p = spearmanr(device_js, device_recall)
            association.append({"seed": seed, "pair_rho": float(pair_rho), "pair_p": float(pair_p),
                                "device_rho": float(device_rho), "device_p": float(device_p)})
            lines.append(f"| {seed} | {pair_rho:+.3f} | {pair_p:.3f} | {device_rho:+.3f} | {device_p:.3f} |")
        machine[family]["association"] = association
        lines += ["",
                  "设备对相关中，负 ρ 表示 JS 增幅越大，双向混淆越少。设备级相关中，正 ρ 表示设备相对其余类别越分离，recall 提升越大。",
                  "这些是诊断性关联，不构成独立因果估计。设备级样本量仅为类别数。", ""]
    lines += [
        "## 综合结论", "",
        "逐流等花费和同一事件序列排除了预算总量、dummy、位置及 Gap 的变化。识别率差异由真实上行包长的分配规则引起。",
        "Random-candidate 没有跨数据集、跨 seed 的稳定劣化，候选集合本身不是主要失败来源。",
        "Targeted-candidate 相对 Random-candidate 在两数据集的 C 视图累计 RF 中均三 seed 变差，且最终设备对包长 JS 均三 seed 上升。",
        "因此当前最强证据支持：确定性的候选内排序把相同预算集中到少数目标桶，并系统性重构防御后的设备条件分布。",
        "UNSW8 同时出现真实上行包长 RF 明显增强、最终包长 JS 上升和迁移张量设备差异增强，支持干预规则制造了更易识别的长度指纹。Deakin10 只在组合 C 视图中增强，包长单视图和 pair 级关联不支持同一解释，更可能涉及特征交互。",
        "所以“确定性规则可能成为新指纹”已经得到数据集内实证支持，但尚未成为跨数据集统一机制。JS 上升是机制线索，不是识别改善的充分条件。",
        "三组 seed 改变 Random 两臂的事件顺序与 Full New5 State 路径。Targeted 的稳定分排序没有随机分支，因此其真实包长分配在三 seed 中相同。三 seed 一致性应理解为相对不同随机参照均成立，不代表三个独立 Targeted 分配实现。",
        "本诊断不支持进入事件数量阶段，也不支持立即更换评分函数。下一步若继续，应先设计不产生设备条件确定性迁移的随机化分配规则，并沿用本报告的三层审计。", "",
    ]
    (ARTIFACTS / "weakener_padding_diagnostic.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "docs/Weakener定向padding机制诊断.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
