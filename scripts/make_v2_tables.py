"""Generate paper tables from v2 result JSON files (single source of truth)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "artifacts" / "es"


def load(name):
    return json.loads((E / name).read_text(encoding="utf-8"))


def f(value, digits=4):
    return f"{value:.{digits}f}"


def main() -> None:
    lines = ["# v2 论文表格（自动生成）", ""]
    lines.append("所有数字来自 `artifacts/es/v2_*.json`；`run_id` 见各文件。")
    lines.append("")

    lines += ["## 表 1 搜索（in-sample 3 折，Δ = Base − Searched）", ""]
    lines.append("| dataset | mode | Base | Searched | Δ | folds | sec |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, label in [
        ("v2_unsw8_position", "UNSW8/position"),
        ("v2_unsw8_gap", "UNSW8/gap"),
        ("v2_unsw8_length", "UNSW8/length"),
        ("v2_cic9_position", "CIC9/position"),
        ("v2_cic9_gap", "CIC9/gap"),
        ("v2_unsw2018_position", "UNSW2018/position"),
        ("v2_unsw2018_gap", "UNSW2018/gap"),
        ("v2_mon_position", "Mon/position"),
    ]:
        j = load(name + ".json")
        lines.append(
            f"| {label} | {j['mode']} | {f(j['base']['mean'])} | {f(j['searched']['mean'])} | "
            f"{f(j['delta_mean'])} ± {f(j['delta_se'])} | {j['folds_improved']}/{j['folds']} | {j['seconds']:.0f} |"
        )
    lines.append("")

    lines += ["## 表 2 Held-out A（Δ = Base − Full；clean / retrained）", ""]
    lines.append("| dataset/mechanism | clean LR | clean RF | clean MLP | retr LR | retr RF | retr MLP |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, label in [
        ("v2_heldout_unsw8_pos", "UNSW/position"),
        ("v2_heldout_unsw8_gap", "UNSW/gap"),
        ("v2_heldout_unsw8_len", "UNSW/length"),
        ("v2_heldout_cic9_pos", "CIC/position"),
        ("v2_heldout_cic9_gap", "CIC/gap"),
        ("v2_heldout_unsw2018_pos", "UNSW2018/position"),
        ("v2_heldout_unsw2018_gap", "UNSW2018/gap"),
    ]:
        j = load(name + ".json")
        cells = []
        for scenario in ("clean", "retrained"):
            for attacker in ("lr", "rf", "mlp"):
                cells.append(f(j["results"]["A"][scenario][attacker]["delta"]["delta_mean"]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 表 2b Held-out A retrained 配对 CI（Δ [95% CI]）", ""]
    lines.append("| dataset/mechanism | LR | RF | MLP |")
    lines.append("| --- | --- | --- | --- |")
    for name, label in [
        ("v2_heldout_unsw8_pos", "UNSW/position"),
        ("v2_heldout_unsw8_gap", "UNSW/gap"),
        ("v2_heldout_unsw8_len", "UNSW/length"),
        ("v2_heldout_cic9_pos", "CIC/position"),
        ("v2_heldout_cic9_gap", "CIC/gap"),
        ("v2_heldout_unsw2018_pos", "UNSW2018/position"),
        ("v2_heldout_unsw2018_gap", "UNSW2018/gap"),
    ]:
        j = load(name + ".json")
        cells = []
        for attacker in ("lr", "rf", "mlp"):
            d = j["results"]["A"]["retrained"][attacker]["delta"]
            cells.append(f"{d['delta_mean']:.4f} [{d['delta_ci_low']:.4f},{d['delta_ci_high']:.4f}]")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 表 2c 机制分解（LR，同一批 8k；正 = 防御降低 BA）", ""]
    lines.append("| dataset | CC | CD_complete | CD_visible | CD_merged | representation | displacement | dummy_mixing | total |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for name, label in [
        ("v2_mechanism_unsw8", "UNSW"),
        ("v2_mechanism_cic9", "CIC"),
        ("v2_mechanism_unsw2018", "UNSW2018"),
        ("v2_mechanism_deakin60", "Deakin60"),
    ]:
        j = load(name + ".json")
        v = j["variants"]; c = j["components"]
        def comp(key):
            item = c[key]
            return f"{item['value']:.4f} [{item['ci'][0]:.4f},{item['ci'][1]:.4f}]"
        lines.append(
            f"| {label} | {v['CC']['ba']:.4f} | {v['CD_complete']['ba']:.4f} | "
            f"{v['CD_visible']['ba']:.4f} | {v['CD_merged']['ba']:.4f} | "
            f"{comp('representation')} | {comp('displacement')} | {comp('dummy_mixing')} | {comp('total')} |"
        )
    lines.append("")

    lines += ["## 表 2d representation 审计（真实包 padding）", ""]
    lines.append("| dataset | split | changed flows | changed events | mean added B | hist identical | count mismatch |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name, label in [
        ("v2_representation_unsw8", "UNSW"),
        ("v2_representation_cic9", "CIC"),
        ("v2_representation_unsw2018", "UNSW2018"),
        ("v2_representation_deakin60", "Deakin60"),
    ]:
        j = load(name + ".json")
        for split in ("train", "test"):
            s = j[split]
            lines.append(
                f"| {label} | {split} | {s['changed_flow_fraction']:.3f} | {s['changed_event_fraction']:.3f} | "
                f"{s['real_added_bytes_mean']:.1f} | {s['hist_identical_fraction']:.3f} | {s['count_mismatch']} |"
            )
    lines.append("")
    lines.append("控制：padding_prob=0 时，UNSW 8000/8000 流 hist 完全一致、changed_flows=0。")
    lines.append("")

    lines += ["## 表 3 长度直方图（同一批 8k，16 bins，LR）", ""]
    lines.append("| version | C→C | C→D merged | D→D merged | C→D real | D→D real |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    versions = [
        ("v2_reconcile_unsw8", "unsw8_v1"),
        ("v2_reconcile_cic9", "cic9_v2"),
        ("v2_reconcile_unsw2018", "unsw2018_v1"),
        ("v2_reconcile_deakin60", "deakin60_v1"),
        ("v2_reconcile_deakin_full", "deakin_full_v1"),
    ]
    for name, label in versions:
        v = load(name + ".json")["variants"]
        lines.append(
            f"| {label} | {f(v['C_to_C']['ba'])} | {f(v['C_to_D_merged']['ba'])} | "
            f"{f(v['D_to_D_merged']['ba'])} | {f(v['C_to_D_real']['ba'])} | {f(v['D_to_D_real']['ba'])} |"
        )
    lines.append("")

    lines += ["## 表 4 dummy-only（support/recall_sum 见 JSON）", ""]
    lines.append("| dataset | LR | RF | MLP |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, label in [
        ("v2_dummy_only_unsw8", "UNSW"),
        ("v2_dummy_only_cic9", "CIC"),
        ("v2_dummy_only_unsw2018", "UNSW2018"),
        ("v2_dummy_only_deakin", "Deakin"),
    ]:
        j = load(name + ".json")
        lines.append(f"| {label} | {f(j['lr']['ba'],3)} | {f(j['rf']['ba'],3)} | {f(j['mlp']['ba'],3)} |")
    lines.append("")

    lines += ["## 表 5 32 窗口位移（base config）", ""]
    lines.append("| dataset | displaced mean | median | p95 | any | retained dummy mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, label in [
        ("v2_window_unsw8", "UNSW"),
        ("v2_window_cic9", "CIC"),
        ("v2_window_unsw2018", "UNSW2018"),
        ("v2_window_deakin", "Deakin"),
    ]:
        j = load(name + ".json")
        d = j["displaced_real"]; r = j["retained_dummy"]
        lines.append(
            f"| {label} | {d['mean']:.3f} | {d['median']:.1f} | {d['p95']:.1f} | "
            f"{d['fraction_any']:.3f} | {r['mean']:.3f} |"
        )
    lines.append("")

    lines += ["## 表 6 Position ablation（BA）", ""]
    j = load("v2_unsw8_position_ablation.json")["summary"]
    lines.append("| attacker | base | class0 | class5 | both | searched |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for attacker in ("lr", "rf", "mlp"):
        s = j[attacker]
        lines.append(
            "| " + attacker + " | " + " | ".join(
                f(s[key]["ba"]) for key in ("base", "class0", "class5", "both", "searched")
            ) + " |"
        )
    lines.append("")

    lines += ["## 表 7 LOCO ΔBA（Full − Full-cX）", ""]
    j = load("v2_unsw8_loco.json")["summary"]
    lines.append("| attacker | Base | c0 | c5 | c6 | c7 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for attacker in ("lr", "rf", "mlp"):
        s = j[attacker]
        lines.append(
            "| " + attacker + " | " + " | ".join(
                f(s[key]["delta_ba"]) for key in ("Base", "Full-c0", "Full-c5", "Full-c6", "Full-c7")
            ) + " |"
        )
    lines.append("")

    lines += ["## 表 8 c5×c6 factorial（BA + class5 interaction）", ""]
    j = load("v2_unsw8_factorial_c5_c6.json")["summary"]
    lines.append("| attacker | 11 | 10 | 01 | 00 | I_class5 [CI] |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for attacker in ("lr", "rf", "mlp"):
        s = j[attacker]
        lines.append(
            f"| {attacker} | " + " | ".join(f(s['ba'][key]) for key in ('11', '10', '01', '00'))
            + f" | {s['class5_interaction']:.3f} [{s['class5_interaction_ci_low']:.3f},{s['class5_interaction_ci_high']:.3f}] |"
        )
    lines.append("")

    lines += ["## 表 9 UNSW 视角/组合（retrained，Δ = Base − Full）", ""]
    j = load("v2_unsw8_views.json")["results"]["retrained"]
    lines.append("| view | LR Δ | RF Δ | MLP Δ |")
    lines.append("| --- | ---: | ---: | ---: |")
    for view in ("combined", "length", "gap", "sequence", "direction", "mask", "length_hist", "gap_hist"):
        x = j[view]
        lines.append(
            f"| {view} | " + " | ".join(f(x[a]["base"]["ba"] - x[a]["full"]["ba"]) for a in ("lr", "rf", "mlp")) + " |"
        )
    j = load("v2_unsw8_combos.json")["results"]["retrained"]
    lines.append("")
    lines.append("| combo | LR Δ | RF Δ | MLP Δ |")
    lines.append("| --- | ---: | ---: | ---: |")
    for combo in ("L", "G", "S", "LG", "LS", "GS", "LGS"):
        x = j[combo]
        lines.append(
            f"| {combo} | " + " | ".join(f(x[a]["base"]["ba"] - x[a]["full"]["ba"]) for a in ("lr", "rf", "mlp")) + " |"
        )
    lines.append("")

    lines += ["## 表 10 其他数据集视角（LR，Δ）", ""]
    lines.append("| dataset | scenario | combined | length | gap | sequence |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for name, label in [("v2_cic9_feature_groups", "CIC"), ("v2_unsw2018_feature_groups", "UNSW2018")]:
        j = load(name + ".json")
        for scenario in ("clean", "retrained"):
            cells = []
            for view in ("combined", "length", "gap", "sequence"):
                x = j["results"][scenario][view]["lr"]
                cells.append(f(x["base"]["ba"] - x["full"]["ba"]))
            lines.append(f"| {label} | {scenario} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## 表 11 Count vs histogram（LR，base BA）", ""]
    lines.append("| dataset | scenario | count | hist8 | hist16 | hist32 | hist64 | smooth |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, label in [
        ("v2_unsw8_count_hist", "UNSW"),
        ("v2_cic9_count_hist", "CIC"),
        ("v2_unsw2018_count_hist", "UNSW2018"),
        ("v2_deakin_count_hist", "Deakin"),
    ]:
        j = load(name + ".json")
        for scenario in ("clean", "retrained"):
            row = []
            for view in ("count", "length_hist8", "length_hist", "length_hist32", "length_hist64", "length_hist_smooth"):
                value = j["results"][scenario].get(view)
                row.append(f(value["lr"]["base"]["ba"]) if value else "—")
            lines.append(f"| {label} | {scenario} | " + " | ".join(row) + " |")
    lines.append("")

    output = ROOT / "docs" / "v2_tables.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
