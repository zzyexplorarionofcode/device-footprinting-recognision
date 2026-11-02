"""Generate phase-1 v2 figures from the authoritative result JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "artifacts" / "es"
OUT = ROOT / "docs" / "figures" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

COLORS = {
    "position": "#2b6cb0",
    "gap": "#dd6b20",
    "length": "#718096",
    "LR": "#2b6cb0",
    "RF": "#2f855a",
    "MLP": "#b83280",
    "representation": "#2b6cb0",
    "displacement": "#dd6b20",
    "dummy_mixing": "#c53030",
    "total": "#4a5568",
    "CC": "#4a5568",
    "CD_merged": "#dd6b20",
    "DD_merged": "#2f855a",
}


def load(name):
    return json.loads((E / name).read_text(encoding="utf-8"))


def fig_search():
    rows = {
        "UNSW": ("v2_unsw8_position", "v2_unsw8_gap", "v2_unsw8_length"),
        "CIC": ("v2_cic9_position", "v2_cic9_gap", None),
        "UNSW2018": ("v2_unsw2018_position", "v2_unsw2018_gap", None),
        "Mon": ("v2_mon_position", None, None),
    }
    dims = ("position", "gap", "length")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(rows))
    width = 0.24
    for offset, dim in enumerate(dims):
        vals, errs = [], []
        for dataset in rows:
            name = rows[dataset][offset]
            if name is None:
                vals.append(0.0)
                errs.append(0.0)
            else:
                j = load(name + ".json")
                vals.append(j["delta_mean"])
                errs.append(j["delta_se"])
        ax.bar(x + (offset - 1) * width, vals, width, yerr=errs, capsize=2,
               label=dim, color=COLORS[dim], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(rows.keys())
    ax.set_ylabel("BA reduction (Base − Searched)")
    ax.set_title("In-sample 3-fold search: position is the only consistent dimension")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_search.png")
    plt.close(fig)


def fig_heldout():
    rows = [
        ("UNSW", "position", "v2_heldout_unsw8_pos"),
        ("UNSW", "gap", "v2_heldout_unsw8_gap"),
        ("UNSW", "length", "v2_heldout_unsw8_len"),
        ("CIC", "position", "v2_heldout_cic9_pos"),
        ("CIC", "gap", "v2_heldout_cic9_gap"),
        ("UNSW2018", "position", "v2_heldout_unsw2018_pos"),
        ("UNSW2018", "gap", "v2_heldout_unsw2018_gap"),
    ]
    labels = [f"{dataset}/{dim}" for dataset, dim, _ in rows]
    attackers = ("lr", "rf", "mlp")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(rows))
    height = 0.24
    for offset, attacker in enumerate(attackers):
        vals, low, high = [], [], []
        for _, _, name in rows:
            d = load(name + ".json")["results"]["A"]["retrained"][attacker]["delta"]
            vals.append(d["delta_mean"])
            low.append(d["delta_mean"] - d["delta_ci_low"])
            high.append(d["delta_ci_high"] - d["delta_mean"])
        ax.errorbar(vals, y + (offset - 1) * height, xerr=[low, high], fmt="o",
                    color=COLORS[attacker.upper()], capsize=2, markersize=4, label=attacker.upper())
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Held-out Δ BA (retrained; positive = defense helps)")
    ax.set_title("Held-out A paired CIs: effects are classifier-dependent")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_heldout_ci.png")
    plt.close(fig)


def fig_hist():
    versions = [
        ("UNSW", "v2_reconcile_unsw8"),
        ("CIC", "v2_reconcile_cic9"),
        ("UNSW2018", "v2_reconcile_unsw2018"),
        ("Deakin60", "v2_reconcile_deakin60"),
        ("Deakin-full", "v2_reconcile_deakin_full"),
    ]
    keys = (("C_to_C", "C→C"), ("C_to_D_merged", "C→D merged"), ("D_to_D_merged", "D→D merged"))
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(versions))
    width = 0.26
    for offset, (key, label) in enumerate(keys):
        vals = [load(name + ".json")["variants"][key]["ba"] for _, name in versions]
        color = COLORS.get(key.split("_")[0] + "_" + key.split("_")[-1], None)
        ax.bar(x + (offset - 1) * width, vals, width, label=label,
               color=["#4a5568", "#dd6b20", "#2f855a"][offset], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in versions])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Order-invariant length histogram: strong and dataset-dependent")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_histogram.png")
    plt.close(fig)


def fig_mechanism():
    rows = [
        ("UNSW", "v2_mechanism_unsw8"),
        ("CIC", "v2_mechanism_cic9"),
        ("UNSW2018", "v2_mechanism_unsw2018"),
        ("Deakin60", "v2_mechanism_deakin60"),
    ]
    comps = ("representation", "displacement", "dummy_mixing")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(rows))
    width = 0.2
    for offset, comp in enumerate(comps):
        vals, errs = [], []
        for _, name in rows:
            c = load(name + ".json")["components"][comp]
            vals.append(c["value"])
            errs.append((c["ci"][1] - c["ci"][0]) / 2)
        ax.bar(x + (offset - 1) * width, vals, width, yerr=errs, capsize=2,
               label=comp, color=COLORS[comp], edgecolor="white", linewidth=0.5)
    total = [load(name + ".json")["components"]["total"]["value"] for _, name in rows]
    ax.scatter(x + width, total, color=COLORS["total"], zorder=5, label="total", marker="D", s=28)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in rows])
    ax.set_ylabel("BA contribution")
    ax.set_title("Mechanism: dummy mixing dominates; real-packet displacement is small")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_mechanism.png")
    plt.close(fig)


def fig_window():
    rows = [
        ("UNSW", "v2_window_unsw8"),
        ("CIC", "v2_window_cic9"),
        ("UNSW2018", "v2_window_unsw2018"),
        ("Deakin", "v2_window_deakin"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    labels = [label for label, _ in rows]
    displaced = [load(name + ".json")["displaced_real"]["mean"] for _, name in rows]
    any_frac = [100 * load(name + ".json")["displaced_real"]["fraction_any"] for _, name in rows]
    axes[0].bar(labels, displaced, color="#2b6cb0")
    axes[0].set_ylabel("mean displaced real packets")
    axes[0].set_title("Displacement per flow")
    axes[1].bar(labels, any_frac, color="#dd6b20")
    axes[1].set_ylabel("% flows with ≥1 displaced real")
    axes[1].set_title("Share of affected flows")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("32-event post-insertion window: CIC loses the most real packets", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_window.png", bbox_inches="tight")
    plt.close(fig)


def fig_views():
    j = load("v2_unsw8_views.json")["results"]["retrained"]
    views = ("combined", "length", "gap", "sequence", "direction", "mask", "length_hist", "gap_hist")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(views))
    width = 0.24
    for offset, attacker in enumerate(("lr", "rf", "mlp")):
        vals = [j[view][attacker]["base"]["ba"] - j[view][attacker]["full"]["ba"] for view in views]
        ax.bar(x + (offset - 1) * width, vals, width, label=attacker.upper(),
               color=COLORS[attacker.upper()], edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=25, ha="right")
    ax.set_ylabel("Δ BA (Base − Full)")
    ax.set_title("What the defense changes: sequence-sensitive views, not length histograms")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_views.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_search()
    fig_heldout()
    fig_hist()
    fig_mechanism()
    fig_window()
    fig_views()
    print("saved to", OUT)
