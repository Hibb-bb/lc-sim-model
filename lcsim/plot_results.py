#!/usr/bin/env python3
"""Plots: (1) per-latent probe scores for one run, (2) fine-latent R^2 vs quality gap for a sweep.

    python plot_results.py results/base
    python plot_results.py --sweep results/gap_equal results/gap_mid results/base
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [("cls_acc", "class (acc)"), ("logP_r2", "log period (R²)"), ("amp_band_r2", "amplitude (R²)"), ("fine_r2", "fine bump (R²)")]
ORDER = ["contrastive", "lejepa", "split_only", "gate_only", "ours", "aug_only", "single_good", "lejepa_aa", "lejepa_aab", "contrastive_aa", "contrastive_aab"]
COLORS = {"contrastive": "#9aa5b1", "lejepa": "#6b7684", "split_only": "#e8b27a", "gate_only": "#e09a4a",
          "ours": "#c8641a", "aug_only": "#8fb3c9", "single_good": "#3b6e8f", "lejepa_aa": "#3b6e8f", "lejepa_aab": "#6b7684", "contrastive_aa": "#7fa7c4", "contrastive_aab": "#9aa5b1"}


def plot_run(d):
    r = json.load(open(os.path.join(d, "results.json")))
    methods = [m for m in ORDER if m in r["probes"]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (m, title) in zip(axes, METRICS):
        x = np.arange(len(methods))
        a = [r["probes"][v]["A"][m] for v in methods]
        b = [r["probes"][v]["B"][m] for v in methods]
        ax.bar(x - 0.2, a, 0.4, color=[COLORS[v] for v in methods], label="survey A (good)")
        ax.bar(x + 0.2, b, 0.4, color=[COLORS[v] for v in methods], alpha=0.45, hatch="//", label="survey B (bad)")
        if r["supervised"]:
            ax.axhline(r["supervised"]["A"][m], color="#3b6e8f", ls="--", lw=1, label="supervised ceiling, raw A")
            ax.axhline(r["supervised"]["B"][m], color="#9aa5b1", ls=":", lw=1, label="supervised ceiling, raw B")
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.set_ylim(bottom=min(0, min(a + b) - 0.05))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Linear probes on frozen embeddings (solid: good survey, hatched: bad survey)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(d, "probes.png")
    fig.savefig(out, dpi=130); print("wrote", out)

    # shared vs private slice
    rows = [(v, k) for v in methods for k in ("A_zs", "A_zp", "B_zs", "B_zp") if k in r["probes"][v]]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(rows)); w = 0.2
        for i, (m, title) in enumerate(METRICS):
            ax.bar(x + (i - 1.5) * w, [r["probes"][v][k][m] for v, k in rows], w, label=title)
        ax.set_xticks(x); ax.set_xticklabels([f"{v}\n{k}" for v, k in rows], fontsize=8)
        ax.set_title("Probes on the shared slice z_s vs the private slice z_p", fontsize=10)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        out = os.path.join(d, "slices.png"); fig.savefig(out, dpi=130); print("wrote", out)


def plot_sweep(dirs):
    runs = [json.load(open(os.path.join(d, "results.json"))) for d in dirs]
    gaps = [r["args"]["sigma_bad"] / r["args"]["sigma_good"] for r in runs]
    order = np.argsort(gaps); runs = [runs[i] for i in order]; gaps = [gaps[i] for i in order]
    methods = [m for m in ORDER if all(m in r["probes"] for r in runs)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, survey in zip(axes, "AB"):
        for v in methods:
            ax.plot(gaps, [r["probes"][v][survey]["fine_r2"] for r in runs], "o-", color=COLORS[v], label=v)
        if runs[0]["supervised"]:
            ax.plot(gaps, [r["supervised"][survey]["fine_r2"] for r in runs], "k--", lw=1, label="supervised on raw")
        ax.set_xscale("log"); ax.set_xlabel("quality gap  σ_bad / σ_good"); ax.set_ylabel("fine bump R²")
        ax.set_title(f"probed on survey {survey} ({'good' if survey == 'A' else 'bad'})", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(dirs[-1], "sweep_fine.png"); fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("dirs", nargs="+"); p.add_argument("--sweep", action="store_true")
    a = p.parse_args()
    if a.sweep:
        plot_sweep(a.dirs)
    else:
        for d in a.dirs:
            plot_run(d)
