#!/usr/bin/env python3
"""Plot ranking runtime scaling (basic + TC) from benchmark CSV."""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_PATH = Path(__file__).parent / "all_benchmark_results.csv"
OUT_PATH = Path(__file__).parent / "ranking_scaling.pdf"


def safe_float(val):
    if val is None or val.strip() == "":
        return None
    try:
        v = float(val)
        return None if math.isnan(v) else v
    except ValueError:
        return None


def main():
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        rows = list(reader)

    sc_basic_n, sc_basic_t = [], []
    sc_tc_n, sc_tc_t = [], []
    mc_basic_n, mc_basic_t = [], []
    mc_tc_n, mc_tc_t = [], []

    for row in rows:
        # Single-CT basic
        n, t = safe_float(row["ranking.N"]), safe_float(row["ranking.rank_ms"])
        if n is not None and t is not None:
            sc_basic_n.append(int(n))
            sc_basic_t.append(t / 1000.0)

        # Single-CT TC
        n, t = safe_float(row.get("ranking_tc.N", "")), safe_float(row.get("ranking_tc.rank_ms", ""))
        if n is not None and t is not None:
            sc_tc_n.append(int(n))
            sc_tc_t.append(t / 1000.0)

        # Multi-CT basic
        n, t = safe_float(row.get("ranking_multi.N", "")), safe_float(row.get("ranking_multi.rank_ms", ""))
        if n is not None and t is not None:
            mc_basic_n.append(int(n))
            mc_basic_t.append(t / 1000.0)

        # Multi-CT TC
        n, t = safe_float(row.get("ranking_multi_tc.N", "")), safe_float(row.get("ranking_multi_tc.rank_ms", ""))
        if n is not None and t is not None:
            mc_tc_n.append(int(n))
            mc_tc_t.append(t / 1000.0)

    print(f"Single-CT basic: N={sc_basic_n}, t={[f'{x:.3f}' for x in sc_basic_t]}")
    print(f"Single-CT TC:    N={sc_tc_n}, t={[f'{x:.3f}' for x in sc_tc_t]}")
    print(f"Multi-CT basic:  N={mc_basic_n}, t={[f'{x:.3f}' for x in mc_basic_t]}")
    print(f"Multi-CT TC:     N={mc_tc_n}, t={[f'{x:.3f}' for x in mc_tc_t]}")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot single-CT segments
    ax.plot(sc_basic_n, sc_basic_t, "o-", color="#1f77b4", markersize=5,
            linewidth=1.5, label="Ranking (basic)", zorder=3)
    ax.plot(sc_tc_n, sc_tc_t, "s-", color="#d62728", markersize=5,
            linewidth=1.5, label="Ranking (tie-corrected)", zorder=3)

    # Plot multi-CT segments (no extra legend entry)
    ax.plot(mc_basic_n, mc_basic_t, "o-", color="#1f77b4", markersize=5,
            linewidth=1.5, zorder=3)
    ax.plot(mc_tc_n, mc_tc_t, "s-", color="#d62728", markersize=5,
            linewidth=1.5, zorder=3)

    # Dashed connector across the single-CT / multi-CT boundary
    if sc_basic_n and mc_basic_n:
        ax.plot([sc_basic_n[-1], mc_basic_n[0]],
                [sc_basic_t[-1], mc_basic_t[0]],
                "--", color="#1f77b4", linewidth=1, zorder=2)
    if sc_tc_n and mc_tc_n:
        ax.plot([sc_tc_n[-1], mc_tc_n[0]],
                [sc_tc_t[-1], mc_tc_t[0]],
                "--", color="#d62728", linewidth=1, zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Input size $N$", fontsize=11)
    ax.set_ylabel("Runtime (s)", fontsize=11)
    ax.set_title("Ranking Runtime Scaling (NVIDIA L40)", fontsize=12)

    # Boundary annotation (placed after data so ylim is known)
    boundary_x = math.sqrt(128 * 256)
    ylo, yhi = ax.get_ylim()
    label_y = ylo * 1.5
    ax.axvline(boundary_x, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(boundary_x * 0.70, label_y, "single-CT", fontsize=8, color="grey",
            ha="right", va="bottom")
    ax.text(boundary_x * 1.45, label_y, "multi-CT", fontsize=8, color="grey",
            ha="left", va="bottom")

    # X-axis: powers of 2
    all_n = sorted(set(sc_basic_n + sc_tc_n + mc_basic_n + mc_tc_n))
    ax.set_xticks(all_n)
    ax.set_xticklabels([str(n) for n in all_n], fontsize=8, rotation=45)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:.0f}" if y >= 1 else f"{y:.2f}" if y >= 0.01 else f"{y:.3f}"
    ))

    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="major", linewidth=0.5, alpha=0.5)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"\nSaved to {OUT_PATH}")

    png_path = OUT_PATH.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()
