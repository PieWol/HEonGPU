#!/usr/bin/env python3
"""Plot single-CT runtime scaling for sorting, minimum, and median."""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = Path(__file__).parent / "all_benchmark_results.csv"
OUT_PATH = Path(__file__).parent / "order_statistics_scaling.pdf"


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

    sort_n, sort_t = [], []
    min_n, min_t = [], []
    med_n, med_t = [], []

    for row in rows:
        n, t = safe_float(row.get("sorting.N", "")), safe_float(row.get("sorting.sort_ms", ""))
        if n is not None and t is not None:
            sort_n.append(int(n))
            sort_t.append(t / 1000.0)

        n, t = safe_float(row.get("minimum.N", "")), safe_float(row.get("minimum.min_ms", ""))
        if n is not None and t is not None:
            min_n.append(int(n))
            min_t.append(t / 1000.0)

        n, t = safe_float(row.get("median.N", "")), safe_float(row.get("median.median_ms", ""))
        if n is not None and t is not None:
            med_n.append(int(n))
            med_t.append(t / 1000.0)

    print(f"Sorting: N={sort_n}, t={[f'{x:.3f}' for x in sort_t]}")
    print(f"Minimum: N={min_n}, t={[f'{x:.3f}' for x in min_t]}")
    print(f"Median:  N={med_n}, t={[f'{x:.3f}' for x in med_t]}")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(sort_n, sort_t, "D-", color="#2ca02c", markersize=5,
            linewidth=1.5, label="Sorting", zorder=3)
    ax.plot(min_n, min_t, "^-", color="#ff7f0e", markersize=5,
            linewidth=1.5, label="Minimum", zorder=3)
    ax.plot(med_n, med_t, "v-", color="#9467bd", markersize=5,
            linewidth=1.5, label="Median", zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Input size $N$", fontsize=11)
    ax.set_ylabel("Runtime (s)", fontsize=11)
    ax.set_title("Order Statistics Runtime Scaling (NVIDIA L40)", fontsize=12)

    all_n = sorted(set(sort_n + min_n + med_n))
    ax.set_xticks(all_n)
    ax.set_xticklabels([str(n) for n in all_n], fontsize=9)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

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
