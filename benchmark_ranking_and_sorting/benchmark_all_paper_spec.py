#!/usr/bin/env python3
"""
Overarching benchmark runner for all HE order-statistics algorithms from:
  "Efficient Ranking, Order Statistics, and Sorting under CKKS"
  Mazzone et al., USENIX Security 2025

Runs ranking (basic + tie-corrected), sorting, minimum, and median for each N
and saves a combined CSV that allows direct comparison of timings across all
algorithms.  Per-algorithm N caps are enforced automatically:
  - ranking:    N <= 128
  - ranking_tc: N <= 64  (tie correction adds 2 levels)
  - sorting:    N <= 256
  - minimum:    N <= 128
  - median:     N <= 128

Usage:
    python3 benchmark_all.py [--n-values N1 N2 ...] [--runs R] [--output FILE]

    # Quick test:
    python3 benchmark_all.py --n-values 8 16 --runs 1

    # Full paper-equivalent sweep (default — includes N=256 for sorting):
    python3 benchmark_all.py

    # Custom N set, 5 runs each:
    python3 benchmark_all.py --n-values 8 16 32 64 128 256 --runs 5
"""

import subprocess
import csv
import sys
import argparse
import statistics
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BIN_DIR    = _SCRIPT_DIR.parent / "build/bin/examples/ranking_and_sorting"

# Each entry: (label, binary name, timing field, extra args, max N)
BENCHMARKS = [
    ("ranking",    "23_ckks_ranking_tie_correction",  "rank_ms",   [],                  128),
    ("ranking_tc", "23_ckks_ranking_tie_correction",  "rank_ms",   ["--tie-correction"], 64),
    ("sorting",    "22_ckks_sorting_paper",           "sort_ms",   [],                  256),
    ("minimum",    "19_ckks_minimum",                 "min_ms",    [],                  128),
    ("median",     "20_ckks_median",                  "median_ms", [],                  128),
]

N_VALUES_DEFAULT = [8, 16, 32, 64, 128, 256]


def run_once(binary: Path, n: int, extra_args: list[str] = []) -> dict | None:
    """Run binary for N in bench mode; return parsed BENCH: fields or None."""
    try:
        result = subprocess.run(
            [str(binary), str(n), "--bench"] + extra_args,
            capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"    ERROR (exit {result.returncode}): {result.stderr[:200]}",
              file=sys.stderr)
        return None

    for line in result.stdout.splitlines():
        if line.startswith("BENCH:"):
            data = {}
            for token in line[6:].strip().split():
                k, v = token.split("=")
                try:
                    data[k] = float(v) if "." in v else int(v)
                except ValueError:
                    data[k] = v
            return data

    print(f"    No BENCH: line found. stdout:\n{result.stdout[:200]}",
          file=sys.stderr)
    return None


def benchmark_one(binary: Path, label: str, timing_field: str,
                  n: int, runs: int, extra_args: list[str] = []) -> dict | None:
    """Run `runs` trials of one binary for one N; return averaged stats."""
    run_results = []
    for run_idx in range(runs):
        print(f"    [{label}] run {run_idx + 1}/{runs} ...", end=" ", flush=True)
        data = run_once(binary, n, extra_args)
        if data is None:
            print("FAILED")
            continue
        t = data.get(timing_field, 0)
        print(f"{timing_field}={t:.1f}ms  peak={data.get('gpu_peak_mib','?')}MiB")
        run_results.append(data)

    if not run_results:
        return None

    numeric_keys = [k for k in run_results[0]
                    if isinstance(run_results[0][k], (int, float))]
    avg  = {f"{label}.{k}": statistics.mean(r[k] for r in run_results)
            for k in numeric_keys}
    if len(run_results) > 1:
        avg.update({f"{label}.{k}_stdev": statistics.stdev(r[k] for r in run_results)
                    for k in numeric_keys})
    avg[f"{label}.runs"] = len(run_results)
    # Convenience: plain timing in seconds
    avg[f"{label}.{timing_field[:-3]}_s"] = avg[f"{label}.{timing_field}"] / 1000.0
    return avg


def print_comparison_table(results: list[dict]) -> None:
    """Print a side-by-side timing comparison table."""
    labels        = [b[0] for b in BENCHMARKS]
    timing_fields = [b[2] for b in BENCHMARKS]

    col_w = 14
    header = f"{'N':>6}  " + "  ".join(
        f"{f'{l}_ms':>{col_w}}" for l in labels
    ) + "  " + "  ".join(
        f"{f'{l}_s':>{col_w}}" for l in labels
    )
    sep = "=" * len(header)

    print(f"\n{sep}")
    print(header)
    print("-" * len(header))
    for r in results:
        ms_cols = "  ".join(
            f"{r.get(f'{l}.{tf}', float('nan')):>{col_w}.1f}"
            for l, tf in zip(labels, timing_fields)
        )
        s_cols = "  ".join(
            f"{r.get(f'{l}.{tf[:-3]}_s', float('nan')):>{col_w}.3f}"
            for l, tf in zip(labels, timing_fields)
        )
        print(f"{r['n']:>6}  {ms_cols}  {s_cols}")
    print(sep)

    ref_label = "ranking"
    ref_field = next(tf for l, _, tf, _, _ in BENCHMARKS if l == ref_label)
    print(f"\nRelative cost (normalised to {ref_label}):")
    print(f"{'N':>6}  " + "  ".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (8 + (col_w + 2) * len(labels)))
    for r in results:
        base = r.get(f"{ref_label}.{ref_field}", None)
        if not base or base != base:  # None or NaN
            continue
        rel_cols = "  ".join(
            f"{r.get(f'{l}.{tf}', float('nan')) / base:>{col_w}.3f}"
            for l, tf in zip(labels, timing_fields)
        )
        print(f"{r['n']:>6}  {rel_cols}")


def save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    all_keys = list(dict.fromkeys(k for r in results for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCombined results saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all HE order-statistics algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=N_VALUES_DEFAULT,
        metavar="N",
        help="Vector lengths to benchmark (powers of 2; per-algorithm caps apply)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed runs per algorithm per N (default: 3)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=_SCRIPT_DIR / "all_benchmark_results.csv",
        help="Combined CSV output file (default: all_benchmark_results.csv)"
    )
    parser.add_argument(
        "--bin-dir", type=Path, default=_BIN_DIR,
        help=f"Directory containing compiled binaries (default: {_BIN_DIR})"
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        metavar="LABEL",
        choices=[b[0] for b in BENCHMARKS],
        help="Skip one or more benchmarks (e.g. --skip sorting median)"
    )
    args = parser.parse_args()

    # Resolve binaries and validate they exist
    binaries: dict[str, Path] = {}
    active_benchmarks = [(l, b, tf, ea, mn) for l, b, tf, ea, mn in BENCHMARKS
                         if l not in args.skip]
    missing = []
    for label, binary_name, _, _, _ in active_benchmarks:
        path = args.bin_dir / binary_name
        binaries[label] = path
        if not path.exists():
            missing.append((label, path))

    if missing:
        for label, path in missing:
            print(f"Binary not found [{label}]: {path}")
        print("\nBuild missing binaries with:")
        for label, _, _, _, _ in active_benchmarks:
            bin_name = next(b for l, b, _, _, _ in BENCHMARKS if l == label)
            print(f"  cmake --build build --target {bin_name}")
        sys.exit(1)

    # Validate N values
    max_n = max(mn for _, _, _, _, mn in active_benchmarks)
    for n in args.n_values:
        if n <= 0 or (n & (n - 1)) != 0:
            print(f"Error: N={n} is not a positive power of 2")
            sys.exit(1)

    n_values = [n for n in args.n_values if n <= max_n]
    if not n_values:
        print("No valid N values.")
        sys.exit(1)

    active_labels = [l for l, _, _, _, _ in active_benchmarks]
    print(f"Algorithms : {', '.join(active_labels)}")
    print(f"N range    : {n_values}")
    print(f"Runs       : {args.runs} per algorithm per N")
    print(f"Bin dir    : {args.bin_dir}")

    all_results = []
    for n in n_values:
        print(f"\n{'='*60}")
        print(f"N = {n}")
        print(f"{'='*60}")
        row = {"n": n}
        for label, _, timing_field, extra_args, max_n in active_benchmarks:
            if n > max_n:
                print(f"    [{label}] skipped (N={n} > max {max_n})")
                row[f"{label}.{timing_field}"]        = float("nan")
                row[f"{label}.{timing_field[:-3]}_s"] = float("nan")
                continue
            result = benchmark_one(binaries[label], label, timing_field,
                                   n, args.runs, extra_args)
            if result is None:
                print(f"  All runs failed for [{label}] N={n}", file=sys.stderr)
                row[f"{label}.{timing_field}"]        = float("nan")
                row[f"{label}.{timing_field[:-3]}_s"] = float("nan")
                row[f"{label}.gpu_peak_mib"]           = float("nan")
            else:
                row.update(result)
        all_results.append(row)

        # Per-N summary line
        active_for_n = [(l, tf) for l, _, tf, _, mn in active_benchmarks if n <= mn]
        summary = "  Summary: " + "  |  ".join(
            f"{l}={row.get(f'{l}.{tf}', float('nan')):.1f}ms"
            for l, tf in active_for_n
        )
        print(summary)

    if all_results:
        print_comparison_table(all_results)
        save_csv(all_results, args.output)
    else:
        print("\nNo results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
