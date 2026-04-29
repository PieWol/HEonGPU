#!/usr/bin/env python3
"""
Benchmark HE median for varying N, mimicking the experimental setup of:
  "Efficient Ranking, Order Statistics, and Sorting under CKKS"
  Mazzone et al., USENIX Security 2025

Usage:
    python3 benchmark_median.py [--n-values N1 N2 ...] [--runs R] [--output FILE]

    # Quick test:
    python3 benchmark_median.py --n-values 8 16 32 --runs 1

    # Full paper-equivalent single-ciphertext sweep (default):
    python3 benchmark_median.py

    # Custom N set, 5 runs each:
    python3 benchmark_median.py --n-values 8 16 32 64 128 --runs 5
"""

import subprocess
import csv
import sys
import argparse
import statistics
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BINARY_DEFAULT = _SCRIPT_DIR.parent / "build/bin/examples/ranking_and_sorting/20_ckks_median"

N_VALUES_DEFAULT = [8, 16, 32, 64, 128]


def run_once(binary: Path, n: int) -> dict | None:
    cmd = [str(binary), str(n), "--bench"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for N={n}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}) for N={n}:\n"
              f"    stderr: {result.stderr[:300]}", file=sys.stderr)
        return None

    # Format: BENCH: N=8 ctx_ms=... keygen_ms=... median_ms=... gpu_keys_mib=... gpu_median_mib=... gpu_peak_mib=...
    for line in result.stdout.splitlines():
        if line.startswith("BENCH:"):
            data = {"n": n}
            for token in line[6:].strip().split():
                k, v = token.split("=")
                data[k] = float(v) if "." in v else int(v)
            return data

    print(f"  No BENCH: line found for N={n}. stdout:\n{result.stdout[:300]}",
          file=sys.stderr)
    return None


def benchmark_n(binary: Path, n: int, runs: int) -> dict | None:
    run_results = []
    for run_idx in range(runs):
        print(f"  run {run_idx + 1}/{runs} ...", end=" ", flush=True)
        data = run_once(binary, n)

        if data is None:
            print("FAILED")
            continue

        total_s = (data.get('ctx_ms', 0) + data.get('keygen_ms', 0) + data.get('median_ms', 0)) / 1000
        print(f"ctx_ms={data.get('ctx_ms', 0):.1f}  "
              f"median_ms={data.get('median_ms', 0):.1f}  "
              f"total={total_s:.2f}s  "
              f"gpu_keys={data.get('gpu_keys_mib', '?')}MiB  "
              f"gpu_median={data.get('gpu_median_mib', '?')}MiB  "
              f"gpu_peak={data.get('gpu_peak_mib', '?')}MiB")
        run_results.append(data)

    if not run_results:
        return None

    keys = [k for k in run_results[0] if k != "n"]
    avg = {"n": n, "runs": len(run_results)}
    for k in keys:
        vals = [r[k] for r in run_results]
        avg[k] = statistics.mean(vals)
        if len(vals) > 1:
            avg[f"{k}_stdev"] = statistics.stdev(vals)

    avg["median_s"] = avg["median_ms"] / 1000.0
    avg["keygen_s"] = avg["keygen_ms"] / 1000.0
    avg["total_s"]  = (avg["ctx_ms"] + avg["keygen_ms"] + avg["median_ms"]) / 1000.0
    return avg


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 125)
    print(f"{'N':>6}  {'ctx_ms':>10}  {'keygen_ms':>12}  {'median_ms':>12}  "
          f"{'median_s':>10}  {'total_s':>10}  {'keys_MiB':>10}  {'med_MiB':>10}  {'peak_MiB':>10}")
    print("-" * 125)
    for r in results:
        cx  = r.get("ctx_ms", 0)
        ks  = r.get("keygen_ms", 0)
        mds = r.get("median_ms", 0)
        ts  = r.get("total_s", 0)
        gk  = r.get("gpu_keys_mib", float("nan"))
        gm  = r.get("gpu_median_mib", float("nan"))
        gp  = r.get("gpu_peak_mib", float("nan"))
        print(f"{r['n']:>6}  {cx:>10.1f}  {ks:>12.1f}  {mds:>12.1f}  "
              f"{mds/1000:>10.3f}  {ts:>10.1f}  {gk:>10.0f}  {gm:>10.0f}  {gp:>10.0f}")
    print("=" * 125)


def save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    all_keys = list(dict.fromkeys(k for r in results for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark HE median for varying N",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--binary", type=Path, default=BINARY_DEFAULT,
        help="Path to 20_ckks_median binary"
    )
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=N_VALUES_DEFAULT,
        metavar="N",
        help="Vector lengths to benchmark (must be powers of 2, N<=128)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed runs per N (default: 3)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=_SCRIPT_DIR / "median_benchmark_results.csv",
        help="CSV output file (default: median_benchmark_results.csv)"
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}")
        print("Build first with: cmake --build build --target 20_ckks_median")
        sys.exit(1)

    for n in args.n_values:
        if n <= 0 or (n & (n - 1)) != 0:
            print(f"Error: N={n} is not a positive power of 2")
            sys.exit(1)
        if n > 128:
            print(f"Warning: N={n} exceeds single-CT limit (128), skipping")

    n_values = [n for n in args.n_values if n <= 128]
    if not n_values:
        print("No valid N values to benchmark.")
        sys.exit(1)

    print(f"Binary : {args.binary}")
    print(f"N range: {n_values}")
    print(f"Runs   : {args.runs} per N")

    all_results = []
    for n in n_values:
        print(f"\nN={n}:")
        result = benchmark_n(args.binary, n, args.runs)
        if result:
            all_results.append(result)
            median_ms = result.get("median_ms", 0)
            print(f"  → avg median: {median_ms:.1f} ms ({median_ms/1000:.3f} s)")
        else:
            print(f"  → all runs failed for N={n}")

    if all_results:
        print_table(all_results)
        save_csv(all_results, args.output)
    else:
        print("\nNo results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
