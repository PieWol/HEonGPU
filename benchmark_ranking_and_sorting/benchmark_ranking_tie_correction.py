#!/usr/bin/env python3
"""
Benchmark HE ranking with and without tie correction for varying N.

Runs both modes (basic, tie-corrected) side by side and reports the
overhead introduced by Algorithm 6 (Mazzone et al., USENIX Security 2025).

Tie correction adds 2 extra levels (sign^2 + mask*E).
With f,g at n=65536 (depth=24), both basic and tie-corrected
support N up to 128 (single-ciphertext limit).

Usage:
    python3 benchmark_ranking_tie_correction.py [--n-values N1 N2 ...] [--runs R] [--output FILE]

    # Quick test:
    python3 benchmark_ranking_tie_correction.py --n-values 8 16 --runs 1

    # Full sweep (default):
    python3 benchmark_ranking_tie_correction.py
"""

import subprocess
import csv
import sys
import argparse
import statistics
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BINARY_DEFAULT = _SCRIPT_DIR.parent / "build/bin/examples/ranking_and_sorting/23_ckks_ranking_tie_correction"

N_VALUES_DEFAULT = [8, 16, 32, 64, 128]


def run_once(binary: Path, n: int, tie_correction: bool) -> dict | None:
    cmd = [str(binary), str(n), "--bench"]
    if tie_correction:
        cmd.append("--tie-correction")
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

    for line in result.stdout.splitlines():
        if line.startswith("BENCH:"):
            data = {"n": n}
            for token in line[6:].strip().split():
                k, v = token.split("=")
                try:
                    data[k] = float(v)
                except ValueError:
                    data[k] = v
            return data

    print(f"  No BENCH: line found for N={n}. stdout:\n{result.stdout[:300]}",
          file=sys.stderr)
    return None


def benchmark_n(binary: Path, n: int, tie_correction: bool, runs: int) -> dict | None:
    mode_label = "tie_corr" if tie_correction else "basic"
    run_results = []
    for run_idx in range(runs):
        print(f"  [{mode_label:>8}] run {run_idx + 1}/{runs} ...", end=" ", flush=True)
        data = run_once(binary, n, tie_correction)

        if data is None:
            print("FAILED")
            continue

        rank_ms = float(data.get('rank_ms', 0))
        gpu_peak = data.get('gpu_peak_mib', '?')
        print(f"rank_ms={rank_ms:.1f}  gpu_peak={gpu_peak}MiB")
        run_results.append(data)

    if not run_results:
        return None

    numeric_keys = [k for k in run_results[0]
                    if k not in ("n", "mode") and not isinstance(run_results[0][k], str)]
    avg = {"n": n, "mode": mode_label, "runs": len(run_results)}
    for k in numeric_keys:
        vals = [float(r[k]) for r in run_results]
        avg[k] = statistics.mean(vals)
        if len(vals) > 1:
            avg[f"{k}_stdev"] = statistics.stdev(vals)

    avg["rank_s"] = avg["rank_ms"] / 1000.0
    return avg


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 110)
    print(f"{'N':>6}  {'mode':>10}  {'ctx_ms':>10}  {'keygen_ms':>10}  "
          f"{'rank_ms':>10}  {'rank_s':>8}  {'keys_MiB':>10}  {'peak_MiB':>10}  {'overhead':>10}")
    print("-" * 110)

    basic_by_n = {r["n"]: r for r in results if r["mode"] == "basic"}

    for r in results:
        cx = r.get("ctx_ms", 0)
        ks = r.get("keygen_ms", 0)
        rs = r.get("rank_ms", 0)
        gk = r.get("gpu_keys_mib", float("nan"))
        gp = r.get("gpu_peak_mib", float("nan"))

        overhead = ""
        if r["mode"] == "tie_corr" and r["n"] in basic_by_n:
            basic_ms = basic_by_n[r["n"]].get("rank_ms", 0)
            if basic_ms > 0:
                pct = (rs - basic_ms) / basic_ms * 100
                overhead = f"{pct:+.1f}%"

        print(f"{r['n']:>6}  {r['mode']:>10}  {cx:>10.1f}  {ks:>10.1f}  "
              f"{rs:>10.1f}  {rs/1000:>8.3f}  {gk:>10.0f}  {gp:>10.0f}  {overhead:>10}")

        if r["mode"] == "tie_corr":
            print()

    print("=" * 110)


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
        description="Benchmark HE ranking: basic vs tie-corrected",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--binary", type=Path, default=BINARY_DEFAULT,
        help="Path to 23_ckks_ranking_tie_correction binary"
    )
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=N_VALUES_DEFAULT,
        metavar="N",
        help="Vector lengths to benchmark (default: 8 16 32 64 128)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of timed runs per (N, mode) pair (default: 3)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=_SCRIPT_DIR / "ranking_tie_correction_results.csv",
        help="CSV output file"
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}")
        print("Build first with: cmake --build build --target 23_ckks_ranking_tie_correction")
        sys.exit(1)

    for n in args.n_values:
        if n <= 0 or (n & (n - 1)) != 0:
            print(f"Error: N={n} is not a positive power of 2")
            sys.exit(1)

    max_tie_corr = 128
    n_values = [n for n in args.n_values if n <= 128]
    if not n_values:
        print("No valid N values to benchmark.")
        sys.exit(1)

    print(f"Binary : {args.binary}")
    print(f"N range: {n_values}")
    print(f"Modes  : basic (all N), tie-corrected (N <= {max_tie_corr})")
    print(f"Runs   : {args.runs} per (N, mode)")

    all_results = []
    for n in n_values:
        print(f"\nN={n}:")

        result_basic = benchmark_n(args.binary, n, False, args.runs)
        if result_basic:
            all_results.append(result_basic)
            print(f"  -> basic avg: {result_basic['rank_ms']:.1f} ms")

        if n <= max_tie_corr:
            result_tc = benchmark_n(args.binary, n, True, args.runs)
            if result_tc:
                all_results.append(result_tc)
                print(f"  -> tie_corr avg: {result_tc['rank_ms']:.1f} ms")

                if result_basic:
                    overhead = (result_tc['rank_ms'] - result_basic['rank_ms']) / result_basic['rank_ms'] * 100
                    print(f"  -> overhead: {overhead:+.1f}%")
        else:
            print(f"  -> tie correction skipped (N > {max_tie_corr})")

    if all_results:
        print_table(all_results)
        save_csv(all_results, args.output)
    else:
        print("\nNo results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
