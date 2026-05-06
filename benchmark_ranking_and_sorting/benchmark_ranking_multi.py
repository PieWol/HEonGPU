#!/usr/bin/env python3
"""
Benchmark multi-ciphertext HE ranking for large N.

Both basic and tie-corrected modes use B=128 blocks.
TC mode uses extended ring dimension (n=65536) for degree 2047 + TC ops.

Usage:
    # Basic ranking (default):
    python3 benchmark_ranking_multi.py                          # N=[256,512], 3 runs
    python3 benchmark_ranking_multi.py --n-values 256 --runs 1  # quick test

    # Tie-corrected ranking:
    python3 benchmark_ranking_multi.py --tie-correction                     # N=[256,512], 3 runs
    python3 benchmark_ranking_multi.py --tie-correction --n-values 256 512  # explicit N
"""

import subprocess
import csv
import sys
import argparse
import statistics
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BINARY_DEFAULT = _SCRIPT_DIR.parent / "build/bin/examples/ranking_and_sorting/21_ckks_ranking_multi"

N_VALUES_BASIC = [256, 512]
N_VALUES_TC    = [256, 512]


def run_once(binary: Path, n: int, tie_correction: bool) -> dict | None:
    cmd = [str(binary), str(n), "--bench"]
    if tie_correction:
        cmd.append("--tie-correction")
    repo_root = _SCRIPT_DIR.parent
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200,
                                cwd=str(repo_root))
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

    print(f"  No BENCH: line for N={n}. stdout:\n{result.stdout[:300]}", file=sys.stderr)
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


def print_table(results: list[dict], block_size: int) -> None:
    print("\n" + "=" * 110)
    print(f"{'N':>6}  {'mode':>10}  {'M':>4}  {'comps':>6}  {'ctx_ms':>10}  "
          f"{'keygen_ms':>10}  {'rank_ms':>10}  {'rank_s':>8}  {'peak_MiB':>10}")
    print("-" * 110)
    for r in results:
        n = int(r['n'])
        m = n // block_size
        comps = m * (m + 1) // 2
        cx = r.get("ctx_ms", 0)
        ks = r.get("keygen_ms", 0)
        rs = r.get("rank_ms", 0)
        gp = r.get("gpu_peak_mib", float("nan"))
        mode = r.get("mode", "?")
        print(f"{n:>6}  {mode:>10}  {m:>4}  {comps:>6}  {cx:>10.1f}  "
              f"{ks:>10.1f}  {rs:>10.1f}  {rs/1000:>8.3f}  {gp:>10.0f}")
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
        description="Benchmark multi-ciphertext HE ranking (basic and tie-corrected)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--binary", type=Path, default=BINARY_DEFAULT)
    parser.add_argument("--n-values", type=int, nargs="+", default=None, metavar="N",
                        help="Vector lengths to benchmark (default depends on mode)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tie-correction", action="store_true",
                        help="Enable tie correction (B=64 blocks, degree 1023)")
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV output file (default: auto-named)")
    args = parser.parse_args()

    block_size = 128
    n_defaults = N_VALUES_TC if args.tie_correction else N_VALUES_BASIC
    n_values = args.n_values if args.n_values else n_defaults

    if args.output is None:
        suffix = "tc" if args.tie_correction else "basic"
        args.output = _SCRIPT_DIR / f"ranking_multi_{suffix}_benchmark_results.csv"

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}")
        print("Build: cmake --build build --target 21_ckks_ranking_multi")
        sys.exit(1)

    valid = []
    for n in n_values:
        if n <= 0 or (n & (n - 1)) != 0:
            print(f"Warning: N={n} is not a power of 2, skipping")
        elif n % block_size != 0:
            print(f"Warning: N={n} is not a multiple of {block_size}, skipping")
        elif n <= block_size:
            print(f"Warning: N={n} <= {block_size}; use single-CT benchmark, skipping")
        else:
            valid.append(n)

    if not valid:
        print("No valid N values.")
        sys.exit(1)

    mode_str = "tie-corrected (B=128, n=65536)" if args.tie_correction else "basic (B=128)"
    print(f"Binary : {args.binary}")
    print(f"Mode   : {mode_str}")
    print(f"N range: {valid}")
    print(f"Runs   : {args.runs} per N")

    all_results = []
    for n in valid:
        M = n // block_size
        print(f"\nN={n}  (M={M} blocks, {M*(M+1)//2} comparisons):")
        result = benchmark_n(args.binary, n, args.tie_correction, args.runs)
        if result:
            all_results.append(result)
            print(f"  -> avg rank: {result.get('rank_ms', 0):.1f} ms")
        else:
            print(f"  -> all runs failed for N={n}")

    if all_results:
        print_table(all_results, block_size)
        save_csv(all_results, args.output)
    else:
        print("\nNo results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
