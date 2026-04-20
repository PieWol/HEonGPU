#!/usr/bin/env python3
"""
Benchmark multi-ciphertext HE ranking for N > 128.

Mirrors benchmark_ranking.py but targets 21_ckks_ranking_multi.
N must be a multiple of 128 and a power of 2: 256, 512, 1024, ...

Usage:
    python3 benchmark_ranking_multi.py                       # default N=[256,512], 3 runs
    python3 benchmark_ranking_multi.py --n-values 256 --runs 1   # quick smoke test
"""

import subprocess
import csv
import sys
import argparse
import statistics
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BINARY_DEFAULT = _SCRIPT_DIR.parent / "build/bin/examples/basic/21_ckks_ranking_multi"

N_VALUES_DEFAULT = [256, 512]


def run_once(binary: Path, n: int) -> dict | None:
    cmd = [str(binary), str(n), "--bench"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
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
                data[k] = float(v) if "." in v else int(v)
            return data

    print(f"  No BENCH: line for N={n}. stdout:\n{result.stdout[:300]}", file=sys.stderr)
    return None


def benchmark_n(binary: Path, n: int, runs: int) -> dict | None:
    run_results = []
    for run_idx in range(runs):
        print(f"  run {run_idx + 1}/{runs} ...", end=" ", flush=True)
        data = run_once(binary, n)
        if data is None:
            print("FAILED")
            continue
        total_s = (data.get('ctx_ms', 0) + data.get('keygen_ms', 0)
                   + data.get('rank_ms', 0)) / 1000
        print(f"ctx_ms={data.get('ctx_ms', 0):.1f}  "
              f"rank_ms={data.get('rank_ms', 0):.1f}  "
              f"total={total_s:.2f}s  "
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

    avg["rank_s"]  = avg["rank_ms"] / 1000.0
    avg["total_s"] = (avg["ctx_ms"] + avg["keygen_ms"] + avg["rank_ms"]) / 1000.0
    return avg


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print(f"{'N':>6}  {'ctx_ms':>10}  {'keygen_ms':>12}  {'rank_ms':>12}  "
          f"{'rank_s':>10}  {'total_s':>10}  {'peak_MiB':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['n']:>6}  {r.get('ctx_ms',0):>10.1f}  "
              f"{r.get('keygen_ms',0):>12.1f}  {r.get('rank_ms',0):>12.1f}  "
              f"{r.get('rank_s',0):>10.3f}  {r.get('total_s',0):>10.1f}  "
              f"{r.get('gpu_peak_mib',float('nan')):>10.0f}")
    print("=" * 100)


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
        description="Benchmark multi-ciphertext HE ranking (N > 128)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--binary", type=Path, default=BINARY_DEFAULT)
    parser.add_argument("--n-values", type=int, nargs="+", default=N_VALUES_DEFAULT, metavar="N")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=_SCRIPT_DIR / "ranking_multi_benchmark_results.csv")
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}")
        print("Build: cmake --build build --target 21_ckks_ranking_multi")
        sys.exit(1)

    from math import log2
    valid = []
    for n in args.n_values:
        if n <= 0 or (n & (n - 1)) != 0:
            print(f"Warning: N={n} is not a power of 2, skipping")
        elif n % 128 != 0:
            print(f"Warning: N={n} is not a multiple of 128, skipping")
        elif n <= 128:
            print(f"Warning: N={n} ≤ 128; use benchmark_ranking.py instead, skipping")
        else:
            valid.append(n)

    if not valid:
        print("No valid N values.")
        sys.exit(1)

    print(f"Binary : {args.binary}")
    print(f"N range: {valid}")
    print(f"Runs   : {args.runs} per N")

    all_results = []
    for n in valid:
        M = n // 128
        print(f"\nN={n}  (M={M} ciphertexts, {M*(M+1)//2} compare ops):")
        result = benchmark_n(args.binary, n, args.runs)
        if result:
            all_results.append(result)
            print(f"  → avg rank: {result.get('rank_ms',0):.1f} ms")
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
