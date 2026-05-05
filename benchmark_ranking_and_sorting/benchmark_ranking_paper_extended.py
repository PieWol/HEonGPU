#!/usr/bin/env python3
"""
Benchmark extended single-ciphertext ranking beyond paper spec.

The paper's CKKS setup (n=32768) supports:
  - Basic ranking up to N=128
  - Tie-corrected ranking up to N=64

By increasing the ring dimension we extend to:
  - Basic ranking: N=256 (n=131072), N=512 (n=524288)
  - Tie-corrected:  N=128 (n=32768), N=256 (n=131072), N=512 (n=524288)

This script benchmarks ONLY what the paper-spec setup cannot do:
  - Basic (no tie correction): N=256, N=512
  - Tie-corrected:             N=128, N=256, N=512

Usage:
    # Full sweep (default):
    CUDA_VISIBLE_DEVICES=1 python3 benchmark_ranking_paper_extended.py

    # Quick single pass:
    CUDA_VISIBLE_DEVICES=1 python3 benchmark_ranking_paper_extended.py --runs 1
"""

import subprocess
import csv
import sys
import argparse
import statistics
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BINARY_DEFAULT = (
    _SCRIPT_DIR.parent
    / "build/bin/examples/ranking_and_sorting/24_ckks_ranking_tie_correction_extended"
)


def run_once(binary: Path, n: int, tie_correction: bool) -> dict | None:
    cmd = [str(binary), str(n), "--bench"]
    if tie_correction:
        cmd.append("--tie-correction")
    else:
        cmd.append("--no-tie-correction")

    timeout = 1800 if n >= 512 else 600

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for N={n}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(
            f"  ERROR (exit {result.returncode}) for N={n}:\n"
            f"    stderr: {result.stderr[:300]}",
            file=sys.stderr,
        )
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

    print(
        f"  No BENCH: line found for N={n}. stdout:\n{result.stdout[:500]}",
        file=sys.stderr,
    )
    return None


def benchmark_n(
    binary: Path, n: int, tie_correction: bool, runs: int
) -> dict | None:
    mode_label = "tie_corr" if tie_correction else "basic"
    run_results = []
    for run_idx in range(runs):
        print(
            f"  [{mode_label:>8}] run {run_idx + 1}/{runs} ...",
            end=" ",
            flush=True,
        )
        data = run_once(binary, n, tie_correction)

        if data is None:
            print("FAILED")
            continue

        rank_ms = float(data.get("rank_ms", 0))
        gpu_peak = data.get("gpu_peak_mib", "?")
        print(f"rank_ms={rank_ms:.1f}  gpu_peak={gpu_peak}MiB")
        run_results.append(data)

    if not run_results:
        return None

    numeric_keys = [
        k
        for k in run_results[0]
        if k not in ("n", "mode") and not isinstance(run_results[0][k], str)
    ]
    avg = {"n": n, "mode": mode_label, "runs": len(run_results)}
    for k in numeric_keys:
        vals = [float(r[k]) for r in run_results]
        avg[k] = statistics.mean(vals)
        if len(vals) > 1:
            avg[f"{k}_stdev"] = statistics.stdev(vals)

    avg["rank_s"] = avg["rank_ms"] / 1000.0
    return avg


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 120)
    print(
        f"{'N':>6}  {'mode':>10}  {'poly_deg':>10}  {'ctx_ms':>10}  "
        f"{'keygen_ms':>10}  {'rank_ms':>10}  {'rank_s':>8}  "
        f"{'keys_MiB':>10}  {'peak_MiB':>10}"
    )
    print("-" * 120)

    poly_deg_map = {128: 32768, 256: 131072, 512: 524288}

    for r in results:
        cx = r.get("ctx_ms", 0)
        ks = r.get("keygen_ms", 0)
        rs = r.get("rank_ms", 0)
        gk = r.get("gpu_keys_mib", float("nan"))
        gp = r.get("gpu_peak_mib", float("nan"))
        pd = poly_deg_map.get(r["n"], "?")

        print(
            f"{r['n']:>6}  {r['mode']:>10}  {pd:>10}  {cx:>10.1f}  "
            f"{ks:>10.1f}  {rs:>10.1f}  {rs/1000:>8.3f}  "
            f"{gk:>10.0f}  {gp:>10.0f}"
        )

    print("=" * 120)


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
        description="Benchmark extended single-ciphertext ranking beyond paper spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=BINARY_DEFAULT,
        help="Path to 24_ckks_ranking_tie_correction_extended binary",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs per (N, mode) pair (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_SCRIPT_DIR / "ranking_extended_results.csv",
        help="CSV output file",
    )
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}")
        print(
            "Build first with: cmake --build build "
            "--target 24_ckks_ranking_tie_correction_extended"
        )
        sys.exit(1)

    # Benchmark plan:
    #   Basic (no tie correction): N=256, 512 only
    #     (N=128 basic already covered by paper-spec benchmark)
    #   Tie-corrected: N=128, 256, 512
    #     (paper-spec only supports tie correction up to N=64)
    basic_n_values = [256, 512]
    tie_corr_n_values = [128, 256, 512]

    print(f"Binary        : {args.binary}")
    print(f"Basic (no TC) : N={basic_n_values}")
    print(f"Tie-corrected : N={tie_corr_n_values}")
    print(f"Runs          : {args.runs} per (N, mode)")
    print()

    all_results = []

    print("=" * 60)
    print("  BASIC RANKING (no tie correction)")
    print("=" * 60)
    for n in basic_n_values:
        print(f"\nN={n}:")
        result = benchmark_n(args.binary, n, False, args.runs)
        if result:
            all_results.append(result)
            print(f"  -> avg: {result['rank_ms']:.1f} ms ({result['rank_s']:.3f} s)")

    print()
    print("=" * 60)
    print("  TIE-CORRECTED RANKING")
    print("=" * 60)
    for n in tie_corr_n_values:
        print(f"\nN={n}:")
        result = benchmark_n(args.binary, n, True, args.runs)
        if result:
            all_results.append(result)
            print(f"  -> avg: {result['rank_ms']:.1f} ms ({result['rank_s']:.3f} s)")

    if all_results:
        print_table(all_results)
        save_csv(all_results, args.output)
    else:
        print("\nNo results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
