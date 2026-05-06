#!/usr/bin/env python3
"""
Sweep multi-CT ranking accuracy across all N and mode combinations.

Runs each (N, mode) once in --bench mode, parses max_err and mismatches,
and writes a concise table + CSV for feeding back into parameter tuning.

Usage:
    python3 sweep_accuracy.py                       # defaults: N=256..4096, both modes
    python3 sweep_accuracy.py --max-n 2048          # cap at 2048
    python3 sweep_accuracy.py --modes basic         # basic only
    python3 sweep_accuracy.py --modes tc             # tie-corrected only
"""

import subprocess
import csv
import sys
import argparse
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
BINARY_DEFAULT = _SCRIPT_DIR.parent / "build/bin/examples/ranking_and_sorting/21_ckks_ranking_multi"

BLOCK_SIZE = 128


def n_values_up_to(max_n: int) -> list[int]:
    vals = []
    n = BLOCK_SIZE * 2  # minimum: 256 (M=2)
    while n <= max_n:
        vals.append(n)
        n *= 2
    return vals


def run_once(binary: Path, n: int, tie_correction: bool) -> dict | None:
    cmd = [str(binary), str(n), "--bench"]
    if tie_correction:
        cmd.append("--tie-correction")
    repo_root = _SCRIPT_DIR.parent
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                cwd=str(repo_root))
    except subprocess.TimeoutExpired:
        return {"n": n, "mode": "tc" if tie_correction else "basic", "status": "TIMEOUT"}

    if result.returncode != 0:
        return {
            "n": n,
            "mode": "tc" if tie_correction else "basic",
            "status": f"ERROR({result.returncode})",
            "stderr": result.stderr[:200],
        }

    for line in result.stdout.splitlines():
        if line.startswith("BENCH:"):
            data = {"n": n, "mode": "tc" if tie_correction else "basic", "status": "OK"}
            for token in line[6:].strip().split():
                k, v = token.split("=", 1)
                try:
                    data[k] = float(v)
                except ValueError:
                    data[k] = v
            return data

    return {"n": n, "mode": "tc" if tie_correction else "basic", "status": "NO_BENCH_LINE"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep multi-CT ranking accuracy")
    parser.add_argument("--binary", type=Path, default=BINARY_DEFAULT)
    parser.add_argument("--max-n", type=int, default=4096)
    parser.add_argument("--modes", nargs="+", default=["basic", "tc"],
                        choices=["basic", "tc"])
    parser.add_argument("--output", type=Path,
                        default=_SCRIPT_DIR / "sweep_accuracy_results.csv")
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"Binary not found: {args.binary}")
        print("Build: cmake --build build --target 21_ckks_ranking_multi")
        sys.exit(1)

    n_vals = n_values_up_to(args.max_n)
    if not n_vals:
        print(f"No valid N values (min is {BLOCK_SIZE * 2})")
        sys.exit(1)

    modes = []
    if "basic" in args.modes:
        modes.append(False)
    if "tc" in args.modes:
        modes.append(True)

    print(f"Binary : {args.binary}")
    print(f"N range: {n_vals}")
    print(f"Modes  : {args.modes}")
    print()

    # Header
    hdr = f"{'N':>6}  {'mode':>6}  {'M':>4}  {'use_fg':>6}  {'max_err':>10}  {'mismatches':>10}  {'rank_s':>8}  {'peak_MiB':>10}  {'status':>8}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for tc in modes:
        for n in n_vals:
            m = n // BLOCK_SIZE
            use_fg = "yes" if m > 2 else "no"
            mode_str = "tc" if tc else "basic"
            print(f"{n:>6}  {mode_str:>6}  {m:>4}  {use_fg:>6}  ", end="", flush=True)

            data = run_once(args.binary, n, tc)
            results.append(data)

            status = data.get("status", "?")
            if status == "OK":
                me = data.get("max_err", -1)
                mm = int(data.get("mismatches", -1))
                rs = data.get("rank_ms", 0) / 1000.0
                gp = data.get("gpu_peak_mib", 0)
                print(f"{me:>10.4f}  {mm:>10}  {rs:>8.2f}  {gp:>10.0f}  {status:>8}")
            else:
                print(f"{'—':>10}  {'—':>10}  {'—':>8}  {'—':>10}  {status:>8}")
                if "stderr" in data:
                    print(f"         stderr: {data['stderr']}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY :\n")
    print(f"{'N':>6}  {'mode':>6}  {'max_err':>10}  {'mismatches':>10}  {'rank_s':>8}  {'verdict':>8}")
    print("-" * 60)
    for r in results:
        if r["status"] != "OK":
            print(f"{int(r['n']):>6}  {r['mode']:>6}  {'FAIL':>10}  {'—':>10}  {'—':>8}  {'FAIL':>8}")
            continue
        me = r.get("max_err", -1)
        mm = int(r.get("mismatches", -1))
        rs = r.get("rank_ms", 0) / 1000.0
        verdict = "PASS" if mm == 0 else ("WARN" if me < 1.5 else "FAIL")
        print(f"{int(r['n']):>6}  {r['mode']:>6}  {me:>10.4f}  {mm:>10}  {rs:>8.2f}  {verdict:>8}")

    # CSV
    if results:
        all_keys = list(dict.fromkeys(k for r in results for k in r))
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved to {args.output}")


if __name__ == "__main__":
    main()
