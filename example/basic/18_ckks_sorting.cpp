// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file 18_ckks_sorting.cpp
 *
 * Homomorphic sorting via Algorithm 5 from:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone, Everts, Hahn, Peter — USENIX Security 2025
 *
 * Implements the sortFG() variant from the reference OpenFHE code
 * (openfhe-statistics/src/sorting.cpp) using the fg-composite sign
 * approximation (Cheon et al., ASIACRYPT '20) instead of high-degree Chebyshev.
 *
 * fg-sign primitives (all on domain [-1, 1]):
 *   g3(x) = (4589x - 16577x³ + 25614x⁵ - 12860x⁷) / 1024
 *     "gap amplifier": pushes values away from 0 toward ±1
 *   f3(x) = (35x - 35x³ + 21x⁵ - 5x⁷) / 16
 *     "output refiner": Newton step, drives approximation toward exact ±1
 *   f3_final(x) = f3(x)/2 + 0.5
 *     final step: maps [-1,1] sign range → [0,1] comparison range
 *
 *   signAdv(ct, dg, df):  g3 × dg,  f3 × (df-1),  f3_final × 1
 *     → ≈1 where slot > 0,  ≈0 where slot < 0
 *   compareAdv(a, b):  signAdv(a - b)
 *     → ≈1 if a > b,  ≈0 if a < b,  ≈0.5 for ties
 *   indicatorAdv(ct, N):  s1*(1-s2)  where
 *     s1 = signAdv((ct + 0.5) / N),  s2 = signAdv((ct - 0.5) / N)
 *     → ≈1 if |ct| < 0.5,  ≈0 otherwise
 *
 * Sorting algorithm (directly mirrors sortFG()):
 *   1. VR = ReplR(V)                         // N×N: row-replicated input
 *   2. VC = ReplC(TransR(V))                 // N×N: column-replicated input
 *   3. C  = compareAdv(VR, VC)               // C[k,j] ≈ 1 if v[j] > v[k]
 *   4a. R  = SumR(C)                         // suffix-sum butterfly, full sum in row 0
 *   4b. R  = MaskRow0(R)                     // discard partial sums; 1 level
 *   4c. R  = ReplR(R)                        // copy row 0 to ALL rows; no depth
 *       → R[k,j] = rank_0(v[j]) + 0.5  for every row k
 *   5. subMask[k,j] = -(k + 0.5)             // encrypted, mod-dropped to R level
 *   6. M  = indicatorAdv(R + subMask, N)     // M[k,j] ≈ 1 iff rank_0(v[j]) = k
 *   7. VR = ReplR(V),  mod-drop to M level   // fresh copy for phase 3
 *   8. S  = SumC(M · VR)                     // S[k, col0] = (k+1)-th order stat
 *
 * Output: slot k*N in the decoded result holds the (k+1)-th order statistic.
 * No factor-of-2 correction is needed (indicator returns values in [0,1]).
 *
 * Depth budget (n=65536, 32 Q primes → 31 usable levels):
 *   Each degree-7 evaluate_poly(lead=true) costs 4 levels (not 3): the
 *   Chebyshev T_3 basis element sits 2 levels below the input, forcing the
 *   polynomial-basis evaluation down 1 extra level; lead=true ensures the
 *   outer rescale fires so the output has rescale_required_=false.
 *   1  : TransR (multiply_plain mask + rescale)
 *   12 : compareAdv (DG_C+DF_C=3 evals × 4 levels)
 *   1  : MaskRow0 (multiply_plain + rescale)
 *   1  : indicatorAdv normalize (×1/N + rescale)
 *   12 : indicatorAdv signAdv × 2 parallel (DG_I+DF_I=3 evals × 4 levels)
 *   1  : indicatorAdv multiply s1·(1−s2) + rescale
 *   1  : multiply(M, VR) + rescale
 *   Total: 29 ≤ 31 ✓  (2 spare levels)
 *
 * Usage:  18_ckks_sorting [N] [--bench]
 *   N       : vector length, power of 2, default 8
 *   --bench : machine-readable timing output only
 */

#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include "../example_util.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

static bool g_verbose = true;
constexpr auto Scheme = heongpu::Scheme::CKKS;

// fg-sign composition parameters.
//
// Each degree-7 Chebyshev evaluate_poly call (with lead=true) costs 4 levels:
//   - BSGS for degree-7 needs T_3 which is 2 levels below input (Chebyshev
//     sub-recursion), forcing the polynomial-basis result down 1 level from the
//     target, then the conditional internal rescale drops 1 more.
//   - The final scalar rescale inside evaluate_poly fires because lead=true
//     makes the scale ≈ 2^80 after the giant-step multiply, so scale/qi ≥ Δ/2.
//   - Output: rescale_required_=false, level drops by 4.  (lead=false only
//     drops 3 levels but leaves rescale_required_=true, breaking chaining.)
//
// signAdv(dg, df) = (dg + df) calls × 4 levels each.
//
// Depth budget (n=65536, 32 Q primes → 31 usable levels):
//   1  : TransR
//   12 : compareAdv  (DG_C+DF_C=3 calls × 4 levels)
//   1  : MaskRow0 (plain multiply + rescale)
//   1  : indicatorAdv normalize
//   12 : indicatorAdv signAdv × 2 (each DG_I+DF_I=3 × 4, but run in parallel)
//   1  : indicatorAdv s1×(1-s2) multiply + rescale
//   1  : phase-3 M×VR multiply + rescale
//   Total: 29 ≤ 31 ✓  (2 spare levels)
constexpr int DG_C = 2; // g-compositions for compare
constexpr int DF_C = 1; // f-compositions for compare
constexpr int DG_I = 2; // g-compositions for indicator
constexpr int DF_I = 1; // f-compositions for indicator

// ---------------------------------------------------------------------------
// CKKSPolyEvaluator — exposes protected evaluate_poly
// ---------------------------------------------------------------------------
class CKKSPolyEvaluator : public heongpu::HEArithmeticOperator<Scheme>
{
  public:
    CKKSPolyEvaluator(heongpu::HEContext<Scheme> ctx,
                      heongpu::HEEncoder<Scheme>& enc)
        : heongpu::HEArithmeticOperator<Scheme>(ctx, enc)
    {
    }

    heongpu::Ciphertext<Scheme>
    eval_chebyshev(heongpu::Ciphertext<Scheme>& ct, double target_scale,
                   const std::vector<Complex64>& coeffs, int degree,
                   heongpu::Relinkey<Scheme>& rk,
                   double a = -1.0, double b = 1.0)
    {
        // lead=true: makes the outer rescale inside evaluate_poly fire, so
        // the output has rescale_required_=false and can be chained.
        Polynomial poly(degree, coeffs, /*lead=*/true,
                        heongpu::PolyType::CHEBYSHEV, a, b);
        return evaluate_poly(ct, target_scale, poly, rk,
                             heongpu::ExecutionOptions());
    }
};

// ---------------------------------------------------------------------------
// GPU timer
// ---------------------------------------------------------------------------
class GPUTimer
{
    cudaEvent_t start_, stop_;
  public:
    GPUTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GPUTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void startTimer() { cudaEventRecord(start_); }
    float stopTimer()
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

static size_t getGPUUsedMiB() {
    return heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
           / (1024ULL * 1024ULL);
}
static size_t getPeakGPUMiB() {
    return heongpu::MemoryPool::instance().get_peak_device_pool_memory_usage()
           / (1024ULL * 1024ULL);
}

// ---------------------------------------------------------------------------
// Input normalization (client-side, before encrypt)
// ---------------------------------------------------------------------------
std::vector<double> normalizeToUnit(const std::vector<double>& v)
{
    double lo = *std::min_element(v.begin(), v.end());
    double hi = *std::max_element(v.begin(), v.end());
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); i++)
        out[i] = (v[i] - lo) / (hi - lo);
    return out;
}

// ---------------------------------------------------------------------------
// Galois shift helpers
// ---------------------------------------------------------------------------
std::vector<int> rowGaloisShifts(int N)   // ReplR: -(N/2)*N, ..., -N
{
    std::vector<int> s;
    for (int i = N / 2; i > 0; i /= 2) s.push_back(-(i * N));
    return s;
}
std::vector<int> colGaloisShifts(int N)   // ReplC: -1, -2, ..., -(N/2)
{
    std::vector<int> s;
    for (int i = 1; i < N; i *= 2) s.push_back(-i);
    return s;
}
std::vector<int> sumrGaloisShifts(int N)  // SumR: +N, +2N, ..., +N*(N/2)
{
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    std::vector<int> s;
    for (int i = 0; i < logN; i++) s.push_back(N * (1 << i));
    return s;
}
std::vector<int> transrGaloisShifts(int N) // TransR: -(N*(N-1)/2^i) i=1..logN
{
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    std::vector<int> s;
    for (int i = 1; i <= logN; i++)
        s.push_back(-((N * (N - 1)) / (1 << i)));
    return s;
}
std::vector<int> sumcGaloisShifts(int N)  // SumC: +1, +2, +4, ..., +N/2
{
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    std::vector<int> s;
    for (int i = 0; i < logN; i++) s.push_back(1 << i);
    return s;
}

// ---------------------------------------------------------------------------
// fg-sign primitives
// ---------------------------------------------------------------------------
/**
 * @brief Apply g3 once: amplifies values away from 0.
 *
 * g3(x) = (4589x - 16577x³ + 25614x⁵ - 12860x⁷) / 1024
 * Degree 7 → 3 levels consumed per call.
 */
static heongpu::Ciphertext<Scheme>
applyG3(heongpu::Ciphertext<Scheme>& ct, CKKSPolyEvaluator& pe,
        heongpu::Relinkey<Scheme>& rk, double scale)
{
    auto fn = [](Complex64 x) -> Complex64 {
        double t = x.real(), t2 = t * t;
        return {t * (4589.0 + t2 * (-16577.0 + t2 * (25614.0 - 12860.0 * t2)))
                / 1024.0, 0.0};
    };
    auto coeffs = heongpu::approximate_function(fn, -1.0, 1.0, 7);
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk);
}

/**
 * @brief Apply f3 once: Newton step, refines approximation toward exact ±1.
 *
 * f3(x) = (35x - 35x³ + 21x⁵ - 5x⁷) / 16
 * Degree 7 → 3 levels consumed per call.
 */
static heongpu::Ciphertext<Scheme>
applyF3(heongpu::Ciphertext<Scheme>& ct, CKKSPolyEvaluator& pe,
        heongpu::Relinkey<Scheme>& rk, double scale)
{
    auto fn = [](Complex64 x) -> Complex64 {
        double t = x.real(), t2 = t * t;
        return {t * (35.0 + t2 * (-35.0 + t2 * (21.0 - 5.0 * t2))) / 16.0, 0.0};
    };
    auto coeffs = heongpu::approximate_function(fn, -1.0, 1.0, 7);
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk);
}

/**
 * @brief Apply f3_final: last refinement step, maps [-1,1] → [0,1].
 *
 * f3_final(x) = f3(x)/2 + 0.5
 * Output ≈ 1 where x > 0, ≈ 0 where x < 0.
 * Degree 7 → 3 levels consumed.
 */
static heongpu::Ciphertext<Scheme>
applyF3Final(heongpu::Ciphertext<Scheme>& ct, CKKSPolyEvaluator& pe,
             heongpu::Relinkey<Scheme>& rk, double scale)
{
    auto fn = [](Complex64 x) -> Complex64 {
        double t = x.real(), t2 = t * t;
        double f3 = t * (35.0 + t2 * (-35.0 + t2 * (21.0 - 5.0 * t2))) / 16.0;
        return {f3 * 0.5 + 0.5, 0.0};
    };
    auto coeffs = heongpu::approximate_function(fn, -1.0, 1.0, 7);
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk);
}

/**
 * @brief fg-composite sign approximation.
 *
 * Input:  ct with slot values in [-1, 1].
 * Output: ≈1 where slot > 0,  ≈0 where slot < 0.
 * Depth:  (dg + df) × 3 levels.
 *
 * Applies g3 dg times (amplify gap), then f3 (df−1) times and f3_final once
 * (refine and shift output). Matches signAdv() in the reference implementation.
 */
static heongpu::Ciphertext<Scheme>
signAdv(heongpu::Ciphertext<Scheme> ct, int dg, int df,
        CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk, double scale)
{
    for (int i = 0; i < dg; i++)
        ct = applyG3(ct, pe, rk, scale);
    for (int i = 0; i < df - 1; i++)
        ct = applyF3(ct, pe, rk, scale);
    return applyF3Final(ct, pe, rk, scale);
}

/**
 * @brief Compare two ciphertexts using fg-composite sign.
 *
 * Computes signAdv(a − b): ≈1 if a > b,  ≈0 if a < b,  ≈0.5 for ties.
 * Inputs must be normalized so that (a − b) ∈ [−1, 1].
 * Depth: (DG_C + DF_C) × 3 levels.
 */
static heongpu::Ciphertext<Scheme>
compareAdv(const heongpu::Ciphertext<Scheme>& a,
           const heongpu::Ciphertext<Scheme>& b,
           CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
           heongpu::HEContext<Scheme>& ctx, double scale)
{
    if (g_verbose) std::cout << "  compareAdv (DG_C=" << DG_C << " DF_C=" << DF_C << ")\n";
    heongpu::Ciphertext<Scheme> a_copy = a;
    heongpu::Ciphertext<Scheme> b_copy = b;
    heongpu::Ciphertext<Scheme> diff(ctx);
    pe.sub(a_copy, b_copy, diff);
    return signAdv(diff, DG_C, DF_C, pe, rk, scale);
}

/**
 * @brief Indicator function: ≈1 iff |ct| < 0.5,  ≈0 otherwise.
 *
 * Implements indicatorAdv() from the reference:
 *   tmp = ct / N           (normalize to [−1, 1], costs 1 rescale)
 *   s1  = signAdv(tmp + 0.5/N)   (≈1 when ct > −0.5)
 *   s2  = signAdv(tmp − 0.5/N)   (≈1 when ct >  0.5)
 *   return s1 × (1 − s2)         (≈1 iff −0.5 < ct < 0.5)
 *
 * Both signAdv calls start from the same ciphertext level, so their depths
 * are independent (max, not sum). Multiply costs 1 extra level.
 *
 * Input:  ct with integer-valued slots in [−N, N] (rank differences).
 * Depth:  1 (normalize) + (DG_I + DF_I) × 3 (signAdv) + 1 (multiply).
 */
static heongpu::Ciphertext<Scheme>
indicatorAdv(heongpu::Ciphertext<Scheme>& ct, int N,
             CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
             heongpu::HEContext<Scheme>& ctx, double scale)
{
    if (g_verbose)
        std::cout << "  indicatorAdv N=" << N
                  << " (DG_I=" << DG_I << " DF_I=" << DF_I << ")\n";

    double inv_N      = 1.0 / N;
    double half_inv_N = 0.5 / N;

    // Normalize to [−1, 1]: costs 1 rescale
    heongpu::Ciphertext<Scheme> tmp = ct;
    pe.multiply_plain_inplace(tmp, inv_N, scale);
    pe.rescale_inplace(tmp);

    // Shift by ±0.5/N (free)
    heongpu::Ciphertext<Scheme> c1 = tmp;
    pe.add_plain_inplace(c1,  half_inv_N);
    heongpu::Ciphertext<Scheme> c2 = tmp;
    pe.add_plain_inplace(c2, -half_inv_N);

    // Two independent sign evaluations (same starting level)
    heongpu::Ciphertext<Scheme> s1 = signAdv(c1, DG_I, DF_I, pe, rk, scale);
    heongpu::Ciphertext<Scheme> s2 = signAdv(c2, DG_I, DF_I, pe, rk, scale);

    // Build (1 − s2): negate is free, add_plain is free
    pe.negate_inplace(s2);
    pe.add_plain_inplace(s2, 1.0);

    // s1 × (1 − s2): ciphertext multiply + relin + rescale
    heongpu::Ciphertext<Scheme> result(ctx);
    pe.multiply(s1, s2, result);
    pe.relinearize_inplace(result, rk);
    pe.rescale_inplace(result);
    return result;
}

// ---------------------------------------------------------------------------
// Matrix primitives (identical to 16/17)
// ---------------------------------------------------------------------------

// ReplR: replicate row 0 across all N rows. No depth.
static heongpu::Ciphertext<Scheme>
replicateRow(const heongpu::Ciphertext<Scheme>& row, int N,
             heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe)
{
    heongpu::Ciphertext<Scheme> r = row;
    if (g_verbose) std::cout << "  ReplR:";
    for (int i = N / 2; i > 0; i /= 2)
    {
        int shift = -(i * N);
        if (g_verbose) std::cout << " " << shift;
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, shift);
        pe.add_inplace(r, rot);
    }
    if (g_verbose) std::cout << "\n";
    return r;
}

// ReplC: replicate column 0 across all N columns. No depth.
static heongpu::Ciphertext<Scheme>
replicateColumn(const heongpu::Ciphertext<Scheme>& col, int N,
                heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe)
{
    heongpu::Ciphertext<Scheme> r = col;
    if (g_verbose) std::cout << "  ReplC:";
    for (int i = 1; i < N; i *= 2)
    {
        int shift = -i;
        if (g_verbose) std::cout << " " << shift;
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, shift);
        pe.add_inplace(r, rot);
    }
    if (g_verbose) std::cout << "\n";
    return r;
}

// TransR: transpose row 0 into column 0. Depth: 1 (MaskC multiply_plain + rescale).
static heongpu::Ciphertext<Scheme>
transposeRowToColumn(const heongpu::Ciphertext<Scheme>& row, int N,
                     heongpu::Galoiskey<Scheme>& gk,
                     CKKSPolyEvaluator& pe,
                     heongpu::HEEncoder<Scheme>& enc,
                     heongpu::HEContext<Scheme>& ctx, double scale)
{
    heongpu::Ciphertext<Scheme> r = row;
    int logN = static_cast<int>(std::ceil(std::log2(N)));

    if (g_verbose) std::cout << "  TransR:";
    for (int i = 1; i <= logN; i++)
    {
        int shift = -((N * (N - 1)) / (1 << i));
        if (g_verbose) std::cout << " " << shift;
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, shift);
        pe.add_inplace(r, rot);
    }
    if (g_verbose) std::cout << "\n";

    // MaskC: keep column 0 only (positions k*N)
    size_t slots = ctx->get_poly_modulus_degree() / 2;
    std::vector<double> mask(slots, 0.0);
    for (int k = 0; k < N; k++) mask[k * N] = 1.0;
    heongpu::Plaintext<Scheme> pt(ctx);
    enc.encode(pt, mask, scale);
    pe.multiply_plain_inplace(r, pt);
    pe.rescale_inplace(r);
    return r;
}

// SumR: fold all N rows into every row. Shifts: +N, +2N, .... No depth.
static heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& m, int N,
        heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe)
{
    heongpu::Ciphertext<Scheme> r = m;
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    if (g_verbose) std::cout << "  SumR:";
    for (int i = 0; i < logN; i++)
    {
        int shift = N * (1 << i);
        if (g_verbose) std::cout << " +" << shift;
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, shift);
        pe.add_inplace(r, rot);
    }
    if (g_verbose) std::cout << "\n";
    return r;
}

// SumC: fold all N columns into column 0. Shifts: +1, +2, +4, .... No depth.
static heongpu::Ciphertext<Scheme>
sumColumns(const heongpu::Ciphertext<Scheme>& m, int N,
           heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe)
{
    heongpu::Ciphertext<Scheme> r = m;
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    if (g_verbose) std::cout << "  SumC:";
    for (int i = 0; i < logN; i++)
    {
        int shift = 1 << i;
        if (g_verbose) std::cout << " +" << shift;
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, shift);
        pe.add_inplace(r, rot);
    }
    if (g_verbose) std::cout << "\n";
    return r;
}

// Debug helper: decrypt + print first N×N slots as a matrix
static void debugMatrix(const char* label,
                        heongpu::Ciphertext<Scheme>& ct,
                        heongpu::HEDecryptor<Scheme>& dec,
                        heongpu::HEEncoder<Scheme>& enc,
                        heongpu::HEContext<Scheme>& ctx,
                        int N)
{
    heongpu::Plaintext<Scheme> pt(ctx);
    dec.decrypt(pt, ct);
    std::vector<double> raw;
    enc.decode(raw, pt);
    std::cout << label << " (level=" << ct.level() << "):\n";
    for (int k = 0; k < N; k++) {
        std::cout << "  row " << k << ": [";
        for (int j = 0; j < N; j++)
            std::cout << std::fixed << std::setprecision(3) << raw[k*N+j]
                      << (j<N-1?", ":"");
        std::cout << "]\n";
    }
}

// ---------------------------------------------------------------------------
// homomorphicSortFG — full Algorithm 5
// ---------------------------------------------------------------------------
heongpu::Ciphertext<Scheme>
homomorphicSortFG(const heongpu::Ciphertext<Scheme>& ct_vector, int N,
                  heongpu::Galoiskey<Scheme>& row_key,
                  heongpu::Galoiskey<Scheme>& col_key,
                  heongpu::Galoiskey<Scheme>& sumr_key,
                  heongpu::Galoiskey<Scheme>& transr_key,
                  heongpu::Galoiskey<Scheme>& sumc_key,
                  heongpu::Relinkey<Scheme>& rk,
                  CKKSPolyEvaluator& pe,
                  heongpu::HEEncoder<Scheme>& enc,
                  heongpu::HEEncryptor<Scheme>& encryptor,
                  heongpu::HEDecryptor<Scheme>& dec,
                  heongpu::HEContext<Scheme>& ctx,
                  double scale)
{
    size_t slots = ctx->get_poly_modulus_degree() / 2;

    // ── Phase 1: rank matrix R ───────────────────────────────────────────────
    if (g_verbose) std::cout << "\n=== Phase 1: rank matrix ===\n";

    // VR[k,j] = v[j] for all rows k
    if (g_verbose) std::cout << "Step 1: ReplR(V)\n";
    heongpu::Ciphertext<Scheme> VR = replicateRow(ct_vector, N, row_key, pe);

    // VC[k,j] = v[k] for all columns j  (TransR depth 1; mod_drop VR to match)
    if (g_verbose) std::cout << "Step 2: TransR(V) + ReplC\n";
    heongpu::Ciphertext<Scheme> col_t =
        transposeRowToColumn(ct_vector, N, transr_key, pe, enc, ctx, scale);
    heongpu::Ciphertext<Scheme> VC = replicateColumn(col_t, N, col_key, pe);

    while (VR.level() > VC.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(VR, tmp);
        VR = std::move(tmp);
    }

    // C[k,j] ≈ 1 if v[j] > v[k],  ≈ 0 if v[j] < v[k]
    if (g_verbose) std::cout << "Step 3: compareAdv(VR, VC)\n";
    heongpu::Ciphertext<Scheme> C = compareAdv(VR, VC, pe, rk, ctx, scale);

    // SumR: left-rotation butterfly sums all rows into row 0 (suffix sum).
    // Row 0 gets the full sum; other rows get partial (suffix) sums.
    if (g_verbose) std::cout << "Step 4a: SumR(C) -> full sum in row 0\n";
    heongpu::Ciphertext<Scheme> R = sumRows(C, N, sumr_key, pe);

    // Mask row 0 to discard partial sums from rows 1..N-1.
    // Costs 1 level (ct×ct multiply + rescale).
    // HEonGPU multiply_plain requires plaintext.depth_ == ciphertext.depth_,
    // but encoding always gives depth_=0.  Encrypting the mask and mod-dropping
    // to R's level avoids the mismatch.
    if (g_verbose) std::cout << "Step 4b: MaskRow0\n";
    {
        std::vector<double> row0_mask(slots, 0.0);
        for (int j = 0; j < N; j++) row0_mask[j] = 1.0;
        heongpu::Plaintext<Scheme> pt_r0(ctx);
        enc.encode(pt_r0, row0_mask, scale);
        heongpu::Ciphertext<Scheme> ct_mask(ctx);
        encryptor.encrypt(ct_mask, pt_r0);
        while (ct_mask.level() > R.level())
        {
            heongpu::Ciphertext<Scheme> tmp(ctx);
            pe.mod_drop(ct_mask, tmp);
            ct_mask = std::move(tmp);
        }
        heongpu::Ciphertext<Scheme> R_masked(ctx);
        pe.multiply(R, ct_mask, R_masked);
        pe.relinearize_inplace(R_masked, rk);
        pe.rescale_inplace(R_masked);
        R = std::move(R_masked);
    }

    // ReplR: right-rotation butterfly replicates row 0 to all N rows.
    // After this, R[k,j] = rank_0(v[j]) + 0.5 for every row k. No depth cost.
    if (g_verbose) std::cout << "Step 4c: ReplR(R)\n";
    R = replicateRow(R, N, row_key, pe);

    if (g_verbose)
        std::cout << "  R level=" << R.level() << "\n";
    if (g_verbose) debugMatrix("R", R, dec, enc, ctx, N);

    // ── Phase 2: one-hot indicator M ────────────────────────────────────────
    if (g_verbose) std::cout << "\n=== Phase 2: one-hot indicator ===\n";

    // subMask[k,j] = -(k+0.5): after ReplR, R[k,j] = rank_0(v[j]) + 0.5
    // R + subMask = rank_0(v[j]) + 0.5 - (k+0.5) = rank_0(v[j]) - k
    // indicatorAdv fires iff |rank_0(v[j]) - k| < 0.5, i.e. rank_0(v[j]) == k
    std::vector<double> sub_vals(slots, 0.0);
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            sub_vals[k * N + j] = -(static_cast<double>(k) + 0.5);

    heongpu::Plaintext<Scheme> pt_sub(ctx);
    enc.encode(pt_sub, sub_vals, scale);
    heongpu::Ciphertext<Scheme> ct_sub(ctx);
    encryptor.encrypt(ct_sub, pt_sub);

    while (ct_sub.level() > R.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(ct_sub, tmp);
        ct_sub = std::move(tmp);
    }

    heongpu::Ciphertext<Scheme> ct_diff(ctx);
    pe.add(R, ct_sub, ct_diff);

    // M[k,j] ≈ 1 iff rank_0(v[j]) = k
    if (g_verbose) std::cout << "Step 5: indicatorAdv(R + subMask)\n";
    heongpu::Ciphertext<Scheme> M =
        indicatorAdv(ct_diff, N, pe, rk, ctx, scale);

    if (g_verbose)
        std::cout << "  M level=" << M.level() << "\n";
    if (g_verbose) debugMatrix("M", M, dec, enc, ctx, N);

    // ── Phase 3: reconstruct sorted values ──────────────────────────────────
    if (g_verbose) std::cout << "\n=== Phase 3: reconstruct ===\n";

    // Fresh ReplR(V), mod-dropped to M's level
    if (g_verbose) std::cout << "Step 6: ReplR(V) fresh + mod_drop\n";
    heongpu::Ciphertext<Scheme> VR2 = replicateRow(ct_vector, N, row_key, pe);
    while (VR2.level() > M.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(VR2, tmp);
        VR2 = std::move(tmp);
    }

    // S[k, col0] = (k+1)-th order statistic
    if (g_verbose) std::cout << "Step 7: multiply(M, VR) + SumC\n";
    heongpu::Ciphertext<Scheme> product(ctx);
    pe.multiply(M, VR2, product);
    pe.relinearize_inplace(product, rk);
    pe.rescale_inplace(product);

    heongpu::Ciphertext<Scheme> S = sumColumns(product, N, sumc_key, pe);

    if (g_verbose)
        std::cout << "Sort complete. result level=" << S.level() << "\n";
    return S;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int  N          = 8;
    bool bench_mode = false;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--bench")
            bench_mode = true;
        else if (!arg.empty() && std::isdigit(static_cast<unsigned char>(arg[0])))
            N = std::stoi(arg);
    }
    g_verbose = !bench_mode;

    if (N <= 0 || (N & (N - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << N << ")\n";
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);

    // ── HE context ──────────────────────────────────────────────────────────
    // n=65536 → 32768 slots → max N=128 (128²=16384 ≤ 32768).
    // 32 Q primes (1 special 60-bit + 31 computation 40-bit) → 31 usable levels.
    // 7 P primes (60-bit) for hybrid keyswitching.
    // Total Q bits = 60 + 31×40 = 1300; P bits = 420; Q+P = 1720 < 128-bit limit ✓
    heongpu::HEContext<Scheme> ctx = heongpu::GenHEContext<Scheme>();
    const size_t poly_modulus_degree = 65536;
    ctx->set_poly_modulus_degree(poly_modulus_degree);
    ctx->set_coeff_modulus_bit_sizes(
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
        {60, 60, 60, 60, 60, 60, 60});
    double scale = std::pow(2.0, 40);

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    ctx->generate();
    float ctx_ms = ctx_timer.stopTimer();

    const int slots = static_cast<int>(poly_modulus_degree / 2);
    if (N * N > slots)
    {
        std::cerr << "Error: N=" << N << " needs " << (N*N)
                  << " slots but only " << slots << " available.\n";
        return EXIT_FAILURE;
    }
    if (g_verbose)
        std::cout << "N=" << N << "  matrix=" << N << "×" << N
                  << "  slots_used=" << (N*N) << "/" << slots << "\n";

    // ── Key generation ───────────────────────────────────────────────────────
    heongpu::HEKeyGenerator<Scheme> keygen(ctx);
    heongpu::Secretkey<Scheme>  sk(ctx);  keygen.generate_secret_key(sk);
    heongpu::Publickey<Scheme>  pk(ctx);  keygen.generate_public_key(pk, sk);

    heongpu::HEEncoder<Scheme>    enc(ctx);
    heongpu::HEEncryptor<Scheme>  encryptor(ctx, pk);
    heongpu::HEDecryptor<Scheme>  decryptor(ctx, sk);
    CKKSPolyEvaluator             pe(ctx, enc);

    int logN = static_cast<int>(std::ceil(std::log2(N)));

    const size_t kMiB = 1024ULL * 1024ULL;
    size_t gpu_baseline =
        heongpu::MemoryPool::instance().get_current_device_pool_memory_usage();

    GPUTimer keygen_timer;
    keygen_timer.startTimer();

    auto rshifts = rowGaloisShifts(N);
    auto cshifts = colGaloisShifts(N);
    auto sshifts = sumrGaloisShifts(N);
    auto tshifts = transrGaloisShifts(N);
    auto scshifts= sumcGaloisShifts(N);

    heongpu::Galoiskey<Scheme> row_key(ctx, rshifts);
    keygen.generate_galois_key(row_key, sk);
    heongpu::Galoiskey<Scheme> col_key(ctx, cshifts);
    keygen.generate_galois_key(col_key, sk);
    heongpu::Galoiskey<Scheme> sumr_key(ctx, sshifts);
    keygen.generate_galois_key(sumr_key, sk);
    heongpu::Galoiskey<Scheme> transr_key(ctx, tshifts);
    keygen.generate_galois_key(transr_key, sk);
    heongpu::Galoiskey<Scheme> sumc_key(ctx, scshifts);
    keygen.generate_galois_key(sumc_key, sk);
    heongpu::Relinkey<Scheme> rk(ctx);
    keygen.generate_relin_key(rk, sk);

    float keygen_ms = keygen_timer.stopTimer();
    size_t gpu_keys_mib = (heongpu::MemoryPool::instance()
                               .get_current_device_pool_memory_usage()
                           - gpu_baseline) / kMiB;
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms
                  << " ms  (" << gpu_keys_mib << " MiB VRAM)\n";

    // ── Input ────────────────────────────────────────────────────────────────
    std::vector<double> input(N);
    if (bench_mode)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 100.0);
        for (int i = 0; i < N; i++) input[i] = dist(rng);
    }
    else
    {
        // Reversed sequence: sorted result should be 0,1,...,N-1
        for (int i = 0; i < N; i++) input[i] = static_cast<double>(N - 1 - i);
    }

    std::vector<double> expected_sorted = input;
    std::sort(expected_sorted.begin(), expected_sorted.end());

    std::vector<double> normalized = normalizeToUnit(input);

    if (g_verbose)
    {
        std::cout << "Input:          "; display_vector(input, N);
        std::cout << "Normalized:     "; display_vector(normalized, N);
        std::cout << "Expected sorted:"; display_vector(expected_sorted, N);
    }

    std::vector<double> slot_buf(slots, 0.0);
    for (int i = 0; i < N; i++) slot_buf[i] = normalized[i];

    heongpu::Plaintext<Scheme>  pt(ctx);
    enc.encode(pt, slot_buf, scale);
    heongpu::Ciphertext<Scheme> ct(ctx);
    encryptor.encrypt(ct, pt);

    // ── Sort (timed) ─────────────────────────────────────────────────────────
    GPUTimer sort_timer;
    sort_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_sorted =
        homomorphicSortFG(ct, N,
                          row_key, col_key, sumr_key, transr_key, sumc_key,
                          rk, pe, enc, encryptor, decryptor, ctx, scale);

    float sort_ms = sort_timer.stopTimer();
    size_t gpu_sort_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // ── Decrypt ───────────────────────────────────────────────────────────────
    heongpu::Plaintext<Scheme> pt_result(ctx);
    decryptor.decrypt(pt_result, ct_sorted);
    std::vector<double> raw;
    enc.decode(raw, pt_result);

    // Slot k*N = (k+1)-th order statistic (normalized, no factor-of-2 correction)
    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;

    std::vector<double> he_sorted(N), he_sorted_orig(N);
    for (int k = 0; k < N; k++)
    {
        he_sorted[k]      = raw[k * N];
        he_sorted_orig[k] = he_sorted[k] * range + lo;
    }

    // ── Output ───────────────────────────────────────────────────────────────
    if (bench_mode)
    {
        std::cout << "BENCH:"
                  << " N="           << N
                  << " ctx_ms="      << ctx_ms
                  << " keygen_ms="   << keygen_ms
                  << " sort_ms="     << sort_ms
                  << " gpu_keys_mib="<< gpu_keys_mib
                  << " gpu_sort_mib="<< gpu_sort_mib
                  << " gpu_peak_mib="<< gpu_peak_mib << "\n";
    }
    else
    {
        std::cout << "\n=== Sorting Results ===\n";
        std::cout << "HE sorted (normalized):     "; display_vector(he_sorted, N);
        std::cout << "HE sorted (original scale): "; display_vector(he_sorted_orig, N);

        std::cout << "\nVerification:\n";
        bool monotone    = true;
        bool all_correct = true;
        for (int k = 0; k < N; k++)
        {
            double expected = expected_sorted[k];
            double actual   = he_sorted_orig[k];
            double err      = std::abs(actual - expected);
            bool   correct  = err < (0.5 * range / N + 0.5);
            if (k > 0 && he_sorted_orig[k] < he_sorted_orig[k-1] - 0.1)
                monotone = false;
            if (!correct) all_correct = false;
            std::cout << "  sorted[" << k << "]: expected=" << expected
                      << "  actual=" << std::fixed << std::setprecision(4)
                      << actual
                      << (correct ? "" : "  INCORRECT") << "\n";
        }
        std::cout << (monotone    ? "  Monotone: YES\n" : "  Monotone: NO\n");
        std::cout << (all_correct ? "  Values:   all correct\n"
                                  : "  Values:   some incorrect\n");

        std::cout << "\nTiming:\n";
        std::cout << "  Context gen  : " << ctx_ms     << " ms\n";
        std::cout << "  Key gen      : " << keygen_ms  << " ms\n";
        std::cout << "  Sort         : " << sort_ms    << " ms  ("
                  << (sort_ms / 1000.0) << " s)\n";
        std::cout << "\nVRAM (above context baseline):\n";
        std::cout << "  Keys  : " << gpu_keys_mib << " MiB\n";
        std::cout << "  Sort  : " << gpu_sort_mib << " MiB\n";
        std::cout << "  Peak  : " << gpu_peak_mib << " MiB\n";
    }

    return EXIT_SUCCESS;
}
