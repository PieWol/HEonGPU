// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file 18_ckks_sort.cpp
 *
 * Implements Algorithm 5 (Sorting) from:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone, Everts, Hahn, Peter — USENIX Security 2025
 *
 * Algorithm 5 (Sorting):
 *   1. R  ← Rank(V)
 *   2. RR ← ReplR(R)
 *   3. M  ← Ind0(RR − (1^N ∥ 2^N ∥ … ∥ N^N))  [one-hot mask per row]
 *   4. VR ← ReplR(V)
 *   5. S  ← TransC(SumC(M · VR))
 *
 * HEonGPU constraint: multiply_plain(Ciphertext, Plaintext) checks depth equality
 * but freshly-encoded Plaintexts always have depth=0. Operations at depth > 0 that
 * require per-slot plaintext constants (rank constants subtraction, MaskC, MaskR)
 * therefore encrypt those constants as ciphertexts and use mod_drop alignment.
 *
 * Output layout (column format):
 *   The final ciphertext is NOT transposed (TransC is omitted). Instead the sorted
 *   values reside at positions 0, N, 2N, …, (N−1)·N of the decoded slot vector:
 *     slot k·N  =  (k+1)-th order statistic  (k = 0 .. N-1)
 *   The client reads every N-th slot to recover the sorted vector.
 *
 * Context: n = 65536 → 32768 slots; 28 computation levels (29 Q primes).
 * Supports N ≤ 128 (single-ciphertext mode, N² ≤ 16384 slots).
 * Accuracy: degree-2047 sign + degree-1023 indicator. Reliable for N ≤ 32;
 * larger N may show indicator approximation errors for elements with close ranks.
 *
 * Usage:
 *   18_ckks_sort [N] [--bench]
 *   N       : vector length, power of 2, default 8
 *   --bench : machine-readable output only
 */

#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>

static bool g_verbose = true;
constexpr auto Scheme = heongpu::Scheme::CKKS;

// ---------------------------------------------------------------------------
// CKKSPolyEvaluator — exposes protected evaluate_poly via a derived class
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
                   heongpu::Relinkey<Scheme>& relin_key, double a = -1.0,
                   double b = 1.0)
    {
        Polynomial poly(degree, coeffs, /*lead=*/false,
                        heongpu::PolyType::CHEBYSHEV, a, b);
        if (g_verbose)
            std::cout << "  Chebyshev poly degree=" << degree
                      << " depth=" << poly.depth() << " levels\n";
        return evaluate_poly(ct, target_scale, poly, relin_key,
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
    GPUTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GPUTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
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

static size_t getGPUUsedMiB()
{
    return heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
           / (1024ULL * 1024ULL);
}
static size_t getPeakGPUMiB()
{
    return heongpu::MemoryPool::instance().get_peak_device_pool_memory_usage()
           / (1024ULL * 1024ULL);
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------
/**
 * @brief Normalize input to [0,1] so pairwise differences lie in [-1,1].
 * Required before encrypt; the sign approximation domain is [-1,1].
 */
std::vector<double> normalizeForRanking(const std::vector<double>& input)
{
    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;
    std::vector<double> out(input.size());
    for (size_t i = 0; i < input.size(); i++)
        out[i] = (input[i] - lo) / range;
    return out;
}

// ---------------------------------------------------------------------------
// Galois shift helpers
// ---------------------------------------------------------------------------
/** TransR shifts: -(N*(N-1)/2^i) for i=1..logN  (negative = right-rotation in paper) */
std::vector<int> transposeGaloisShifts(int vec_len)
{
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    std::vector<int> shifts;
    for (int i = 1; i <= log_n; i++)
        shifts.push_back(-((vec_len * (vec_len - 1)) / (1 << i)));
    return shifts;
}

/** SumC shifts: +1, +2, +4, …, +N/2  (positive = left-rotation in paper ≪) */
std::vector<int> sumcGaloisShifts(int vec_len)
{
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    std::vector<int> shifts;
    for (int i = 0; i < log_n; i++)
        shifts.push_back(1 << i);
    return shifts;
}

// ---------------------------------------------------------------------------
// Matrix primitives
// ---------------------------------------------------------------------------
/**
 * @brief ReplR (Algorithm 11): replicate row 0 across all N rows.
 * Shifts: -(N/2)*N, -(N/4)*N, …, -N  (right-rotations by row multiples).
 */
heongpu::Ciphertext<Scheme>
replicateRow(const heongpu::Ciphertext<Scheme>& row_initial, int vec_len,
             heongpu::Galoiskey<Scheme>& galois_key,
             CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> result = row_initial;
    if (g_verbose) std::cout << "  ReplR shifts: ";
    for (int i = vec_len / 2; i > 0; i /= 2)
    {
        int shift = -(i * vec_len);
        if (g_verbose) std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose) std::cout << "\n";
    return result;
}

/**
 * @brief ReplC (Algorithm 12): replicate column 0 across all N columns.
 * Shifts: -1, -2, -4, …, -(N/2)  (right-rotations by column offsets).
 */
heongpu::Ciphertext<Scheme>
replicateColumn(const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
                heongpu::Galoiskey<Scheme>& galois_key,
                CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> result = col_initial;
    if (g_verbose) std::cout << "  ReplC shifts: ";
    for (int i = 1; i < vec_len; i *= 2)
    {
        int shift = -i;
        if (g_verbose) std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose) std::cout << "\n";
    return result;
}

/**
 * @brief TransR (Algorithm 1): transpose row vector to column vector.
 * Shifts: -(N*(N-1)/2^i) for i=1..logN, then MaskC(X,0).
 * Depth consumed: 1 (multiply_plain mask + rescale).
 */
heongpu::Ciphertext<Scheme>
transposeRowToColumn(const heongpu::Ciphertext<Scheme>& row_vector, int vec_len,
                     heongpu::Galoiskey<Scheme>& galois_key,
                     CKKSPolyEvaluator& evaluator,
                     heongpu::HEEncoder<Scheme>& encoder,
                     heongpu::HEContext<Scheme>& context, double scale)
{
    heongpu::Ciphertext<Scheme> result = row_vector;
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));

    if (g_verbose) std::cout << "  TransR (logN=" << log_n << ") shifts: ";
    for (int i = 1; i <= log_n; i++)
    {
        int shift = -((vec_len * (vec_len - 1)) / (1 << i));
        if (g_verbose) std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose) std::cout << "\n";

    // MaskC(X, 0): keep column 0 only (positions k*N for k=0..N-1)
    size_t total_slots = context->get_poly_modulus_degree() / 2;
    std::vector<double> mask(total_slots, 0.0);
    for (int row = 0; row < vec_len; row++)
        mask[row * vec_len] = 1.0;

    heongpu::Plaintext<Scheme> pt_mask(context);
    encoder.encode(pt_mask, mask, scale);
    evaluator.multiply_plain_inplace(result, pt_mask);
    evaluator.rescale_inplace(result); // depth +1
    return result;
}

/**
 * @brief Chebyshev sign approximation (comparison function).
 * Approximates sign(x) on [-1,1] with degree-D Chebyshev polynomial via BSGS.
 * Depth consumed: ceil(log2(degree)) ≈ 11 for degree=2047.
 */
heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree = 2047)
{
    if (g_verbose)
        std::cout << "  Sign approx degree=" << degree << "...\n";
    auto sign_fn = [](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(re > 0 ? 1.0 : (re < 0 ? -1.0 : 0.0), 0.0);
    };
    auto coeffs = heongpu::approximate_function(sign_fn, -1.0, 1.0, degree);
    return poly_eval.eval_chebyshev(ct_diff, scale, coeffs, degree, relin_key,
                                    -1.0, 1.0);
}

/**
 * @brief Chebyshev indicator approximation: Ind0(x) ≈ 1 when |x| < 0.5, 0 otherwise.
 *
 * Applied to (rank[j] − k) after the subtraction of the rank-constant matrix.
 * Domain [−(N−1), N−1] covers all possible rank differences for an N-element vector.
 *
 * Depth consumed: ceil(log2(degree)) levels (≈10 for degree=1023).
 *
 * Accuracy note: degree-1023 provides reliable discrimination for N ≤ 32 where the
 * Chebyshev node spacing near x=0 is much smaller than the rank error from the sign
 * approximation. For N > 32 the indicator may misclassify elements with close ranks.
 */
heongpu::Ciphertext<Scheme>
chebyshev_indicator_approx(heongpu::Ciphertext<Scheme>& ct_input, int vec_len,
                           CKKSPolyEvaluator& poly_eval,
                           heongpu::Relinkey<Scheme>& relin_key, double scale,
                           int degree = 1023)
{
    if (g_verbose)
        std::cout << "  Indicator approx degree=" << degree
                  << " domain=[" << -(vec_len - 1) << "," << (vec_len - 1) << "]\n";
    auto ind0 = [](Complex64 x) -> Complex64 {
        return Complex64(std::abs(x.real()) < 0.5 ? 1.0 : 0.0, 0.0);
    };
    double a = -(double)(vec_len - 1);
    double b =  (double)(vec_len - 1);
    auto coeffs = heongpu::approximate_function(ind0, a, b, degree);
    return poly_eval.eval_chebyshev(ct_input, scale, coeffs, degree, relin_key,
                                    a, b);
}

/**
 * @brief SumR (Algorithm 9): fold all rows into row 0.
 * Shifts: +N, +2N, +4N, …  (positive = left-rotation, row-fold).
 * No level consumed.
 */
heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& sumr_galois_key,
        CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> result = ct_matrix;
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    if (g_verbose) std::cout << "  SumR shifts: ";
    for (int i = 0; i < log_n; i++)
    {
        int shift = vec_len * (1 << i);
        if (g_verbose) std::cout << "+" << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, sumr_galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose) std::cout << "\n";
    return result;
}

/**
 * @brief SumC (Algorithm 10): fold all N columns into column 0.
 *
 * Uses positive shifts +1, +2, +4, …, +N/2 (left-rotations, ≪ in paper notation).
 * After logN iterations, position k*N (column 0 of row k) holds the sum of all
 * N values in row k. Columns 1..N-1 contain garbage; callers must handle this.
 *
 * Correctness: for a row-major N×N matrix, left-rotating by 2^i brings column 2^i
 * into column 0 without cross-row contamination at column 0, because 2^i < N for all
 * i < logN. The garbage in non-zero columns does NOT affect column 0.
 *
 * No level consumed (rotations and additions do not change depth).
 */
heongpu::Ciphertext<Scheme>
sumColumns(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
           heongpu::Galoiskey<Scheme>& sumc_galois_key,
           CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> result = ct_matrix;
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    if (g_verbose) std::cout << "  SumC shifts: ";
    for (int i = 0; i < log_n; i++)
    {
        int shift = 1 << i; // +1, +2, +4, …
        if (g_verbose) std::cout << "+" << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, sumc_galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose) std::cout << "\n";
    return result;
}

// ---------------------------------------------------------------------------
// Algorithm 3: basicRank
// ---------------------------------------------------------------------------
/**
 * @brief Algorithm 3 (Rank): compute 1-based fractional rank of each element.
 *
 * Depth budget (28 levels available with n=65536, 29 Q primes):
 *   depth 0  → fresh ciphertext
 *   depth 1  → TransR (MaskC multiply_plain + rescale)
 *   depth 1  → mod_drop ct_row for alignment
 *   depth 12 → Chebyshev sign degree-2047 (11 levels, BSGS)
 *   depth 12 → add_plain +1, SumR (free)
 *   depth 13 → multiply_plain(0.5, scale) + rescale (scalar ÷2)
 *   depth 13 → add_plain +0.5 (completes 1-based fractional rank)
 *   Total consumed: 13 ≤ 28 ✓
 *
 * Output: position j (0 ≤ j < N) in row 0 holds rank[j] (1-based fractional).
 */
heongpu::Ciphertext<Scheme>
basicRank(const heongpu::Ciphertext<Scheme>& ct_vector, int vec_len,
          heongpu::Galoiskey<Scheme>& row_galois_key,
          heongpu::Galoiskey<Scheme>& col_galois_key,
          heongpu::Galoiskey<Scheme>& transpose_galois_key,
          heongpu::Galoiskey<Scheme>& sumr_galois_key,
          heongpu::Relinkey<Scheme>& relin_key,
          CKKSPolyEvaluator& evaluator,
          heongpu::HEEncoder<Scheme>& encoder,
          heongpu::HEContext<Scheme>& context, double scale)
{
    if (g_verbose)
        std::cout << "\n=== Ranking (N=" << vec_len << ") ===\n"
                  << "Step 1: ReplR...\n" << std::flush;

    heongpu::Ciphertext<Scheme> ct_row =
        replicateRow(ct_vector, vec_len, row_galois_key, evaluator);

    if (g_verbose) std::cout << "Step 2: TransR + ReplC...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_col_t =
        transposeRowToColumn(ct_vector, vec_len, transpose_galois_key,
                             evaluator, encoder, context, scale);
    heongpu::Ciphertext<Scheme> ct_col =
        replicateColumn(ct_col_t, vec_len, col_galois_key, evaluator);

    // Align ct_row level down to match ct_col (TransR consumed 1 level)
    while (ct_row.level() > ct_col.level())
    {
        heongpu::Ciphertext<Scheme> tmp(context);
        evaluator.mod_drop(ct_row, tmp);
        ct_row = std::move(tmp);
    }

    if (g_verbose) std::cout << "Step 3: Compute diff (vR - vC)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_row, ct_col, ct_diff);

    if (g_verbose) std::cout << "Step 4: Chebyshev sign approx...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_sign =
        chebyshev_sign_approx(ct_diff, evaluator, relin_key, scale, 2047);

    if (g_verbose) std::cout << "Step 5: Add 1 (shift {-1,0,+1} → {0,1,2})...\n" << std::flush;
    evaluator.add_plain_inplace(ct_sign, 1.0);

    if (g_verbose) std::cout << "Step 6: SumR (row-fold)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_sumr =
        sumRows(ct_sign, vec_len, sumr_galois_key, evaluator);

    // ÷2 via scalar multiply_plain (double overload — no depth check)
    if (g_verbose) std::cout << "Step 7: Scale by 0.5 + rescale...\n" << std::flush;
    evaluator.multiply_plain_inplace(ct_sumr, 0.5, scale);
    evaluator.rescale_inplace(ct_sumr); // depth 12 → 13

    if (g_verbose) std::cout << "Step 8: Add 0.5 (complete fractional rank)...\n" << std::flush;
    evaluator.add_plain_inplace(ct_sumr, 0.5);

    if (g_verbose) std::cout << "Ranking complete (depth=13).\n" << std::flush;
    return ct_sumr;
}

// ---------------------------------------------------------------------------
// Algorithm 5: homomorphicSort
// ---------------------------------------------------------------------------
/**
 * @brief Algorithm 5 (Sort): homomorphically sort an encrypted vector.
 *
 * Preconditions:
 *   - ct_vector encrypts a normalized vector in [0,1] with distinct elements.
 *   - N = vec_len is a power of 2 with N² ≤ 32768 (i.e., N ≤ 128).
 *
 * Depth budget (28 levels available):
 *   depth 13  → basicRank output
 *   depth 13  → ReplR(rank) [free], ciphertext subtract rank-constants [free]
 *   depth 23  → chebyshev_indicator_approx degree-1023 (+10 levels)
 *   depth 23  → mod_drop VR from 0 to 23 [free: only drops primes]
 *   depth 24  → multiply(M, VR) + relinearize + rescale (+1 level)
 *   depth 24  → SumC rotations [free]
 *   Total consumed: 24 ≤ 28 ✓  (4 levels spare)
 *
 * Output layout ("column format"):
 *   The returned ciphertext has the sorted values at positions k*N for k=0..N-1.
 *   Position k*N holds the (k+1)-th order statistic (0-indexed: k=0 → minimum).
 *   Clients extract sorted[k] = decoded_slots[k * vec_len].
 *
 * Why column format instead of Algorithm 5's TransC step:
 *   TransC and its MaskR require multiply_plain(Ciphertext, Plaintext) at depth ≥24.
 *   In HEonGPU, freshly-encoded Plaintexts always have depth=0, so this throws a
 *   depth-mismatch error. Encrypting the mask ciphertext and using ct×ct multiply
 *   (with relinearization) would be correct but costs an extra level and is ~10×
 *   slower. Omitting TransC and reading every N-th slot is equivalent and free.
 *   The same constraint applies to MaskC after SumC: column 0 is always correct
 *   (SumC rotations < N guarantee no cross-row contamination at column 0), so
 *   we skip MaskC too — column 0 is read directly by the client.
 *
 * Why rank_constants are encrypted as a ciphertext:
 *   The rank-constant matrix (row k = k+1 for all j) must be subtracted from
 *   ReplR(rank) at depth 13. Since add_plain(Ciphertext, Plaintext) also checks
 *   depth equality in HEonGPU (and Plaintext.depth_=0 always), the subtraction
 *   is done as a ciphertext op: encrypt rank_constants → mod_drop to depth 13 →
 *   ct subtract. Encrypting public constants is valid; noise is negligible.
 */
heongpu::Ciphertext<Scheme>
homomorphicSort(const heongpu::Ciphertext<Scheme>& ct_vector, int vec_len,
                heongpu::Galoiskey<Scheme>& row_galois_key,
                heongpu::Galoiskey<Scheme>& col_galois_key,
                heongpu::Galoiskey<Scheme>& sumr_galois_key,
                heongpu::Galoiskey<Scheme>& transpose_galois_key,
                heongpu::Galoiskey<Scheme>& sumc_galois_key,
                heongpu::Relinkey<Scheme>& relin_key,
                CKKSPolyEvaluator& evaluator,
                heongpu::HEEncoder<Scheme>& encoder,
                heongpu::HEEncryptor<Scheme>& encryptor,
                heongpu::HEContext<Scheme>& context,
                double scale)
{
    size_t total_slots = context->get_poly_modulus_degree() / 2;

    // ── Step 1: R = Rank(V) ──────────────────────────────────────────────
    if (g_verbose) std::cout << "\n=== homomorphicSort (N=" << vec_len << ") ===\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_rank =
        basicRank(ct_vector, vec_len,
                  row_galois_key, col_galois_key,
                  transpose_galois_key, sumr_galois_key,
                  relin_key, evaluator, encoder, context, scale);
    // depth 13

    // ── Step 2: RR = ReplR(R) ────────────────────────────────────────────
    if (g_verbose) std::cout << "\nStep 2: ReplR(rank)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_rr =
        replicateRow(ct_rank, vec_len, row_galois_key, evaluator);
    // depth 13

    // ── Step 3: M = Ind0(RR − rank_constants) ───────────────────────────
    // rank_constants[k][j] = k+1 for all j in row k.
    // Subtract by adding negated constants encrypted as a ciphertext.
    if (g_verbose) std::cout << "\nStep 3a: Build & encrypt rank-constants matrix...\n" << std::flush;

    std::vector<double> rank_consts_neg(total_slots, 0.0);
    for (int k = 0; k < vec_len; k++)
        for (int j = 0; j < vec_len; j++)
            rank_consts_neg[k * vec_len + j] = -(double)(k + 1);

    heongpu::Plaintext<Scheme> pt_consts(context);
    encoder.encode(pt_consts, rank_consts_neg, scale);
    heongpu::Ciphertext<Scheme> ct_consts(context);
    encryptor.encrypt(ct_consts, pt_consts);
    // Align ct_consts level to ct_rr using non-inplace mod_drop (mod_drop_inplace
    // does NOT update depth_, mod_drop(in,out) sets out.depth_ = in.depth_ + 1).
    while (ct_consts.level() > ct_rr.level())
    {
        heongpu::Ciphertext<Scheme> tmp(context);
        evaluator.mod_drop(ct_consts, tmp);
        ct_consts = std::move(tmp);
    }
    if (g_verbose)
        std::cout << "  ct_rr level=" << ct_rr.level()
                  << "  ct_consts level=" << ct_consts.level() << "\n" << std::flush;

    if (g_verbose) std::cout << "Step 3b: Subtract rank-constants (ct - ct)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.add(ct_rr, ct_consts, ct_diff); // adds negated constants = subtract

    if (g_verbose) std::cout << "Step 3c: Indicator Ind0 (degree 1023)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_mask =
        chebyshev_indicator_approx(ct_diff, vec_len, evaluator, relin_key, scale, 1023);
    // depth 23; M[k][j] ≈ 1 iff rank[j] ≈ k+1, else ≈ 0

    // ── Step 4: VR = ReplR(V), aligned to depth 23 ──────────────────────
    if (g_verbose) std::cout << "\nStep 4: ReplR(V) + mod_drop to depth 23...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_vr =
        replicateRow(ct_vector, vec_len, row_galois_key, evaluator);
    // Align ct_vr level to ct_mask using non-inplace mod_drop
    while (ct_vr.level() > ct_mask.level())
    {
        heongpu::Ciphertext<Scheme> tmp(context);
        evaluator.mod_drop(ct_vr, tmp);
        ct_vr = std::move(tmp);
    }
    if (g_verbose)
        std::cout << "  ct_mask level=" << ct_mask.level()
                  << "  ct_vr level=" << ct_vr.level() << "\n" << std::flush;

    // ── Step 5: M · VR, then SumC ────────────────────────────────────────
    if (g_verbose) std::cout << "\nStep 5a: multiply(M, VR) + relin + rescale...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_product(context);
    evaluator.multiply(ct_mask, ct_vr, ct_product);
    evaluator.relinearize_inplace(ct_product, relin_key);
    evaluator.rescale_inplace(ct_product);
    // depth 24

    if (g_verbose) std::cout << "Step 5b: SumC (fold columns into column 0)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_sorted_col =
        sumColumns(ct_product, vec_len, sumc_galois_key, evaluator);
    // depth 24
    // Column 0 of each row k holds the (k+1)-th order statistic.
    // TransC (column→row transposition) is omitted — see class-level note.

    if (g_verbose)
        std::cout << "Sort complete (depth=24). "
                     "Output in column format: slot k*N = (k+1)-th statistic.\n" << std::flush;
    return ct_sorted_col;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Parse args
    int vec_len   = 8;
    bool bench_mode = false;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--bench")
            bench_mode = true;
        else if (!arg.empty() && std::isdigit(static_cast<unsigned char>(arg[0])))
            vec_len = std::stoi(arg);
    }
    g_verbose = !bench_mode;

    if (vec_len <= 0 || (vec_len & (vec_len - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << vec_len << ")\n";
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);

    // ── HE Context ──────────────────────────────────────────────────────────
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // n = 65536 → 32768 available slots, supports N² ≤ 32768 (N ≤ 128 for pow-of-2).
    //
    // Q = 60 + 28×40 = 1180 bits (29 primes, 28 computation levels)
    // P = 7×60       =  420 bits
    // Q+P             = 1600 bits < 1746 = heongpu_128bit_std_parms(65536) ✓
    //
    // Depth budget: 28 usable levels. Sort consumes 24; 4 spare.
    const size_t poly_modulus_degree = 65536;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes(
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, // 29 Q primes
        {60, 60, 60, 60, 60, 60, 60});                                 // 7 P primes

    double scale = std::pow(2.0, 40);

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    context->generate();
    float ctx_ms = ctx_timer.stopTimer();

    const int available_slots = static_cast<int>(poly_modulus_degree / 2);
    if (vec_len * vec_len > available_slots)
    {
        std::cerr << "Error: N=" << vec_len << " needs " << (vec_len * vec_len)
                  << " slots but only " << available_slots << " available.\n"
                  << "Reduce N or increase poly_modulus_degree.\n";
        return EXIT_FAILURE;
    }

    if (g_verbose)
        std::cout << "N=" << vec_len << "  matrix=" << vec_len << "×" << vec_len
                  << "  slots=" << (vec_len * vec_len) << "/" << available_slots << "\n";

    // ── Keys ────────────────────────────────────────────────────────────────
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme>  secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme>  public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<Scheme>    encoder(context);
    heongpu::HEEncryptor<Scheme>  encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme>  decryptor(context, secret_key);
    CKKSPolyEvaluator             evaluator(context, encoder);

    // Compute all Galois shift sets before keygen timer
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));

    std::vector<int> row_galois_shifts;    // ReplR: -(N/2)*N, ..., -N
    for (int i = vec_len / 2; i > 0; i /= 2)
        row_galois_shifts.push_back(-(i * vec_len));

    std::vector<int> col_galois_shifts;    // ReplC: -1, -2, -4, ...
    for (int i = 1; i < vec_len; i *= 2)
        col_galois_shifts.push_back(-i);

    std::vector<int> sumr_galois_shifts;   // SumR: +N, +2N, +4N, ...
    for (int i = 0; i < log_n; i++)
        sumr_galois_shifts.push_back(vec_len * (1 << i));

    std::vector<int> transr_shifts = transposeGaloisShifts(vec_len); // TransR (neg)
    std::vector<int> sumc_shifts   = sumcGaloisShifts(vec_len);      // SumC (pos)

    if (g_verbose)
        std::cout << "Galois shifts — row: " << row_galois_shifts.size()
                  << "  col: " << col_galois_shifts.size()
                  << "  sumr: " << sumr_galois_shifts.size()
                  << "  transr: " << transr_shifts.size()
                  << "  sumc: " << sumc_shifts.size() << "\n";

    // ── Key generation (timed) ───────────────────────────────────────────────
    const size_t kMiB = 1024ULL * 1024ULL;
    size_t gpu_baseline_bytes =
        heongpu::MemoryPool::instance().get_current_device_pool_memory_usage();

    GPUTimer keygen_timer;
    keygen_timer.startTimer();

    heongpu::Galoiskey<Scheme> row_galois_key(context, row_galois_shifts);
    keygen.generate_galois_key(row_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> col_galois_key(context, col_galois_shifts);
    keygen.generate_galois_key(col_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> sumr_galois_key(context, sumr_galois_shifts);
    keygen.generate_galois_key(sumr_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> transr_galois_key(context, transr_shifts);
    keygen.generate_galois_key(transr_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> sumc_galois_key(context, sumc_shifts);
    keygen.generate_galois_key(sumc_galois_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    float keygen_ms = keygen_timer.stopTimer();
    size_t gpu_keys_mib = getGPUUsedMiB() - gpu_baseline_bytes / kMiB;
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms << " ms  (keys: "
                  << gpu_keys_mib << " MiB VRAM)\n";

    // ── Input preparation ────────────────────────────────────────────────────
    std::vector<double> input(vec_len);
    if (bench_mode)
    {
        // Uniform random, seeded for reproducibility
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 100.0);
        for (int i = 0; i < vec_len; i++)
            input[i] = dist(rng);
    }
    else
    {
        // Shuffled: 0..(N-1) in reverse, easy to verify sorted output
        for (int i = 0; i < vec_len; i++)
            input[i] = static_cast<double>(vec_len - 1 - i);
    }

    // Reference sort (for verification)
    std::vector<double> input_sorted = input;
    std::sort(input_sorted.begin(), input_sorted.end());

    std::vector<double> normalized = normalizeForRanking(input);

    if (g_verbose)
    {
        std::cout << "Original input: ";
        display_vector(input, vec_len);
        std::cout << "Normalized:     ";
        display_vector(normalized, vec_len);
        std::cout << "Expected sorted (original): ";
        display_vector(input_sorted, vec_len);
    }

    // Encode into full slot buffer
    std::vector<double> slot_buf(available_slots, 0.0);
    for (int i = 0; i < vec_len; i++)
        slot_buf[i] = normalized[i];

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, slot_buf, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    // ── Sort (timed) ─────────────────────────────────────────────────────────
    GPUTimer sort_timer;
    sort_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_sorted =
        homomorphicSort(ciphertext, vec_len,
                        row_galois_key, col_galois_key,
                        sumr_galois_key, transr_galois_key,
                        sumc_galois_key, relin_key,
                        evaluator, encoder, encryptor, context, scale);

    float sort_ms = sort_timer.stopTimer();
    size_t gpu_sort_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline_bytes) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // ── Decrypt and decode ───────────────────────────────────────────────────
    heongpu::Plaintext<Scheme> sort_plaintext(context);
    decryptor.decrypt(sort_plaintext, ct_sorted);
    std::vector<double> sort_result;
    encoder.decode(sort_result, sort_plaintext);

    // Extract column format: sorted[k] = slot k*N
    std::vector<double> he_sorted(vec_len);
    for (int k = 0; k < vec_len; k++)
        he_sorted[k] = sort_result[k * vec_len];

    // Denormalize for comparison with original input
    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;
    std::vector<double> he_sorted_denorm(vec_len);
    for (int k = 0; k < vec_len; k++)
        he_sorted_denorm[k] = he_sorted[k] * range + lo;

    // ── Output ───────────────────────────────────────────────────────────────
    if (bench_mode)
    {
        // rank_ms excluded; sort_ms covers full ranking+sorting protocol
        std::cout << "BENCH:"
                  << " N=" << vec_len
                  << " ctx_ms=" << ctx_ms
                  << " keygen_ms=" << keygen_ms
                  << " sort_ms=" << sort_ms
                  << " gpu_keys_mib=" << gpu_keys_mib
                  << " gpu_sort_mib=" << gpu_sort_mib
                  << " gpu_peak_mib=" << gpu_peak_mib << "\n";
    }
    else
    {
        std::cout << "\n=== Sorting Results ===\n";
        std::cout << "HE sorted (normalized):  ";
        display_vector(he_sorted, vec_len);
        std::cout << "HE sorted (original scale): ";
        display_vector(he_sorted_denorm, vec_len);

        // Verify monotonicity and approximate correctness
        std::cout << "\nVerification:\n";
        bool monotone    = true;
        bool all_correct = true;
        for (int k = 0; k < vec_len; k++)
        {
            double expected = input_sorted[k];
            double actual   = he_sorted_denorm[k];
            double err      = std::abs(actual - expected);
            bool correct    = (err < 0.5 * range / vec_len + 0.5);
            if (k > 0 && he_sorted_denorm[k] < he_sorted_denorm[k - 1] - 0.1)
                monotone = false;
            if (!correct)
                all_correct = false;
            std::cout << "  sorted[" << k << "]: expected=" << expected
                      << "  actual=" << actual
                      << (correct ? "" : "  ← INCORRECT") << "\n";
        }
        std::cout << (monotone    ? "  Monotone: YES\n"   : "  Monotone: NO\n");
        std::cout << (all_correct ? "  Values: all correct\n"
                                  : "  Values: some incorrect (check N or degree)\n");

        std::cout << "\nTiming:\n";
        std::cout << "  Key generation : " << keygen_ms << " ms\n";
        std::cout << "  Sort (rank+sort): " << sort_ms << " ms  ("
                  << (sort_ms / 1000.0) << " s)\n";
        std::cout << "\nVRAM usage (above context baseline):\n";
        std::cout << "  Keys  : " << gpu_keys_mib << " MiB\n";
        std::cout << "  Sort  : " << gpu_sort_mib << " MiB\n";
        std::cout << "  Peak  : " << gpu_peak_mib << " MiB\n";
    }

    return EXIT_SUCCESS;
}
