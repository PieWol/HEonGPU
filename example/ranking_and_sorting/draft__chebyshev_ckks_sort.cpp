
/**
 * @file 17_ckks_sort.cpp
 *
 * Implements sorting via Algorithm 5 from:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone, Everts, Hahn, Peter — USENIX Security 2025
 *
 * The algorithm has three phases:
 *
 *   Phase 1 — Rank matrix (Algorithm 3):
 *     VR = ReplR(V)          // N×N: row k = V for all k
 *     VC = ReplC(TransR(V))  // N×N: col j = V[j] for all rows
 *     C  = compare(VR,VC)    // C[k,j] ≈ 1 if v[j]>v[k], ≈0 if v[j]<v[k], ≈0.5 if equal
 *     R  = SumR(C)           // R[k,j] = rank_0(v[j]) + 0.5  in EVERY row k
 *                            // (SumR cyclic tree-fold fills all N rows identically)
 *
 *   Phase 2 — Order statistics (Algorithm 5, steps 2-4):
 *     subMask[k,j] = -(k + 0.5)   for k=0..N-1, j=0..N-1
 *     M = Ind0(R + subMask)        // M[k,j] ≈ 1 iff rank_0(v[j]) = k  (one-hot)
 *     → row k of M selects the (k+1)-th smallest element
 *
 *   Phase 3 — Reconstruct sorted values:
 *     VR = ReplR(V)                // fresh copy, mod-dropped to level of M
 *     S  = SumC(M · VR)            // S[k, col0] = (k+1)-th order statistic
 *
 * Key invariant: SumR distributes the column sum to ALL N rows (cyclic tree-fold
 * with shifts +N,+2N,+4N,... wraps, so every row ends with the same total).
 * Therefore no MaskR or ReplR of the rank ciphertext is needed — the rank matrix
 * from Phase 1 can be fed directly into the subMask subtraction of Phase 2.
 *
 * HEonGPU constraint: multiply_plain(Ciphertext, Plaintext) checks depth equality
 * but freshly-encoded Plaintexts always have depth=0.  Per-slot plaintext constants
 * (subMask, TransR MaskC) are therefore encrypted as ciphertexts and mod-dropped
 * to the required level before use.
 *
 * Indicator: the composite-sign approach is used:
 *   Ind0(x) ≈ 0.5 * (sign(x + 0.5) − sign(x − 0.5))
 * After normalizing x by 1/N both sign calls live on [-1,1], avoiding the
 * Gibbs/Runge overflow that would occur with a direct polynomial approximation
 * of the indicator on [-N,N].  The function returns 2·Ind0 (factor of 2 because
 * sign(+ε)−sign(−ε) = 1−(−1) = 2); callers divide decoded output by 2.
 *
 * Output layout (column format, TransC omitted):
 *   Slot k·N  =  (k+1)-th order statistic   (k = 0 .. N-1)
 *   Client reads every N-th slot to recover the sorted vector.
 *
 * Depth budget (28 levels available with n=65536, 31 Q-computation primes):
 *   depth  0  → fresh ciphertext
 *   depth  1  → Phase 1: TransR MaskC (multiply_plain + rescale)
 *   depth  1  → mod_drop VR to match VC
 *   depth 10  → sign degree-255, 9 levels (BSGS)
 *   depth 10  → add_plain +1, SumR (both free)
 *   depth 11  → multiply_plain 0.5 + rescale  → R[j] = rank_0(j)+0.5
 *   depth 11  → Phase 2: encrypt subMask, mod_drop, add (free)
 *   depth 12  → normalize by 1/N + rescale (indicator pre-step)
 *   depth 21  → sign degree-255 ×2 (parallel, costs 9 levels)
 *   depth 21  → subtract signs (free)
 *   depth 21  → Phase 3: mod_drop fresh VR to level 21 (free)
 *   depth 22  → multiply(M, VR) + relin + rescale
 *   depth 22  → SumC (free)
 *   Total: 22 levels ≤ 28 ✓  (6 levels spare)
 *
 * Usage:
 *   17_ckks_sort [N] [--bench]
 *   N       : vector length, power of 2, default 8
 *   --bench : machine-readable timing output only
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
#include <iomanip>

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
/** TransR shifts: -(N*(N-1)/2^i) for i=1..logN */
std::vector<int> transposeGaloisShifts(int vec_len)
{
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    std::vector<int> shifts;
    for (int i = 1; i <= log_n; i++)
        shifts.push_back(-((vec_len * (vec_len - 1)) / (1 << i)));
    return shifts;
}

/** SumC shifts: +1, +2, +4, …, +N/2 */
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
 *
 * Precondition: only row 0 of the input is non-zero.
 * Shifts: -(N/2)*N, -(N/4)*N, …, -N (right-rotations by row multiples).
 * No depth consumed.
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
 *
 * Precondition: only column 0 is non-zero.
 * Shifts: -1, -2, -4, …, -(N/2).
 * No depth consumed.
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
 * @brief TransR (Algorithm 1): transpose row 0 into a column vector.
 *
 * Shifts: -(N*(N-1)/2^i) for i=1..logN, then MaskC to zero all but column 0.
 * Depth consumed: 1 (MaskC multiply_plain + rescale).
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
 * @brief Chebyshev sign approximation on domain [a, b].
 *
 * Depth consumed: ceil(log2(degree)) levels (BSGS).
 * Default domain [-1, 1]; the input must lie within [a, b].
 */
heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree = 255, double a = -1.0, double b = 1.0)
{
    if (g_verbose)
        std::cout << "  Sign approx degree=" << degree
                  << " domain=[" << a << "," << b << "]...\n";
    auto sign_fn = [](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(re > 0 ? 1.0 : (re < 0 ? -1.0 : 0.0), 0.0);
    };
    auto coeffs = heongpu::approximate_function(sign_fn, a, b, degree);
    return poly_eval.eval_chebyshev(ct, scale, coeffs, degree, relin_key, a, b);
}

/**
 * @brief Composite-sign indicator: Ind0(x) ≈ 0.5*(sign(x+0.5) − sign(x−0.5)).
 *
 * Returns 2·Ind0(x): ≈ 2 when x ≈ 0, ≈ 0 when |x| ≥ 1.
 * Callers must divide decoded output by 2.
 *
 * Why composite sign: a direct Chebyshev polynomial approximation of
 * 1_{|x|<0.5} on [-N,N] suffers the Gibbs/Runge phenomenon — coefficients
 * blow up, and CKKS Chebyshev evaluation interprets the input as already in
 * [-1,1] internally (T_d(ct) overflows for |ct|>1).  The composite-sign form
 * avoids both issues: sign is bounded by construction, and after normalizing
 * ct by 1/N the two calls land on the well-conditioned [-1,1] domain.
 *
 * Input ct_input: integer-valued (rank_0(j) - k), range [-(N-1), N-1].
 * Depth consumed: 1 (normalize rescale) + ceil(log2(degree)) (sign).
 */
heongpu::Ciphertext<Scheme>
chebyshev_indicator_approx(heongpu::Ciphertext<Scheme>& ct_input, int vec_len,
                           CKKSPolyEvaluator& poly_eval,
                           heongpu::Relinkey<Scheme>& relin_key, double scale,
                           heongpu::HEContext<Scheme>& context,
                           int degree = 255)
{
    double invN    = 1.0 / vec_len;
    double halfInv = 0.5 / vec_len;

    if (g_verbose)
        std::cout << "  Indicator degree=" << degree
                  << " (normalize ×1/" << vec_len
                  << ", threshold ±" << halfInv << ")\n";

    // Normalize to [-1,1] (×1/N + rescale, costs 1 level)
    heongpu::Ciphertext<Scheme> ct_norm = ct_input;
    poly_eval.multiply_plain_inplace(ct_norm, invN, scale);
    poly_eval.rescale_inplace(ct_norm);

    // Shift by ±0.5/N (add_plain, free)
    heongpu::Ciphertext<Scheme> ct_plus  = ct_norm;
    poly_eval.add_plain_inplace(ct_plus,  halfInv);
    heongpu::Ciphertext<Scheme> ct_minus = ct_norm;
    poly_eval.add_plain_inplace(ct_minus, -halfInv);

    // Two sign evaluations on [-1,1]; both start from the same level
    if (g_verbose) std::cout << "  sign(x/" << vec_len << " + " << halfInv << "):\n";
    heongpu::Ciphertext<Scheme> sign_plus =
        chebyshev_sign_approx(ct_plus,  poly_eval, relin_key, scale, degree);
    if (g_verbose) std::cout << "  sign(x/" << vec_len << " - " << halfInv << "):\n";
    heongpu::Ciphertext<Scheme> sign_minus =
        chebyshev_sign_approx(ct_minus, poly_eval, relin_key, scale, degree);

    // sign_plus - sign_minus ≈ 2 * Ind0(ct_input)
    heongpu::Ciphertext<Scheme> result(context);
    poly_eval.sub(sign_plus, sign_minus, result);
    return result;
}

/**
 * @brief SumR: fold all N rows into every row by cyclic tree-fold.
 *
 * Shifts: +N, +2N, +4N, … (left-rotations by row multiples).
 * After logN steps, position [k,j] of the result equals the sum of column j
 * across ALL N input rows — this is true for every row k, not just row 0.
 * No depth consumed.
 */
heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& galois_key,
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
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose) std::cout << "\n";
    return result;
}

/**
 * @brief SumC: fold all N columns into column 0 by cyclic tree-fold.
 *
 * Shifts: +1, +2, +4, …, +N/2 (left-rotations).
 * After logN steps, position [k,0] holds the sum of row k across all N columns.
 * Columns 1..N-1 contain garbage after the fold; read only column 0.
 * No depth consumed.
 */
heongpu::Ciphertext<Scheme>
sumColumns(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
           heongpu::Galoiskey<Scheme>& galois_key,
           CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> result = ct_matrix;
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    if (g_verbose) std::cout << "  SumC shifts: ";
    for (int i = 0; i < log_n; i++)
    {
        int shift = 1 << i;
        if (g_verbose) std::cout << "+" << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose) std::cout << "\n";
    return result;
}

// ---------------------------------------------------------------------------
// Phase 1: computeRankMatrix
// ---------------------------------------------------------------------------
/**
 * @brief Compute the N×N rank matrix R.
 *
 * Output: R[k,j] = rank_0(v[j]) + 0.5  for every row k.
 *   rank_0(v[j]) = #{i : v[i] < v[j]}  (0-based, 0 for minimum, N-1 for maximum)
 *
 * How: compare(VR,VC) gives ≈ 1 if v[j]>v[k], ≈ 0 if v[j]<v[k], ≈ 0.5 if equal.
 * Summing over k: sum_k compare(v[j],v[k]) = rank_0(v[j]) + 0.5.
 * After SumR the result is the same in EVERY row (cyclic fold).
 *
 * Depth budget:
 *   depth 0  → fresh
 *   depth 1  → TransR (MaskC multiply_plain + rescale)
 *   depth 1  → mod_drop VR to match VC
 *   depth 10 → sign degree-255 (9 levels, BSGS)
 *   depth 10 → add_plain +1, SumR (free)
 *   depth 11 → multiply_plain 0.5 + rescale
 *   Total: 11 levels consumed.
 */
static heongpu::HEContext<Scheme>* g_dbg_ctx = nullptr;
static void dbg_decrypt(const std::string& label,
                        heongpu::Ciphertext<Scheme>& ct,
                        heongpu::HEDecryptor<Scheme>& dec,
                        heongpu::HEEncoder<Scheme>& enc, int N)
{
    heongpu::Plaintext<Scheme> pt(*g_dbg_ctx);
    dec.decrypt(pt, ct);
    std::vector<double> vals;
    enc.decode(vals, pt);
    std::cout << "[DBG] " << label << " (first " << N*N << " slots):\n";
    for (int k = 0; k < N; k++) {
        std::cout << "  row " << k << ": ";
        for (int j = 0; j < N; j++)
            std::cout << std::setw(8) << std::setprecision(3) << std::fixed
                      << vals[k*N+j] << " ";
        std::cout << "\n";
    }
    std::cout << std::flush;
}

heongpu::Ciphertext<Scheme>
computeRankMatrix(const heongpu::Ciphertext<Scheme>& ct_vector, int vec_len,
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
        std::cout << "\n=== computeRankMatrix (N=" << vec_len << ") ===\n"
                  << "Step 1: ReplR(V)...\n" << std::flush;

    heongpu::Ciphertext<Scheme> ct_vr =
        replicateRow(ct_vector, vec_len, row_galois_key, evaluator);

    if (g_verbose) std::cout << "Step 2: TransR(V) + ReplC(V)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_col_t =
        transposeRowToColumn(ct_vector, vec_len, transpose_galois_key,
                             evaluator, encoder, context, scale);
    heongpu::Ciphertext<Scheme> ct_vc =
        replicateColumn(ct_col_t, vec_len, col_galois_key, evaluator);

    // TransR consumed 1 level; align VR down to match VC
    while (ct_vr.level() > ct_vc.level())
    {
        heongpu::Ciphertext<Scheme> tmp(context);
        evaluator.mod_drop(ct_vr, tmp);
        ct_vr = std::move(tmp);
    }

    // compare(VR, VC) = 0.5*(sign(VR-VC) + 1): use sign then shift
    if (g_verbose) std::cout << "Step 3: diff = VR - VC...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_vr, ct_vc, ct_diff);

    if (g_verbose) std::cout << "Step 4: sign(diff)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_sign =
        chebyshev_sign_approx(ct_diff, evaluator, relin_key, scale, 255);

    // shift sign ∈ [-1,1] → compare ∈ [0,1]: add 1 (free), then ×0.5
    if (g_verbose) std::cout << "Step 5: add 1 (sign → compare range), SumR...\n" << std::flush;
    evaluator.add_plain_inplace(ct_sign, 1.0);

    heongpu::Ciphertext<Scheme> ct_r =
        sumRows(ct_sign, vec_len, sumr_galois_key, evaluator);
    // ct_r[k,j] = sum_i (sign(v[j]-v[i])+1) = rank_0(v[j])*2 + 1  in all rows

    if (g_verbose) std::cout << "Step 6: ×0.5 + rescale → R[j] = rank_0(j)+0.5...\n" << std::flush;
    evaluator.multiply_plain_inplace(ct_r, 0.5, scale);
    evaluator.rescale_inplace(ct_r); // depth +1

    if (g_verbose)
        std::cout << "computeRankMatrix done. R[k,j] = rank_0(v[j])+0.5 in all rows."
                  << " level=" << ct_r.level() << "\n" << std::flush;
    return ct_r;
}

// ---------------------------------------------------------------------------
// Phase 2+3: orderStatistics (= homomorphicSort)
// ---------------------------------------------------------------------------
/**
 * @brief Compute all N order statistics of an encrypted vector.
 *
 * Preconditions:
 *   - ct_vector: normalized to [0,1], N distinct values.
 *   - N = vec_len, power of 2, N² ≤ 32768 (N ≤ 128).
 *
 * Algorithm (directly follows OpenFHE reference sort()):
 *   1. R = computeRankMatrix(V)      // depth 11; R[k,j]=rank_0(j)+0.5 in ALL rows
 *   2. subMask[k,j] = -(k+0.5)       // encrypted, mod-dropped to R's level
 *   3. ct_diff = R + subMask          // ct_diff[k,j] = rank_0(j)-k  (integer)
 *   4. M = Ind0(ct_diff)              // M[k,j] ≈ 1 iff rank_0(j)=k  (one-hot)
 *   5. VR = ReplR(V), mod_drop        // fresh VR at level of M
 *   6. S = SumC(M · VR)               // S[k,col0] = (k+1)-th order statistic
 *
 * No MaskR or ReplR of the rank matrix: SumR already fills all rows identically,
 * so step 1's output is ready for step 2 without any masking or replication.
 * This matches the OpenFHE reference implementation.
 *
 * Depth budget:
 *   depth 11 → computeRankMatrix output
 *   depth 11 → add subMask (ct+ct, free)
 *   depth 12 → indicator normalize rescale (+1)
 *   depth 21 → indicator sign×2 (+9 levels)
 *   depth 21 → mod_drop VR (free)
 *   depth 22 → multiply(M, VR) + relin + rescale (+1)
 *   depth 22 → SumC (free)
 *   Total: 22 ≤ 28 ✓
 *
 * Output: column format — slot k*N holds the (k+1)-th order statistic.
 * Decoded output must be divided by 2 (indicator returns 2·Ind0).
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
                heongpu::HEDecryptor<Scheme>& decryptor,
                heongpu::HEContext<Scheme>& context,
                double scale)
{
    size_t total_slots = context->get_poly_modulus_degree() / 2;

    // ── Phase 1: R = RankMatrix(V) ───────────────────────────────────────────
    if (g_verbose) std::cout << "\n=== homomorphicSort (N=" << vec_len << ") ===\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_rank =
        computeRankMatrix(ct_vector, vec_len,
                          row_galois_key, col_galois_key,
                          transpose_galois_key, sumr_galois_key,
                          relin_key, evaluator, encoder, context, scale);
    // ct_rank[k,j] = rank_0(v[j]) + 0.5  in EVERY row k (no masking needed)
    if (g_verbose) dbg_decrypt("ct_rank (all rows)", ct_rank, decryptor, encoder, vec_len);

    // ── Phase 2: One-hot mask M = Ind0(R − subMask) ──────────────────────────
    // subMask[k,j] = -(k+0.5) so that R[k,j] + subMask[k,j] = rank_0(j) - k.
    // M[k,j] ≈ 1 iff rank_0(v[j]) = k, i.e. v[j] is the (k+1)-th smallest.
    if (g_verbose) std::cout << "\nPhase 2: build & subtract rank-target matrix...\n" << std::flush;

    std::vector<double> sub_mask_vals(total_slots, 0.0);
    for (int k = 0; k < vec_len; k++)
        for (int j = 0; j < vec_len; j++)
            sub_mask_vals[k * vec_len + j] = -(k + 0.5);

    heongpu::Plaintext<Scheme> pt_sub(context);
    encoder.encode(pt_sub, sub_mask_vals, scale);
    heongpu::Ciphertext<Scheme> ct_sub(context);
    encryptor.encrypt(ct_sub, pt_sub);
    // mod_drop to match ct_rank's level
    while (ct_sub.level() > ct_rank.level())
    {
        heongpu::Ciphertext<Scheme> tmp(context);
        evaluator.mod_drop(ct_sub, tmp);
        ct_sub = std::move(tmp);
    }
    if (g_verbose)
        std::cout << "  ct_rank level=" << ct_rank.level()
                  << "  ct_sub level=" << ct_sub.level() << "\n" << std::flush;

    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.add(ct_rank, ct_sub, ct_diff);
    // ct_diff[k,j] = rank_0(v[j]) - k  (integer ∈ {-(N-1), …, 0, …, N-1})

    if (g_verbose) dbg_decrypt("ct_diff (rank - target)", ct_diff, decryptor, encoder, vec_len);
    if (g_verbose) std::cout << "Phase 2: indicator (composite sign, degree=255)...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_mask =
        chebyshev_indicator_approx(ct_diff, vec_len, evaluator, relin_key, scale, context);
    // ct_mask[k,j] ≈ 2 iff rank_0(v[j])=k, else ≈ 0
    if (g_verbose) dbg_decrypt("ct_mask (one-hot indicator)", ct_mask, decryptor, encoder, vec_len);

    // ── Phase 3: Reconstruct sorted values ───────────────────────────────────
    if (g_verbose) std::cout << "\nPhase 3: ReplR(V) + mod_drop to indicator level...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_vr =
        replicateRow(ct_vector, vec_len, row_galois_key, evaluator);
    while (ct_vr.level() > ct_mask.level())
    {
        heongpu::Ciphertext<Scheme> tmp(context);
        evaluator.mod_drop(ct_vr, tmp);
        ct_vr = std::move(tmp);
    }
    if (g_verbose)
        std::cout << "  ct_mask level=" << ct_mask.level()
                  << "  ct_vr level=" << ct_vr.level() << "\n" << std::flush;
    if (g_verbose) dbg_decrypt("ct_vr", ct_vr, decryptor, encoder, vec_len);

    if (g_verbose) std::cout << "Phase 3: multiply(M, VR) + relin + rescale...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_product(context);
    evaluator.multiply(ct_mask, ct_vr, ct_product);
    evaluator.relinearize_inplace(ct_product, relin_key);
    evaluator.rescale_inplace(ct_product);

    if (g_verbose) std::cout << "Phase 3: SumC...\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_sorted =
        sumColumns(ct_product, vec_len, sumc_galois_key, evaluator);
    // ct_sorted[k, col0] = (k+1)-th order statistic × 2
    // (the ×2 comes from the indicator returning 2·Ind0)

    if (g_verbose)
        std::cout << "Sort complete. depth=" << ct_sorted.level()
                  << ". Slot k*N = (k+1)-th statistic × 2.\n" << std::flush;
    return ct_sorted;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int vec_len    = 8;
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

    // n = 65536 → 32768 slots; 31 computation levels available.
    // Algorithm consumes 22 levels; 9 spare.
    const size_t poly_modulus_degree = 65536;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes(
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
             40, 40, 40}, // 32 Q primes (31 computation levels)
        {60, 60, 60, 60, 60, 60, 60});                               // 7 P primes

    double scale = std::pow(2.0, 40);

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    context->generate();
    g_dbg_ctx = &context;
    float ctx_ms = ctx_timer.stopTimer();

    const int available_slots = static_cast<int>(poly_modulus_degree / 2);
    if (vec_len * vec_len > available_slots)
    {
        std::cerr << "Error: N=" << vec_len << " needs " << (vec_len * vec_len)
                  << " slots but only " << available_slots << " available.\n";
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

    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));

    std::vector<int> row_shifts;    // ReplR
    for (int i = vec_len / 2; i > 0; i /= 2)
        row_shifts.push_back(-(i * vec_len));

    std::vector<int> col_shifts;    // ReplC
    for (int i = 1; i < vec_len; i *= 2)
        col_shifts.push_back(-i);

    std::vector<int> sumr_shifts;   // SumR
    for (int i = 0; i < log_n; i++)
        sumr_shifts.push_back(vec_len * (1 << i));

    std::vector<int> transr_shifts = transposeGaloisShifts(vec_len);
    std::vector<int> sumc_shifts   = sumcGaloisShifts(vec_len);

    if (g_verbose)
        std::cout << "Galois shifts — row: " << row_shifts.size()
                  << "  col: " << col_shifts.size()
                  << "  sumr: " << sumr_shifts.size()
                  << "  transr: " << transr_shifts.size()
                  << "  sumc: " << sumc_shifts.size() << "\n";

    // ── Key generation ───────────────────────────────────────────────────────
    const size_t kMiB = 1024ULL * 1024ULL;
    size_t gpu_baseline =
        heongpu::MemoryPool::instance().get_current_device_pool_memory_usage();

    GPUTimer keygen_timer;
    keygen_timer.startTimer();

    heongpu::Galoiskey<Scheme> row_galois_key(context, row_shifts);
    keygen.generate_galois_key(row_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> col_galois_key(context, col_shifts);
    keygen.generate_galois_key(col_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> sumr_galois_key(context, sumr_shifts);
    keygen.generate_galois_key(sumr_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> transr_galois_key(context, transr_shifts);
    keygen.generate_galois_key(transr_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> sumc_galois_key(context, sumc_shifts);
    keygen.generate_galois_key(sumc_galois_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    float keygen_ms = keygen_timer.stopTimer();
    size_t gpu_keys_mib = getGPUUsedMiB() - gpu_baseline / kMiB;
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms << " ms  (keys: "
                  << gpu_keys_mib << " MiB VRAM)\n";

    // ── Input preparation ────────────────────────────────────────────────────
    std::vector<double> input(vec_len);
    if (bench_mode)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 100.0);
        for (int i = 0; i < vec_len; i++)
            input[i] = dist(rng);
    }
    else
    {
        for (int i = 0; i < vec_len; i++)
            input[i] = static_cast<double>(vec_len - 1 - i);
    }

    std::vector<double> input_sorted = input;
    std::sort(input_sorted.begin(), input_sorted.end());

    std::vector<double> normalized = normalizeForRanking(input);

    if (g_verbose)
    {
        std::cout << "Original input:          ";
        display_vector(input, vec_len);
        std::cout << "Normalized [0,1]:        ";
        display_vector(normalized, vec_len);
        std::cout << "Expected sorted (orig):  ";
        display_vector(input_sorted, vec_len);
    }

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
                        evaluator, encoder, encryptor, decryptor, context, scale);

    float sort_ms = sort_timer.stopTimer();
    size_t gpu_sort_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // ── Decrypt and decode ───────────────────────────────────────────────────
    heongpu::Plaintext<Scheme> sort_plaintext(context);
    decryptor.decrypt(sort_plaintext, ct_sorted);
    std::vector<double> sort_result;
    encoder.decode(sort_result, sort_plaintext);

    // Extract column format: sorted[k] = slot k*N.
    // Divide by 2: indicator returns 2·Ind0.
    std::vector<double> he_sorted(vec_len);
    for (int k = 0; k < vec_len; k++)
        he_sorted[k] = sort_result[k * vec_len] / 2.0;

    // Denormalize
    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;
    std::vector<double> he_sorted_denorm(vec_len);
    for (int k = 0; k < vec_len; k++)
        he_sorted_denorm[k] = he_sorted[k] * range + lo;

    // ── Output ───────────────────────────────────────────────────────────────
    if (bench_mode)
    {
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
        std::cout << "HE sorted (normalized):     ";
        display_vector(he_sorted, vec_len);
        std::cout << "HE sorted (original scale): ";
        display_vector(he_sorted_denorm, vec_len);

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
                                  : "  Values: some incorrect\n");

        std::cout << "\nTiming:\n";
        std::cout << "  Key generation : " << keygen_ms << " ms\n";
        std::cout << "  Sort           : " << sort_ms << " ms  ("
                  << (sort_ms / 1000.0) << " s)\n";
        std::cout << "\nVRAM (above context baseline):\n";
        std::cout << "  Keys : " << gpu_keys_mib << " MiB\n";
        std::cout << "  Sort : " << gpu_sort_mib << " MiB\n";
        std::cout << "  Peak : " << gpu_peak_mib << " MiB\n";
    }

    return EXIT_SUCCESS;
}
