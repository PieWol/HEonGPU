/**
 * @file 21_ckks_ranking_multi.cpp
 *
 * Multi-ciphertext homomorphic ranking for large N, following:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone et al., USENIX Security 2025  (Algorithm 7, complOpt=true)
 *
 * Both modes use block size L=128.  M = N/L block ciphertexts.
 *
 * Comparison method (paper §6.1):
 *   M ≤ 2 (N ≤ 256): Chebyshev degree 2047
 *   M > 2 (N > 256):  f,g composition (dg=3, df=2)
 *
 * Ring dimension and dnum are auto-selected via selectMultiCTParams()
 * to minimise key-switching noise (mimicking OpenFHE's ringDim=0).
 * TC modes target dnum=1 by using a larger ring dimension.
 *
 * Parameter table (128-bit security):
 *
 *   Basic + Cheby 2047:  n=32768   Q={60,45×14}  P={60×3}   dnum=5
 *   Basic + f,g:         n=65536   Q={60,45×22}  P={60×11}  dnum=3
 *   TC + Cheby 2047:     n=65536   Q={60,45×15}  P={60×17}  dnum=1
 *   TC + f,g:            n=131072  Q={60,45×24}  P={60×39}  dnum=1
 *
 * Tie-correction algorithm (same as 23_ckks_ranking_tie_correction.cpp):
 *   E = 1 − sign²  (equality indicator)
 *   Diagonal blocks:  E × within-block adjusted mask (±0.5)
 *   Cross-block pairs: E × uniform +0.5 (complement gets negated)
 *   TC offset = sumR(accumulated masked E) − 0.5
 *
 * Usage: 21_ckks_ranking_multi [N] [--tie-correction] [--bench]
 */

#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

static bool g_verbose = true;
constexpr auto Scheme = heongpu::Scheme::CKKS;

// ---------------------------------------------------------------------------
// CKKS parameter selection — mimics OpenFHE's ringDim=0 auto-selection.
// Picks the smallest ring dimension that fits the required depth while
// maximising P primes to minimise dnum (and thus key-switching noise).
// TC modes target dnum=1 for accuracy at deep comparison circuits.
// ---------------------------------------------------------------------------
struct CKKSParams {
    size_t poly_modulus_degree;
    std::vector<int> q_bits;
    std::vector<int> p_bits;
    int scale_bits;
    int dnum;
};

static CKKSParams selectMultiCTParams(bool tie_correction, bool use_fg)
{
    auto make_q = [](int first, int rest_val, int rest_count) {
        std::vector<int> q = {first};
        for (int i = 0; i < rest_count; i++) q.push_back(rest_val);
        return q;
    };

    if (tie_correction && use_fg)
    {
        // TC + f,g: depth 23, n=131072 (budget 3500) → dnum=1
        // Q={60,45×24}=1140, P={60×39}=2340, total=3480 ≤ 3500
        return {131072, make_q(60, 45, 24), std::vector<int>(39, 60), 45, 1};
    }
    if (tie_correction)
    {
        // TC + Cheby 2047: depth 14, n=65536 (budget 1761) → dnum=1
        // Q={60,45×15}=735, P={60×17}=1020, total=1755 ≤ 1761
        return {65536, make_q(60, 45, 15), std::vector<int>(17, 60), 45, 1};
    }
    if (use_fg)
    {
        // Basic + f,g: depth 22, n=65536 (budget 1761) → dnum=3
        // Q={60,45×22}=1050, P={60×11}=660, total=1710 ≤ 1761
        return {65536, make_q(60, 45, 22), std::vector<int>(11, 60), 45, 3};
    }
    // Basic + Cheby 2047: depth 13, n=32768 (budget 881) → dnum=5
    // Q={60,45×14}=690, P={60×3}=180, total=870 ≤ 881
    return {32768, make_q(60, 45, 14), {60, 60, 60}, 45, 5};
}

// ---------------------------------------------------------------------------
// CKKSPolyEvaluator — exposes protected evaluate_poly for BSGS Chebyshev
// ---------------------------------------------------------------------------
class CKKSPolyEvaluator : public heongpu::HEArithmeticOperator<Scheme>
{
  public:
    CKKSPolyEvaluator(heongpu::HEContext<Scheme> ctx,
                      heongpu::HEEncoder<Scheme>& enc)
        : heongpu::HEArithmeticOperator<Scheme>(ctx, enc)
    {}

    heongpu::Ciphertext<Scheme>
    eval_chebyshev(heongpu::Ciphertext<Scheme>& ct, double target_scale,
                   const std::vector<Complex64>& coeffs, int degree,
                   heongpu::Relinkey<Scheme>& rk,
                   bool lead = false,
                   double a = -1.0, double b = 1.0)
    {
        Polynomial poly(degree, coeffs, lead,
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
// Input normalization
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
std::vector<int> rowGaloisShifts(int L)    // ReplR: -(L/2)*L, ..., -L
{
    std::vector<int> s;
    for (int i = L / 2; i > 0; i /= 2) s.push_back(-(i * L));
    return s;
}
std::vector<int> colGaloisShifts(int L)    // ReplC: -1, -2, ..., -(L/2)
{
    std::vector<int> s;
    for (int i = 1; i < L; i *= 2) s.push_back(-i);
    return s;
}
std::vector<int> sumrGaloisShifts(int L)   // SumR: +L, +2L, ..., +L*(L/2)
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 0; i < logL; i++) s.push_back(L * (1 << i));
    return s;
}
std::vector<int> transrGaloisShifts(int L) // TransR: -(L*(L-1)/2^i) i=1..logL
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 1; i <= logL; i++)
        s.push_back(-((L * (L - 1)) / (1 << i)));
    return s;
}
std::vector<int> sumcGaloisShifts(int L)   // SumC: +1, +2, ..., +L/2
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 0; i < logL; i++) s.push_back(1 << i);
    return s;
}
std::vector<int> transpcGaloisShifts(int L) // TransposeC: +(L*(L-1)/2^i) i=1..logL
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 1; i <= logL; i++)
        s.push_back((L * (L - 1)) / (1 << i));
    return s;
}

// ---------------------------------------------------------------------------
// Matrix primitives
// ---------------------------------------------------------------------------

// ReplR: replicate row 0 to all L rows. Shifts: -(L/2)*L, ..., -L. No depth.
static heongpu::Ciphertext<Scheme>
replicateRow(const heongpu::Ciphertext<Scheme>& row, int L,
             heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe,
             heongpu::HEContext<Scheme>& ctx)
{
    heongpu::Ciphertext<Scheme> r = row;
    for (int i = L / 2; i > 0; i /= 2)
    {
        heongpu::Ciphertext<Scheme> rot(ctx);
        pe.rotate_rows(r, rot, gk, -(i * L));
        pe.add_inplace(r, rot);
    }
    return r;
}

// ReplC: replicate column 0 to all L columns. Shifts: -1,-2,...,-(L/2). No depth.
static heongpu::Ciphertext<Scheme>
replicateColumn(const heongpu::Ciphertext<Scheme>& col, int L,
                heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe,
                heongpu::HEContext<Scheme>& ctx)
{
    heongpu::Ciphertext<Scheme> r = col;
    for (int i = 1; i < L; i *= 2)
    {
        heongpu::Ciphertext<Scheme> rot(ctx);
        pe.rotate_rows(r, rot, gk, -i);
        pe.add_inplace(r, rot);
    }
    return r;
}

// TransR: transpose row 0 to column 0. Depth: 1 (MaskC multiply_plain + rescale).
static heongpu::Ciphertext<Scheme>
transposeRowToColumn(const heongpu::Ciphertext<Scheme>& row, int L,
                     heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe,
                     heongpu::HEEncoder<Scheme>& enc,
                     heongpu::HEContext<Scheme>& ctx, double scale)
{
    heongpu::Ciphertext<Scheme> r = row;
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    for (int i = 1; i <= logL; i++)
    {
        int shift = -((L * (L - 1)) / (1 << i));
        heongpu::Ciphertext<Scheme> rot(ctx);
        pe.rotate_rows(r, rot, gk, shift);
        pe.add_inplace(r, rot);
    }
    // MaskC: keep column 0 (positions k*L for k=0..L-1)
    size_t slots = ctx->get_poly_modulus_degree() / 2;
    std::vector<double> mask(slots, 0.0);
    for (int k = 0; k < L; k++) mask[k * L] = 1.0;
    heongpu::Plaintext<Scheme> pt(ctx);
    enc.encode(pt, mask, scale);
    pe.multiply_plain_inplace(r, pt);
    pe.rescale_inplace(r);
    return r;
}

// SumR: fold all L rows into row 0. Shifts: +L, +2L, ..., +L*(L/2). No depth.
static heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& m, int L,
        heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe,
        heongpu::HEContext<Scheme>& ctx)
{
    heongpu::Ciphertext<Scheme> r = m;
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    for (int i = 0; i < logL; i++)
    {
        heongpu::Ciphertext<Scheme> rot(ctx);
        pe.rotate_rows(r, rot, gk, L * (1 << i));
        pe.add_inplace(r, rot);
    }
    return r;
}

// SumC: fold all L columns into column 0. Shifts: +1, +2, ..., +L/2. No depth.
static heongpu::Ciphertext<Scheme>
sumColumns(const heongpu::Ciphertext<Scheme>& m, int L,
           heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe,
           heongpu::HEContext<Scheme>& ctx)
{
    heongpu::Ciphertext<Scheme> r = m;
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    for (int i = 0; i < logL; i++)
    {
        heongpu::Ciphertext<Scheme> rot(ctx);
        pe.rotate_rows(r, rot, gk, 1 << i);
        pe.add_inplace(r, rot);
    }
    return r;
}

// MaskColumn0: zero out everything except column 0 (positions k*L for k=0..L-1).
// Depth: 1 (multiply_plain + rescale). Needed between sumColumns and transposeColumn.
static heongpu::Ciphertext<Scheme>
maskColumn0(heongpu::Ciphertext<Scheme>& ct, int L,
            CKKSPolyEvaluator& pe, heongpu::HEEncoder<Scheme>& enc,
            heongpu::HEContext<Scheme>& ctx, double scale)
{
    size_t slots = ctx->get_poly_modulus_degree() / 2;
    std::vector<double> mask(slots, 0.0);
    for (int k = 0; k < L; k++) mask[k * L] = 1.0;

    heongpu::Plaintext<Scheme> pt(ctx);
    enc.encode(pt, mask, scale);

    int target_depth = ct.depth();
    while (pt.depth() < target_depth)
        pe.mod_drop_inplace(pt);

    heongpu::Ciphertext<Scheme> out = ct;
    pe.multiply_plain_inplace(out, pt);
    pe.rescale_inplace(out);
    return out;
}

// TransposeC: transpose column 0 to row 0. Shifts: +(L*(L-1)/2^i). No depth.
// Row 0 holds the transposed result; other rows contain partial (garbage) sums.
static heongpu::Ciphertext<Scheme>
transposeColumn(const heongpu::Ciphertext<Scheme>& col, int L,
                heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe,
                heongpu::HEContext<Scheme>& ctx)
{
    heongpu::Ciphertext<Scheme> r = col;
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    for (int i = 1; i <= logL; i++)
    {
        int shift = (L * (L - 1)) / (1 << i);
        heongpu::Ciphertext<Scheme> rot(ctx);
        pe.rotate_rows(r, rot, gk, shift);
        pe.add_inplace(r, rot);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Chebyshev sign approximation (raw, no normalize)
// Input: ct with values in [-1, 1]
// Output: ≈+1 where ct > 0,  ≈-1 where ct < 0
// Normalization to [0,1] is deferred to after decryption to save 1 level
// for maskColumn0 in Phase 3.
// ---------------------------------------------------------------------------
static heongpu::Ciphertext<Scheme>
compareUnit(heongpu::Ciphertext<Scheme>& ct_diff,
            CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
            double scale, int degree = 2047)
{
    auto sign_fn = [](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(re > 0.0 ? 1.0 : (re < 0.0 ? -1.0 : 0.0), 0.0);
    };
    std::vector<Complex64> coeffs =
        heongpu::approximate_function(sign_fn, -1.0, 1.0, degree);

    return pe.eval_chebyshev(ct_diff, scale, coeffs, degree, rk);
}

// ---------------------------------------------------------------------------
// Fractional rank computation (for verification)
// ---------------------------------------------------------------------------
static std::vector<double> computeFractionalRanks(const std::vector<double>& input)
{
    int n = static_cast<int>(input.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return input[a] < input[b]; });

    std::vector<double> ranks(n);
    int i = 0;
    while (i < n)
    {
        int j = i;
        while (j < n && input[idx[j]] == input[idx[i]]) j++;
        double mean_rank = 0.0;
        for (int k = i; k < j; k++) mean_rank += (k + 1);
        mean_rank /= (j - i);
        for (int k = i; k < j; k++) ranks[idx[k]] = mean_rank;
        i = j;
    }
    return ranks;
}

static std::vector<double> computeOrdinalRanks(const std::vector<double>& input)
{
    int n = static_cast<int>(input.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) {
                  if (input[a] != input[b]) return input[a] < input[b];
                  return a < b;
              });
    std::vector<double> ranks(n);
    for (int k = 0; k < n; k++) ranks[idx[k]] = k + 1;
    return ranks;
}

// ---------------------------------------------------------------------------
// f,g sign composition (from Cheon et al., used for N > 256 per paper §6.1)
// g(x) = (4589x − 16577x³ + 25614x⁵ − 12860x⁷)/1024
// f(x) = (35x − 35x³ + 21x⁵ − 5x⁷)/16
// sign(x) ≈ f^df(g^dg(x)), output ∈ [-1, +1]
// ---------------------------------------------------------------------------
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
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk, /*lead=*/true);
}

static heongpu::Ciphertext<Scheme>
applyF3(heongpu::Ciphertext<Scheme>& ct, CKKSPolyEvaluator& pe,
        heongpu::Relinkey<Scheme>& rk, double scale)
{
    auto fn = [](Complex64 x) -> Complex64 {
        double t = x.real(), t2 = t * t;
        return {t * (35.0 + t2 * (-35.0 + t2 * (21.0 - 5.0 * t2))) / 16.0, 0.0};
    };
    auto coeffs = heongpu::approximate_function(fn, -1.0, 1.0, 7);
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk, /*lead=*/true);
}

static heongpu::Ciphertext<Scheme>
compareFG(heongpu::Ciphertext<Scheme>& ct_diff, int dg, int df,
          CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
          double scale)
{
    heongpu::Ciphertext<Scheme> ct = ct_diff;
    for (int i = 0; i < dg; i++)
        ct = applyG3(ct, pe, rk, scale);
    for (int i = 0; i < df; i++)
        ct = applyF3(ct, pe, rk, scale);
    return ct;
}

// ---------------------------------------------------------------------------
// Multi-ciphertext ranking (Algorithm 7 from paper, complementary opt.)
// With optional tie correction (Algorithm 6).
// ---------------------------------------------------------------------------
struct MultiCTResult {
    std::vector<heongpu::Ciphertext<Scheme>> ranks;
    std::vector<heongpu::Ciphertext<Scheme>> tc_offsets; // empty if !tie_correction
};

static MultiCTResult
multiCiphertextRank(
    const std::vector<heongpu::Ciphertext<Scheme>>& blocks,
    int L, int cheby_degree, bool tie_correction,
    heongpu::Galoiskey<Scheme>& row_key,
    heongpu::Galoiskey<Scheme>& col_key,
    heongpu::Galoiskey<Scheme>& transr_key,
    heongpu::Galoiskey<Scheme>& sumr_key,
    heongpu::Galoiskey<Scheme>& sumc_key,
    heongpu::Galoiskey<Scheme>& transpc_key,
    heongpu::Relinkey<Scheme>& rk,
    CKKSPolyEvaluator& pe,
    heongpu::HEEncoder<Scheme>& enc,
    heongpu::HEContext<Scheme>& ctx,
    double scale,
    bool use_fg = false, int fg_dg = 3, int fg_df = 2)
{
    const int M = static_cast<int>(blocks.size());

    // ── Pre-encode TC masks (once, reused for every block pair) ─────────────
    size_t slots = ctx->get_poly_modulus_degree() / 2;
    heongpu::Plaintext<Scheme> diag_mask_pt(ctx);
    heongpu::Plaintext<Scheme> cross_mask_pt(ctx);

    if (tie_correction)
    {
        // Diagonal (j==k): within-block adjusted mask
        std::vector<double> diag_mask(slots, 0.0);
        for (int r = 0; r < L; r++)
            for (int c = 0; c < L; c++)
                diag_mask[r * L + c] = (c >= r) ? 0.5 : -0.5;
        enc.encode(diag_mask_pt, diag_mask, scale);

        // Cross-block (j<k): uniform +0.5 (all cols have higher global index)
        std::vector<double> cross_mask(slots, 0.0);
        for (int r = 0; r < L; r++)
            for (int c = 0; c < L; c++)
                cross_mask[r * L + c] = 0.5;
        enc.encode(cross_mask_pt, cross_mask, scale);
    }

    // ── Phase 1: replicate ──────────────────────────────────────────────────
    if (g_verbose) std::cout << "\n=== Phase 1: Replicate ===\n";

    std::vector<heongpu::Ciphertext<Scheme>> replR(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<heongpu::Ciphertext<Scheme>> replC(M, heongpu::Ciphertext<Scheme>(ctx));

    for (int j = 0; j < M; j++)
    {
        if (g_verbose) std::cout << "  Block " << j << ": ReplR + TransR+ReplC\n";
        replR[j] = replicateRow(blocks[j], L, row_key, pe, ctx);
        heongpu::Ciphertext<Scheme> col_t =
            transposeRowToColumn(blocks[j], L, transr_key, pe, enc, ctx, scale);
        replC[j] = replicateColumn(col_t, L, col_key, pe, ctx);
    }

    // ── Phase 2: compare upper triangle ────────────────────────────────────
    if (g_verbose) std::cout << "\n=== Phase 2: Compare (" << (M*(M+1)/2) << " pairs) ===\n";

    std::vector<heongpu::Ciphertext<Scheme>> Cv(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<heongpu::Ciphertext<Scheme>> Ch(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<bool> Cv_init(M, false), Ch_init(M, false);

    // TC accumulators: masked equality indicators (vertical only)
    std::vector<heongpu::Ciphertext<Scheme>> Ev(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<bool> Ev_init(M, false);

    for (int j = 0; j < M; j++)
    {
        for (int k = j; k < M; k++)
        {
            if (g_verbose) std::cout << "  Compare (" << j << "," << k << ")\n";

            heongpu::Ciphertext<Scheme> rj = replR[j];
            while (rj.level() > replC[k].level())
            {
                heongpu::Ciphertext<Scheme> tmp(ctx);
                pe.mod_drop(rj, tmp);
                rj = std::move(tmp);
            }

            heongpu::Ciphertext<Scheme> diff(ctx);
            pe.sub(rj, replC[k], diff);

            heongpu::Ciphertext<Scheme> Cjk = use_fg
                ? compareFG(diff, fg_dg, fg_df, pe, rk, scale)
                : compareUnit(diff, pe, rk, scale, cheby_degree);

            // Accumulate sign into Cv[j]
            if (!Cv_init[j]) { Cv[j] = Cjk;                Cv_init[j] = true; }
            else              { pe.add_inplace(Cv[j], Cjk);                    }

            // Complementary: Ch[k] += -Cjk
            if (j != k)
            {
                heongpu::Ciphertext<Scheme> Ckj = Cjk;
                pe.negate_inplace(Ckj);
                if (!Ch_init[k]) { Ch[k] = Ckj;                Ch_init[k] = true; }
                else             { pe.add_inplace(Ch[k], Ckj);                    }
            }

            // ── Tie-correction: E = 1 − sign², then apply mask ────────────
            if (tie_correction)
            {
                if (g_verbose) std::cout << "    TC: sign² + mask·E\n";

                // sign²
                heongpu::Ciphertext<Scheme> Cjk_copy = Cjk;
                heongpu::Ciphertext<Scheme> sign_sq(ctx);
                pe.multiply(Cjk, Cjk_copy, sign_sq);
                pe.relinearize_inplace(sign_sq, rk);
                pe.rescale_inplace(sign_sq);

                // E = 1 − sign²
                pe.negate_inplace(sign_sq);
                pe.add_plain_inplace(sign_sq, 1.0);

                // Apply position-dependent mask
                heongpu::Plaintext<Scheme> mask_pt = (j == k) ? diag_mask_pt : cross_mask_pt;
                while (mask_pt.depth() < sign_sq.depth())
                    pe.mod_drop_inplace(mask_pt);
                pe.multiply_plain_inplace(sign_sq, mask_pt);
                pe.rescale_inplace(sign_sq);

                // Accumulate into Ev[j]
                if (!Ev_init[j]) { Ev[j] = sign_sq;                  Ev_init[j] = true; }
                else             { pe.add_inplace(Ev[j], sign_sq);                       }

                // Complement for cross-block: block k gets negated contribution
                if (j != k)
                {
                    heongpu::Ciphertext<Scheme> neg_masked = sign_sq;
                    pe.negate_inplace(neg_masked);
                    if (!Ev_init[k]) { Ev[k] = neg_masked;                  Ev_init[k] = true; }
                    else             { pe.add_inplace(Ev[k], neg_masked);                       }
                }
            }
        }
    }

    // ── Phase 3: sum ────────────────────────────────────────────────────────
    if (g_verbose) std::cout << "\n=== Phase 3: Sum ===\n";

    std::vector<heongpu::Ciphertext<Scheme>> result(M, heongpu::Ciphertext<Scheme>(ctx));

    for (int j = 0; j < M; j++)
    {
        heongpu::Ciphertext<Scheme> sv = sumRows(Cv[j], L, sumr_key, pe, ctx);
        result[j] = sv;

        if (j > 0 && Ch_init[j])
        {
            if (g_verbose) std::cout << "  Block " << j << ": sumColumns + maskCol0 + transposeColumn\n";
            heongpu::Ciphertext<Scheme> sh = sumColumns(Ch[j], L, sumc_key, pe, ctx);
            sh = maskColumn0(sh, L, pe, enc, ctx, scale);
            sh = transposeColumn(sh, L, transpc_key, pe, ctx);

            while (result[j].level() > sh.level())
            {
                heongpu::Ciphertext<Scheme> tmp(ctx);
                pe.mod_drop(result[j], tmp);
                result[j] = std::move(tmp);
            }
            pe.add_inplace(result[j], sh);
        }
    }

    // TC Phase 3: sumR of masked-E accumulators (no maskC needed — vertical only)
    std::vector<heongpu::Ciphertext<Scheme>> tc_result;
    if (tie_correction)
    {
        tc_result.resize(M, heongpu::Ciphertext<Scheme>(ctx));
        for (int j = 0; j < M; j++)
        {
            if (Ev_init[j])
                tc_result[j] = sumRows(Ev[j], L, sumr_key, pe, ctx);
        }
    }

    return {result, tc_result};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int  N              = 256;
    bool bench_mode     = false;
    bool tie_correction = false;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--bench")
            bench_mode = true;
        else if (arg == "--tie-correction")
            tie_correction = true;
        else if (!arg.empty() && std::isdigit(static_cast<unsigned char>(arg[0])))
            N = std::stoi(arg);
    }
    g_verbose = !bench_mode;

    const int L = 128;

    if (N <= 0 || (N & (N - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << N << ")\n";
        return EXIT_FAILURE;
    }
    if (N % L != 0)
    {
        std::cerr << "Error: N=" << N << " must be a multiple of " << L << "\n";
        return EXIT_FAILURE;
    }
    if (N <= L)
    {
        std::cerr << "Error: N=" << N << " <= " << L
                  << "; use 23_ckks_ranking_tie_correction for single-ciphertext mode.\n";
        return EXIT_FAILURE;
    }
    const int M = N / L;

    cudaSetDevice(0);

    // ── HE context ──────────────────────────────────────────────────────────
    const bool use_fg = (M > 2);
    const int fg_dg = 3, fg_df = 2;
    const int cheby_degree = 2047;

    CKKSParams params = selectMultiCTParams(tie_correction, use_fg);
    double scale = std::pow(2.0, params.scale_bits);

    heongpu::HEContext<Scheme> ctx = heongpu::GenHEContext<Scheme>();
    ctx->set_poly_modulus_degree(params.poly_modulus_degree);
    ctx->set_coeff_modulus_bit_sizes(params.q_bits, params.p_bits);

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    ctx->generate();
    float ctx_ms = ctx_timer.stopTimer();

    if (g_verbose)
    {
        std::cout << "N=" << N << "  M=" << M << " ciphertexts"
                  << "  L=" << L
                  << "  mode=" << (tie_correction ? "tie-corrected" : "basic") << "\n";
        std::cout << "Compare method: "
                  << (use_fg ? "f,g (dg=3, df=2)" : "Chebyshev 2047")
                  << "  n=" << params.poly_modulus_degree
                  << "  scale=2^" << params.scale_bits
                  << "  dnum=" << params.dnum << "\n";
    }

    // ── Key generation ───────────────────────────────────────────────────────
    heongpu::HEKeyGenerator<Scheme> keygen(ctx);
    heongpu::Secretkey<Scheme>  sk(ctx);  keygen.generate_secret_key(sk);
    heongpu::Publickey<Scheme>  pk(ctx);  keygen.generate_public_key(pk, sk);

    heongpu::HEEncoder<Scheme>    enc(ctx);
    heongpu::HEEncryptor<Scheme>  encryptor(ctx, pk);
    heongpu::HEDecryptor<Scheme>  decryptor(ctx, sk);
    CKKSPolyEvaluator             pe(ctx, enc);

    const size_t kMiB = 1024ULL * 1024ULL;
    size_t gpu_baseline =
        heongpu::MemoryPool::instance().get_current_device_pool_memory_usage();

    GPUTimer keygen_timer;
    keygen_timer.startTimer();

    auto rshifts    = rowGaloisShifts(L);
    auto cshifts    = colGaloisShifts(L);
    auto sshifts    = sumrGaloisShifts(L);
    auto tshifts    = transrGaloisShifts(L);
    auto scshifts   = sumcGaloisShifts(L);
    auto tpcshifts  = transpcGaloisShifts(L);

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

    heongpu::Galoiskey<Scheme> transpc_key(ctx, tpcshifts);
    keygen.generate_galois_key(transpc_key, sk);

    heongpu::Relinkey<Scheme> rk(ctx);
    keygen.generate_relin_key(rk, sk);

    float keygen_ms = keygen_timer.stopTimer();
    size_t gpu_keys_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms << " ms  ("
                  << gpu_keys_mib << " MiB VRAM)\n";

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
        for (int i = 0; i < N; i++) input[i] = static_cast<double>(i);
    }

    std::vector<double> norm = normalizeForRanking(input);

    if (g_verbose)
    {
        std::cout << "Input (first 8): ";
        display_vector(input, std::min(N, 8));
    }

    const int slots = static_cast<int>(ctx->get_poly_modulus_degree() / 2);
    std::vector<heongpu::Ciphertext<Scheme>> blocks;
    blocks.reserve(M);

    for (int j = 0; j < M; j++)
    {
        std::vector<double> buf(slots, 0.0);
        for (int i = 0; i < L; i++)
            buf[i] = norm[j * L + i];

        heongpu::Plaintext<Scheme> pt(ctx);
        enc.encode(pt, buf, scale);
        heongpu::Ciphertext<Scheme> ct(ctx);
        encryptor.encrypt(ct, pt);
        blocks.push_back(std::move(ct));
    }

    // ── Ranking (timed) ──────────────────────────────────────────────────────
    GPUTimer rank_timer;
    rank_timer.startTimer();

    MultiCTResult rank_result =
        multiCiphertextRank(blocks, L, cheby_degree, tie_correction,
                            row_key, col_key, transr_key, sumr_key,
                            sumc_key, transpc_key, rk, pe, enc, ctx, scale,
                            use_fg, fg_dg, fg_df);

    float rank_ms = rank_timer.stopTimer();
    size_t gpu_rank_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // ── Decrypt & convert raw sign sums to ranks ────────────────────────────
    // raw = Σ sign(x_i − x_j) → rank = (raw + N + 1) / 2
    // TC:  rank += sumR(masked_E) − 0.5
    std::vector<double> all_ranks(N);
    for (int j = 0; j < M; j++)
    {
        heongpu::Plaintext<Scheme> pt(ctx);
        decryptor.decrypt(pt, rank_result.ranks[j]);
        std::vector<double> raw;
        enc.decode(raw, pt);

        if (tie_correction)
        {
            heongpu::Plaintext<Scheme> tc_pt(ctx);
            decryptor.decrypt(tc_pt, rank_result.tc_offsets[j]);
            std::vector<double> tc_raw;
            enc.decode(tc_raw, tc_pt);

            for (int i = 0; i < L; i++)
            {
                double basic_rank = (raw[i] + N + 1) / 2.0;
                double tc_offset  = tc_raw[i] - 0.5;
                all_ranks[j * L + i] = basic_rank + tc_offset;
            }
        }
        else
        {
            for (int i = 0; i < L; i++)
                all_ranks[j * L + i] = (raw[i] + N + 1) / 2.0;
        }
    }

    // ── Output ───────────────────────────────────────────────────────────────
    if (bench_mode)
    {
        std::cout << "BENCH:"
                  << " N="            << N
                  << " mode="         << (tie_correction ? "tie_corr" : "basic")
                  << " ctx_ms="       << ctx_ms
                  << " keygen_ms="    << keygen_ms
                  << " rank_ms="      << rank_ms
                  << " gpu_keys_mib=" << gpu_keys_mib
                  << " gpu_rank_mib=" << gpu_rank_mib
                  << " gpu_peak_mib=" << gpu_peak_mib << "\n";
    }
    else
    {
        std::cout << "\n=== Multi-Ciphertext Ranking Results ("
                  << (tie_correction ? "tie-corrected" : "basic") << ") ===\n";
        std::cout << "Input (first 8): ";
        display_vector(input, std::min(N, 8));
        std::cout << "Ranks (first 8): ";
        display_vector(all_ranks, std::min(N, 8));

        std::vector<double> expected_ranks = tie_correction
            ? computeOrdinalRanks(input)
            : computeFractionalRanks(input);

        std::cout << "\nVerification:\n";
        double max_err = 0.0;
        std::vector<int> show_indices;
        for (int i = 0; i < std::min(N, 8); i++) show_indices.push_back(i);
        if (N > L + 2)
        {
            show_indices.push_back(L - 2);
            show_indices.push_back(L - 1);
            show_indices.push_back(L);
            show_indices.push_back(L + 1);
        }
        for (int i = std::max(0, N - 4); i < N; i++) show_indices.push_back(i);
        std::sort(show_indices.begin(), show_indices.end());
        show_indices.erase(std::unique(show_indices.begin(), show_indices.end()),
                           show_indices.end());
        int prev = -1;
        for (int i : show_indices)
        {
            if (prev >= 0 && i > prev + 1) std::cout << "  ...\n";
            double expected = expected_ranks[i];
            double actual   = all_ranks[i];
            double err      = std::abs(actual - expected);
            if (err > max_err) max_err = err;
            std::cout << "  [" << i << "] expected=" << expected
                      << "  actual=" << actual
                      << "  err=" << err << "\n";
            prev = i;
        }
        std::cout << "Max error: " << max_err << (max_err < 1.5 ? " (OK)" : " (HIGH)") << "\n";

        std::cout << "\nTiming:\n";
        std::cout << "  Context gen : " << ctx_ms    << " ms\n";
        std::cout << "  Key gen     : " << keygen_ms << " ms\n";
        std::cout << "  Ranking     : " << rank_ms   << " ms  ("
                  << (rank_ms / 1000.0) << " s)\n";
        std::cout << "\nVRAM (above context baseline):\n";
        std::cout << "  Keys  : " << gpu_keys_mib << " MiB\n";
        std::cout << "  Rank  : " << gpu_rank_mib << " MiB\n";
        std::cout << "  Peak  : " << gpu_peak_mib << " MiB\n";
    }

    return EXIT_SUCCESS;
}
