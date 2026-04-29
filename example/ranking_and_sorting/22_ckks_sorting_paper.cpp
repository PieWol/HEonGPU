/**
 * @file 22_ckks_sorting_paper.cpp
 *
 * Paper-exact homomorphic sorting via Algorithm 5 from:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone, Everts, Hahn, Peter — USENIX Security 2025
 *
 * Uses the paper's exact fg-composite parameters at n=131072:
 *   dg_c=3, df_c=2 (compare)
 *   dg_i=(log2(N)+1)/2 (adaptive indicator), df_i=2
 *
 * Scale adaptation for fixed-scale CKKS
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * The paper uses OpenFHE's FLEXIBLEAUTO with decimalPrecision=59, giving
 * a depth formula of 4*(dg_c+df_c+dg_i+df_i)+4.  Our fixed-scale CKKS
 * needs one extra level for the encrypted MaskRow0 step (+5 base), but
 * our optimized tie correction saves one level vs the paper's Algorithm 6
 * (+2 instead of +3).  These cancel out, giving the same total depth:
 *
 *   Paper: 4*(sum)+4 (base) + 3 (tie correction) = 4*(sum)+7
 *   Ours:  4*(sum)+5 (base) + 2 (optimized TC)   = 4*(sum)+7
 *
 * Although depths match, FLEXIBLEAUTO uses variable-sized primes that
 * average below the nominal 59 bits, leaving more room for P primes
 * (and thus smaller dnum / keys).  Fixed-scale CKKS with uniform 59-bit
 * primes needs 3069 Q-bits at depth=51, yielding dnum=8 — too large.
 * We find the largest scale (starting from 59) that keeps dnum ≤ 5
 * (the paper's maximum for single-CT sorting).  This gives scale=59
 * where the budget allows (N≤16) and reduces to ~54 at deep chains.
 *
 * Example parameter budgets at n=131072 (security limit 3500 bits):
 *                      ours (fixed-scale)           paper (FLEXIBLEAUTO)
 *   N=4:   depth=39, Q=40, scale=57, dnum=2       depth=39, Q=40, dnum=2
 *   N=8:   depth=43, Q=44, scale=59, dnum=3       depth=43, Q=44, dnum=3
 *   N=64:  depth=47, Q=48, scale=57, dnum=4       depth=47, Q=48, dnum=4
 *   N=256: depth=51, Q=52, scale=54, dnum=5       depth=51, Q=52, dnum=5
 *
 * Tie correction (Algorithm 6) is always enabled: the paper proves sorting
 * correctness only when ranks form a permutation of (1,..,N), which requires
 * the tie-correction offset.  This adds +2 levels to the depth budget
 * (optimized: the ×4 factor and mask-0.5 are folded into a single adjusted
 * plaintext multiply).
 *
 * Usage:  22_ckks_sorting_paper [N] [--bench] [--ties]
 *   N       : vector length, power of 2, default 4
 *   --bench : machine-readable timing output only
 *   --ties  : use paired tied input to verify tie correction
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

// ---------------------------------------------------------------------------
// CKKSPolyEvaluator
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
// Input normalization
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
std::vector<int> rowGaloisShifts(int N)
{
    std::vector<int> s;
    for (int i = N / 2; i > 0; i /= 2) s.push_back(-(i * N));
    return s;
}
std::vector<int> colGaloisShifts(int N)
{
    std::vector<int> s;
    for (int i = 1; i < N; i *= 2) s.push_back(-i);
    return s;
}
std::vector<int> sumrGaloisShifts(int N)
{
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    std::vector<int> s;
    for (int i = 0; i < logN; i++) s.push_back(N * (1 << i));
    return s;
}
std::vector<int> transrGaloisShifts(int N)
{
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    std::vector<int> s;
    for (int i = 1; i <= logN; i++)
        s.push_back(-((N * (N - 1)) / (1 << i)));
    return s;
}
std::vector<int> sumcGaloisShifts(int N)
{
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    std::vector<int> s;
    for (int i = 0; i < logN; i++) s.push_back(1 << i);
    return s;
}

// ---------------------------------------------------------------------------
// fg-sign primitives (identical to 18_ckks_sorting.cpp)
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
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk);
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
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk);
}

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

static heongpu::Ciphertext<Scheme>
compareAdv(const heongpu::Ciphertext<Scheme>& a,
           const heongpu::Ciphertext<Scheme>& b,
           int dg_c, int df_c,
           CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
           heongpu::HEContext<Scheme>& ctx, double scale)
{
    if (g_verbose) std::cout << "  compareAdv (dg_c=" << dg_c << " df_c=" << df_c << ")\n";
    heongpu::Ciphertext<Scheme> a_copy = a;
    heongpu::Ciphertext<Scheme> b_copy = b;
    heongpu::Ciphertext<Scheme> diff(ctx);
    pe.sub(a_copy, b_copy, diff);
    return signAdv(diff, dg_c, df_c, pe, rk, scale);
}

static heongpu::Ciphertext<Scheme>
indicatorAdv(heongpu::Ciphertext<Scheme>& ct, int N,
             int dg_i, int df_i,
             CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
             heongpu::HEContext<Scheme>& ctx, double scale)
{
    if (g_verbose)
        std::cout << "  indicatorAdv N=" << N
                  << " (dg_i=" << dg_i << " df_i=" << df_i << ")\n";

    double inv_N      = 1.0 / N;
    double half_inv_N = 0.5 / N;

    heongpu::Ciphertext<Scheme> tmp = ct;
    pe.multiply_plain_inplace(tmp, inv_N, scale);
    pe.rescale_inplace(tmp);

    heongpu::Ciphertext<Scheme> c1 = tmp;
    pe.add_plain_inplace(c1,  half_inv_N);
    heongpu::Ciphertext<Scheme> c2 = tmp;
    pe.add_plain_inplace(c2, -half_inv_N);

    heongpu::Ciphertext<Scheme> s1 = signAdv(c1, dg_i, df_i, pe, rk, scale);
    heongpu::Ciphertext<Scheme> s2 = signAdv(c2, dg_i, df_i, pe, rk, scale);

    pe.negate_inplace(s2);
    pe.add_plain_inplace(s2, 1.0);

    heongpu::Ciphertext<Scheme> result(ctx);
    pe.multiply(s1, s2, result);
    pe.relinearize_inplace(result, rk);
    pe.rescale_inplace(result);
    return result;
}

// ---------------------------------------------------------------------------
// Matrix primitives
// ---------------------------------------------------------------------------
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

    size_t slots = ctx->get_poly_modulus_degree() / 2;
    std::vector<double> mask(slots, 0.0);
    for (int k = 0; k < N; k++) mask[k * N] = 1.0;
    heongpu::Plaintext<Scheme> pt(ctx);
    enc.encode(pt, mask, scale);
    pe.multiply_plain_inplace(r, pt);
    pe.rescale_inplace(r);
    return r;
}

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
// homomorphicSortFG
// ---------------------------------------------------------------------------
heongpu::Ciphertext<Scheme>
homomorphicSortFG(const heongpu::Ciphertext<Scheme>& ct_vector, int N,
                  int dg_c, int df_c, int dg_i, int df_i,
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

    // Phase 1: rank matrix
    if (g_verbose) std::cout << "\n=== Phase 1: rank matrix ===\n";

    if (g_verbose) std::cout << "Step 1: ReplR(V)\n";
    heongpu::Ciphertext<Scheme> VR = replicateRow(ct_vector, N, row_key, pe);

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

    if (g_verbose) std::cout << "Step 3: compareAdv(VR, VC)\n";
    heongpu::Ciphertext<Scheme> C = compareAdv(VR, VC, dg_c, df_c, pe, rk, ctx, scale);

    if (g_verbose) std::cout << "Step 4a: SumR(C) -> full sum in row 0\n";
    heongpu::Ciphertext<Scheme> R = sumRows(C, N, sumr_key, pe);

    // --- Tie-correction offset (Algorithm 6 from Mazzone et al.) ---
    // correction = SumR(E·(mask - 0.5)) where E = 4·C·(1-C), mask = δ_{j≥i}
    //
    // Optimization: fold the ×4 and the mask-0.5 into one plaintext multiply
    // on e_raw = C·(1-C), saving 1 level and 1 SumR vs the naive approach.
    //   correction = SumR(e_raw · adjusted)   where adjusted[i,j] = 2 if j≥i, -2 otherwise
    if (g_verbose) std::cout << "Step 4-tc1: e_raw = C*(1-C)\n";
    {
        heongpu::Ciphertext<Scheme> C_neg = C;
        pe.negate_inplace(C_neg);
        pe.add_plain_inplace(C_neg, 1.0);

        heongpu::Ciphertext<Scheme> e_raw(ctx);
        pe.multiply(C_neg, C, e_raw);
        pe.relinearize_inplace(e_raw, rk);
        pe.rescale_inplace(e_raw);

        if (g_verbose)
            std::cout << "  e_raw level=" << e_raw.level() << "\n";

        if (g_verbose) std::cout << "Step 4-tc2: e_raw * adjusted mask (4·(δ_{j≥i} - 0.5))\n";
        std::vector<double> adj_mask(slots, 0.0);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                adj_mask[i * N + j] = (j >= i) ? 2.0 : -2.0;

        heongpu::Plaintext<Scheme> pt_adj(ctx);
        enc.encode(pt_adj, adj_mask, scale);
        while (pt_adj.depth() < e_raw.depth())
            pe.mod_drop_inplace(pt_adj);

        pe.multiply_plain_inplace(e_raw, pt_adj);
        pe.rescale_inplace(e_raw);

        if (g_verbose) std::cout << "Step 4-tc3: correction = SumR(e_raw * adj_mask)\n";
        heongpu::Ciphertext<Scheme> correction = sumRows(e_raw, N, sumr_key, pe);

        while (R.level() > correction.level())
        {
            heongpu::Ciphertext<Scheme> tmp(ctx);
            pe.mod_drop(R, tmp);
            R = std::move(tmp);
        }

        pe.add_inplace(R, correction);
        if (g_verbose)
            std::cout << "  corrected R level=" << R.level() << "\n";
    }

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

    if (g_verbose) std::cout << "Step 4c: ReplR(R)\n";
    R = replicateRow(R, N, row_key, pe);

    if (g_verbose)
        std::cout << "  R level=" << R.level() << "\n";
    if (g_verbose) debugMatrix("R", R, dec, enc, ctx, N);

    // Phase 2: one-hot indicator
    if (g_verbose) std::cout << "\n=== Phase 2: one-hot indicator ===\n";

    // Corrected ranks are integers {1,..,N} → shift row k by -(k+1).
    std::vector<double> sub_vals(slots, 0.0);
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            sub_vals[k * N + j] = -(static_cast<double>(k) + 1.0);

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

    if (g_verbose) std::cout << "Step 5: indicatorAdv(R + subMask)\n";
    heongpu::Ciphertext<Scheme> M =
        indicatorAdv(ct_diff, N, dg_i, df_i, pe, rk, ctx, scale);

    if (g_verbose)
        std::cout << "  M level=" << M.level() << "\n";
    if (g_verbose) debugMatrix("M", M, dec, enc, ctx, N);

    // Phase 3: reconstruct sorted values
    if (g_verbose) std::cout << "\n=== Phase 3: reconstruct ===\n";

    if (g_verbose) std::cout << "Step 6: ReplR(V) fresh + mod_drop\n";
    heongpu::Ciphertext<Scheme> VR2 = replicateRow(ct_vector, N, row_key, pe);
    while (VR2.level() > M.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(VR2, tmp);
        VR2 = std::move(tmp);
    }

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
    int  N          = 4;
    bool bench_mode = false;
    bool use_ties   = false;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--bench")
            bench_mode = true;
        else if (arg == "--ties")
            use_ties = true;
        else if (!arg.empty() && std::isdigit(static_cast<unsigned char>(arg[0])))
            N = std::stoi(arg);
    }
    g_verbose = !bench_mode;

    if (N <= 0 || (N & (N - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << N << ")\n";
        return EXIT_FAILURE;
    }

    int logN = static_cast<int>(std::ceil(std::log2(std::max(N, 2))));

    cudaSetDevice(0);

    // Paper-exact fg params
    const int dg_c = 3;
    const int df_c = 2;
    const int dg_i = (logN + 1) / 2;
    const int df_i = 2;
    const int actual_depth = 4 * (dg_c + df_c + dg_i + df_i) + 7;
    const int Q_size = actual_depth + 1;
    const int security_bits = 3500;

    // FLEXIBLEAUTO uses variable-sized primes, achieving lower dnum than
    // fixed-scale CKKS at the same depth.  Find the largest scale that
    // keeps dnum <= the paper's value for this N.
    // Paper dnum targets (Table 2, Mazzone et al.):
    //   N=4:2  N=8:3  N=16:3  N=32:3  N=64:4  N=128:5  N=256:5
    const int paper_dnum[] = {0,0,2,3,3,3,4,5,5};  // indexed by logN
    const int max_dnum = (logN >= 2 && logN <= 8) ? paper_dnum[logN] : 5;
    int scale_bits = 0, P_size = 0, dnum = 0, Q_bits = 0;
    for (int s = 59; s >= 45; s--)
    {
        int q = 60 + (Q_size - 1) * s;
        int p = (security_bits - q) / 60;
        if (p < 1) continue;

        while (p > 1)
        {
            int total_P = p * 60;
            bool valid = true;
            for (int i = 0; i < Q_size; i += p)
            {
                int group_sum = 0;
                for (int j = i; j < std::min(i + p, Q_size); j++)
                    group_sum += (j == 0 ? 60 : s);
                if (group_sum > total_P) { valid = false; break; }
            }
            if (valid) break;
            p--;
        }

        int d = (Q_size + p - 1) / p;
        if (d <= max_dnum)
        {
            scale_bits = s;
            P_size = p;
            dnum = d;
            Q_bits = q;
            break;
        }
    }

    if (scale_bits == 0)
    {
        std::cerr << "Error: cannot achieve dnum<=" << max_dnum
                  << " for depth=" << actual_depth << " within "
                  << security_bits << "-bit budget\n";
        return EXIT_FAILURE;
    }

    int total_bits = Q_bits + P_size * 60;

    if (g_verbose)
    {
        std::cout << "Paper-exact sorting: n=131072 (with tie correction)\n";
        std::cout << "Scale adapted for paper dnum<=" << max_dnum
                  << " (N=" << N << "):  depth=" << actual_depth
                  << "  scale=2^" << scale_bits << "\n";
        std::cout << "fg params: dg_c=" << dg_c << " df_c=" << df_c
                  << " dg_i=" << dg_i << " df_i=" << df_i
                  << "  depth=" << actual_depth
                  << "  Q_size=" << Q_size
                  << "  P_size=" << P_size
                  << "  dnum=" << dnum << "\n";
        std::cout << "Q+P bits=" << total_bits << " / " << security_bits
                  << " (128-bit bound)\n";
    }

    if (total_bits > security_bits)
    {
        std::cerr << "Error: Q+P=" << total_bits
                  << " exceeds " << security_bits
                  << "-bit security bound for n=131072\n";
        return EXIT_FAILURE;
    }

    if (dnum >= 7)
        std::cerr << "Warning: dnum=" << dnum
                  << " — keys alone likely exceed 48 GB VRAM.\n";
    else if (dnum >= 4)
        std::cerr << "Warning: dnum=" << dnum
                  << " — keys may exceed 16 GB VRAM. 48 GB GPU recommended.\n";
    else if (dnum >= 3)
        std::cerr << "Note: dnum=" << dnum
                  << " — requires >16 GB VRAM (e.g. L40 48 GB).\n";

    // n=131072 is the minimum ring dimension for 128-bit security at every N:
    // even N=4 needs depth 37 → Q≈2243 bits, which exceeds the 1770-bit
    // budget of n=65536. The depth (not the slot count) is the binding constraint.
    const size_t poly_modulus_degree = 131072;

    std::vector<int> q_bits = {60};
    for (int i = 1; i < Q_size; i++) q_bits.push_back(scale_bits);
    std::vector<int> p_bits(P_size, 60);

    heongpu::HEContext<Scheme> ctx = heongpu::GenHEContext<Scheme>();
    ctx->set_poly_modulus_degree(poly_modulus_degree);
    ctx->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    double scale = std::pow(2.0, scale_bits);

    heongpu::MemoryPoolConfig pool_config;
    pool_config.initial_device_fraction = 90.0f;
    pool_config.max_device_fraction = 100.0f;

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    ctx->generate(pool_config);
    float ctx_ms = ctx_timer.stopTimer();

    const int slots = static_cast<int>(poly_modulus_degree / 2);
    if (N * N > slots)
    {
        std::cerr << "Error: N=" << N << " needs " << (N*N)
                  << " slots but only " << slots << " available.\n";
        return EXIT_FAILURE;
    }
    if (g_verbose)
        std::cout << "N=" << N << "  matrix=" << N << "x" << N
                  << "  slots_used=" << (N*N) << "/" << slots << "\n";

    // Key generation
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

    // Input
    std::vector<double> input(N);
    if (bench_mode)
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 100.0);
        for (int i = 0; i < N; i++) input[i] = dist(rng);
    }
    else if (use_ties)
    {
        for (int i = 0; i < N; i++)
            input[i] = static_cast<double>((N - 1 - i) / 2 + 1);
    }
    else
    {
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

    // Sort (timed)
    GPUTimer sort_timer;
    sort_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_sorted =
        homomorphicSortFG(ct, N, dg_c, df_c, dg_i, df_i,
                          row_key, col_key, sumr_key, transr_key, sumc_key,
                          rk, pe, enc, encryptor, decryptor, ctx, scale);

    float sort_ms = sort_timer.stopTimer();
    size_t gpu_sort_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // Decrypt
    heongpu::Plaintext<Scheme> pt_result(ctx);
    decryptor.decrypt(pt_result, ct_sorted);
    std::vector<double> raw;
    enc.decode(raw, pt_result);

    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;

    std::vector<double> he_sorted(N), he_sorted_orig(N);
    for (int k = 0; k < N; k++)
    {
        he_sorted[k]      = raw[k * N];
        he_sorted_orig[k] = he_sorted[k] * range + lo;
    }

    // Output
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
        std::cout << "\n=== Sorting Results"
                  << (use_ties ? " (tied input)" : "") << " ===\n";
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
                      << "  err=" << std::setprecision(6) << err
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
