/**
 * @file 21_ckks_ranking_multi.cpp
 *
 * Multi-ciphertext homomorphic ranking for large N, following:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone et al., USENIX Security 2025  (Algorithm 7, complOpt=true)
 *
 * Derived from the proven-correct single-CT sorting implementation
 * (22_ckks_sorting_paper.cpp), stripped down to ranking only.
 *
 * Block size L=128, M = N/L block ciphertexts.
 * Sign approximation: fg-composite (dg=3, df=2), output in [0,1].
 * Ring dimension: n=65536 with adaptive dnum (typically 3-4).
 *
 * Tie correction (Algorithm 6):
 *   e_raw = C*(1-C), adjusted mask folds ×4 and δ bias,
 *   correction = SumR(e_raw * adj_mask).
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
#include <numeric>
#include <fstream>

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
    {}

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
// Load reference input data
// ---------------------------------------------------------------------------
static std::vector<double> loadPoints1D(int n)
{
    for (const char* path : {"data/points1d.csv", "../data/points1d.csv",
                             "example/ranking_and_sorting/data/points1d.csv"})
    {
        std::ifstream f(path);
        if (!f.is_open()) continue;
        std::vector<double> v;
        v.reserve(n);
        std::string line;
        while (std::getline(f, line) && static_cast<int>(v.size()) < n)
        {
            try { v.push_back(std::stod(line)); }
            catch (...) {}
        }
        if (static_cast<int>(v.size()) == n) return v;
    }
    std::cerr << "Warning: points1d.csv not found, using sequential input\n";
    std::vector<double> v(n);
    for (int i = 0; i < n; i++) v[i] = static_cast<double>(i);
    return v;
}

// ---------------------------------------------------------------------------
// Input normalization
// ---------------------------------------------------------------------------
static std::vector<double> normalizeToUnit(const std::vector<double>& v)
{
    double lo = *std::min_element(v.begin(), v.end());
    double hi = *std::max_element(v.begin(), v.end());
    std::vector<double> out(v.size());
    for (size_t i = 0; i < v.size(); i++)
        out[i] = (v[i] - lo) / (hi - lo);
    return out;
}

// ---------------------------------------------------------------------------
// Parameter derivation (adapted from 22_ckks_sorting_paper.cpp)
//
// Finds the largest scale (starting from 59) that keeps dnum within budget
// at n=65536 (security limit 1761 bits).
// ---------------------------------------------------------------------------
struct CKKSParams {
    size_t poly_modulus_degree;
    std::vector<int> q_bits;
    std::vector<int> p_bits;
    int scale_bits;
    int dnum;
};

static CKKSParams deriveParams(bool tie_correction)
{
    // fg-composite: dg=3, df=2 → 5 degree-7 evals.
    // Empirical depth (matches working single-CT implementations):
    //   basic: 24,  TC: 27 (= 24 + 3 for sign² + mask + maskC0)
    const int depth = tie_correction ? 27 : 24;
    const int Q_size = depth + 1;
    const int security_bits = 1761;  // n=65536, 128-bit security
    const int max_dnum = 5;

    int scale_bits = 0, P_size = 0, dnum = 0;
    for (int s = 59; s >= 40; s--)
    {
        int q_total = 60 + (Q_size - 1) * s;
        int p_avail = (security_bits - q_total) / 60;
        if (p_avail < 1) continue;

        int d = (Q_size + p_avail - 1) / p_avail;
        if (d <= max_dnum)
        {
            scale_bits = s;
            P_size = p_avail;
            dnum = d;
            break;
        }
    }

    if (scale_bits == 0)
    {
        std::cerr << "Error: cannot fit depth=" << depth
                  << " within " << security_bits << "-bit budget at n=65536\n";
        std::exit(EXIT_FAILURE);
    }

    std::vector<int> q_bits = {60};
    for (int i = 1; i < Q_size; i++) q_bits.push_back(scale_bits);
    std::vector<int> p_bits(P_size, 60);

    return {65536, q_bits, p_bits, scale_bits, dnum};
}

// ---------------------------------------------------------------------------
// Galois shift helpers (block size L)
// ---------------------------------------------------------------------------
static std::vector<int> rowGaloisShifts(int L)
{
    std::vector<int> s;
    for (int i = L / 2; i > 0; i /= 2) s.push_back(-(i * L));
    return s;
}
static std::vector<int> colGaloisShifts(int L)
{
    std::vector<int> s;
    for (int i = 1; i < L; i *= 2) s.push_back(-i);
    return s;
}
static std::vector<int> sumrGaloisShifts(int L)
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 0; i < logL; i++) s.push_back(L * (1 << i));
    return s;
}
static std::vector<int> transrGaloisShifts(int L)
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 1; i <= logL; i++)
        s.push_back(-((L * (L - 1)) / (1 << i)));
    return s;
}
static std::vector<int> sumcGaloisShifts(int L)
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 0; i < logL; i++) s.push_back(1 << i);
    return s;
}
static std::vector<int> transpcGaloisShifts(int L)
{
    int logL = static_cast<int>(std::ceil(std::log2(L)));
    std::vector<int> s;
    for (int i = 1; i <= logL; i++)
        s.push_back((L * (L - 1)) / (1 << i));
    return s;
}

// ---------------------------------------------------------------------------
// fg-sign primitives (from 22_ckks_sorting_paper.cpp)
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

// sign(x) ≈ f^df(g^dg(x)), output ∈ [0, 1] via F3Final
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

// compareAdv: returns C ∈ [0,1] where C≈1 means a>b
static heongpu::Ciphertext<Scheme>
compareAdv(heongpu::Ciphertext<Scheme>& a, heongpu::Ciphertext<Scheme>& b,
           int dg, int df, CKKSPolyEvaluator& pe,
           heongpu::Relinkey<Scheme>& rk,
           heongpu::HEContext<Scheme>& ctx, double scale)
{
    heongpu::Ciphertext<Scheme> diff(ctx);
    pe.sub(a, b, diff);
    return signAdv(diff, dg, df, pe, rk, scale);
}

// ---------------------------------------------------------------------------
// Matrix primitives
// ---------------------------------------------------------------------------
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
    size_t slots = ctx->get_poly_modulus_degree() / 2;
    std::vector<double> mask(slots, 0.0);
    for (int k = 0; k < L; k++) mask[k * L] = 1.0;
    heongpu::Plaintext<Scheme> pt(ctx);
    enc.encode(pt, mask, scale);
    pe.multiply_plain_inplace(r, pt);
    pe.rescale_inplace(r);
    return r;
}

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
    while (pt.depth() < ct.depth())
        pe.mod_drop_inplace(pt);

    heongpu::Ciphertext<Scheme> out = ct;
    pe.multiply_plain_inplace(out, pt);
    pe.rescale_inplace(out);
    return out;
}

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
// Verification helpers
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
// Multi-ciphertext ranking
//
// Layout: replR[j][r,c] = block_j[c], replC[k][r,c] = block_k[r]
// C_{j,k}[r,c] = sign(block_j[c] - block_k[r]) * 0.5 + 0.5  ∈ [0,1]
//   ≈ 1 when block_j[c] > block_k[r]
//
// Rank of block_j element c:
//   sumRows gives Σ_r C[r,c] = #{elements beaten by j[c]} + 0.5 (self)
//   rank = total_sum + 0.5
//
// Tie correction (from 22_ckks_sorting_paper.cpp):
//   e_raw = C*(1-C), adjusted mask folds ×4 factor:
//   diagonal: (c >= r) ? 2.0 : -2.0
//   cross-block (k > j): -2.0 uniformly (all partners have higher global index)
//   Complement for block k: negate masked result → +2.0
// ---------------------------------------------------------------------------
struct MultiCTResult {
    std::vector<heongpu::Ciphertext<Scheme>> ranks;
    std::vector<heongpu::Ciphertext<Scheme>> tc_offsets;
};

static MultiCTResult
multiCiphertextRank(
    const std::vector<heongpu::Ciphertext<Scheme>>& blocks,
    int L, bool tie_correction,
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
    double scale)
{
    const int M = static_cast<int>(blocks.size());
    const int dg = 3, df = 2;
    size_t slots = ctx->get_poly_modulus_degree() / 2;

    // Pre-encode TC masks
    heongpu::Plaintext<Scheme> diag_mask_pt(ctx);
    heongpu::Plaintext<Scheme> cross_mask_pt(ctx);

    if (tie_correction)
    {
        // Diagonal: adjusted mask from 22_ with ×4 folded
        // (c >= r) ? 2.0 : -2.0
        std::vector<double> diag_mask(slots, 0.0);
        for (int r = 0; r < L; r++)
            for (int c = 0; c < L; c++)
                diag_mask[r * L + c] = (c >= r) ? 2.0 : -2.0;
        enc.encode(diag_mask_pt, diag_mask, scale);

        // Cross-block: uniform -2.0 (all partners have higher global index)
        std::vector<double> cross_mask(slots, 0.0);
        for (int r = 0; r < L; r++)
            for (int c = 0; c < L; c++)
                cross_mask[r * L + c] = -2.0;
        enc.encode(cross_mask_pt, cross_mask, scale);
    }

    // Phase 1: replicate
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

    // Phase 2: compare upper triangle + accumulate
    if (g_verbose) std::cout << "\n=== Phase 2: Compare (" << (M*(M+1)/2) << " pairs) ===\n";

    // Rank accumulators (basic rank via sumRows / sumC+maskC0+transpC)
    std::vector<heongpu::Ciphertext<Scheme>> Cv(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<heongpu::Ciphertext<Scheme>> Ch(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<bool> Cv_init(M, false), Ch_init(M, false);

    // TC accumulators
    std::vector<heongpu::Ciphertext<Scheme>> Ev(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<heongpu::Ciphertext<Scheme>> Eh(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<bool> Ev_init(M, false), Eh_init(M, false);

    for (int j = 0; j < M; j++)
    {
        for (int k = j; k < M; k++)
        {
            if (g_verbose) std::cout << "  Compare (" << j << "," << k << ")\n";

            // Level-match replR[j] to replC[k]
            heongpu::Ciphertext<Scheme> rj = replR[j];
            while (rj.level() > replC[k].level())
            {
                heongpu::Ciphertext<Scheme> tmp(ctx);
                pe.mod_drop(rj, tmp);
                rj = std::move(tmp);
            }

            // C_{j,k} = compareAdv(replR[j], replC[k]) ∈ [0,1]
            heongpu::Ciphertext<Scheme> Cjk =
                compareAdv(rj, replC[k], dg, df, pe, rk, ctx, scale);

            // Accumulate into Cv[j] (block j's rank via sumRows)
            if (!Cv_init[j]) { Cv[j] = Cjk;                Cv_init[j] = true; }
            else             { pe.add_inplace(Cv[j], Cjk);                     }

            // Complement for block k: (1-C) counts wins for block_k elements
            if (j != k)
            {
                heongpu::Ciphertext<Scheme> comp = Cjk;
                pe.negate_inplace(comp);
                pe.add_plain_inplace(comp, 1.0);
                if (!Ch_init[k]) { Ch[k] = comp;                Ch_init[k] = true; }
                else             { pe.add_inplace(Ch[k], comp);                     }
            }

            // Tie correction: e_raw = C*(1-C), then apply adjusted mask
            if (tie_correction)
            {
                if (g_verbose) std::cout << "    TC: e_raw = C*(1-C)\n";

                heongpu::Ciphertext<Scheme> one_minus_C = Cjk;
                pe.negate_inplace(one_minus_C);
                pe.add_plain_inplace(one_minus_C, 1.0);

                heongpu::Ciphertext<Scheme> e_raw(ctx);
                pe.multiply(Cjk, one_minus_C, e_raw);
                pe.relinearize_inplace(e_raw, rk);
                pe.rescale_inplace(e_raw);

                // Apply adjusted mask
                heongpu::Plaintext<Scheme> mask_pt =
                    (j == k) ? diag_mask_pt : cross_mask_pt;
                while (mask_pt.depth() < e_raw.depth())
                    pe.mod_drop_inplace(mask_pt);
                pe.multiply_plain_inplace(e_raw, mask_pt);
                pe.rescale_inplace(e_raw);

                // Accumulate into Ev[j] (processed via sumRows in Phase 3)
                if (!Ev_init[j]) { Ev[j] = e_raw;                  Ev_init[j] = true; }
                else             { pe.add_inplace(Ev[j], e_raw);                       }

                // Cross-block: block k gets negated mask (uniform +2.0)
                if (j != k)
                {
                    heongpu::Ciphertext<Scheme> e_neg = e_raw;
                    pe.negate_inplace(e_neg);
                    if (!Eh_init[k]) { Eh[k] = e_neg;                  Eh_init[k] = true; }
                    else             { pe.add_inplace(Eh[k], e_neg);                       }
                }
            }
        }
    }

    // Phase 3: reduce accumulators to per-element rank vectors
    if (g_verbose) std::cout << "\n=== Phase 3: Reduce ===\n";

    std::vector<heongpu::Ciphertext<Scheme>> result(M, heongpu::Ciphertext<Scheme>(ctx));

    for (int j = 0; j < M; j++)
    {
        // Vertical contribution: sumRows(Cv[j])
        heongpu::Ciphertext<Scheme> sv = sumRows(Cv[j], L, sumr_key, pe, ctx);
        result[j] = sv;

        // Horizontal contribution: sumC + maskC0 + transpC of Ch[j]
        if (Ch_init[j])
        {
            if (g_verbose) std::cout << "  Block " << j << ": sumC + maskC0 + transpC\n";
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

    // TC Phase 3: sumR(Ev) + sumC-maskC0-transpC(Eh)
    std::vector<heongpu::Ciphertext<Scheme>> tc_result;
    if (tie_correction)
    {
        tc_result.resize(M, heongpu::Ciphertext<Scheme>(ctx));
        for (int j = 0; j < M; j++)
        {
            if (Ev_init[j])
                tc_result[j] = sumRows(Ev[j], L, sumr_key, pe, ctx);

            if (Eh_init[j])
            {
                heongpu::Ciphertext<Scheme> eh =
                    sumColumns(Eh[j], L, sumc_key, pe, ctx);
                eh = maskColumn0(eh, L, pe, enc, ctx, scale);
                eh = transposeColumn(eh, L, transpc_key, pe, ctx);

                if (Ev_init[j])
                {
                    while (tc_result[j].level() > eh.level())
                    {
                        heongpu::Ciphertext<Scheme> tmp(ctx);
                        pe.mod_drop(tc_result[j], tmp);
                        tc_result[j] = std::move(tmp);
                    }
                    pe.add_inplace(tc_result[j], eh);
                }
                else
                    tc_result[j] = eh;
            }
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

    // Derive parameters (adaptive scale/dnum at n=65536)
    CKKSParams params = deriveParams(tie_correction);
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
        int total_bits = params.q_bits[0];
        for (size_t i = 1; i < params.q_bits.size(); i++)
            total_bits += params.q_bits[i];
        for (size_t i = 0; i < params.p_bits.size(); i++)
            total_bits += params.p_bits[i];

        std::cout << "N=" << N << "  M=" << M << " blocks"
                  << "  L=" << L
                  << "  mode=" << (tie_correction ? "tie-corrected" : "basic") << "\n";
        std::cout << "fg-composite (dg=3, df=2)"
                  << "  n=" << params.poly_modulus_degree
                  << "  scale=2^" << params.scale_bits
                  << "  depth=" << (static_cast<int>(params.q_bits.size()) - 1)
                  << "  dnum=" << params.dnum
                  << "  Q+P=" << total_bits << "/1761\n";
    }

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

    // Input
    std::vector<double> input = loadPoints1D(N);
    std::vector<double> norm = normalizeToUnit(input);

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

    // Ranking (timed)
    GPUTimer rank_timer;
    rank_timer.startTimer();

    MultiCTResult rank_result =
        multiCiphertextRank(blocks, L, tie_correction,
                            row_key, col_key, transr_key, sumr_key,
                            sumc_key, transpc_key, rk, pe, enc, ctx, scale);

    float rank_ms = rank_timer.stopTimer();
    size_t gpu_rank_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // Decrypt & convert to ranks
    // C ∈ [0,1]: rank = total_sum + 0.5
    // TC correction adds directly (already in correct scale from adjusted mask)
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

            // TC correction includes +0.5 via self-comparison (e_raw=0.25, mask=2.0)
            for (int i = 0; i < L; i++)
                all_ranks[j * L + i] = raw[i] + tc_raw[i];
        }
        else
        {
            for (int i = 0; i < L; i++)
                all_ranks[j * L + i] = raw[i] + 0.5;
        }
    }

    // Verification
    std::vector<double> expected_ranks = tie_correction
        ? computeOrdinalRanks(input)
        : computeFractionalRanks(input);
    double max_err = 0.0;
    int mismatches = 0;
    for (int i = 0; i < N; i++)
    {
        double err = std::abs(all_ranks[i] - expected_ranks[i]);
        if (err > max_err) max_err = err;
        double rounded = tie_correction
            ? std::round(all_ranks[i])
            : std::round(all_ranks[i] * 2.0) / 2.0;
        if (std::abs(rounded - expected_ranks[i]) > 0.01) mismatches++;
    }

    // Output
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
                  << " gpu_peak_mib=" << gpu_peak_mib
                  << " max_err="      << max_err
                  << " mismatches="   << mismatches << "\n";
    }
    else
    {
        std::cout << "\n=== Multi-Ciphertext Ranking Results ("
                  << (tie_correction ? "tie-corrected" : "basic") << ") ===\n";
        std::cout << "Input (first 8): ";
        display_vector(input, std::min(N, 8));
        std::cout << "Ranks (first 8): ";
        display_vector(all_ranks, std::min(N, 8));

        std::cout << "\nVerification:\n";
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
            std::cout << "  [" << i << "] expected=" << expected_ranks[i]
                      << "  actual=" << all_ranks[i]
                      << "  err=" << std::abs(all_ranks[i] - expected_ranks[i]) << "\n";
            prev = i;
        }
        std::cout << "Max error: " << max_err << (max_err < 1.5 ? " (OK)" : " (HIGH)")
                  << "  Mismatches: " << mismatches << "/" << N << "\n";

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
