/**
 * @file 20_ckks_median.cpp
 *
 * Paper-exact homomorphic median via Algorithm 4 (Order Statistic) from:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone, Everts, Hahn, Peter — USENIX Security 2025
 *
 * Algorithm 4 for median computes:
 *   rankWithCorrectionFG(v) -> indicatorAdvShifted(rank, N) -> mask
 * The median element(s) are identified by a mask in the output.
 * For even N (all our benchmarks), two elements are detected.
 *
 * The correction term e = 4*(1-C)*C adjusts comparison-based ranks
 * to near-integer values, which is needed for the indicator to
 * cleanly distinguish the median rank from adjacent ranks.
 *
 * fg-composite parameters at n=131072 (matching paper's test-median.cpp):
 *   dg_c=3, df_c=2 (compare, no bias — median uses unbiased ranks)
 *   dg_i=log2(N)/2 (adaptive indicator), df_i=1
 *
 * Depth formula (matching paper's budget for OpenFHE Paterson-Stockmeyer):
 *   depth = 4*(dg_c + df_c + dg_i + df_i) + 6
 *
 * Note: HEonGPU's Chebyshev BSGS actually uses only 3 levels per degree-7
 * iteration (vs OpenFHE's 4), but the tighter formula 3*sum+6 causes a
 * systematic offset in evaluate_poly at low absolute levels. We use the
 * paper's 4*sum+6 formula which is correct and paper-compliant.
 *
 * The +6 overhead (vs +3 for minimum) comes from:
 *   +1 TransR mask, +3 correction (ct*ct, *4, *triMask),
 *   +1 indicator scale, +1 indicator multiply
 *
 * Parameter budgets at n=131072 (security limit 3500 bits):
 *   N=8:   dg_i=1, depth=34, Q=35, scale=59, dnum=2
 *   N=16:  dg_i=2, depth=38, Q=39, scale=59, dnum=2
 *   N=32:  dg_i=2, depth=38, Q=39, scale=59, dnum=2
 *   N=64:  dg_i=3, depth=42, Q=43, scale=59, dnum=3
 *   N=128: dg_i=3, depth=42, Q=43, scale=59, dnum=3
 *
 * Usage:  20_ckks_median [N] [--bench]
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
#include <fstream>

static bool g_verbose = true;

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
constexpr auto Scheme = heongpu::Scheme::CKKS;

// ---------------------------------------------------------------------------
// CKKSPolyEvaluator
// ---------------------------------------------------------------------------
class CKKSPolyEvaluator : public heongpu::HEArithmeticOperator<Scheme>
{
  public:
    CKKSPolyEvaluator(heongpu::HEContext<Scheme> ctx,
                      heongpu::HEEncoder<Scheme>& enc)
        : heongpu::HEArithmeticOperator<Scheme>(ctx, enc) {}

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

// ---------------------------------------------------------------------------
// fg-sign primitives
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

// indicatorAdvShifted: detects rank ≈ N/2 and N/2+1 for even N
static heongpu::Ciphertext<Scheme>
indicatorAdvShifted(heongpu::Ciphertext<Scheme>& ct, int N,
                    int dg_i, int df_i,
                    CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
                    heongpu::HEContext<Scheme>& ctx, double scale)
{
    if (g_verbose)
        std::cout << "  indicatorAdvShifted N=" << N
                  << " (dg_i=" << dg_i << " df_i=" << df_i << ")\n";

    double sf = 2.0 / (N + 1);

    heongpu::Ciphertext<Scheme> c1 = ct;
    pe.multiply_plain_inplace(c1, sf, scale);
    pe.rescale_inplace(c1);
    pe.add_plain_inplace(c1, sf - 1.0);

    heongpu::Ciphertext<Scheme> c2 = ct;
    pe.multiply_plain_inplace(c2, -sf, scale);
    pe.rescale_inplace(c2);
    pe.add_plain_inplace(c2, sf + 1.0);

    heongpu::Ciphertext<Scheme> s1 = signAdv(c1, dg_i, df_i, pe, rk, scale);
    heongpu::Ciphertext<Scheme> s2 = signAdv(c2, dg_i, df_i, pe, rk, scale);

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

// ---------------------------------------------------------------------------
// homomorphicMedianFG — Algorithm 4 for k=N/2 (median)
// Uses rankWithCorrectionFG + indicatorAdvShifted
// ---------------------------------------------------------------------------
heongpu::Ciphertext<Scheme>
homomorphicMedianFG(const heongpu::Ciphertext<Scheme>& ct_vector, int N,
                    int dg_c, int df_c, int dg_i, int df_i,
                    heongpu::Galoiskey<Scheme>& row_key,
                    heongpu::Galoiskey<Scheme>& col_key,
                    heongpu::Galoiskey<Scheme>& sumr_key,
                    heongpu::Galoiskey<Scheme>& transr_key,
                    heongpu::Relinkey<Scheme>& rk,
                    CKKSPolyEvaluator& pe,
                    heongpu::HEEncoder<Scheme>& enc,
                    heongpu::HEContext<Scheme>& ctx,
                    double scale)
{
    size_t slots = ctx->get_poly_modulus_degree() / 2;

    // Phase 1: comparison matrix (unbiased for median)
    if (g_verbose) std::cout << "\n=== Phase 1: comparison matrix ===\n";

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

    if (g_verbose) std::cout << "Step 3: compareAdv(VR, VC) — no bias\n";
    heongpu::Ciphertext<Scheme> C =
        compareAdv(VR, VC, dg_c, df_c, pe, rk, ctx, scale);

    // Phase 2: rankWithCorrectionFG
    if (g_verbose) std::cout << "\n=== Phase 2: rank with correction ===\n";

    // Correction: e = 4*(1-C)*C — peaks at C=0.5 (self-comparison)
    if (g_verbose) std::cout << "Step 4a: correction term e = 4*(1-C)*C\n";
    heongpu::Ciphertext<Scheme> C_neg = C;
    pe.negate_inplace(C_neg);
    pe.add_plain_inplace(C_neg, 1.0);

    heongpu::Ciphertext<Scheme> e_raw(ctx);
    pe.multiply(C_neg, C, e_raw);
    pe.relinearize_inplace(e_raw, rk);
    pe.rescale_inplace(e_raw);

    heongpu::Ciphertext<Scheme> e = e_raw;
    pe.multiply_plain_inplace(e, 4.0, scale);
    pe.rescale_inplace(e);

    if (g_verbose)
        std::cout << "  e level=" << e.level() << "\n";

    // Upper triangular mask: triMask[k,j] = 1 if j >= k
    if (g_verbose) std::cout << "Step 4b: e * triangularMask\n";
    std::vector<double> tri_mask(slots, 0.0);
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            if (j >= k) tri_mask[k * N + j] = 1.0;

    heongpu::Plaintext<Scheme> pt_tri(ctx);
    enc.encode(pt_tri, tri_mask, scale);

    // mod_drop plaintext to match ciphertext depth
    while (pt_tri.depth() < e.depth())
    {
        heongpu::Plaintext<Scheme> tmp(ctx);
        pe.mod_drop(pt_tri, tmp);
        pt_tri = std::move(tmp);
    }

    heongpu::Ciphertext<Scheme> e_tri = e;
    pe.multiply_plain_inplace(e_tri, pt_tri);
    pe.rescale_inplace(e_tri);

    // correctionOffset = sumRows(e * triMask) - 0.5 * sumRows(e)
    if (g_verbose) std::cout << "Step 4c: correction offset\n";
    heongpu::Ciphertext<Scheme> sumR_e_tri = sumRows(e_tri, N, sumr_key, pe);

    heongpu::Ciphertext<Scheme> sumR_e = sumRows(e, N, sumr_key, pe);
    pe.multiply_plain_inplace(sumR_e, 0.5, scale);
    pe.rescale_inplace(sumR_e);

    // Align levels before subtraction
    while (sumR_e_tri.level() > sumR_e.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(sumR_e_tri, tmp);
        sumR_e_tri = std::move(tmp);
    }

    heongpu::Ciphertext<Scheme> correction(ctx);
    pe.sub(sumR_e_tri, sumR_e, correction);

    // R = sumRows(C) + correction
    if (g_verbose) std::cout << "Step 4d: corrected rank\n";
    heongpu::Ciphertext<Scheme> sumR_C = sumRows(C, N, sumr_key, pe);

    while (sumR_C.level() > correction.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(sumR_C, tmp);
        sumR_C = std::move(tmp);
    }

    heongpu::Ciphertext<Scheme> R(ctx);
    pe.add(sumR_C, correction, R);

    if (g_verbose)
        std::cout << "  R level=" << R.level() << "\n";

    // Phase 3: indicatorAdvShifted detects median rank(s)
    if (g_verbose) std::cout << "\n=== Phase 3: indicator (detect median rank) ===\n";
    heongpu::Ciphertext<Scheme> mask =
        indicatorAdvShifted(R, N, dg_i, df_i, pe, rk, ctx, scale);

    if (g_verbose)
        std::cout << "  mask level=" << mask.level() << "\n";
    return mask;
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

    int logN = static_cast<int>(std::ceil(std::log2(std::max(N, 2))));

    cudaSetDevice(0);

    const int dg_c = 3;
    const int df_c = 2;
    const int dg_i = logN / 2;
    const int df_i = 1;
    const int actual_depth = 4 * (dg_c + df_c + dg_i + df_i) + 6;
    const int Q_size = actual_depth + 1;
    const int security_bits = 3500;
    const int scale_bits = 59;

    int Q_bits = 60 + (Q_size - 1) * scale_bits;
    int P_size = (security_bits - Q_bits) / 60;

    while (P_size > 1)
    {
        int total_P = P_size * 60;
        bool valid = true;
        for (int i = 0; i < Q_size; i += P_size)
        {
            int group_sum = 0;
            for (int j = i; j < std::min(i + P_size, Q_size); j++)
                group_sum += (j == 0 ? 60 : scale_bits);
            if (group_sum > total_P) { valid = false; break; }
        }
        if (valid) break;
        P_size--;
    }

    int dnum = (Q_size + P_size - 1) / P_size;
    int total_bits = Q_bits + P_size * 60;

    if (g_verbose)
    {
        std::cout << "Paper-exact median (Algorithm 4): n=131072\n";
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

    if (dnum >= 4)
        std::cerr << "Warning: dnum=" << dnum
                  << " — keys may exceed 16 GB VRAM. 48 GB GPU recommended.\n";

    const size_t poly_modulus_degree = 131072;

    std::vector<int> q_bits = {60};
    for (int i = 1; i < Q_size; i++) q_bits.push_back(scale_bits);
    std::vector<int> p_bits(P_size, 60);

    heongpu::HEContext<Scheme> ctx = heongpu::GenHEContext<Scheme>();
    ctx->set_poly_modulus_degree(poly_modulus_degree);
    ctx->set_coeff_modulus_bit_sizes(q_bits, p_bits);
    double scale = std::pow(2.0, scale_bits);

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
        std::cout << "N=" << N << "  matrix=" << N << "x" << N
                  << "  slots_used=" << (N*N) << "/" << slots << "\n";

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

    auto rshifts  = rowGaloisShifts(N);
    auto cshifts  = colGaloisShifts(N);
    auto sshifts  = sumrGaloisShifts(N);
    auto tshifts  = transrGaloisShifts(N);

    heongpu::Galoiskey<Scheme> row_key(ctx, rshifts);
    keygen.generate_galois_key(row_key, sk);
    heongpu::Galoiskey<Scheme> col_key(ctx, cshifts);
    keygen.generate_galois_key(col_key, sk);
    heongpu::Galoiskey<Scheme> sumr_key(ctx, sshifts);
    keygen.generate_galois_key(sumr_key, sk);
    heongpu::Galoiskey<Scheme> transr_key(ctx, tshifts);
    keygen.generate_galois_key(transr_key, sk);
    heongpu::Relinkey<Scheme> rk(ctx);
    keygen.generate_relin_key(rk, sk);

    float keygen_ms = keygen_timer.stopTimer();
    size_t gpu_keys_mib = (heongpu::MemoryPool::instance()
                               .get_current_device_pool_memory_usage()
                           - gpu_baseline) / kMiB;
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms
                  << " ms  (" << gpu_keys_mib << " MiB VRAM)\n";

    // Input (same CSV as OpenFHE reference)
    std::vector<double> input = loadPoints1D(N);

    // Expected median: average of two middle values for even N
    std::vector<double> input_sorted = input;
    std::sort(input_sorted.begin(), input_sorted.end());
    double expected_median = (input_sorted[N/2 - 1] + input_sorted[N/2]) / 2.0;

    std::vector<double> normalized = normalizeToUnit(input);

    if (g_verbose)
    {
        std::cout << "Input:           "; display_vector(input, N);
        std::cout << "Expected median: " << expected_median << "\n";
    }

    std::vector<double> slot_buf(slots, 0.0);
    for (int i = 0; i < N; i++) slot_buf[i] = normalized[i];
    heongpu::Plaintext<Scheme>  pt(ctx);
    enc.encode(pt, slot_buf, scale);
    heongpu::Ciphertext<Scheme> ct(ctx);
    encryptor.encrypt(ct, pt);

    // Compute median (timed)
    GPUTimer median_timer;
    median_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_mask =
        homomorphicMedianFG(ct, N, dg_c, df_c, dg_i, df_i,
                            row_key, col_key, sumr_key, transr_key,
                            rk, pe, enc, ctx, scale);

    float median_ms = median_timer.stopTimer();
    size_t gpu_median_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // Decrypt mask and extract median via weighted average
    heongpu::Plaintext<Scheme> pt_result(ctx);
    decryptor.decrypt(pt_result, ct_mask);
    std::vector<double> raw;
    enc.decode(raw, pt_result);

    double he_median = 0, mask_sum = 0;
    for (int i = 0; i < N; i++)
    {
        double m = raw[i];
        he_median += m * input[i];
        mask_sum += m;
    }
    he_median = (mask_sum > 0.01) ? he_median / mask_sum : 0;

    if (bench_mode)
    {
        std::cout << "BENCH:"
                  << " N="               << N
                  << " ctx_ms="          << ctx_ms
                  << " keygen_ms="       << keygen_ms
                  << " median_ms="       << median_ms
                  << " gpu_keys_mib="    << gpu_keys_mib
                  << " gpu_median_mib="  << gpu_median_mib
                  << " gpu_peak_mib="    << gpu_peak_mib << "\n";
    }
    else
    {
        double err     = std::abs(he_median - expected_median);
        double lo      = *std::min_element(input.begin(), input.end());
        double hi      = *std::max_element(input.begin(), input.end());
        double range   = hi - lo;
        bool   correct = err < (0.5 * range / N + 0.5);

        std::cout << "\nMask values: [";
        for (int i = 0; i < N; i++)
            std::cout << std::fixed << std::setprecision(4) << raw[i]
                      << (i < N-1 ? ", " : "");
        std::cout << "]\n";

        std::cout << "\n=== Median Result ===\n";
        std::cout << "Expected median : " << expected_median << "\n";
        std::cout << "HE median       : " << std::fixed << std::setprecision(4)
                  << he_median << (correct ? "" : "  INCORRECT") << "\n";
        std::cout << "Error           : " << err << "\n";
        std::cout << "\nTiming:\n";
        std::cout << "  Context gen : " << ctx_ms     << " ms\n";
        std::cout << "  Key gen     : " << keygen_ms  << " ms\n";
        std::cout << "  Median      : " << median_ms  << " ms  ("
                  << (median_ms / 1000.0) << " s)\n";
        std::cout << "\nVRAM (above context baseline):\n";
        std::cout << "  Keys   : " << gpu_keys_mib   << " MiB\n";
        std::cout << "  Median : " << gpu_median_mib << " MiB\n";
        std::cout << "  Peak   : " << gpu_peak_mib   << " MiB\n";
    }

    return EXIT_SUCCESS;
}
