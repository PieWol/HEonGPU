/**
 * @file 19_ckks_minimum.cpp
 *
 * Homomorphic minimum (0th order statistic) via Algorithm 5 from:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone, Everts, Hahn, Peter — USENIX Security 2025
 *
 * Runs the full sorting pipeline (homomorphicSortFG) and reads slot 0,
 * which holds the 1st order statistic = minimum of the input vector.
 *
 * Output: slot 0 of the decoded result = normalized minimum.
 * Depth budget: identical to 18_ckks_sorting (29 levels).
 *
 * Usage:  19_ckks_minimum [N] [--bench]
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

constexpr int DG_C = 2;
constexpr int DF_C = 1;
constexpr int DG_I = 2;
constexpr int DF_I = 1;

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
// Normalization
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
    for (int i = 0; i < dg; i++) ct = applyG3(ct, pe, rk, scale);
    for (int i = 0; i < df - 1; i++) ct = applyF3(ct, pe, rk, scale);
    return applyF3Final(ct, pe, rk, scale);
}

static heongpu::Ciphertext<Scheme>
compareAdv(const heongpu::Ciphertext<Scheme>& a,
           const heongpu::Ciphertext<Scheme>& b,
           CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
           heongpu::HEContext<Scheme>& ctx, double scale)
{
    heongpu::Ciphertext<Scheme> a_copy = a, b_copy = b, diff(ctx);
    pe.sub(a_copy, b_copy, diff);
    return signAdv(diff, DG_C, DF_C, pe, rk, scale);
}

static heongpu::Ciphertext<Scheme>
indicatorAdv(heongpu::Ciphertext<Scheme>& ct, int N,
             CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
             heongpu::HEContext<Scheme>& ctx, double scale)
{
    double inv_N = 1.0 / N, half_inv_N = 0.5 / N;
    heongpu::Ciphertext<Scheme> tmp = ct;
    pe.multiply_plain_inplace(tmp, inv_N, scale);
    pe.rescale_inplace(tmp);
    heongpu::Ciphertext<Scheme> c1 = tmp, c2 = tmp;
    pe.add_plain_inplace(c1,  half_inv_N);
    pe.add_plain_inplace(c2, -half_inv_N);
    heongpu::Ciphertext<Scheme> s1 = signAdv(c1, DG_I, DF_I, pe, rk, scale);
    heongpu::Ciphertext<Scheme> s2 = signAdv(c2, DG_I, DF_I, pe, rk, scale);
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
    for (int i = N / 2; i > 0; i /= 2)
    {
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, -(i * N));
        pe.add_inplace(r, rot);
    }
    return r;
}

static heongpu::Ciphertext<Scheme>
replicateColumn(const heongpu::Ciphertext<Scheme>& col, int N,
                heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe)
{
    heongpu::Ciphertext<Scheme> r = col;
    for (int i = 1; i < N; i *= 2)
    {
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, -i);
        pe.add_inplace(r, rot);
    }
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
    for (int i = 1; i <= logN; i++)
    {
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, -((N * (N - 1)) / (1 << i)));
        pe.add_inplace(r, rot);
    }
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
    for (int i = 0; i < logN; i++)
    {
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, N * (1 << i));
        pe.add_inplace(r, rot);
    }
    return r;
}

static heongpu::Ciphertext<Scheme>
sumColumns(const heongpu::Ciphertext<Scheme>& m, int N,
           heongpu::Galoiskey<Scheme>& gk, CKKSPolyEvaluator& pe)
{
    heongpu::Ciphertext<Scheme> r = m;
    int logN = static_cast<int>(std::ceil(std::log2(N)));
    for (int i = 0; i < logN; i++)
    {
        heongpu::Ciphertext<Scheme> rot = r;
        pe.rotate_rows_inplace(rot, gk, 1 << i);
        pe.add_inplace(r, rot);
    }
    return r;
}

// ---------------------------------------------------------------------------
// homomorphicSortFG — full Algorithm 5 (identical to 18_ckks_sorting)
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
                  heongpu::HEContext<Scheme>& ctx,
                  double scale)
{
    size_t slots = ctx->get_poly_modulus_degree() / 2;

    // Phase 1: rank matrix
    heongpu::Ciphertext<Scheme> VR = replicateRow(ct_vector, N, row_key, pe);
    heongpu::Ciphertext<Scheme> col_t =
        transposeRowToColumn(ct_vector, N, transr_key, pe, enc, ctx, scale);
    heongpu::Ciphertext<Scheme> VC = replicateColumn(col_t, N, col_key, pe);
    while (VR.level() > VC.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(VR, tmp);
        VR = std::move(tmp);
    }
    heongpu::Ciphertext<Scheme> C = compareAdv(VR, VC, pe, rk, ctx, scale);
    heongpu::Ciphertext<Scheme> R = sumRows(C, N, sumr_key, pe);

    // MaskRow0 + ReplR
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
    R = replicateRow(R, N, row_key, pe);

    // Phase 2: indicator
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
    heongpu::Ciphertext<Scheme> M =
        indicatorAdv(ct_diff, N, pe, rk, ctx, scale);

    // Phase 3: reconstruct
    heongpu::Ciphertext<Scheme> VR2 = replicateRow(ct_vector, N, row_key, pe);
    while (VR2.level() > M.level())
    {
        heongpu::Ciphertext<Scheme> tmp(ctx);
        pe.mod_drop(VR2, tmp);
        VR2 = std::move(tmp);
    }
    heongpu::Ciphertext<Scheme> product(ctx);
    pe.multiply(M, VR2, product);
    pe.relinearize_inplace(product, rk);
    pe.rescale_inplace(product);
    return sumColumns(product, N, sumc_key, pe);
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
    auto scshifts = sumcGaloisShifts(N);

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
    else
    {
        for (int i = 0; i < N; i++) input[i] = static_cast<double>(N - 1 - i);
    }

    double expected_min = *std::min_element(input.begin(), input.end());
    std::vector<double> normalized = normalizeToUnit(input);

    if (g_verbose)
    {
        std::cout << "Input:         "; display_vector(input, N);
        std::cout << "Expected min:  " << expected_min << "\n";
    }

    std::vector<double> slot_buf(slots, 0.0);
    for (int i = 0; i < N; i++) slot_buf[i] = normalized[i];
    heongpu::Plaintext<Scheme>  pt(ctx);
    enc.encode(pt, slot_buf, scale);
    heongpu::Ciphertext<Scheme> ct(ctx);
    encryptor.encrypt(ct, pt);

    // Compute minimum (timed)
    GPUTimer min_timer;
    min_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_sorted =
        homomorphicSortFG(ct, N,
                          row_key, col_key, sumr_key, transr_key, sumc_key,
                          rk, pe, enc, encryptor, ctx, scale);

    float min_ms = min_timer.stopTimer();
    size_t gpu_min_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // Decrypt and extract slot 0 = minimum
    heongpu::Plaintext<Scheme> pt_result(ctx);
    decryptor.decrypt(pt_result, ct_sorted);
    std::vector<double> raw;
    enc.decode(raw, pt_result);

    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;
    double he_min = raw[0] * range + lo;  // slot 0 = 1st order statistic

    if (bench_mode)
    {
        std::cout << "BENCH:"
                  << " N="            << N
                  << " ctx_ms="       << ctx_ms
                  << " keygen_ms="    << keygen_ms
                  << " min_ms="       << min_ms
                  << " gpu_keys_mib=" << gpu_keys_mib
                  << " gpu_min_mib="  << gpu_min_mib
                  << " gpu_peak_mib=" << gpu_peak_mib << "\n";
    }
    else
    {
        double err     = std::abs(he_min - expected_min);
        bool   correct = err < (0.5 * range / N + 0.5);
        std::cout << "\n=== Minimum Result ===\n";
        std::cout << "Expected min : " << expected_min << "\n";
        std::cout << "HE min       : " << std::fixed << std::setprecision(4)
                  << he_min << (correct ? "" : "  INCORRECT") << "\n";
        std::cout << "Error        : " << err << "\n";
        std::cout << "\nTiming:\n";
        std::cout << "  Context gen : " << ctx_ms    << " ms\n";
        std::cout << "  Key gen     : " << keygen_ms << " ms\n";
        std::cout << "  Minimum     : " << min_ms    << " ms  ("
                  << (min_ms / 1000.0) << " s)\n";
        std::cout << "\nVRAM (above context baseline):\n";
        std::cout << "  Keys : " << gpu_keys_mib << " MiB\n";
        std::cout << "  Min  : " << gpu_min_mib  << " MiB\n";
        std::cout << "  Peak : " << gpu_peak_mib << " MiB\n";
    }

    return EXIT_SUCCESS;
}
