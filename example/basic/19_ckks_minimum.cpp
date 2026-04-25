/**
 * @file 19_ckks_minimum.cpp
 *
 * Paper-exact homomorphic minimum via Algorithm 4 (Order Statistic) from:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone, Everts, Hahn, Peter — USENIX Security 2025
 *
 * Algorithm 4 computes: rank(v) -> indicator(rank, target=1) -> mask
 * The minimum is identified by a one-hot mask in the output.
 *
 * The paper uses Chebyshev comparison and indicator for minimum at N<=256,
 * and fg-composite for N>256. This implementation covers N<=128
 * (single-ciphertext mode) with Chebyshev as the paper specifies.
 *
 * From the paper (Section 6.1): "For ranking and minimum, we use Chebyshev
 * approximation of the comparison function up to degree 2^11 for N <= 256"
 *
 * Chebyshev parameters from test-minimum.cpp (OpenFHE reference):
 *   N<=32:  compareDepth=7 (degree 59), indicatorDepth=7 (degree 59)
 *   N<=128: compareDepth=9 (degree 247), indicatorDepth=7 (degree 59)
 *
 * HEonGPU depth: ceil(log2(degree)) levels per Chebyshev evaluation
 * (vs OpenFHE's depth2degree mapping which is 1 level more expensive).
 * Indicator normalization from [1,N] to [-1,1] costs 1 additional level.
 *
 *   depth = 1 (TransR mask) + compare_levels + 1 (norm) + indicator_levels
 *   N<=32:  1 + 6 + 1 + 6 = 14, Q=15, dnum=1
 *   N<=128: 1 + 8 + 1 + 6 = 16, Q=17, dnum=1
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
// Chebyshev comparison: compareGt(a, b) ≈ 1{a > b}
// Matches paper's compare function with error=0.005 bias for strict >
// ---------------------------------------------------------------------------
static heongpu::Ciphertext<Scheme>
compareGtChebyshev(const heongpu::Ciphertext<Scheme>& a,
                   const heongpu::Ciphertext<Scheme>& b,
                   int degree,
                   CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
                   heongpu::HEContext<Scheme>& ctx, double scale)
{
    if (g_verbose)
        std::cout << "  compareGtChebyshev (degree=" << degree << ")\n";

    heongpu::Ciphertext<Scheme> a_copy = a;
    heongpu::Ciphertext<Scheme> b_copy = b;
    heongpu::Ciphertext<Scheme> diff(ctx);
    pe.sub(a_copy, b_copy, diff);

    auto fn = [](Complex64 x) -> Complex64 {
        double t = x.real();
        return {(t > 0.005) ? 1.0 : 0.0, 0.0};
    };
    auto coeffs = heongpu::approximate_function(fn, -1.0, 1.0, degree);
    return pe.eval_chebyshev(diff, scale, coeffs, degree, rk);
}

// ---------------------------------------------------------------------------
// Chebyshev indicator: detects rank ≈ 1 (minimum)
// Pre-normalizes rank from [1, N] to [-1, 1], then evaluates Chebyshev.
// Matches paper's indicator(c, 0.5, 1.5, 0.5, N+0.5, degree).
// ---------------------------------------------------------------------------
static heongpu::Ciphertext<Scheme>
indicatorChebyshev(heongpu::Ciphertext<Scheme>& ct, int N,
                   int degree,
                   CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
                   double scale)
{
    if (g_verbose)
        std::cout << "  indicatorChebyshev N=" << N
                  << " (degree=" << degree << ")\n";

    // Normalize rank from [1, N] to [-1, 1]:
    //   u = (2*rank - (N+1)) / N
    heongpu::Ciphertext<Scheme> u = ct;
    pe.multiply_plain_inplace(u, 2.0 / N, scale);
    pe.rescale_inplace(u);
    pe.add_plain_inplace(u, -(N + 1.0) / N);

    if (g_verbose)
        std::cout << "  normalized level=" << u.level() << "\n";

    // Indicator target: rank ∈ [0.5, 1.5] maps to u ∈ [-1, (2-N)/N]
    double u_high = (2.0 - N) / N;
    auto fn = [u_high](Complex64 x) -> Complex64 {
        double t = x.real();
        return {(t < u_high) ? 1.0 : 0.0, 0.0};
    };
    auto coeffs = heongpu::approximate_function(fn, -1.0, 1.0, degree);
    return pe.eval_chebyshev(u, scale, coeffs, degree, rk);
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
// homomorphicMin — Algorithm 4 (Order Statistic) for k=1 (minimum)
// Uses Chebyshev comparison + Chebyshev indicator (paper-spec for N<=256)
// ---------------------------------------------------------------------------
heongpu::Ciphertext<Scheme>
homomorphicMin(const heongpu::Ciphertext<Scheme>& ct_vector, int N,
               int degreeC, int degreeI,
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
    // Phase 1: comparison matrix with strict > (compareGt with 0.005 bias)
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

    if (g_verbose) std::cout << "Step 3: compareGtChebyshev(VR, VC)\n";
    heongpu::Ciphertext<Scheme> C =
        compareGtChebyshev(VR, VC, degreeC, pe, rk, ctx, scale);

    if (g_verbose)
        std::cout << "  C level=" << C.level() << "\n";

    // Phase 2: rank = sumRows(C) + 1
    if (g_verbose) std::cout << "\n=== Phase 2: rank via sumRows ===\n";
    heongpu::Ciphertext<Scheme> R = sumRows(C, N, sumr_key, pe);
    pe.add_plain_inplace(R, 1.0);

    if (g_verbose)
        std::cout << "  R level=" << R.level() << "\n";

    // Phase 3: Chebyshev indicator detects rank ≈ 1 (minimum)
    if (g_verbose) std::cout << "\n=== Phase 3: indicator (detect rank=1) ===\n";
    heongpu::Ciphertext<Scheme> mask =
        indicatorChebyshev(R, N, degreeI, pe, rk, scale);

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

    cudaSetDevice(0);

    // Chebyshev degrees matching the paper's test-minimum.cpp:
    //   depth2degree(7)=59, depth2degree(9)=247
    const int degreeC = (N <= 32) ? 59 : 247;
    const int degreeI = 59;

    // HEonGPU levels: ceil(log2(degree)) per Chebyshev eval
    const int compare_levels   = static_cast<int>(std::ceil(std::log2(degreeC)));
    const int indicator_levels = static_cast<int>(std::ceil(std::log2(degreeI)));
    const int actual_depth = 1 + compare_levels + 1 + indicator_levels;
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
        std::cout << "Paper-exact minimum (Algorithm 4): n=131072\n";
        std::cout << "Chebyshev: degreeC=" << degreeC << " (" << compare_levels
                  << " levels), degreeI=" << degreeI << " (" << indicator_levels
                  << " levels)\n";
        std::cout << "depth=" << actual_depth
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

    heongpu::Ciphertext<Scheme> ct_mask =
        homomorphicMin(ct, N, degreeC, degreeI,
                       row_key, col_key, sumr_key, transr_key,
                       rk, pe, enc, ctx, scale);

    float min_ms = min_timer.stopTimer();
    size_t gpu_min_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // Decrypt mask and extract minimum
    heongpu::Plaintext<Scheme> pt_result(ctx);
    decryptor.decrypt(pt_result, ct_mask);
    std::vector<double> raw;
    enc.decode(raw, pt_result);

    double he_min = 0, mask_sum = 0;
    int min_idx = 0;
    double max_mask = -1;
    for (int i = 0; i < N; i++)
    {
        double m = raw[i];
        he_min += m * input[i];
        mask_sum += m;
        if (m > max_mask) { max_mask = m; min_idx = i; }
    }
    he_min = (mask_sum > 0.01) ? he_min / mask_sum : input[min_idx];

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
        double lo      = *std::min_element(input.begin(), input.end());
        double hi      = *std::max_element(input.begin(), input.end());
        double range   = hi - lo;
        bool   correct = err < (0.5 * range / N + 0.5);

        std::cout << "\nMask values: [";
        for (int i = 0; i < N; i++)
            std::cout << std::fixed << std::setprecision(4) << raw[i]
                      << (i < N-1 ? ", " : "");
        std::cout << "]\n";

        std::cout << "\n=== Minimum Result ===\n";
        std::cout << "Expected min : " << expected_min << "\n";
        std::cout << "HE min       : " << std::fixed << std::setprecision(4)
                  << he_min << "  (mask argmax at idx " << min_idx
                  << ", v[" << min_idx << "]=" << input[min_idx] << ")"
                  << (correct ? "" : "  INCORRECT") << "\n";
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
