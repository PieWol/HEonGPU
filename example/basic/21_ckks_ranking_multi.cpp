/**
 * @file 21_ckks_ranking_multi.cpp
 *
 * Multi-ciphertext homomorphic ranking for N > 128, following:
 *   "Efficient Ranking, Order Statistics, and Sorting under CKKS"
 *   Mazzone et al., USENIX Security 2025  (rankFG, complOpt=true)
 *
 * Layout: subVectorLength L=128. Each ciphertext holds L×L=16384 slots.
 * numCiphertext M = N/L. N must be a multiple of 128 and a power of 2.
 *
 * Algorithm (complementary optimization):
 *   Phase 1: for each block j, compute replR[j] and replC[j]
 *   Phase 2: upper-triangle pairs (j,k), j≤k:
 *     Cjk = compare(replR[j], replC[k])  ∈ [0,1]
 *     Cv[j] += Cjk
 *     if j≠k: Ch[k] += (1 − Cjk)
 *   Phase 3:
 *     sv[j] = sumRows(Cv[j])
 *     sh[j] = transposeColumn(sumColumns(Ch[j]))  for j>0
 *     s[j]  = sv[j] + sh[j] + 0.5
 *
 * Output: s[j] row-0, positions 0..L-1 hold fractional ranks for
 *         input elements j*L..(j+1)*L-1.
 *
 * Depth budget (14 levels, same context as 16_ckks_ranking):
 *   1  TransR
 *   12 chebyshev_sign_approx degree=2047
 *   0  sumRows / sumColumns (rotations+adds)
 *   1  maskColumn0 (multiply_plain + rescale, Ch path only)
 *   0  transposeColumn (rotations+adds)
 *   Total: 14 = 14 ✓  (normalize deferred to post-decryption)
 *
 * Usage: 21_ckks_ranking_multi [N] [--bench]
 *   N must be a multiple of 128 and a power of 2 (256, 512, ...)
 */

#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

static bool g_verbose = true;
constexpr auto Scheme = heongpu::Scheme::CKKS;

// Fixed block size — 128² = 16384 slots fits in n=32768 (16384 slots)
constexpr int SUB_VECTOR_LENGTH = 128;

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
                   double a = -1.0, double b = 1.0)
    {
        Polynomial poly(degree, coeffs, /*lead=*/false,
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
// Output: ≈+1 where ct > 0,  ≈-1 where ct < 0  (depth: +12 from Chebyshev)
// Normalization to [0,1] is deferred to after decryption to save 1 level
// for maskColumn0 in Phase 3.
// ---------------------------------------------------------------------------
static heongpu::Ciphertext<Scheme>
compareUnit(heongpu::Ciphertext<Scheme>& ct_diff,
            CKKSPolyEvaluator& pe, heongpu::Relinkey<Scheme>& rk,
            double scale)
{
    constexpr int degree = 2047;

    auto sign_fn = [](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(re > 0.0 ? 1.0 : (re < 0.0 ? -1.0 : 0.0), 0.0);
    };
    std::vector<Complex64> coeffs =
        heongpu::approximate_function(sign_fn, -1.0, 1.0, degree);

    return pe.eval_chebyshev(ct_diff, scale, coeffs, degree, rk);
}

// ---------------------------------------------------------------------------
// Multi-ciphertext ranking (Algorithm 2 from paper, complementary opt.)
// ---------------------------------------------------------------------------
std::vector<heongpu::Ciphertext<Scheme>>
multiCiphertextRank(
    const std::vector<heongpu::Ciphertext<Scheme>>& blocks,
    int L,
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

    // Pre-allocate accumulators; Cv_init/Ch_init track first-write per slot
    std::vector<heongpu::Ciphertext<Scheme>> Cv(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<heongpu::Ciphertext<Scheme>> Ch(M, heongpu::Ciphertext<Scheme>(ctx));
    std::vector<bool> Cv_init(M, false), Ch_init(M, false);

    for (int j = 0; j < M; j++)
    {
        for (int k = j; k < M; k++)
        {
            if (g_verbose) std::cout << "  Compare (" << j << "," << k << ")\n";

            // Align depths: replR[j] starts at depth 0, replC[k] at depth 1
            // (TransR consumed 1 level). Mod-drop replR to match.
            heongpu::Ciphertext<Scheme> rj = replR[j];
            while (rj.level() > replC[k].level())
            {
                heongpu::Ciphertext<Scheme> tmp(ctx);
                pe.mod_drop(rj, tmp);
                rj = std::move(tmp);
            }

            heongpu::Ciphertext<Scheme> diff(ctx);
            pe.sub(rj, replC[k], diff);

            heongpu::Ciphertext<Scheme> Cjk = compareUnit(diff, pe, rk, scale);

            // Accumulate vertical: Cv[j] += Cjk
            if (!Cv_init[j]) { Cv[j] = Cjk;                Cv_init[j] = true; }
            else              { pe.add_inplace(Cv[j], Cjk);                    }

            // Complementary: Ch[k] += -sign_jk (raw sign, no normalize)
            if (j != k)
            {
                heongpu::Ciphertext<Scheme> Ckj = Cjk;
                pe.negate_inplace(Ckj);

                if (!Ch_init[k]) { Ch[k] = Ckj;                Ch_init[k] = true; }
                else             { pe.add_inplace(Ch[k], Ckj);                    }
            }
        }
    }

    // ── Phase 3: sum ────────────────────────────────────────────────────────
    if (g_verbose) std::cout << "\n=== Phase 3: Sum ===\n";

    std::vector<heongpu::Ciphertext<Scheme>> result(M, heongpu::Ciphertext<Scheme>(ctx));

    for (int j = 0; j < M; j++)
    {
        // sv[j]: sumRows folds all L rows into row 0
        heongpu::Ciphertext<Scheme> sv = sumRows(Cv[j], L, sumr_key, pe, ctx);
        result[j] = sv;

        // sh[j]: horizontal contributions from blocks 0..j-1
        if (j > 0 && Ch_init[j])
        {
            if (g_verbose) std::cout << "  Block " << j << ": sumColumns + maskCol0 + transposeColumn\n";
            heongpu::Ciphertext<Scheme> sh = sumColumns(Ch[j], L, sumc_key, pe, ctx);
            sh = maskColumn0(sh, L, pe, enc, ctx, scale);
            sh = transposeColumn(sh, L, transpc_key, pe, ctx);

            // sh is now 1 level below sv; mod_drop sv to match
            while (result[j].level() > sh.level())
            {
                heongpu::Ciphertext<Scheme> tmp(ctx);
                pe.mod_drop(result[j], tmp);
                result[j] = std::move(tmp);
            }
            pe.add_inplace(result[j], sh);
        }

        // No in-HE normalize; conversion done after decryption
    }

    return result;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int  N          = 256;
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
    if (N % SUB_VECTOR_LENGTH != 0)
    {
        std::cerr << "Error: N=" << N << " must be a multiple of " << SUB_VECTOR_LENGTH << "\n";
        return EXIT_FAILURE;
    }
    if (N <= SUB_VECTOR_LENGTH)
    {
        std::cerr << "Error: N=" << N << " ≤ " << SUB_VECTOR_LENGTH
                  << "; use 16_ckks_ranking for single-ciphertext mode.\n";
        return EXIT_FAILURE;
    }

    const int M = N / SUB_VECTOR_LENGTH; // numCiphertext

    cudaSetDevice(0);

    // ── HE context ──────────────────────────────────────────────────────────
    // n=32768 → 16384 slots = 128² → 1 ciphertext per 128-element block
    // Q={60, 40×14}, P={60×4} — same as 16_ckks_ranking
    heongpu::HEContext<Scheme> ctx = heongpu::GenHEContext<Scheme>();
    ctx->set_poly_modulus_degree(32768);
    ctx->set_coeff_modulus_bit_sizes(
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
        {60, 60, 60, 60});
    double scale = std::pow(2.0, 40);

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    ctx->generate();
    float ctx_ms = ctx_timer.stopTimer();

    if (g_verbose)
        std::cout << "N=" << N << "  M=" << M << " ciphertexts"
                  << "  subVectorLength=" << SUB_VECTOR_LENGTH << "\n";

    // ── Key generation ───────────────────────────────────────────────────────
    heongpu::HEKeyGenerator<Scheme> keygen(ctx);
    heongpu::Secretkey<Scheme>  sk(ctx);  keygen.generate_secret_key(sk);
    heongpu::Publickey<Scheme>  pk(ctx);  keygen.generate_public_key(pk, sk);

    heongpu::HEEncoder<Scheme>    enc(ctx);
    heongpu::HEEncryptor<Scheme>  encryptor(ctx, pk);
    heongpu::HEDecryptor<Scheme>  decryptor(ctx, sk);
    CKKSPolyEvaluator             pe(ctx, enc);

    const int L = SUB_VECTOR_LENGTH;
    const size_t kMiB = 1024ULL * 1024ULL;
    size_t gpu_baseline =
        heongpu::MemoryPool::instance().get_current_device_pool_memory_usage();

    GPUTimer keygen_timer;
    keygen_timer.startTimer();

    // Galoiskey constructor requires non-const lvalue ref; store shifts first
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
        // Sorted input: rank[i] should equal i+1
        for (int i = 0; i < N; i++) input[i] = static_cast<double>(i);
    }

    std::vector<double> norm = normalizeForRanking(input);

    if (g_verbose)
    {
        std::cout << "Input (first 8): ";
        display_vector(input, std::min(N, 8));
    }

    // Split normalized input into M blocks of L elements each
    const int slots = static_cast<int>(ctx->get_poly_modulus_degree() / 2); // 16384
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

    std::vector<heongpu::Ciphertext<Scheme>> ct_ranks =
        multiCiphertextRank(blocks, L,
                            row_key, col_key, transr_key, sumr_key,
                            sumc_key, transpc_key, rk, pe, enc, ctx, scale);

    float rank_ms = rank_timer.stopTimer();
    size_t gpu_rank_mib =
        (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
         - gpu_baseline) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // ── Decrypt & convert raw sign sums to ranks ──────────────────────────────
    // compareUnit returns sign ∈ [-1,+1] (no in-HE normalize).
    // raw = Σ sign(x_i − x_j) = (rank−1) − (N−rank) = 2*rank − N − 1
    // ⟹ rank = (raw + N + 1) / 2
    std::vector<double> all_ranks(N);
    for (int j = 0; j < M; j++)
    {
        heongpu::Plaintext<Scheme> pt(ctx);
        decryptor.decrypt(pt, ct_ranks[j]);
        std::vector<double> raw;
        enc.decode(raw, pt);
        for (int i = 0; i < L; i++)
            all_ranks[j * L + i] = (raw[i] + N + 1) / 2.0;
    }

    // ── Output ───────────────────────────────────────────────────────────────
    if (bench_mode)
    {
        std::cout << "BENCH:"
                  << " N="            << N
                  << " ctx_ms="       << ctx_ms
                  << " keygen_ms="    << keygen_ms
                  << " rank_ms="      << rank_ms
                  << " gpu_keys_mib=" << gpu_keys_mib
                  << " gpu_rank_mib=" << gpu_rank_mib
                  << " gpu_peak_mib=" << gpu_peak_mib << "\n";
    }
    else
    {
        std::cout << "\n=== Multi-Ciphertext Ranking Results ===\n";
        std::cout << "Input (first 8): ";
        display_vector(input, std::min(N, 8));
        std::cout << "Ranks (first 8): ";
        display_vector(all_ranks, std::min(N, 8));

        std::cout << "\nVerification (expected rank[i] = i+1 for sorted input):\n";
        double max_err = 0.0;
        // Show first 8, block boundary (L-2..L+1), and last 4
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
        // deduplicate and sort
        std::sort(show_indices.begin(), show_indices.end());
        show_indices.erase(std::unique(show_indices.begin(), show_indices.end()),
                           show_indices.end());
        int prev = -1;
        for (int i : show_indices)
        {
            if (prev >= 0 && i > prev + 1) std::cout << "  ...\n";
            double expected = static_cast<double>(i + 1);
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
