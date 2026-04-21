#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>

// Global verbose flag: false in --bench mode for clean parseable output
static bool g_verbose = true;

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;

/**
 * @brief Thin wrapper around HEArithmeticOperator that promotes the protected
 *        evaluate_poly method to public, enabling BSGS Chebyshev polynomial
 *        evaluation from user code.
 *
 * HEOperator<Scheme::CKKS>::evaluate_poly uses baby-step/giant-step internally
 * and matches the algorithm described in the target paper. By deriving from
 * HEArithmeticOperator (which already inherits evaluate_poly as protected),
 * this class exposes it as a single public call without re-implementing BSGS.
 */
class CKKSPolyEvaluator : public heongpu::HEArithmeticOperator<Scheme>
{
  public:
    CKKSPolyEvaluator(heongpu::HEContext<Scheme> ctx,
                      heongpu::HEEncoder<Scheme>& enc)
        : heongpu::HEArithmeticOperator<Scheme>(ctx, enc)
    {
    }

    /**
     * @brief Evaluate a Chebyshev polynomial on a ciphertext using BSGS.
     *
     * Both Polynomial and evaluate_poly are protected in HEOperator, so they
     * can only be accessed from within a derived class. This method bridges
     * user-supplied coefficients (from approximate_function) into the internal
     * BSGS evaluation without exposing the protected types.
     *
     * @param ct           Input ciphertext (values in [a, b])
     * @param target_scale Desired output scale
     * @param coeffs       Chebyshev coefficients c[0..degree]
     * @param degree       Polynomial degree
     * @param relin_key    Relinearization key
     * @param a            Interval lower bound (default -1)
     * @param b            Interval upper bound (default  1)
     */
    heongpu::Ciphertext<Scheme>
    eval_chebyshev(heongpu::Ciphertext<Scheme>& ct, double target_scale,
                   const std::vector<Complex64>& coeffs, int degree,
                   heongpu::Relinkey<Scheme>& relin_key, double a = -1.0,
                   double b = 1.0)
    {
        // Polynomial is protected in HEOperator — constructible only here
        Polynomial poly(degree, coeffs, /*lead=*/false,
                        heongpu::PolyType::CHEBYSHEV, a, b);
        if (g_verbose)
            std::cout << "  Chebyshev poly degree=" << degree
                      << " depth=" << poly.depth() << " levels\n";
        return evaluate_poly(ct, target_scale, poly, relin_key,
                             heongpu::ExecutionOptions());
    }
};

// Forward declarations
heongpu::Ciphertext<Scheme> replicateRow(
    const heongpu::Ciphertext<Scheme>& row_initial, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator,
    heongpu::HEContext<Scheme>& context);

heongpu::Ciphertext<Scheme> replicateColumn(
    const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator,
    heongpu::HEContext<Scheme>& context);

heongpu::Ciphertext<Scheme> transposeRowToColumn(
    const heongpu::Ciphertext<Scheme>& row_vector, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEContext<Scheme>& context,
    double scale);

heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree = 2047);

heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& sumr_galois_key,
        CKKSPolyEvaluator& evaluator,
        heongpu::HEContext<Scheme>& context);

// Matches the paper's compareDepth table (test-ranking.cpp, no tie correction):
//   N<=8:   compareDepth=7  -> degree=127   (min gap ~ 1/7)
//   N<=16:  compareDepth=8  -> degree=255   (min gap ~ 1/15)
//   N<=64:  compareDepth=10 -> degree=1023  (min gap ~ 1/63)
//   N<=128: compareDepth=11 -> degree=2047  (min gap ~ 1/127)
int selectChebyshevDegree(int N)
{
    if (N <= 8)  return 127;
    if (N <= 16) return 255;
    if (N <= 64) return 1023;
    return 2047;
}

std::vector<double> normalizeForRanking(const std::vector<double>& input)
{
    double lo    = *std::min_element(input.begin(), input.end());
    double hi    = *std::max_element(input.begin(), input.end());
    double range = hi - lo;
    std::vector<double> normalized(input.size());
    for (size_t i = 0; i < input.size(); i++)
        normalized[i] = (input[i] - lo) / range;
    return normalized;
}

heongpu::Ciphertext<Scheme>
basicRank(const heongpu::Ciphertext<Scheme>& ct_vector, int vec_len,
          heongpu::Galoiskey<Scheme>& row_galois_key,
          heongpu::Galoiskey<Scheme>& col_galois_key,
          heongpu::Galoiskey<Scheme>& transpose_galois_key,
          heongpu::Galoiskey<Scheme>& sumr_galois_key,
          heongpu::Relinkey<Scheme>& relin_key,
          CKKSPolyEvaluator& evaluator,
          heongpu::HEEncoder<Scheme>& encoder,
          heongpu::HEContext<Scheme>& context, double scale);

class GPUTimer
{
    cudaEvent_t start, stop;

  public:
    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() { cudaEventRecord(start); }

    float stopTimer()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
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

std::vector<int> transposeGaloisShifts(int vec_len)
{
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    std::vector<int> shifts;
    for (int i = 1; i <= log_n; i++)
        shifts.push_back(-((vec_len * (vec_len - 1)) / (1 << i)));
    return shifts;
}

int main(int argc, char* argv[])
{
    // ----- Parse command line -----
    // Usage: 17_ckks_ranking_paper [N] [--bench]
    //   N       : vector length, must be power of 2 (default: 64)
    //   --bench : suppress verbose output, emit single BENCH: line for scripts
    int vec_len   = 64;
    bool bench_mode = false;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--bench")
        {
            bench_mode = true;
        }
        else if (!arg.empty() &&
                 std::isdigit(static_cast<unsigned char>(arg[0])))
        {
            vec_len = std::stoi(arg);
        }
    }
    g_verbose = !bench_mode;

    if (vec_len <= 0 || (vec_len & (vec_len - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << vec_len
                  << ")\n";
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);

    // ===== HE Context (paper-matching parameters) =====
    //
    // The paper (Mazzone et al., USENIX Security 2025) uses OpenFHE with:
    //   integralPrecision = 1, decimalPrecision = 35 (30 for N<=16)
    //   FLEXIBLEAUTO scaling, HEStd_128_classic, HYBRID key switching
    //
    // HEonGPU uses fixed-scale CKKS (no FLEXIBLEAUTO). For N<=32 the paper's
    // exact 35-bit scaling primes produce correct results. For N>=64 the deeper
    // Chebyshev evaluation (degree>=1023, 10+ levels) accumulates too much
    // noise at 35-bit precision in fixed-scale mode, so we use 40-bit primes.
    //
    // HEonGPU also needs 2 extra levels beyond the paper's multiplicativeDepth
    // (mod_drop for level alignment + explicit x0.5 normalization):
    //   Paper depth: compareDepth + 1  (max: 11+1 = 12 for N=128)
    //   HEonGPU depth: compareDepth + 3  (max: 11+3 = 14 for N=128)
    //
    // N<=32 (paper-exact):
    //   Q = {36, 35x14} = 15 primes, 526 bits, scale=2^35
    //   P = {36x8}       = 8 primes,  288 bits, dnum=2
    //   Q+P = 814 < 881 = heongpu_128bit_std_parms(32768)
    //
    // N>=64 (precision-adjusted):
    //   Q = {60, 45x14} = 15 primes, 690 bits, scale=2^45
    //   P = {60x3}       = 3 primes,  180 bits, dnum=5
    //   Q+P = 870 < 881 = heongpu_128bit_std_parms(32768)
    //   Fixed-scale CKKS needs higher per-level precision than FLEXIBLEAUTO.
    //   45-bit scaling primes give enough headroom for degree-2047 Chebyshev
    //   (11 levels) while fitting within the n=32768 security budget.
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    const size_t poly_modulus_degree = 32768;
    context->set_poly_modulus_degree(poly_modulus_degree);

    int scale_bits;
    if (vec_len <= 32)
    {
        context->set_coeff_modulus_bit_sizes(
            {36, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
            {36, 36, 36, 36, 36, 36, 36, 36});
        scale_bits = 35;
    }
    else
    {
        context->set_coeff_modulus_bit_sizes(
            {60, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
            {60, 60, 60});
        scale_bits = 45;
    }
    double scale = pow(2.0, scale_bits);
    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    context->generate();
    float ctx_ms = ctx_timer.stopTimer();

    int available_slots = static_cast<int>(poly_modulus_degree / 2);
    if (vec_len * vec_len > available_slots)
    {
        std::cerr << "Error: N=" << vec_len << " needs " << (vec_len * vec_len)
                  << " slots but only " << available_slots << " available.\n"
                  << "Supported: N <= 128 with current poly_modulus_degree.\n";
        return EXIT_FAILURE;
    }

    if (g_verbose)
    {
        std::cout << "N=" << vec_len << "  matrix=" << vec_len << "x"
                  << vec_len << "  slots=" << (vec_len * vec_len) << "/"
                  << available_slots << "\n";
        if (vec_len <= 32)
            std::cout << "Paper-exact params: Q={36,35x14}, P={36x8}, "
                      << "scale=2^35, dnum=2\n";
        else
            std::cout << "Precision-adjusted params: Q={60,45x14}, P={60x3}, "
                      << "scale=2^45, dnum=5\n";
    }

    // ===== Key material =====
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    CKKSPolyEvaluator evaluator(context, encoder);

    // Galois key shifts (identical to 16_ckks_ranking.cpp)
    std::vector<int> row_galois_shifts;
    for (int i = vec_len / 2; i > 0; i /= 2)
        row_galois_shifts.push_back(-(i * vec_len));

    std::vector<int> col_galois_shifts;
    for (int i = 1; i < vec_len; i *= 2)
        col_galois_shifts.push_back(-i);

    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    std::vector<int> sumr_galois_shifts;
    for (int i = 0; i < log_n; i++)
        sumr_galois_shifts.push_back(vec_len * (1 << i));

    std::vector<int> transpose_shifts = transposeGaloisShifts(vec_len);

    if (g_verbose)
    {
        std::cout << "Galois shifts — row: " << row_galois_shifts.size()
                  << "  col: " << col_galois_shifts.size()
                  << "  sumr: " << sumr_galois_shifts.size()
                  << "  transpose: " << transpose_shifts.size() << "\n";
    }

    // ===== KEY GENERATION (timed) =====
    const size_t kMiB = 1024ULL * 1024ULL;
    size_t gpu_baseline_bytes = heongpu::MemoryPool::instance().get_current_device_pool_memory_usage();
    size_t gpu_baseline_mib   = gpu_baseline_bytes / kMiB;
    GPUTimer keygen_timer;
    keygen_timer.startTimer();

    heongpu::Galoiskey<Scheme> row_galois_key(context, row_galois_shifts);
    keygen.generate_galois_key(row_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> col_galois_key(context, col_galois_shifts);
    keygen.generate_galois_key(col_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> sumr_galois_key(context, sumr_galois_shifts);
    keygen.generate_galois_key(sumr_galois_key, secret_key);

    heongpu::Galoiskey<Scheme> transpose_galois_key(context, transpose_shifts);
    keygen.generate_galois_key(transpose_galois_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    float keygen_ms = keygen_timer.stopTimer();
    size_t gpu_keys_mib = getGPUUsedMiB() - gpu_baseline_mib;
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms << " ms  (keys: "
                  << gpu_keys_mib << " MiB VRAM)\n";

    // ===== Input preparation =====
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
            input[i] = static_cast<double>(i);
    }

    std::vector<double> normalized_input = normalizeForRanking(input);

    if (g_verbose)
    {
        std::cout << "Original input: ";
        display_vector(input, vec_len);
        std::cout << "Normalized:     ";
        display_vector(normalized_input, vec_len);
    }

    std::vector<double> row_initial(available_slots, 0.0);
    for (int i = 0; i < vec_len; i++)
        row_initial[i] = normalized_input[i];

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, row_initial, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    // ===== RANKING (timed) =====
    GPUTimer rank_timer;
    rank_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_rank =
        basicRank(ciphertext, vec_len, row_galois_key, col_galois_key,
                  transpose_galois_key, sumr_galois_key, relin_key, evaluator,
                  encoder, context, scale);

    float rank_ms = rank_timer.stopTimer();
    size_t gpu_rank_mib = (heongpu::MemoryPool::instance().get_current_device_pool_memory_usage()
                           - gpu_baseline_bytes) / kMiB;
    size_t gpu_peak_mib = getPeakGPUMiB();

    // ===== Decrypt and decode =====
    heongpu::Plaintext<Scheme> rank_plaintext(context);
    decryptor.decrypt(rank_plaintext, ct_rank);
    std::vector<double> rank_result;
    encoder.decode(rank_result, rank_plaintext);

    // ===== Output =====
    if (bench_mode)
    {
        std::cout << "BENCH:"
                  << " N=" << vec_len << " ctx_ms=" << ctx_ms
                  << " keygen_ms=" << keygen_ms
                  << " rank_ms=" << rank_ms
                  << " gpu_keys_mib=" << gpu_keys_mib
                  << " gpu_rank_mib=" << gpu_rank_mib
                  << " gpu_peak_mib=" << gpu_peak_mib << "\n";
    }
    else
    {
        std::cout << "\n=== Ranking Results (paper-matching params) ===\n";
        std::cout << "Input vector:  ";
        display_vector(input, vec_len);

        std::cout << "Rank (1-based fractional):\n";
        for (int i = 0; i < vec_len; i++)
        {
            double decoded_rank = rank_result[i];
            std::cout << "  input[" << i << "] = " << input[i]
                      << " -> rank = " << decoded_rank << "\n";
        }

        std::cout << "\nVerification (expected rank = index + 1):\n";
        bool all_correct = true;
        for (int i = 0; i < vec_len; i++)
        {
            double expected_rank = static_cast<double>(i + 1);
            double actual_rank   = rank_result[i];
            double error         = std::abs(actual_rank - expected_rank);
            bool is_correct      = (error < 0.5);
            std::cout << "  Element " << (i + 1) << ": expected=" << expected_rank
                      << ", actual=" << actual_rank
                      << (is_correct ? "" : " INCORRECT") << "\n";
            if (!is_correct)
                all_correct = false;
        }
        std::cout << (all_correct ? "\nAll ranking results correct!\n"
                                  : "\nSome ranking results are incorrect!\n");

        std::cout << "\nTiming summary:\n";
        std::cout << "  Key generation : " << keygen_ms << " ms\n";
        std::cout << "  Ranking        : " << rank_ms << " ms  ("
                  << (rank_ms / 1000.0) << " s)\n";
        std::cout << "\nVRAM usage (above context baseline):\n";
        std::cout << "  Keys           : " << gpu_keys_mib << " MiB\n";
        std::cout << "  After ranking  : " << gpu_rank_mib << " MiB\n";
        std::cout << "  Peak           : " << gpu_peak_mib << " MiB\n";
    }

    return EXIT_SUCCESS;
}

// ===== Algorithm implementations (identical to 16_ckks_ranking.cpp) =====

heongpu::Ciphertext<Scheme>
replicateRow(const heongpu::Ciphertext<Scheme>& row_initial, int vec_len,
             heongpu::Galoiskey<Scheme>& galois_key,
             CKKSPolyEvaluator& evaluator,
             heongpu::HEContext<Scheme>& context)
{
    heongpu::Ciphertext<Scheme> row_replicated = row_initial;

    if (g_verbose)
        std::cout << "  ReplR shifts: ";
    for (int i = vec_len / 2; i > 0; i = i / 2)
    {
        int shift = -(i * vec_len);
        if (g_verbose)
            std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated(context);
        evaluator.rotate_rows(row_replicated, rotated, galois_key, shift);
        evaluator.add_inplace(row_replicated, rotated);
    }
    if (g_verbose)
        std::cout << "\n";
    return row_replicated;
}

heongpu::Ciphertext<Scheme>
replicateColumn(const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
                heongpu::Galoiskey<Scheme>& galois_key,
                CKKSPolyEvaluator& evaluator,
                heongpu::HEContext<Scheme>& context)
{
    heongpu::Ciphertext<Scheme> col_replicated = col_initial;

    if (g_verbose)
        std::cout << "  ReplC shifts: ";
    for (int i = 1; i < vec_len; i *= 2)
    {
        int shift = -i;
        if (g_verbose)
            std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated(context);
        evaluator.rotate_rows(col_replicated, rotated, galois_key, shift);
        evaluator.add_inplace(col_replicated, rotated);
    }
    if (g_verbose)
        std::cout << "\n";
    return col_replicated;
}

heongpu::Ciphertext<Scheme>
transposeRowToColumn(const heongpu::Ciphertext<Scheme>& row_vector,
                     int vec_len,
                     heongpu::Galoiskey<Scheme>& galois_key,
                     CKKSPolyEvaluator& evaluator,
                     heongpu::HEEncoder<Scheme>& encoder,
                     heongpu::HEContext<Scheme>& context, double scale)
{
    heongpu::Ciphertext<Scheme> result = row_vector;

    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    int N     = vec_len;

    if (g_verbose)
        std::cout << "  TransR (log N=" << log_n << ") shifts: ";
    for (int i = 1; i <= log_n; i++)
    {
        int shift = -((N * (N - 1)) / (1 << i));
        if (g_verbose)
            std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated(context);
        evaluator.rotate_rows(result, rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose)
        std::cout << "\n";

    size_t total_slots = context->get_poly_modulus_degree() / 2;
    std::vector<double> mask_values(total_slots, 0.0);
    for (int row = 0; row < vec_len; row++)
        mask_values[row * vec_len] = 1.0;

    heongpu::Plaintext<Scheme> mask(context);
    encoder.encode(mask, mask_values, scale);
    evaluator.multiply_plain_inplace(result, mask);
    evaluator.rescale_inplace(result);
    return result;
}

heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree)
{
    if (g_verbose)
        std::cout << "  Chebyshev sign approx degree=" << degree << "...\n";

    auto sign_func = [](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(re > 0.0 ? 1.0 : (re < 0.0 ? -1.0 : 0.0), 0.0);
    };

    std::vector<Complex64> cheby_coeffs =
        heongpu::approximate_function(sign_func, -1.0, 1.0, degree);

    return poly_eval.eval_chebyshev(ct_diff, scale, cheby_coeffs, degree,
                                    relin_key, -1.0, 1.0);
}

heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& sumr_galois_key,
        CKKSPolyEvaluator& evaluator,
        heongpu::HEContext<Scheme>& context)
{
    heongpu::Ciphertext<Scheme> result = ct_matrix;

    if (g_verbose)
        std::cout << "  SumR (vec_len=" << vec_len << ") row-fold shifts: ";

    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    for (int i = 0; i < log_n; i++)
    {
        int shift = vec_len * (1 << i);
        if (g_verbose)
            std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated(context);
        evaluator.rotate_rows(result, rotated, sumr_galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }

    if (g_verbose)
        std::cout << "\n";
    return result;
}

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
    {
        std::cout << "\n=== Basic Ranking (N=" << vec_len << ") ===\n";
        std::cout << "Step 1: ReplR...\n";
    }

    heongpu::Ciphertext<Scheme> ct_row =
        replicateRow(ct_vector, vec_len, row_galois_key, evaluator, context);

    if (g_verbose)
        std::cout << "Step 2: TransR + ReplC...\n";
    heongpu::Ciphertext<Scheme> ct_col_transposed =
        transposeRowToColumn(ct_vector, vec_len, transpose_galois_key,
                             evaluator, encoder, context, scale);
    heongpu::Ciphertext<Scheme> ct_col =
        replicateColumn(ct_col_transposed, vec_len, col_galois_key, evaluator, context);

    evaluator.mod_drop_inplace(ct_row);

    if (g_verbose)
        std::cout << "Step 3: Compute differences (vR - vC)...\n";
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_row, ct_col, ct_diff);

    const int cheby_degree = selectChebyshevDegree(vec_len);
    if (g_verbose)
        std::cout << "Step 4: Chebyshev sign approx (N=" << vec_len
                  << " -> degree=" << cheby_degree << ")...\n";
    heongpu::Ciphertext<Scheme> ct_sign = chebyshev_sign_approx(
        ct_diff, evaluator, relin_key, scale, cheby_degree);

    if (g_verbose)
        std::cout << "Step 5: Add 1 to shift sign range...\n";
    evaluator.add_plain_inplace(ct_sign, 1.0);

    if (g_verbose)
        std::cout << "Step 6: SumR (row-folding)...\n";
    heongpu::Ciphertext<Scheme> ct_sumr =
        sumRows(ct_sign, vec_len, sumr_galois_key, evaluator, context);

    if (g_verbose)
        std::cout << "Step 7: Scale by 0.5...\n";
    evaluator.multiply_plain_inplace(ct_sumr, 0.5, scale);
    evaluator.rescale_inplace(ct_sumr);

    if (g_verbose)
        std::cout << "Step 8: Add 0.5 to complete fractional rank...\n";
    evaluator.add_plain_inplace(ct_sumr, 0.5);

    if (g_verbose)
        std::cout << "Ranking complete.\n";
    return ct_sumr;
}
