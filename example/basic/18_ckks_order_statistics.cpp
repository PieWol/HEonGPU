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
        Polynomial poly(degree, coeffs, /*lead=*/false,
                        heongpu::PolyType::CHEBYSHEV, a, b);
        if (g_verbose)
            std::cout << "  Chebyshev poly degree=" << degree
                      << " depth=" << poly.depth() << " levels\n";
        return evaluate_poly(ct, target_scale, poly, relin_key,
                             heongpu::ExecutionOptions());
    }
};

// ===== Forward declarations =====

heongpu::Ciphertext<Scheme> replicateRow(
    const heongpu::Ciphertext<Scheme>& row_initial, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator);

heongpu::Ciphertext<Scheme> replicateColumn(
    const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator);

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
        CKKSPolyEvaluator& evaluator);

heongpu::Ciphertext<Scheme>
sumFirstRow(const heongpu::Ciphertext<Scheme>& ct_vec, int vec_len,
            heongpu::Galoiskey<Scheme>& sumc_galois_key,
            CKKSPolyEvaluator& evaluator);

heongpu::Ciphertext<Scheme>
chebyshev_indicator_approx(heongpu::Ciphertext<Scheme>& ct_rank,
                           int k, int N,
                           CKKSPolyEvaluator& poly_eval,
                           heongpu::Relinkey<Scheme>& relin_key,
                           double scale, int degree = 2047);

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

heongpu::Ciphertext<Scheme>
orderStatistic(const heongpu::Ciphertext<Scheme>& ct_vector,
               int k, int vec_len,
               heongpu::Galoiskey<Scheme>& row_galois_key,
               heongpu::Galoiskey<Scheme>& col_galois_key,
               heongpu::Galoiskey<Scheme>& transpose_galois_key,
               heongpu::Galoiskey<Scheme>& sumr_galois_key,
               heongpu::Galoiskey<Scheme>& sumc_galois_key,
               heongpu::Relinkey<Scheme>& relin_key,
               CKKSPolyEvaluator& evaluator,
               heongpu::HEEncoder<Scheme>& encoder,
               heongpu::HEContext<Scheme>& context,
               double scale, int indicator_degree = 2047);

/**
 * @brief Normalize a plaintext vector to [0,1] before encryption.
 *
 * All pairwise differences (v[k] - v[j]) must lie in [-1,1] for the
 * Chebyshev sign approximation domain. Normalizing to [0,1] guarantees
 * max |v[k]-v[j]| ≤ 1.
 */
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

/**
 * @brief GPU-aware timer using CUDA Events for accurate GPU timing.
 */
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

/**
 * @brief Compute required transpose Galois shifts for TransR.
 *
 * Shifts = -(N*(N-1) / 2^i) for i = 1 .. ceil(log2(N))
 */
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
    // Usage: 18_ckks_order_statistics [N] [k] [--bench]
    //   N       : vector length, must be power of 2 ≤ 128 (default: 8)
    //   k       : order statistic rank to compute, 1-based (default: 1)
    //   --bench : suppress verbose output, emit single BENCH: line for scripts
    int vec_len    = 8;
    int k_target   = 1;
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
            if (vec_len == 8)
                vec_len = std::stoi(arg);    // first positional → N
            else
                k_target = std::stoi(arg);   // second positional → k
        }
    }
    g_verbose = !bench_mode;

    // Validate N
    if (vec_len <= 0 || (vec_len & (vec_len - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << vec_len
                  << ")\n";
        return EXIT_FAILURE;
    }
    if (k_target < 1 || k_target > vec_len)
    {
        std::cerr << "Error: k must be in [1, N] (got k=" << k_target
                  << ", N=" << vec_len << ")\n";
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);

    // ===== HE Context =====
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // poly_modulus_degree=65536 → 32,768 available slots (N=128: N²=16384 ✓)
    // Depth budget: basicRank(13) + chebyshev_indicator(12) + multiply(1) = 26.
    const size_t poly_modulus_degree = 65536;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Q = 60 + 26×40 = 1100 bits; P = 60 bits → Q_tilde = 1160 bits
    // 1160 < 1761 = heongpu_128bit_std_parms(65536) → 128-bit security ✓
    // 27 primes in Q → 26 usable computation levels.
    context->set_coeff_modulus_bit_sizes(
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
         40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, {60});
    double scale = pow(2.0, 40);

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    context->generate();
    float ctx_ms = ctx_timer.stopTimer();

    // Validate N^2 fits in available slots
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
        std::cout << "N=" << vec_len << "  k=" << k_target
                  << "  matrix=" << vec_len << "x" << vec_len
                  << "  slots=" << (vec_len * vec_len) << "/"
                  << available_slots << "\n";
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

    // Compute all needed Galois shifts
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));

    // Row shifts: -(N*2^i) for i=log2(N/2)..0  (ReplR)
    std::vector<int> row_galois_shifts;
    for (int i = vec_len / 2; i > 0; i /= 2)
        row_galois_shifts.push_back(-(i * vec_len));

    // Column shifts: -1,-2,...,-(N/2) for ReplC
    std::vector<int> col_galois_shifts;
    for (int i = 1; i < vec_len; i *= 2)
        col_galois_shifts.push_back(-i);

    // SumR shifts: N, 2N, 4N, ..., N*(N/2)  (row-folding)
    std::vector<int> sumr_galois_shifts;
    for (int i = 0; i < log_n; i++)
        sumr_galois_shifts.push_back(vec_len * (1 << i));

    // Transpose shifts: -(N*(N-1)/2^i) for i=1..logN  (TransR)
    std::vector<int> transpose_shifts = transposeGaloisShifts(vec_len);

    // SumC shifts: +1, +2, +4, ..., +N/2  (sumFirstRow)
    std::vector<int> sumc_galois_shifts;
    for (int i = 0; i < log_n; i++)
        sumc_galois_shifts.push_back(1 << i);

    if (g_verbose)
    {
        std::cout << "Galois shifts — row: " << row_galois_shifts.size()
                  << "  col: " << col_galois_shifts.size()
                  << "  sumr: " << sumr_galois_shifts.size()
                  << "  sumc: " << sumc_galois_shifts.size()
                  << "  transpose: " << transpose_shifts.size() << "\n";
    }

    // ===== KEY GENERATION (timed) =====
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

    heongpu::Galoiskey<Scheme> sumc_galois_key(context, sumc_galois_shifts);
    keygen.generate_galois_key(sumc_galois_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    float keygen_ms = keygen_timer.stopTimer();
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms << " ms\n";

    // ===== Input preparation =====
    // bench mode: uniform random; verbose mode: sorted for easy verification.
    // For sorted input [0..N-1], the k-th order statistic (1-based) is k-1
    // (original scale), or normalized = (k-1)/(N-1).
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
            input[i] = static_cast<double>(i); // 0, 1, 2, ..., N-1
    }

    std::vector<double> normalized_input = normalizeForRanking(input);

    if (g_verbose)
    {
        std::cout << "Original input: ";
        display_vector(input, vec_len);
        std::cout << "Normalized:     ";
        display_vector(normalized_input, vec_len);
    }

    // Encode into a slot buffer padded to available_slots
    std::vector<double> row_initial(available_slots, 0.0);
    for (int i = 0; i < vec_len; i++)
        row_initial[i] = normalized_input[i];

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, row_initial, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    // ===== ORDER STATISTIC (timed) =====
    // Computes the k-th smallest element of the encrypted vector.
    // Key generation time is excluded from os_ms.
    GPUTimer os_timer;
    os_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_os =
        orderStatistic(ciphertext, k_target, vec_len,
                       row_galois_key, col_galois_key,
                       transpose_galois_key, sumr_galois_key, sumc_galois_key,
                       relin_key, evaluator, encoder, context, scale);

    float os_ms = os_timer.stopTimer();

    // Decrypt and decode: position 0 holds the k-th order statistic (normalized)
    heongpu::Plaintext<Scheme> os_plaintext(context);
    decryptor.decrypt(os_plaintext, ct_os);
    std::vector<double> os_result;
    encoder.decode(os_result, os_plaintext);
    double os_decoded = os_result[0];

    // ===== Output =====
    if (bench_mode)
    {
        std::cout << "BENCH:"
                  << " N=" << vec_len
                  << " k=" << k_target
                  << " ctx_ms=" << ctx_ms
                  << " keygen_ms=" << keygen_ms
                  << " os_ms=" << os_ms << "\n";
    }
    else
    {
        // Reconstruct expected value from sorted input
        std::vector<double> sorted_input = input;
        std::sort(sorted_input.begin(), sorted_input.end());
        double expected_original  = sorted_input[k_target - 1];
        double input_min          = input[0]; // already 0 for sorted test input
        double input_max          = input[vec_len - 1];
        double input_range        = input_max - input_min;
        double expected_normalized = (expected_original - input_min) / input_range;

        std::cout << "\n=== Order Statistic Results ===\n";
        std::cout << "Input vector (original): ";
        display_vector(input, vec_len);
        std::cout << "Input vector (normalized): ";
        display_vector(normalized_input, vec_len);
        std::cout << "\nComputing k=" << k_target << " order statistic...\n";
        std::cout << "  Decoded (normalized) : " << os_decoded << "\n";
        std::cout << "  Expected (normalized): " << expected_normalized << "\n";
        double error = std::abs(os_decoded - expected_normalized);
        std::cout << "  Error                : " << error
                  << (error < 0.05 ? "" : " INACCURATE") << "\n";

        // Denormalize back to original scale
        double os_original = os_decoded * input_range + input_min;
        std::cout << "  Decoded (original scale): " << os_original << "\n";
        std::cout << "  Expected (original scale): " << expected_original << "\n";

        std::cout << "\nTiming summary:\n";
        std::cout << "  Key generation: " << keygen_ms << " ms\n";
        std::cout << "  Order statistic: " << os_ms << " ms  ("
                  << (os_ms / 1000.0) << " s)\n";
    }

    return EXIT_SUCCESS;
}

// =============================================================================
// Helper function implementations
// =============================================================================

/**
 * @brief Replicates a row vector homomorphically using logarithmic rotations.
 * Algorithm 11 (ReplR): for i=0..logN-1: X ← X + (X ≫ N·2^i)
 */
heongpu::Ciphertext<Scheme>
replicateRow(const heongpu::Ciphertext<Scheme>& row_initial, int vec_len,
             heongpu::Galoiskey<Scheme>& galois_key,
             CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> row_replicated = row_initial;

    if (g_verbose)
        std::cout << "  ReplR shifts: ";
    for (int i = vec_len / 2; i > 0; i = i / 2)
    {
        int shift = -(i * vec_len);
        if (g_verbose)
            std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = row_replicated;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(row_replicated, rotated);
    }
    if (g_verbose)
        std::cout << "\n";
    return row_replicated;
}

/**
 * @brief Replicates a column vector homomorphically using logarithmic rotations.
 * Algorithm 12 (ReplC): for i=0..logN-1: X ← X + (X ≫ 2^i)
 */
heongpu::Ciphertext<Scheme>
replicateColumn(const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
                heongpu::Galoiskey<Scheme>& galois_key,
                CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> col_replicated = col_initial;

    if (g_verbose)
        std::cout << "  ReplC shifts: ";
    for (int i = 1; i < vec_len; i *= 2)
    {
        int shift = -i;
        if (g_verbose)
            std::cout << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = col_replicated;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(col_replicated, rotated);
    }
    if (g_verbose)
        std::cout << "\n";
    return col_replicated;
}

/**
 * @brief TransR: transpose a row vector to a column vector.
 *
 * Algorithm (Section 2.3, Algorithm 1):
 *   for i = 1,...,⌈log N⌉: X ← X + (X ≫ N(N-1)/2^i)
 *   X ← MaskC(X, 0)
 *
 * Depth consumed: 1 level (multiply_plain mask + rescale).
 */
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
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose)
        std::cout << "\n";

    // MaskC(X, 0): keep only the first column (positions 0, N, 2N, ...)
    size_t total_slots = context->get_poly_modulus_degree() / 2;
    std::vector<double> mask_values(total_slots, 0.0);
    for (int row = 0; row < vec_len; row++)
        mask_values[row * vec_len] = 1.0;

    heongpu::Plaintext<Scheme> mask(context);
    encoder.encode(mask, mask_values, scale);
    evaluator.multiply_plain_inplace(result, mask);
    evaluator.rescale_inplace(result); // depth 0 → 1
    return result;
}

/**
 * @brief Chebyshev sign approximation using built-in BSGS evaluate_poly.
 *
 * Depth: ceil(log2(degree)) levels consumed.
 *   degree=2047 → 11 levels via BSGS.
 */
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
                                    relin_key, /*a=*/-1.0, /*b=*/1.0);
}

/**
 * @brief SumR (Algorithm 9): sum all rows into the first row via row-folding.
 *
 * Left-rotates by N*2^i for i=0..logN-1, accumulating each shifted copy.
 * No level consumed (rotations and additions do not change depth).
 */
heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& sumr_galois_key,
        CKKSPolyEvaluator& evaluator)
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
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, sumr_galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }

    if (g_verbose)
        std::cout << "\n";
    return result;
}

/**
 * @brief Sum all N elements of the first row into position 0.
 *
 * Tree-sum using left rotations: for i=0..logN-1: X ← X + (X ≪ 2^i).
 * After logN steps position 0 = Σ_{j=0}^{N-1} X[j].
 *
 * No level consumed (rotations and additions do not change depth).
 */
heongpu::Ciphertext<Scheme>
sumFirstRow(const heongpu::Ciphertext<Scheme>& ct_vec, int vec_len,
            heongpu::Galoiskey<Scheme>& sumc_galois_key,
            CKKSPolyEvaluator& evaluator)
{
    heongpu::Ciphertext<Scheme> result = ct_vec;
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    if (g_verbose)
        std::cout << "  sumFirstRow shifts: ";
    for (int i = 0; i < log_n; i++)
    {
        int shift = (1 << i); // +1, +2, +4, ..., +N/2
        if (g_verbose)
            std::cout << "+" << shift << " ";
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, sumc_galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }
    if (g_verbose)
        std::cout << "\n";
    return result;
}

/**
 * @brief Chebyshev indicator approximation Ind_k(x; dI).
 *
 * Approximates the indicator function for rank k:
 *   Ind_k(x) ≈ 1  if x ∈ [k - 0.5, k + 0.5]
 *             ≈ 0  otherwise
 * on domain [0.5, N+0.5] covering all 1-based integer ranks 1..N.
 *
 * Depth: 1 (affine rescale to [-1,1]) + ceil(log2(degree)) (BSGS).
 * Total: 12 levels for degree=2047.
 */
heongpu::Ciphertext<Scheme>
chebyshev_indicator_approx(heongpu::Ciphertext<Scheme>& ct_rank,
                           int k, int N,
                           CKKSPolyEvaluator& poly_eval,
                           heongpu::Relinkey<Scheme>& relin_key,
                           double scale, int degree)
{
    if (g_verbose)
        std::cout << "  Chebyshev indicator approx k=" << k
                  << " domain=[0.5," << (N + 0.5) << "] degree=" << degree
                  << "...\n";

    auto indicator_func = [k](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(
            (re >= static_cast<double>(k) - 0.5 &&
             re <= static_cast<double>(k) + 0.5) ? 1.0 : 0.0,
            0.0);
    };

    double a = 0.5;
    double b = static_cast<double>(N) + 0.5;
    std::vector<Complex64> cheby_coeffs =
        heongpu::approximate_function(indicator_func, a, b, degree);

    return poly_eval.eval_chebyshev(ct_rank, scale, cheby_coeffs, degree,
                                    relin_key, a, b);
}

/**
 * @brief Algorithm 3 (Rank): compute fractional ranking of an encrypted vector.
 *
 * Depth budget (26 levels available with poly_modulus_degree=65536):
 *   depth 0  → fresh ciphertext
 *   depth 1  → transposeRowToColumn (multiply_plain mask + rescale)
 *   depth 12 → chebyshev_sign_approx degree-2047 (11 levels via BSGS)
 *   depth 13 → multiply_plain(0.5) + rescale (÷2)
 *   Total: 13 levels consumed.
 *
 * Output: position j holds rank[j] (1-based fractional).
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
    {
        std::cout << "\n=== Basic Ranking (N=" << vec_len << ") ===\n";
        std::cout << "Step 1: ReplR...\n";
    }

    // Step 1: row replication — position (k,j) = v[j]
    heongpu::Ciphertext<Scheme> ct_row =
        replicateRow(ct_vector, vec_len, row_galois_key, evaluator);

    // Step 2: TransR then ReplC — position (k,j) = v[k]
    if (g_verbose)
        std::cout << "Step 2: TransR + ReplC...\n";
    heongpu::Ciphertext<Scheme> ct_col_transposed =
        transposeRowToColumn(ct_vector, vec_len, transpose_galois_key,
                             evaluator, encoder, context, scale);
    heongpu::Ciphertext<Scheme> ct_col =
        replicateColumn(ct_col_transposed, vec_len, col_galois_key, evaluator);

    // Align ct_row depth to match ct_col (TransR rescaled: depth 0→1)
    evaluator.mod_drop_inplace(ct_row);

    // Step 3: diff[k,j] = v[j] - v[k]
    if (g_verbose)
        std::cout << "Step 3: Compute differences (vR - vC)...\n";
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_row, ct_col, ct_diff);

    // Step 4: Chebyshev sign approximation (degree 2047 = 2^11 - 1)
    if (g_verbose)
        std::cout << "Step 4: Chebyshev sign approx...\n";
    heongpu::Ciphertext<Scheme> ct_sign = chebyshev_sign_approx(
        ct_diff, evaluator, relin_key, scale, /*degree=*/2047);

    // Step 5: {-1,0,+1} → {0,1,2}: add 1
    if (g_verbose)
        std::cout << "Step 5: Add 1 to shift sign range...\n";
    evaluator.add_plain_inplace(ct_sign, 1.0);

    // Step 6: SumR — fold rows into row 0; position j = 2*rank[j] - 1
    if (g_verbose)
        std::cout << "Step 6: SumR (row-folding)...\n";
    heongpu::Ciphertext<Scheme> ct_sumr =
        sumRows(ct_sign, vec_len, sumr_galois_key, evaluator);

    // Step 7: ÷2 via scalar multiply_plain + rescale → position j = rank[j]-0.5
    if (g_verbose)
        std::cout << "Step 7: Scale by 0.5 (÷2)...\n";
    evaluator.multiply_plain_inplace(ct_sumr, 0.5, scale);
    evaluator.rescale_inplace(ct_sumr); // depth 12 → 13

    // Step 8: add 0.5 → rank[j]  (1-based fractional rank)
    if (g_verbose)
        std::cout << "Step 8: Add 0.5 to complete fractional rank...\n";
    evaluator.add_plain_inplace(ct_sumr, 0.5);

    if (g_verbose)
        std::cout << "Ranking complete.\n";
    return ct_sumr;
}

/**
 * @brief Algorithm 4 (Order Statistic): extract the k-th smallest value.
 *
 * Steps (paper Section 3.2, Algorithm 4):
 *  1. R ← Rank(V; dC)      — fractional rank vector, depth 0→13
 *  2. O ← Ind_k(R; dI)     — Boolean mask: ≈1 where rank[j]=k, depth 13→25
 *  3. SumC(O · V)           — inner product; position 0 = k-th OS, depth 25→26
 *
 * Division by SumC(O) is omitted (valid for distinct elements, SumC(O)≈1).
 *
 * Depth budget (poly_modulus_degree=65536, 26 usable levels):
 *   basicRank: 13 levels; chebyshev_indicator: 12 levels; multiply+rescale: 1.
 *   Total: 26 ≤ 26 ✓
 */
heongpu::Ciphertext<Scheme>
orderStatistic(const heongpu::Ciphertext<Scheme>& ct_vector,
               int k, int vec_len,
               heongpu::Galoiskey<Scheme>& row_galois_key,
               heongpu::Galoiskey<Scheme>& col_galois_key,
               heongpu::Galoiskey<Scheme>& transpose_galois_key,
               heongpu::Galoiskey<Scheme>& sumr_galois_key,
               heongpu::Galoiskey<Scheme>& sumc_galois_key,
               heongpu::Relinkey<Scheme>& relin_key,
               CKKSPolyEvaluator& evaluator,
               heongpu::HEEncoder<Scheme>& encoder,
               heongpu::HEContext<Scheme>& context,
               double scale, int indicator_degree)
{
    if (g_verbose)
        std::cout << "\n=== Order Statistic k=" << k
                  << " (N=" << vec_len << ") ===\n";

    // Step 1: R ← Rank(V; dC)  →  depth 0..13
    if (g_verbose)
        std::cout << "Step 1: Compute rank vector...\n";
    heongpu::Ciphertext<Scheme> ct_rank =
        basicRank(ct_vector, vec_len, row_galois_key, col_galois_key,
                  transpose_galois_key, sumr_galois_key, relin_key,
                  evaluator, encoder, context, scale);

    // Step 2: O ← Ind_k(R; dI)  →  depth 13..25
    // Ranks in [1,N]; indicator domain [0.5, N+0.5]. Zero-padded garbage slots
    // (positions N..total_slots-1) are suppressed by the zero-padded ct_v_aligned.
    if (g_verbose)
        std::cout << "Step 2: Apply indicator Ind_k...\n";
    heongpu::Ciphertext<Scheme> ct_mask =
        chebyshev_indicator_approx(ct_rank, k, vec_len,
                                   evaluator, relin_key, scale,
                                   indicator_degree);

    // Step 3: value ← SumC(O · V)  →  depth 25..26
    // ct_vector is at depth 0; align by mod_dropping to match ct_mask depth.
    if (g_verbose)
        std::cout << "Step 3: Align depths and multiply O · V...\n";
    const int rank_depth = 13;
    const int ind_depth  = 1 + static_cast<int>(
        std::ceil(std::log2(static_cast<double>(indicator_degree + 1))));
    const int mask_depth = rank_depth + ind_depth;

    heongpu::Ciphertext<Scheme> ct_v_aligned = ct_vector;
    for (int d = 0; d < mask_depth; d++)
        evaluator.mod_drop_inplace(ct_v_aligned);

    heongpu::Ciphertext<Scheme> ct_product(context);
    evaluator.multiply(ct_mask, ct_v_aligned, ct_product, relin_key);
    evaluator.rescale_inplace(ct_product); // depth 25 → 26

    // Sum positions 0..N-1 into position 0 (no depth change)
    if (g_verbose)
        std::cout << "Step 3: Sum product into position 0...\n";
    heongpu::Ciphertext<Scheme> ct_value =
        sumFirstRow(ct_product, vec_len, sumc_galois_key, evaluator);

    if (g_verbose)
        std::cout << "Order statistic complete.\n";
    return ct_value;
}
