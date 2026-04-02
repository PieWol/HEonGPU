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
    CKKSPolyEvaluator& evaluator);

heongpu::Ciphertext<Scheme> replicateColumn(
    const heongpu::Ciphertext<Scheme>& col_initial, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator);

/**
 * @brief TransR: transpose a row vector to a column vector.
 *
 * Accepts a pre-generated Galois key for shifts -(N(N-1)/2^i) i=1..logN.
 * Key must be generated in main before ranking to keep rank timing clean.
 */
heongpu::Ciphertext<Scheme> transposeRowToColumn(
    const heongpu::Ciphertext<Scheme>& row_vector, int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    CKKSPolyEvaluator& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    heongpu::HEContext<Scheme>& context,
    double scale);

/**
 * @brief Chebyshev sign approximation using BSGS polynomial evaluation.
 *
 * Uses heongpu::approximate_function to compute degree-D Chebyshev
 * coefficients for sign(x) on [-1,1], then evaluates via the built-in
 * baby-step/giant-step evaluate_poly (promoted via CKKSPolyEvaluator).
 *
 * Depth consumed: ceil(log2(D)) levels ≈ 11 for D=2048.
 *
 * For the paper's regime (N=64 elements, min pairwise diff ≈ 1/63):
 *   - D=2047 ensures correct sign classification for |x| ≥ 0.016
 *   - D must be 2^k - 1 for BSGS precomputation to cover all required powers
 *   - Scale: input differences in [-1,1] require no extra normalization
 *
 * @param ct_diff  Encrypted difference ct_col - ct_row, at some depth d
 * @param poly_eval CKKSPolyEvaluator (exposes evaluate_poly)
 * @param relin_key Relinearization key
 * @param scale     Encoding scale (must match current ciphertext scale)
 * @param degree    Chebyshev degree; must be 2^k - 1 for BSGS to work
 *                  correctly. 2047 (= 2^11 - 1) matches the paper for N≤256.
 * @return Ciphertext containing sign approximation values ≈ ±1
 */
heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree = 2047);

/**
 * @brief SumR (Algorithm 9): sum all rows into the first row using row-folding.
 *
 * Applies log2(N) left rotations by N, 2N, 4N, ..., N*(N/2), adding each
 * shifted copy to the accumulator. This folds row k into row 0 for all k,
 * so position j in the output holds the column-sum for column j.
 *
 * @param ct_matrix   Encrypted N×N matrix (flattened row-major)
 * @param vec_len     N (matrix side length, must be power of 2)
 * @param sumr_galois_key  Galois key with shifts N, 2N, 4N, ..., N*(N/2)
 * @param evaluator   Arithmetic operator
 * @return Ciphertext; position j holds Σ_k ct_matrix[k,j] (other rows garbage)
 */
heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& sumr_galois_key,
        CKKSPolyEvaluator& evaluator);

/**
 * @brief Normalize a plaintext vector to [0,1] before encryption.
 *
 * This is a client-side operation that MUST be applied before encode/encrypt.
 * The HE ranking protocol requires that all pairwise differences (v[k] - v[j])
 * lie within [-1,1], which is the domain of the Chebyshev sign approximation.
 * Normalizing input to [0,1] guarantees this: the maximum possible difference
 * is 1 - 0 = 1. The server only ever receives normalized, encrypted data.
 *
 * @param input Raw values (any finite range; all-equal input is undefined)
 * @return Values linearly rescaled to [0,1] via min-max normalization
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
 * @brief Basic ranking: computes fractional rank of each element (1-based).
 *
 * Precondition: ct_vector must encrypt values in [0,1]. The client is
 * responsible for calling normalizeForRanking() before encode/encrypt.
 * Passing un-normalized data silently breaks the Chebyshev sign step.
 *
 * All Galois keys must be pre-generated by the caller so that key generation
 * time is excluded from the ranking timing.
 *
 * Depth budget (14 levels available with poly_modulus_degree=32768, 15 primes):
 *   depth 0  → fresh ciphertext
 *   depth 1  → transposeRowToColumn (multiply_plain mask + rescale)
 *   depth 1  → mod_drop ct_row for alignment
 *   depth 12 → chebyshev_sign_approx degree-2047 (11 levels via BSGS)
 *   depth 12 → add_plain +1, SumR (row-folding rotations+adds, no level change)
 *   depth 13 → multiply_plain(half_first_row_mask) + rescale (MaskR + ÷2)
 *   depth 13 → add_plain +0.5 (complete fractional rank, no level change)
 *   Total: 13 ≤ 14 ✓
 *
 * Output: position j in the decrypted result holds rank[j] directly (1-based
 * fractional); positions j >= vec_len are garbage and should be ignored.
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
          heongpu::HEContext<Scheme>& context, double scale);

/**
 * @brief GPU-aware timer using CUDA Events for accurate GPU timing
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
 * Pre-computing these in main keeps all keygen outside the rank timer.
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
    // Usage: 16_ckks_rotation_parallel [N] [--bench]
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

    // Validate N is a positive power of 2
    if (vec_len <= 0 || (vec_len & (vec_len - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << vec_len
                  << ")\n";
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);

    // ===== HE Context =====
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    // poly_modulus_degree=32768 → 16,384 available slots = 128×128
    // This matches the paper's single-ciphertext limit of N=128 for ranking.
    const size_t poly_modulus_degree = 32768;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Q = 60 + 14×40 = 620 bits; P = 60 bits → Q_tilde = 680 bits
    // 680 < 881 = heongpu_128bit_std_parms(32768) → 128-bit security ✓
    // 15 primes in Q → 14 usable computation levels (14 rescales before
    // exhausting Q)
    context->set_coeff_modulus_bit_sizes(
        {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, {60});
    double scale = pow(2.0, 40); // matches 40-bit computation primes
    context->generate();

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
        std::cout << "N=" << vec_len << "  matrix=" << vec_len << "x"
                  << vec_len << "  slots=" << (vec_len * vec_len) << "/"
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

    // Compute all needed Galois shifts before timing keygen
    // Row shifts: -(2^i * N) for i = log2(N/2)..0  (ReplR)
    std::vector<int> row_galois_shifts;
    for (int i = vec_len / 2; i > 0; i /= 2)
        row_galois_shifts.push_back(-(i * vec_len));

    // Column shifts: -1,-2,...,-(N/2) for ReplC only
    std::vector<int> col_galois_shifts;
    for (int i = 1; i < vec_len; i *= 2)
        col_galois_shifts.push_back(-i);

    // SumR shifts: N, 2N, 4N, ..., N*(N/2)  (Algorithm 9, row-folding)
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    std::vector<int> sumr_galois_shifts;
    for (int i = 0; i < log_n; i++)
        sumr_galois_shifts.push_back(vec_len * (1 << i));

    // Transpose shifts: -(N*(N-1)/2^i) for i=1..logN  (TransR)
    std::vector<int> transpose_shifts = transposeGaloisShifts(vec_len);

    if (g_verbose)
    {
        std::cout << "Galois shifts — row: " << row_galois_shifts.size()
                  << "  col: " << col_galois_shifts.size()
                  << "  sumr: " << sumr_galois_shifts.size()
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

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    float keygen_ms = keygen_timer.stopTimer();
    if (g_verbose)
        std::cout << "Key generation: " << keygen_ms << " ms\n";

    // ===== Input preparation =====
    // bench mode: uniform random (realistic); verbose mode: sorted for easy
    // correctness verification (rank[i] should equal i+1).
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

    // Encode into a slot buffer padded to the full available_slots size
    std::vector<double> row_initial(available_slots, 0.0);
    for (int i = 0; i < vec_len; i++)
        row_initial[i] = normalized_input[i];

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, row_initial, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    // ===== RANKING (timed) =====
    // All Galois and relin keys were generated above — this timer covers only
    // the homomorphic ranking computation itself.
    GPUTimer rank_timer;
    rank_timer.startTimer();

    heongpu::Ciphertext<Scheme> ct_rank =
        basicRank(ciphertext, vec_len, row_galois_key, col_galois_key,
                  transpose_galois_key, sumr_galois_key, relin_key, evaluator,
                  encoder, context, scale);

    float rank_ms = rank_timer.stopTimer();

    // ===== Decrypt and decode =====
    heongpu::Plaintext<Scheme> rank_plaintext(context);
    decryptor.decrypt(rank_plaintext, ct_rank);
    std::vector<double> rank_result;
    encoder.decode(rank_result, rank_plaintext);

    // ===== Output =====
    if (bench_mode)
    {
        // Single line parseable by benchmark_ranking.py
        // rank_ms excludes key generation; keygen_ms reported separately
        std::cout << "BENCH:"
                  << " N=" << vec_len << " keygen_ms=" << keygen_ms
                  << " rank_ms=" << rank_ms << "\n";
    }
    else
    {
        // Full verbose output with correctness verification
        std::cout << "\n=== Ranking Results ===\n";
        std::cout << "Input vector:  ";
        display_vector(input, vec_len);

        // Position j holds rank[j] directly (1-based fractional rank)
        std::cout << "Rank (1-based fractional):\n";
        for (int i = 0; i < vec_len; i++)
        {
            double decoded_rank = rank_result[i];
            std::cout << "  input[" << i << "] = " << input[i]
                      << " -> rank = " << decoded_rank << "\n";
        }

        // Verification: for sorted input input[i] = i, rank = i+1 (1-based)
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
    }

    return EXIT_SUCCESS;
}

// The core idea: manipulate the encrypted vector so that only a single
// evaluation of the comparison function is needed to compare all values.
// For vector v = (v1,v2,...,vN): produce
//   vR = (v1,v2,...,vN, v1,v2,...,vN, ...)  [row replication]
//   vC = (v1,v1,...,v1, v2,v2,...,v2, ...)  [column replication]

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
 * The Galois key covering all required shifts must be passed in — it is
 * generated by the caller (main) before the rank timer starts.
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
    // rescale_required_=true after multiply_plain; rescale before any rotation
    evaluator.rescale_inplace(result); // depth 0 → 1
    return result;
}

/**
 * @brief Chebyshev sign approximation using built-in BSGS evaluate_poly.
 *
 * Computes sign(x) ≈ Σ c_k T_k(x) for k=1,3,5,...,degree on [-1,1].
 * The Chebyshev series of the sign function contains only odd-degree terms.
 * heongpu::approximate_function computes minimax-quality coefficients via
 * Chebyshev interpolation.
 *
 * Depth: Polynomial::depth() = ceil(log2(degree)) levels consumed.
 *   degree=2048 → 11 levels; with scale management the total is ≤12 levels.
 */
heongpu::Ciphertext<Scheme>
chebyshev_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
                      CKKSPolyEvaluator& poly_eval,
                      heongpu::Relinkey<Scheme>& relin_key, double scale,
                      int degree)
{
    if (g_verbose)
        std::cout << "  Chebyshev sign approx degree=" << degree << "...\n";

    // sign(x): +1 for x>0, -1 for x<0, 0 at x=0 (odd function)
    // approximate_function uses Chebyshev interpolation at degree+1 nodes
    auto sign_func = [](Complex64 x) -> Complex64 {
        double re = x.real();
        return Complex64(re > 0.0 ? 1.0 : (re < 0.0 ? -1.0 : 0.0), 0.0);
    };

    std::vector<Complex64> cheby_coeffs =
        heongpu::approximate_function(sign_func, -1.0, 1.0, degree);

    // Polynomial is protected in HEOperator — construction happens inside
    // CKKSPolyEvaluator::eval_chebyshev which has protected-member access
    return poly_eval.eval_chebyshev(ct_diff, scale, cheby_coeffs, degree,
                                    relin_key, /*a=*/-1.0, /*b=*/1.0);
}

/**
 * @brief SumR (Algorithm 9): sum all rows into the first row via row-folding.
 *
 * Left-rotates by N*2^i for i=0..logN-1, accumulating each shifted copy.
 * After logN iterations, position j holds Σ_k ct_matrix[k,j] for all rows k.
 * Rows k>0 in the output contain garbage (will be zeroed by MaskR in caller).
 *
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
        int shift = vec_len * (1 << i); // N, 2N, 4N, ...
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
 * @brief Algorithm 3 (Rank): compute fractional ranking of an encrypted vector.
 *
 * Steps:
 *  1. ReplR  → vR matrix (each row = input vector v[j] at position (k,j))
 *  2. TransR → column form, then ReplC → vC matrix (v[k] at position (k,j))
 *  3. ct_diff = vR - vC  (v[j]-v[k]); sign>0 when v[j]>v[k] (paper's Cmp)
 *  4. Cmp via Chebyshev sign approx → values ≈ {-1, 0, +1}
 *  5. Add 1: {-1,0,+1} → {0,1,2} = 2·Cmp (no level consumed)
 *  6. SumR (row-folding): col-sum in row 0; position j = Σ_k 2·Cmp(v[j],v[k])
 *     = 2·(rank[j]-0.5) = 2·rank[j]-1 (no level consumed)
 *  7. multiply_plain(ct, 0.5, scale) + rescale: scalar ÷2 on all positions
 *     → position j = rank[j]-0.5  (depth +1; Plaintext overload can't be used
 *     at depth 12 because HEonGPU requires depth match; double overload is fine)
 *  8. Add 0.5: position j = rank[j]  (1-based fractional rank, no level)
 *
 * Output: position j in the decrypted result holds rank[j] directly.
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

    // Step 3: diff[k,j] = v[j] - v[k]; positive when v[j] > v[k]
    // Paper's Cmp(v[j],v[k]) = sign(v[j]-v[k]) → +1 iff v[j]>v[k]
    if (g_verbose)
        std::cout << "Step 3: Compute differences (vR - vC)...\n";
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_row, ct_col, ct_diff);

    // Step 4: Chebyshev sign approximation (degree 2047 = 2^11 - 1)
    if (g_verbose)
        std::cout << "Step 4: Chebyshev sign approx...\n";
    heongpu::Ciphertext<Scheme> ct_sign = chebyshev_sign_approx(
        ct_diff, evaluator, relin_key, scale, /*degree=*/2047);

    // Step 5: {-1,0,+1} → {0,1,2}: add 1 (no level consumed)
    if (g_verbose)
        std::cout << "Step 5: Add 1 to shift sign range...\n";
    evaluator.add_plain_inplace(ct_sign, 1.0);

    // Step 6: SumR (Algorithm 9) — row-fold into row 0
    // Position j = Σ_k (1 + Cmp(v[j],v[k])) = 2·rank[j] - 1
    if (g_verbose)
        std::cout << "Step 6: SumR (row-folding)...\n";
    heongpu::Ciphertext<Scheme> ct_sumr =
        sumRows(ct_sign, vec_len, sumr_galois_key, evaluator);

    // Step 7: ÷2 via scalar multiply_plain (double overload — no depth check).
    // multiply_plain(ct, 0.5, scale) multiplies plaintext values by 0.5 and
    // sets rescale_required_=true. After rescale: position j = rank[j]-0.5.
    // Applies to all positions uniformly (MaskR masking skipped — positions
    // outside 0..N-1 are garbage but client only reads 0..N-1).
    if (g_verbose)
        std::cout << "Step 7: Scale by 0.5 (÷2)...\n";
    evaluator.multiply_plain_inplace(ct_sumr, 0.5, scale);
    evaluator.rescale_inplace(ct_sumr); // depth 12 → 13

    // Step 8: add 0.5 → rank[j] - 0.5 + 0.5 = rank[j]  (1-based fractional)
    if (g_verbose)
        std::cout << "Step 8: Add 0.5 to complete fractional rank...\n";
    evaluator.add_plain_inplace(ct_sumr, 0.5);

    if (g_verbose)
        std::cout << "Ranking complete.\n";
    return ct_sumr;
}
