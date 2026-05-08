#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <fstream>
#include <omp.h>

// Global verbose flag: false in --bench mode for clean parseable output
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
                   heongpu::Relinkey<Scheme>& relin_key,
                   bool lead = false,
                   double a = -1.0, double b = 1.0)
    {
        Polynomial poly(degree, coeffs, lead,
                        heongpu::PolyType::CHEBYSHEV, a, b);
        if (g_verbose)
            std::cout << "  poly degree=" << degree
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
fg_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
               CKKSPolyEvaluator& poly_eval,
               heongpu::Relinkey<Scheme>& relin_key, double scale,
               int dg = 3, int df = 2);

heongpu::Ciphertext<Scheme>
sumRows(const heongpu::Ciphertext<Scheme>& ct_matrix, int vec_len,
        heongpu::Galoiskey<Scheme>& sumr_galois_key,
        CKKSPolyEvaluator& evaluator,
        heongpu::HEContext<Scheme>& context);

// Paper's compareDepth table (same for both basic and tie-corrected ranking;
// the comparison function is identical, only the post-processing differs):
//   N<=8:   compareDepth=7  -> degree=127
//   N<=16:  compareDepth=8  -> degree=255
//   N<=32:  compareDepth=9  -> degree=511
//   N<=64:  compareDepth=10 -> degree=1023
//   N<=128: compareDepth=11 -> degree=2047
int selectChebyshevDegree(int N)
{
    if (N <= 8)   return 127;
    if (N <= 16)  return 255;
    if (N <= 32)  return 511;
    if (N <= 64)  return 1023;
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

std::vector<double> computeFractionalRanks(const std::vector<double>& input)
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
        while (j < n && input[idx[j]] == input[idx[i]])
            j++;
        double mean_rank = 0.0;
        for (int k = i; k < j; k++)
            mean_rank += (k + 1);
        mean_rank /= (j - i);
        for (int k = i; k < j; k++)
            ranks[idx[k]] = mean_rank;
        i = j;
    }
    return ranks;
}

std::vector<double> computeOrdinalRanks(const std::vector<double>& input)
{
    int n = static_cast<int>(input.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b)
              {
                  if (input[a] != input[b])
                      return input[a] < input[b];
                  return a < b;
              });

    std::vector<double> ranks(n);
    for (int k = 0; k < n; k++)
        ranks[idx[k]] = k + 1;
    return ranks;
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
          heongpu::HEContext<Scheme>& context, double scale,
          bool tie_correction, bool use_fg, int cheby_degree);

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
    // Usage: 23_ckks_ranking_tie_correction [N] [--tie-correction] [--ties] [--bench]
    int vec_len         = 64;
    bool bench_mode     = false;
    bool tie_correction = false;
    bool use_ties       = false;
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (arg == "--bench")
            bench_mode = true;
        else if (arg == "--tie-correction")
            tie_correction = true;
        else if (arg == "--ties")
            use_ties = true;
        else if (!arg.empty() &&
                 std::isdigit(static_cast<unsigned char>(arg[0])))
            vec_len = std::stoi(arg);
    }
    g_verbose = !bench_mode;

    if (vec_len <= 0 || (vec_len & (vec_len - 1)) != 0)
    {
        std::cerr << "Error: N must be a positive power of 2 (got " << vec_len
                  << ")\n";
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);

    // ===== HE Context =====
    //
    // N<=32: Chebyshev sign at n=32768 (sufficient accuracy, small footprint)
    //   depth = ceil(log2(degree+1)) + 2;  TC adds +2 for sign²+mask
    //   n=32768, Q={36,35×14}, P={36×8}, scale=2^35, dnum=2
    //
    // N>32: f,g composition (dg=3, df=2) at n=65536
    //   Better runtime than Chebyshev without hitting memory limits.
    //   Each degree-7 poly = 4 levels, 5 evals = 20 levels for sign.
    //   Basic:  transpose(1) + fg(20) + scale(1) = 22
    //           n=65536, Q={60,45×22}, P={60×11}, scale=2^45, dnum=3
    //   TC:     transpose(1) + fg(20) + sign²(1) + mask(1) + scale(1) = 24
    //           n=65536, Q={60,45×24}, P={60×10}, scale=2^45, dnum=3
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();

    size_t poly_modulus_degree;
    int scale_bits;
    int available_depth;

    // f,g composition for N>32: better runtime than Chebyshev without hitting
    // memory limits. N<=32: Chebyshev at n=32768 (sufficient accuracy, smaller).
    const bool use_fg = (vec_len > 32);
    const int cheby_degree = use_fg ? 0 : selectChebyshevDegree(vec_len);
    int required_depth;
    if (use_fg)
        // TC: transpose(1) + fg(20) + sign²(1) + mask(1) + scale(1) = 24
        // Basic: transpose(1) + fg(20) + scale(1) = 22
        required_depth = tie_correction ? 24 : 22;
    else
        // Chebyshev: transpose(1) + cheby + scale(1), TC adds sign²(1) + mask(1)
        required_depth = static_cast<int>(std::ceil(std::log2(cheby_degree + 1)))
                         + (tie_correction ? 4 : 2);

    if (use_fg && tie_correction)
    {
        // n=65536 (budget 1761): Q={60,45×24}=1140, P={60×10}=600, total=1740, dnum=3
        poly_modulus_degree = 65536;
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes(
            {60, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
             45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
            {60, 60, 60, 60, 60, 60, 60, 60, 60, 60});
        scale_bits = 45;
        available_depth = 24;
    }
    else if (use_fg && !tie_correction)
    {
        // n=65536 (budget 1761): Q={60,45×22}=1050, P={60×11}=660, total=1710, dnum=3
        poly_modulus_degree = 65536;
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes(
            {60, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
             45, 45, 45, 45, 45, 45, 45, 45, 45, 45},
            {60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60});
        scale_bits = 45;
        available_depth = 22;
    }
    else
    {
        // N<=32 Chebyshev: n=32768, Q={36,35×14}, P={36×8}, scale=2^35, dnum=2
        poly_modulus_degree = 32768;
        context->set_poly_modulus_degree(poly_modulus_degree);
        context->set_coeff_modulus_bit_sizes(
            {36, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35},
            {36, 36, 36, 36, 36, 36, 36, 36});
        scale_bits = 35;
        available_depth = 14;
    }
    double scale = pow(2.0, scale_bits);

    if (required_depth > available_depth)
    {
        std::cerr << "Error: N=" << vec_len
                  << (tie_correction ? " with tie correction" : "")
                  << " needs " << required_depth << " levels but only "
                  << available_depth << " available.\n";
        return EXIT_FAILURE;
    }

    GPUTimer ctx_timer;
    ctx_timer.startTimer();
    context->generate();
    float ctx_ms = ctx_timer.stopTimer();

    int available_slots = static_cast<int>(poly_modulus_degree / 2);
    if (vec_len * vec_len > available_slots)
    {
        std::cerr << "Error: N=" << vec_len << " needs " << (vec_len * vec_len)
                  << " slots but only " << available_slots << " available.\n";
        return EXIT_FAILURE;
    }

    if (g_verbose)
    {
        std::cout << "N=" << vec_len << "  mode="
                  << (tie_correction ? "tie-corrected" : "basic");
        if (use_fg)
            std::cout << "  sign=g^3*f^2";
        else
            std::cout << "  degree=" << cheby_degree;
        std::cout << "  depth=" << required_depth << "/" << available_depth
                  << "  n=" << poly_modulus_degree << "\n";
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

    // ===== Input (same CSV as OpenFHE reference) =====
    std::vector<double> input = loadPoints1D(vec_len);

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
                  encoder, context, scale, tie_correction, use_fg, cheby_degree);

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
                  << " N=" << vec_len
                  << " mode=" << (tie_correction ? "tie_corr" : "basic")
                  << " ctx_ms=" << ctx_ms
                  << " keygen_ms=" << keygen_ms
                  << " rank_ms=" << rank_ms
                  << " gpu_keys_mib=" << gpu_keys_mib
                  << " gpu_rank_mib=" << gpu_rank_mib
                  << " gpu_peak_mib=" << gpu_peak_mib << "\n";
    }
    else
    {
        std::cout << "\n=== Ranking Results ("
                  << (tie_correction ? "tie-corrected" : "basic, no tie correction")
                  << (use_ties ? ", tied input" : "") << ") ===\n";
        std::cout << "Input vector:  ";
        display_vector(input, vec_len);

        std::vector<double> expected_ranks = tie_correction
                                                ? computeOrdinalRanks(input)
                                                : computeFractionalRanks(input);

        std::cout << "Rank (1-based):\n";
        for (int i = 0; i < vec_len; i++)
            std::cout << "  input[" << i << "] = " << input[i]
                      << " -> rank = " << rank_result[i]
                      << "  (expected " << expected_ranks[i] << ")\n";

        std::cout << "\nVerification:\n";
        bool all_correct = true;
        int n_ties_wrong = 0;
        for (int i = 0; i < vec_len; i++)
        {
            double error    = std::abs(rank_result[i] - expected_ranks[i]);
            bool is_correct = (error < 0.5);
            if (!is_correct)
            {
                all_correct = false;
                n_ties_wrong++;
                std::cout << "  MISMATCH input[" << i << "]=" << input[i]
                          << ": expected=" << expected_ranks[i]
                          << ", actual=" << rank_result[i]
                          << ", error=" << error << "\n";
            }
        }
        if (all_correct)
            std::cout << "  All " << vec_len << " ranks correct!\n";
        else
            std::cout << "  " << n_ties_wrong << "/" << vec_len
                      << " ranks incorrect.\n";

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

// ===== Algorithm implementations (based on 17_ckks_ranking_paper.cpp) =====

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

// f,g composition sign approximation (outputs [-1,+1])
// g3(t) = t*(4589 + t²*(-16577 + t²*(25614 - 12860*t²))) / 1024
// f3(t) = t*(35 + t²*(-35 + t²*(21 - 5*t²))) / 16
// Each is degree 7 → 4 levels per evaluation. Total: (dg+df)*4 = 20 levels.
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
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk, true);
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
    return pe.eval_chebyshev(ct, scale, coeffs, 7, rk, true);
}

heongpu::Ciphertext<Scheme>
fg_sign_approx(heongpu::Ciphertext<Scheme>& ct_diff,
               CKKSPolyEvaluator& poly_eval,
               heongpu::Relinkey<Scheme>& relin_key, double scale,
               int dg, int df)
{
    if (g_verbose)
        std::cout << "  f,g sign approx (dg=" << dg << ", df=" << df << ")...\n";
    heongpu::Ciphertext<Scheme> ct = ct_diff;
    for (int i = 0; i < dg; i++)
        ct = applyG3(ct, poly_eval, relin_key, scale);
    for (int i = 0; i < df; i++)
        ct = applyF3(ct, poly_eval, relin_key, scale);
    return ct;
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
          heongpu::HEContext<Scheme>& context, double scale,
          bool tie_correction, bool use_fg, int cheby_degree)
{
    if (g_verbose)
    {
        std::cout << "\n=== Ranking (N=" << vec_len << ", "
                  << (tie_correction ? "tie-corrected" : "basic") << ") ===\n";
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
        replicateColumn(ct_col_transposed, vec_len, col_galois_key, evaluator,
                        context);

    evaluator.mod_drop_inplace(ct_row);

    if (g_verbose)
        std::cout << "Step 3: Compute differences (vR - vC)...\n";
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_row, ct_col, ct_diff);

    heongpu::Ciphertext<Scheme> ct_sign(context);
    if (use_fg)
    {
        if (g_verbose)
            std::cout << "Step 4: f,g sign approx (dg=3, df=2)...\n";
        ct_sign = fg_sign_approx(ct_diff, evaluator, relin_key, scale, 3, 2);
    }
    else
    {
        if (g_verbose)
            std::cout << "Step 4: Chebyshev sign approx (N=" << vec_len
                      << " -> degree=" << cheby_degree << ")...\n";
        ct_sign = chebyshev_sign_approx(ct_diff, evaluator, relin_key, scale,
                                        cheby_degree);
    }

    // --- Save raw sign matrix for tie correction before modifying ---
    heongpu::Ciphertext<Scheme> ct_sign_raw;
    if (tie_correction)
        ct_sign_raw = ct_sign;

    // --- Basic ranking (identical to 17_ckks_ranking_paper.cpp) ---

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

    if (!tie_correction)
    {
        if (g_verbose)
            std::cout << "Basic ranking complete (no tie correction).\n";
        return ct_sumr;
    }

    // --- Tie-correction offset (Algorithm 6 from Mazzone et al.) ---
    //
    // Equality indicator from the sign matrix: E = 1 - sign(x)^2
    //   sign ≈ +1 or -1 → E ≈ 0  (distinct elements)
    //   sign ≈  0        → E ≈ 1  (tied elements)
    // Equivalent to the paper's e = 4*c*(1-c) since c = (sign+1)/2.
    //
    // Offset: F = SumR(E * adjusted_mask) - 0.5
    //   where adjusted_mask[i,j] = 0.5 if j >= i, -0.5 if j < i
    //   This combines U = SumR(E * upper_mask) and T = SumR(E) into one
    //   plaintext multiply: SumR(E * (mask - 0.5)) = U - 0.5*T

    if (g_verbose)
        std::cout << "Step 9: Compute equality E = 1 - sign^2...\n";

    // sign^2: 1 level (multiply + relin + rescale)
    heongpu::Ciphertext<Scheme> ct_sign_copy = ct_sign_raw;
    heongpu::Ciphertext<Scheme> ct_sign_sq(context);
    evaluator.multiply(ct_sign_raw, ct_sign_copy, ct_sign_sq);
    evaluator.relinearize_inplace(ct_sign_sq, relin_key);
    evaluator.rescale_inplace(ct_sign_sq);

    // E = 1 - sign^2
    evaluator.negate_inplace(ct_sign_sq);
    evaluator.add_plain_inplace(ct_sign_sq, 1.0);

    if (g_verbose)
        std::cout << "Step 10: Multiply E by adjusted mask and SumR...\n";

    // Build adjusted mask: δ_{j≥i} (upper triangle) as in Algorithm 6.
    // Our SumR sums over the row dimension (column-wise), same as the paper.
    size_t total_slots = context->get_poly_modulus_degree() / 2;
    std::vector<double> adj_mask_values(total_slots, 0.0);
    for (int i = 0; i < vec_len; i++)
        for (int j = 0; j < vec_len; j++)
            adj_mask_values[i * vec_len + j] = (j >= i) ? 0.5 : -0.5;

    heongpu::Plaintext<Scheme> adj_mask_pt(context);
    encoder.encode(adj_mask_pt, adj_mask_values, scale);

    // Align plaintext depth to ciphertext depth (encode sets depth=0,
    // but ct_sign_sq is at compareDepth+2 after Chebyshev + sign^2)
    while (adj_mask_pt.depth() < ct_sign_sq.depth())
        evaluator.mod_drop_inplace(adj_mask_pt);

    // E * adjusted_mask: 1 level (multiply_plain + rescale)
    evaluator.multiply_plain_inplace(ct_sign_sq, adj_mask_pt);
    evaluator.rescale_inplace(ct_sign_sq);

    // SumR of masked equality
    heongpu::Ciphertext<Scheme> ct_offset =
        sumRows(ct_sign_sq, vec_len, sumr_galois_key, evaluator, context);

    // F = SumR(E * adjusted_mask) - 0.5
    evaluator.sub_plain_inplace(ct_offset, 0.5);

    if (g_verbose)
        std::cout << "Step 11: Add tie-correction offset to basic rank...\n";

    // Align ct_sumr to ct_offset depth (ct_sumr is 1 level shallower)
    while (ct_sumr.depth() < ct_offset.depth())
        evaluator.mod_drop_inplace(ct_sumr);

    evaluator.add_inplace(ct_sumr, ct_offset);

    if (g_verbose)
        std::cout << "Tie-corrected ranking complete.\n";
    return ct_sumr;
}
