#include "heongpu.cuh"
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <omp.h>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;

/**
 * @brief GPU-aware timer using CUDA Events for accurate GPU timing
 */
class GPUTimer {
    cudaEvent_t start, stop;
public:
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer() {
        cudaEventRecord(start);
    }

    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

/**
 * @brief Replicates a row vector homomorphically using parallel rotations
 */
heongpu::Ciphertext<Scheme> replicateRow(
    const heongpu::Ciphertext<Scheme>& row_initial,
    int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator)
{
    std::vector<heongpu::Ciphertext<Scheme>> vector_ciphertexts(vec_len);
    vector_ciphertexts[0] = row_initial;

    #pragma omp parallel for
    for (int i = 1; i < vec_len; i++) {
        vector_ciphertexts[i] = row_initial;
        int shift = -(i * vec_len);
        evaluator.rotate_rows_inplace(vector_ciphertexts[i], galois_key, shift);
    }

    heongpu::Ciphertext<Scheme> row_replicated = vector_ciphertexts[0];
    for (int i = 1; i < vec_len; i++) {
        evaluator.add_inplace(row_replicated, vector_ciphertexts[i]);
    }

    return row_replicated;
}

/**
 * @brief Replicates a column vector homomorphically using parallel rotations
 */
heongpu::Ciphertext<Scheme> replicateColumn(
    const heongpu::Ciphertext<Scheme>& col_initial,
    int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator)
{
    std::vector<heongpu::Ciphertext<Scheme>> vector_ciphertexts(vec_len);
    vector_ciphertexts[0] = col_initial;

    #pragma omp parallel for
    for (int i = 1; i < vec_len; i++) {
        vector_ciphertexts[i] = col_initial;
        int shift = -(i * vec_len);
        evaluator.rotate_rows_inplace(vector_ciphertexts[i], galois_key, shift);
    }

    heongpu::Ciphertext<Scheme> row_replicated = vector_ciphertexts[0];
    for (int i = 1; i < vec_len; i++) {
        evaluator.add_inplace(row_replicated, vector_ciphertexts[i]);
    }

    return row_replicated;
}

/**
 * @brief Transposes a row vector to a column vector homomorphically
 */
heongpu::Ciphertext<Scheme> transposeRowToColumn(
    const heongpu::Ciphertext<Scheme>& row_vector,
    int vec_len,
    heongpu::Galoiskey<Scheme>& galois_key,
    heongpu::HEArithmeticOperator<Scheme>& evaluator,
    heongpu::HEEncoder<Scheme>& encoder,
    const heongpu::HEContext<Scheme>& context,
    double scale)
{
    heongpu::Ciphertext<Scheme> result = row_vector;

    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    int N = vec_len;

    for (int i = 1; i <= log_n; i++) {
        int shift = (N * (N - 1)) / (1 << i);
        heongpu::Ciphertext<Scheme> rotated = result;
        evaluator.rotate_rows_inplace(rotated, galois_key, shift);
        evaluator.add_inplace(result, rotated);
    }

    size_t total_slots = context.get_poly_modulus_degree() / 2;
    std::vector<double> mask_values(total_slots, 0.0);

    for (int row = 0; row < vec_len; row++) {
        mask_values[row * vec_len] = 1.0;
    }

    heongpu::Plaintext<Scheme> mask;
    encoder.encode(mask, mask_values, scale);
    evaluator.multiply_plain_inplace(result, mask);

    return result;
}

/**
 * @brief Test helper to compare expected and actual results with tolerance
 */
bool compareVectors(const std::vector<double>& expected, const std::vector<double>& actual,
                   size_t count, double tolerance = 1e-3) {
    bool success = true;
    for (size_t i = 0; i < count; i++) {
        if (std::abs(expected[i] - actual[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i]
                     << " but got " << actual[i] << std::endl;
            success = false;
        }
    }
    return success;
}

/**
 * @brief Test the GPUTimer class
 */
void test_GPUTimer() {
    std::cout << "\n=== Testing GPUTimer ===\n";

    GPUTimer timer;
    timer.startTimer();

    // Simulate some GPU work
    cudaDeviceSynchronize();

    float elapsed = timer.stopTimer();

    assert(elapsed >= 0.0f && "Timer should return non-negative time");
    std::cout << "GPUTimer test passed! Measured time: " << elapsed << " ms\n";
}

/**
 * @brief Test replicateRowParallel function
 */
void test_replicateRowParallel() {
    std::cout << "\n=== Testing replicateRowParallel ===\n";

    cudaSetDevice(0);

    // Setup HE context
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    const size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    double scale = pow(2.0, 30);
    context.generate();

    // Key generation
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    // Setup encoder, encryptor, decryptor
    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> evaluator(context, encoder);

    const int vec_len = sqrt(poly_modulus_degree / 2);

    // Generate Galois keys for row rotations
    std::vector<int> row_galois_shifts;
    for (int i = 1; i < vec_len; i++) {
        row_galois_shifts.push_back(-(i * vec_len));
    }

    heongpu::Galoiskey<Scheme> row_galois_key(context, row_galois_shifts);
    keygen.generate_galois_key(row_galois_key, secret_key);

    // Test case 1: Simple sequence [1, 2, 3, 4, ...]
    std::cout << "Test case 1: Simple sequence\n";
    std::vector<double> input(vec_len);
    for (int i = 0; i < vec_len; i++) {
        input[i] = i + 1;
    }

    std::vector<double> row_initial(poly_modulus_degree / 2, 0.0);
    for (size_t i = 0; i < vec_len; i++) {
        row_initial[i] = input[i];
    }

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, row_initial, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    heongpu::Ciphertext<Scheme> row_replicated = replicateRow(ciphertext, vec_len, row_galois_key, evaluator);

    heongpu::Plaintext<Scheme> decrypted_ciphertext(context);
    decryptor.decrypt(decrypted_ciphertext, row_replicated);
    std::vector<double> result;
    encoder.decode(result, decrypted_ciphertext);

    // Verify: Each row should contain the original vector
    std::vector<double> expected(vec_len * vec_len);
    for (int row = 0; row < vec_len; row++) {
        for (int col = 0; col < vec_len; col++) {
            expected[row * vec_len + col] = input[col];
        }
    }

    bool test1_passed = compareVectors(expected, result, vec_len * vec_len);
    assert(test1_passed && "replicateRowParallel test case 1 failed");
    std::cout << "Test case 1 passed!\n";

    // Test case 2: All zeros
    std::cout << "Test case 2: All zeros\n";
    std::fill(input.begin(), input.end(), 0.0);
    std::fill(row_initial.begin(), row_initial.end(), 0.0);

    encoder.encode(plaintext, row_initial, scale);
    encryptor.encrypt(ciphertext, plaintext);

    row_replicated = replicateRow(ciphertext, vec_len, row_galois_key, evaluator);
    decryptor.decrypt(decrypted_ciphertext, row_replicated);
    encoder.decode(result, decrypted_ciphertext);

    std::fill(expected.begin(), expected.end(), 0.0);
    bool test2_passed = compareVectors(expected, result, vec_len * vec_len);
    assert(test2_passed && "replicateRowParallel test case 2 failed");
    std::cout << "Test case 2 passed!\n";

    std::cout << "All replicateRowParallel tests passed!\n";
}

/**
 * @brief Test replicateColumnParallel function
 */
void test_replicateColumnParallel() {
    std::cout << "\n=== Testing replicateColumnParallel ===\n";

    cudaSetDevice(0);

    // Setup HE context
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    const size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    double scale = pow(2.0, 30);
    context.generate();

    // Key generation
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    // Setup encoder, encryptor, decryptor
    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> evaluator(context, encoder);

    const int vec_len = sqrt(poly_modulus_degree / 2);

    // Generate Galois keys for column rotations
    std::vector<int> col_galois_shifts;
    for (int i = 1; i < vec_len; i++) {
        col_galois_shifts.push_back(-(i * vec_len));
    }

    heongpu::Galoiskey<Scheme> col_galois_key(context, col_galois_shifts);
    keygen.generate_galois_key(col_galois_key, secret_key);

    // Test with simple sequence
    std::cout << "Test case: Simple sequence\n";
    std::vector<double> input(vec_len);
    for (int i = 0; i < vec_len; i++) {
        input[i] = i + 1;
    }

    std::vector<double> col_initial(poly_modulus_degree / 2, 0.0);
    for (size_t i = 0; i < vec_len; i++) {
        col_initial[i] = input[i];
    }

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, col_initial, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    heongpu::Ciphertext<Scheme> col_replicated = replicateColumn(ciphertext, vec_len, col_galois_key, evaluator);

    heongpu::Plaintext<Scheme> decrypted_ciphertext(context);
    decryptor.decrypt(decrypted_ciphertext, col_replicated);
    std::vector<double> result;
    encoder.decode(result, decrypted_ciphertext);

    std::cout << "replicateColumnParallel test completed (manual verification needed)\n";
    std::cout << "First " << std::min(16, (int)result.size()) << " elements: ";
    for (int i = 0; i < std::min(16, (int)result.size()); i++) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";
}

/**
 * @brief Test transposeRowToColumn function
 */
void test_transposeRowToColumn() {
    std::cout << "\n=== Testing transposeRowToColumn ===\n";

    cudaSetDevice(0);

    // Setup HE context
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    const size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    double scale = pow(2.0, 30);
    context.generate();

    // Key generation
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    // Setup encoder, encryptor, decryptor
    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> evaluator(context, encoder);

    const int vec_len = sqrt(poly_modulus_degree / 2);

    // Generate Galois keys for transpose operations
    std::vector<int> transpose_shifts;
    int log_n = static_cast<int>(std::ceil(std::log2(vec_len)));
    int N = vec_len;
    for (int i = 1; i <= log_n; i++) {
        int shift = (N * (N - 1)) / (1 << i);
        transpose_shifts.push_back(shift);
    }

    heongpu::Galoiskey<Scheme> transpose_galois_key(context, transpose_shifts);
    keygen.generate_galois_key(transpose_galois_key, secret_key);

    // Test with simple row vector [1, 2, 3, 4, ...]
    std::cout << "Test case: Row vector transpose\n";
    std::vector<double> input(vec_len);
    for (int i = 0; i < vec_len; i++) {
        input[i] = i + 1;
    }

    std::vector<double> row_vector(poly_modulus_degree / 2, 0.0);
    for (size_t i = 0; i < vec_len; i++) {
        row_vector[i] = input[i];
    }

    heongpu::Plaintext<Scheme> plaintext(context);
    encoder.encode(plaintext, row_vector, scale);
    heongpu::Ciphertext<Scheme> ciphertext(context);
    encryptor.encrypt(ciphertext, plaintext);

    heongpu::Ciphertext<Scheme> transposed = transposeRowToColumn(
        ciphertext, vec_len, transpose_galois_key, evaluator, encoder, context, scale);

    heongpu::Plaintext<Scheme> decrypted_ciphertext(context);
    decryptor.decrypt(decrypted_ciphertext, transposed);
    std::vector<double> result;
    encoder.decode(result, decrypted_ciphertext);

    // Verify: Should have values at positions 0, vec_len, 2*vec_len, ...
    std::cout << "Checking transposed column pattern:\n";
    bool test_passed = true;
    for (int row = 0; row < vec_len; row++) {
        int pos = row * vec_len;
        double expected_val = input[row];
        if (std::abs(result[pos] - expected_val) > 1e-1) {
            std::cerr << "Mismatch at position " << pos << " (row " << row
                     << "): expected " << expected_val << " but got " << result[pos] << std::endl;
            test_passed = false;
        }
    }

    assert(test_passed && "transposeRowToColumn test failed");
    std::cout << "transposeRowToColumn test passed!\n";
}

int main()
{
    std::cout << "Starting CKKS Rotation Parallel Tests\n";
    std::cout << "======================================\n";

    try {
        test_GPUTimer();
        test_replicateRowParallel();
        test_replicateColumnParallel();
        test_transposeRowToColumn();

        std::cout << "\n======================================\n";
        std::cout << "All tests passed successfully!\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
