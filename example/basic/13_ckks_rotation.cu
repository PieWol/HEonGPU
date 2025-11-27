#include "heongpu.cuh"
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// Set up HE Scheme
constexpr auto Scheme = heongpu::Scheme::CKKS;


// The core idea of our design is to manipulate the encrypted vector in suc4h a way that only a single evaluation of the comparison function is needed to compare all values
// vector v = (v1,v2,v3), we produce vR = (v1,v2,v3,v1,v2,v3,v1,v2,v3), vC = (v1,v1,v1,v2,v2,v2,v3,v3,v3).

int main()
{
    cudaSetDevice(0);

    // HE Context initialisieren
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    const size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    double scale = pow(2.0, 30);
    context.generate();

    // Schlüsselerzeugung
    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);
    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    // Encoder, Encryptor, Decryptor
    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> evaluator(context, encoder);

    // Vektor befüllen. Aktuell fixed length
    std::vector<double> input = { 1, 2, 3, 4};
    int vec_len = input.size();


    // Calculate total slots needed for the matrix
    int matrix_slots = vec_len * vec_len;  // n = √n × √n

    // Validate that matrix fits in available slots
    int available_slots = poly_modulus_degree / 2;
    if (matrix_slots > available_slots) {
        std::cerr << "Error: Matrix size " << vec_len << "x" << vec_len
                << " (" << matrix_slots << " slots) exceeds available slots ("
                << available_slots << ")" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Matrix size: " << vec_len << "x" << vec_len
              << " (using " << matrix_slots << "/" << available_slots << " slots)\n";

    // Generate Galois keys for rotations
    // For row replication: need rotation by -vec_len (rotate left to fill next row)
    std::vector<int> galois_shift;
    galois_shift.push_back(-vec_len);

    std::cout << "Generating Galois key for rotation by: " << (-vec_len) << "\n";

    heongpu::Galoiskey<Scheme> galois_key(context, galois_shift);
    keygen.generate_galois_key(galois_key, secret_key);

    // we adopt the row- by-row approach [20],
    // which consists of concatenating each row into a single vector and then encrypting it. For a square matrix of size N, we have the requirement that N2 ≤ n/2, where n
    // is the ring dimension, otherwise multiple ciphertexts are needed to store the entire matrix.

    // ===== Row Replication via Homomorphic Rotations =====
    std::cout << "\n=== Homomorphic Row Replication ===\n";
    std::cout << "Original input vector: ";
    display_vector(input, vec_len);

    // Prepare initial row vector: first vec_len slots contain input, rest is zeros
    std::vector<double> row_initial(poly_modulus_degree / 2, 0.0);
    for (size_t i = 0; i < vec_len; i++) {
        row_initial[i] = input[i];
    }

    // Encode and encrypt the initial row
    heongpu::Plaintext<Scheme> row_plaintext(context);
    encoder.encode(row_plaintext, row_initial, scale);
    heongpu::Ciphertext<Scheme> row_ciphertext(context);
    encryptor.encrypt(row_ciphertext, row_plaintext);

    // Replicate row using rotations and additions
    // Result = original + rotate(vec_len) + rotate(2*vec_len) + ... + rotate((vec_len-1)*vec_len)
    heongpu::Ciphertext<Scheme> row_replicated(context);
    encryptor.encrypt(row_replicated, row_plaintext);  // Start with a fresh copy of the original

    std::cout << "Applying rotations: ";
    heongpu::Ciphertext<Scheme> temp_rotated(context);
    encryptor.encrypt(temp_rotated, row_plaintext);  // Start with a copy to rotate

    for (int i = 1; i < vec_len; i++) {
        std::cout << (i * vec_len) << " ";

        evaluator.rotate_rows_inplace(temp_rotated, galois_key, -vec_len);  // Negative rotation!
        evaluator.add_inplace(row_replicated, temp_rotated);
    }
    std::cout << "\n";

    // Decrypt and verify the row replication
    heongpu::Plaintext<Scheme> decrypted_ciphertext(context);
    decryptor.decrypt(decrypted_ciphertext, row_replicated);
    std::vector<double> row_result;
    encoder.decode(row_result, decrypted_ciphertext);

    std::cout << "Replicated row vector:\n";
    display_vector(row_result, vec_len * vec_len);

    std::cout << "\nExpected pattern (VR): each element repeated across all rows\n";
    std::cout << "Row 0: [" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << "]\n";
    std::cout << "Row 1: [" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << "]\n";
    std::cout << "Row 2: [" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << "]\n";
    std::cout << "Row 3: [" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << "]\n";


    // Todo: compare values

    return EXIT_SUCCESS;
}