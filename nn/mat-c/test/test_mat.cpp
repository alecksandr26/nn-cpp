#include <gtest/gtest.h>
#include <algorithm> // for std::copy
#include <cstring>   // for std::memcmp

extern "C" {
#include "../include/mat.h"
}

/* Helper: check two arrays element-wise with EXPECT_FLOAT_EQ */
static void expect_array_eq(const float *expected, const float *actual, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		EXPECT_FLOAT_EQ(expected[i], actual[i]) << "mismatch at index " << i;
	}
}

/* Helper: check two arrays element-wise with EXPECT_NEAR and given eps */
static void expect_array_near(const float *expected, const float *actual, size_t n, double eps = 1e-6) {
	for (size_t i = 0; i < n; ++i) {
		EXPECT_NEAR(expected[i], actual[i], eps) << "mismatch at index " << i;
	}
}

/* Helper: fill array with an arithmetic sequence start, step */
static void fill_seq(float *A, size_t n, float start = 1.0f, float step = 1.0f) {
	float v = start;
	for (size_t i = 0; i < n; ++i) {
		A[i] = v;
		v += step;
	}
}

/* --- Tests --- */

// Basic fill: set all elements to a constant
TEST(Matf32Test, FillMatrix) {
	float A[2 * 3];
	Matf32_fill(A, 2, 3, 5.0f);

	float expected[6] = {5,5,5,5,5,5};
	expect_array_eq(expected, A, 6);
}

// add_scalar and mul_scalar basic functionality
TEST(Matf32Test, AddAndMulScalar) {
	float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	float expected_add[4] = {2.5f, 3.5f, 4.5f, 5.5f};
	float expected_mul[4] = {5.0f, 7.0f, 9.0f, 11.0f};

	// add scalar
	Matf32_add_scalar(A, 2, 2, 1.5f);
	expect_array_eq(expected_add, A, 4);
	
	// multiply by scalar (in-place)
	Matf32_mul_scalar(A, 2, 2, 2.f);
	expect_array_eq(expected_mul, A, 4);
}

// Element-wise add and multiply (Hadamard)
TEST(Matf32Test, ElementWiseAddAndMul) {
	float A[4];
	float B[4];
	float C[4];

	fill_seq(A, 4, 1.0f); // [1,2,3,4]
	fill_seq(B, 4, 5.0f); // [5,6,7,8]

	// expected results
	float expected_add[4];
	float expected_mul[4];
	for (size_t i = 0; i < 4; ++i) {
		expected_add[i] = A[i] + B[i];
		expected_mul[i] = A[i] * B[i];
	}

	Matf32_add(A, B, C, 2, 2);
	expect_array_eq(expected_add, C, 4);

	Matf32_mul(A, B, C, 2, 2);
	expect_array_eq(expected_mul, C, 4);
}

// In-place element-wise addition: use C == A (destination same as first operand)
TEST(Matf32Test, InPlaceAdd) {
	float A[4];
	float B[4];
	float oldA[4];

	fill_seq(A, 4, 1.0f); // [1,2,3,4]
	fill_seq(B, 4, 10.0f); // [10,11,12,13]
	std::copy(A, A+4, oldA);

	// Do A = A + B in-place
	Matf32_add(A, B, A, 2, 2);

	float expected[4];
	for (size_t i = 0; i < 4; ++i) expected[i] = oldA[i] + B[i];
	expect_array_eq(expected, A, 4);
}

// Dot product: small 2x3 * 3x2 example (exact integers)
TEST(Matf32Test, DotProductSmall) {
	float A[6] = {
		1, 2, 3,
		4, 5, 6
	}; // 2x3

	float B[6] = {
		7, 8,
		9, 10,
		11, 12
	}; // 3x2

	float C[4];
	Matf32_dot(A, B, C, 2, 3, 2);

	// Expected:
	// [58, 64]
	// [139, 154]
	float expected[4] = {58.0f, 64.0f, 139.0f, 154.0f};
	expect_array_near(expected, C, 4);
}

// Dot product with identity matrix -> result equals the original matrix
TEST(Matf32Test, DotProductIdentityRight) {
	const size_t N = 4;
	float A[N * N];
	float I[N * N];
	float C[N * N];

	// fill A with 1..16
	fill_seq(A, N*N, 1.0f);

	// create identity I
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			I[i * N + j] = (i == j) ? 1.0f : 0.0f;
	
	Matf32_dot(A, I, C, N, N, N);
	expect_array_near(A, C, N*N);
}

// Zero-dimension operations should be no-ops (leave buffer unchanged)
TEST(Matf32Test, ZeroSizeNoOp) {
	float buf[6] = { 42.0f, 42.0f, 42.0f, 42.0f, 42.0f, 42.0f };
	float expected[6];
	std::copy(buf, buf+6, expected);

	// Zero rows
	Matf32_fill(buf, 0, 6, 7.0f);           // no elements should be modified
	Matf32_add_scalar(buf, 0, 6, 1.0f);
	Matf32_mul_scalar(buf, 0, 6, 2.0f);

	expect_array_eq(expected, buf, 6);

	// Zero cols
	Matf32_fill(buf, 6, 0, 9.0f);
	Matf32_add_scalar(buf, 6, 0, 1.0f);
	Matf32_mul_scalar(buf, 6, 0, 2.0f);
	expect_array_eq(expected, buf, 6);
}

/* Optional: larger-ish test that checks a 3x2 * 2x5 multiply against manual compute */
TEST(Matf32Test, DotProductNonSquare) {
	const size_t R = 3, K = 2, Cc = 5;
	float A[R*K];
	float B[K*Cc];
	float out[R*Cc];
	float expected[R*Cc];

	fill_seq(A, R*K, 1.0f); // 1..6
	fill_seq(B, K*Cc, 10.0f); // 10..19

	// compute expected result (naive)
	for (size_t i = 0; i < R; ++i) {
		for (size_t j = 0; j < Cc; ++j) {
			float s = 0.0f;
			for (size_t k = 0; k < K; ++k) {
				s += A[i*K + k] * B[k*Cc + j];
			}
			expected[i*Cc + j] = s;
		}
	}

	Matf32_dot(A, B, out, R, K, Cc);
	expect_array_near(expected, out, R*Cc);
}


// --- Test for Matf32_copy ---
TEST(Matf32Test, CopyMatrix) {
	const size_t rows = 2;
	const size_t cols = 3;
	const size_t total = rows * cols;

	float src[total];
	float dst[total];

	fill_seq(src, total, 1.0f, 2.0f); // 1,3,5,7,9,11
	for (size_t i = 0; i < total; ++i) dst[i] = 0.0f;
	
	Matf32_copy(src, dst, rows, cols);
	expect_array_eq(src, dst, total);
}

TEST(Matf32Test, MixedValues) {
	float A[5] = {2.5f, -1.5f, 0.0f, 4.0f, -2.0f};
	EXPECT_FLOAT_EQ(Matf32_grand_sum(A, 1, 5), 3.0f); // 2.5-1.5+0+4-2 = 3
}

TEST(Matf32Test, SubScalarBasic) {
	float A[4] = {5.0f, 6.0f, 7.0f, 8.0f};
	Matf32_sub_scalar(A, 2, 2, 2.0f);  // subtract 2 from each element

	float expected[4] = {3.0f, 4.0f, 5.0f, 6.0f};
	expect_array_eq(expected, A, 4);
}

TEST(Matf32Test, DivScalarBasic) {
	float A[4] = {2.0f, 4.0f, 6.0f, 8.0f};
	Matf32_div_scalar(A, 2, 2, 2.0f);  // divide each element by 2

	float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	expect_array_eq(expected, A, 4);
}


TEST(Matf32Test, SubBasic) {
	float A[4] = {5.0f, 7.0f, 9.0f, 11.0f};
	float B[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	float C[4];

	Matf32_sub(A, B, C, 2, 2);

	float expected[4] = {4.0f, 5.0f, 6.0f, 7.0f};
	expect_array_eq(expected, C, 4);
}


TEST(Matf32Test, DivBasic) {
	float A[4] = {10.0f, 20.0f, 30.0f, 40.0f};
	float B[4] = {2.0f, 4.0f, 5.0f, 10.0f};
	float C[4];

	Matf32_div(A, B, C, 2, 2);

	float expected[4] = {5.0f, 5.0f, 6.0f, 4.0f};
	expect_array_eq(expected, C, 4);
}

TEST(Matf32Test, TransposeSquareMatrix) {
	float A[4] = {1, 2,
		      3, 4}; // 2x2 matrix
	float B[4] = {0};

	Matf32_transpose(A, B, 2, 2);

	float expected[4] = {1, 3,
			     2, 4};
	expect_array_eq(expected, B, 4);
}

TEST(Matf32Test, SumSquareMatrix) {
	float A[4] = {1, 2, 3, 4}; // 2x2 matrix
	float sum = Matf32_grand_sum(A, 2, 2);
	EXPECT_FLOAT_EQ(sum, 10.0f);
}
