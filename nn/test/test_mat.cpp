#include <gtest/gtest.h>
#include "../include/mat.hpp"  // Adjust path to your Mat class header

using namespace nn::mathops;

// Helper: check two arrays element-wise
static void expect_array_eq(const float* expected, const float* actual, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		EXPECT_FLOAT_EQ(expected[i], actual[i]) << "Mismatch at index " << i;
	}
}

TEST(MatTest, ConstructorRowsCols) {
	Mat<float> A(2, 3);
	const Shape &s = A.get_shape();
	EXPECT_EQ(s.rows, 2);
	EXPECT_EQ(s.cols, 3);

	float *raw = const_cast<float*>(A.get_mat_raw());
	EXPECT_NE(raw, nullptr);  // matrix memory allocated
}

TEST(MatTest, ConstructorShape) {
	Shape shape(2, 3);
	Mat<float> A(shape);
	const Shape &s = A.get_shape();
	EXPECT_EQ(s.rows, 2);
	EXPECT_EQ(s.cols, 3);

	float *raw = const_cast<float*>(A.get_mat_raw());
	EXPECT_NE(raw, nullptr);
}

TEST(MatTest, CopyConstructor) {
	Mat<float> A = {
		{1.0f, 2.0f},
		{3.0f, 4.0f}
	};
	
	// Fill matrix with some values
	Mat<float> B(A);  // Copy constructor
	const Shape &s = B.get_shape();
	EXPECT_EQ(s.rows, 2);
	EXPECT_EQ(s.cols, 2);
	expect_array_eq(A.get_mat_raw(), B.get_mat_raw(), 4);
}

TEST(MatTest, MoveConstructor) {
	Mat<float> A(2, 2);
	float *raw_ptr = const_cast<float*>(A.get_mat_raw());

	Mat<float> B(std::move(A));  // Move constructor
	const Shape &s = B.get_shape();
	EXPECT_EQ(s.rows, 2);
	EXPECT_EQ(s.cols, 2);
	EXPECT_EQ(raw_ptr, B.get_mat_raw());  // Memory pointer transferred
	EXPECT_EQ(A.get_mat_raw(), nullptr);
}


// TODO: Find the way of test this
// TEST(MatTest, Destructor) {
// 	Mat<float> *A = new Mat<float>(2, 2);
// 	EXPECT_NE(A->get_mat_raw(), nullptr);
// 	delete A;
// 	EXPECT_EQ(A->get_mat_raw(), nullptr);
// }

TEST(MatTest, Dot2x2) {
	Mat<float> A(2, 2);
	Mat<float> B(2, 2);

	// A = [1 2; 3 4]
	float a_vals[] = {1, 2, 3, 4};
	std::copy(a_vals, a_vals + 4, A.get_mat_raw());

	// B = [5 6; 7 8]
	float b_vals[] = {5, 6, 7, 8};
	std::copy(b_vals, b_vals + 4, B.get_mat_raw());

	Mat<float> C = A.dot(B);

	EXPECT_FLOAT_EQ(C.get_mat_raw()[0], 19); // row0 col0
	EXPECT_FLOAT_EQ(C.get_mat_raw()[1], 22); // row0 col1
	EXPECT_FLOAT_EQ(C.get_mat_raw()[2], 43); // row1 col0
	EXPECT_FLOAT_EQ(C.get_mat_raw()[3], 50); // row1 col1
}


TEST(MatTest, Dot2x2V2) {
	Mat<float> A(2, 2);
	Mat<float> B(2, 2);

	// A = [1 2; 3 4]
	float a_vals[] = {1, 2, 3, 4};
	std::copy(a_vals, a_vals + 4, A.get_mat_raw());

	// B = [5 6; 7 8]
	float b_vals[] = {5, 6, 7, 8};
	std::copy(b_vals, b_vals + 4, B.get_mat_raw());

	Mat<float> C;
	C = A.dot(B);

	EXPECT_FLOAT_EQ(C.get_mat_raw()[0], 19); // row0 col0
	EXPECT_FLOAT_EQ(C.get_mat_raw()[1], 22); // row0 col1
	EXPECT_FLOAT_EQ(C.get_mat_raw()[2], 43); // row1 col0
	EXPECT_FLOAT_EQ(C.get_mat_raw()[3], 50); // row1 col1
}


TEST(MatTest, DotAndAssign2x2) {
	Mat<float> A(2, 2);
	Mat<float> B(2, 2);

	float a_vals[] = {1, 2, 3, 4};
	std::copy(a_vals, a_vals + 4, A.get_mat_raw());

	float b_vals[] = {5, 6, 7, 8};
	std::copy(b_vals, b_vals + 4, B.get_mat_raw());

	A.dot_and_assign(B);

	EXPECT_FLOAT_EQ(A.get_mat_raw()[0], 19);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[1], 22);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[2], 43);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[3], 50);
}


TEST(MatTest, MoveOperatorAssign) {
	Mat<float> A(2, 2);
	Mat<float> B(2, 2);

	float a_vals[] = {1, 2, 3, 4};
	std::copy(a_vals, a_vals + 4, A.get_mat_raw());

	float b_vals[] = {5, 6, 7, 8};
	std::copy(b_vals, b_vals + 4, B.get_mat_raw());

	Mat<float> C1;
	C1 = std::move(A);
	EXPECT_EQ(A.get_mat_raw(), nullptr);
}

TEST(MatTest, AddScalar) {
	Mat<float> A(2, 2);
	float* raw = const_cast<float*>(A.get_mat_raw());
	raw[0] = 1.0f; raw[1] = 2.0f;
	raw[2] = 3.0f; raw[3] = 4.0f;

	Mat<float> B = A + 5.0f;

	EXPECT_FLOAT_EQ(B.get_mat_raw()[0], 6.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[1], 7.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[2], 8.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[3], 9.0f);
}

TEST(MatTest, AddAssignScalar) {
	Mat<float> A(2, 2);
	float* raw = const_cast<float*>(A.get_mat_raw());
	raw[0] = 1.0f; raw[1] = 2.0f;
	raw[2] = 3.0f; raw[3] = 4.0f;

	A += 10.0f;

	EXPECT_FLOAT_EQ(A.get_mat_raw()[0], 11.0f);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[1], 12.0f);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[2], 13.0f);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[3], 14.0f);
}

TEST(MatTest, SubScalar) {
	Mat<float> A(1, 3);
	float* raw = const_cast<float*>(A.get_mat_raw());
	raw[0] = 5.0f; raw[1] = 10.0f; raw[2] = 15.0f;

	Mat<float> B = A - 5.0f;

	EXPECT_FLOAT_EQ(B.get_mat_raw()[0], 0.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[1], 5.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[2], 10.0f);
}

TEST(MatTest, MulScalar) {
	Mat<float> A(2, 2);
	float* raw = const_cast<float*>(A.get_mat_raw());
	raw[0] = 1.0f; raw[1] = -2.0f;
	raw[2] = 3.0f; raw[3] = -4.0f;

	Mat<float> B = A * 2.0f;

	EXPECT_FLOAT_EQ(B.get_mat_raw()[0], 2.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[1], -4.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[2], 6.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[3], -8.0f);
}

TEST(MatTest, DivScalar) {
	Mat<float> A(2, 2);
	float* raw = const_cast<float*>(A.get_mat_raw());
	raw[0] = 10.0f; raw[1] = 20.0f;
	raw[2] = 30.0f; raw[3] = 40.0f;

	Mat<float> B = A / 10.0f;

	EXPECT_FLOAT_EQ(B.get_mat_raw()[0], 1.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[1], 2.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[2], 3.0f);
	EXPECT_FLOAT_EQ(B.get_mat_raw()[3], 4.0f);
}

TEST(MatTest, DivAssignScalar) {
	Mat<float> A(2, 2);
	float* raw = const_cast<float*>(A.get_mat_raw());
	raw[0] = 10.0f; raw[1] = 20.0f;
	raw[2] = 30.0f; raw[3] = 40.0f;

	A /= 10.0f;

	EXPECT_FLOAT_EQ(A.get_mat_raw()[0], 1.0f);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[1], 2.0f);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[2], 3.0f);
	EXPECT_FLOAT_EQ(A.get_mat_raw()[3], 4.0f);
}


TEST(MatTest, GrandSumBasic) {
	Mat<float> A(2, 3);  // 2x3 matrix
	float* raw = const_cast<float*>(A.get_mat_raw());

	// Fill matrix with known values
	raw[0] = 1.0f; raw[1] = 2.0f; raw[2] = 3.0f;
	raw[3] = 4.0f; raw[4] = 5.0f; raw[5] = 6.0f;

	float sum = A.grand_sum();
	EXPECT_FLOAT_EQ(sum, 21.0f); // 1+2+3+4+5+6
}


TEST(MatTest, TransposeInPlace) {
	Mat<float> A(2, 3);  // 2x3 matrix
	float* raw = const_cast<float*>(A.get_mat_raw());

	// Fill matrix
	raw[0] = 1; raw[1] = 2; raw[2] = 3;
	raw[3] = 4; raw[4] = 5; raw[5] = 6;
	
	A.transpose();  // In-place transpose, now should be 3x2
	
	float* t_raw = const_cast<float*>(A.get_mat_raw());

	// Expected layout after transpose:
	// [1 4]
	// [2 5]
	// [3 6]
	EXPECT_EQ(A.get_shape().rows, 3);
	EXPECT_EQ(A.get_shape().cols, 2);

	EXPECT_FLOAT_EQ(t_raw[0], 1.0f);
	EXPECT_FLOAT_EQ(t_raw[1], 4.0f);
	EXPECT_FLOAT_EQ(t_raw[2], 2.0f);
	EXPECT_FLOAT_EQ(t_raw[3], 5.0f);
	EXPECT_FLOAT_EQ(t_raw[4], 3.0f);
	EXPECT_FLOAT_EQ(t_raw[5], 6.0f);
}


TEST(MatTest, ElementAccessOperator) {
	Mat<float> A(2, 3); // 2x3 matrix
	float* raw = const_cast<float*>(A.get_mat_raw());

	// Fill matrix manually
	raw[0] = 1.0f; raw[1] = 2.0f; raw[2] = 3.0f;
	raw[3] = 4.0f; raw[4] = 5.0f; raw[5] = 6.0f;

	// Test element access via operator()
	EXPECT_FLOAT_EQ(A(0, 0), 1.0f);
	EXPECT_FLOAT_EQ(A(0, 1), 2.0f);
	EXPECT_FLOAT_EQ(A(0, 2), 3.0f);
	EXPECT_FLOAT_EQ(A(1, 0), 4.0f);
	EXPECT_FLOAT_EQ(A(1, 1), 5.0f);
	EXPECT_FLOAT_EQ(A(1, 2), 6.0f);

	// Test modifying elements via operator()
	A(0, 0) = 10.0f;
	EXPECT_FLOAT_EQ(A(0, 0), 10.0f);
	EXPECT_FLOAT_EQ(raw[0], 10.0f); // underlying raw storage updated
}


