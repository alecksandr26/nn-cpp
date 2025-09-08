#include <assert.h>
#include <stddef.h>

/* TODO: Implement the optional versions */

/* Matf32_fill: set all elements of matrix `A` of `nrows` and `ncols` to value a */
void Matf32_fill(float *A, size_t nrows, size_t ncols, float a) {
	assert(A && "A can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		A[i] = a;
	}
}

/* Matf32_add_scalar: add scalar a to all elements of matrix `A` of `nrows` and `ncols` */
void Matf32_add_scalar(float *A, size_t nrows, size_t ncols, float a) {
	assert(A && "A can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		A[i] += a;
	}
}

/* Matf32_sub_scalar: sub scalar a to all elements of matrix `A` of `nrows` and `ncols` */
void Matf32_sub_scalar(float *A, size_t nrows, size_t ncols, float a) {
	assert(A && "A can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		A[i] -= a;
	}
}

/* Matf32_mul_scalar: multiply all elements of matrix `A` of `nrows` and `ncols` by scalar a */
void Matf32_mul_scalar(float *A, size_t nrows, size_t ncols, float a) {
	assert(A && "A can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		A[i] *= a;
	}
}

/* Matf32_div_scalar: divide all elements of matrix `A` of `nrows` and `ncols` by scalar a */
void Matf32_div_scalar(float *A, size_t nrows, size_t ncols, float a) {
	assert(A && "A can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		A[i] /= a;
	}
}

/* Matf32_add: element-wise addition of matrices A and B, result in C (all of size nrows x ncols) */
void Matf32_add(const float *A, const float *B, float *C, size_t nrows, size_t ncols) {
	assert(A && "A can't be null");
	assert(B && "B can't be null");
	assert(C && "C can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		C[i] = A[i] + B[i];
	}
}


/* Matf32_sub: element-wise subtraction of matrices A and B, result in C (all of size nrows x ncols) */
void Matf32_sub(const float *A, const float *B, float *C, size_t nrows,
		size_t ncols)
{
	assert(A && "A can't be null");
	assert(B && "B can't be null");
	assert(C && "C can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		C[i] = A[i] - B[i];
	}
}

/* Matf32_mul: element-wise (Hadamard) multiplication of matrices A and B, result in C (all of size nrows x ncols) */
void Matf32_mul(const float *A, const float *B, float *C, size_t nrows, size_t ncols)
{
	assert(A && "A can't be null");
	assert(B && "B can't be null");
	assert(C && "C can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		C[i] = A[i] * B[i];
	}
}


/* Matf32_div: element-wise (Hadamard) division of matrices A and B, result in C (all of size nrows x ncols) */
void Matf32_div(const float *A, const float *B, float *C, size_t nrows, size_t ncols)
{
	assert(A && "A can't be null");
	assert(B && "B can't be null");
	assert(C && "C can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		C[i] = A[i] / B[i];
	}
}


/* Matf32_copy: copy matrix src (nrows x ncols) into dst */
void Matf32_copy(const float *src, float *dst, size_t nrows, size_t ncols) {
	assert(src && "Can't src be null");
	assert(dst && "Can't src be null");
	
	for (size_t i = 0; i < nrows; ++i) {
		for (size_t j = 0; j < ncols; j++) {
			dst[i * ncols + j] = src[i * ncols + j];
		}
	}
}

/* Matf32_grand_sum: Compute and return the sum of all elements in the matrix */
float Matf32_grand_sum(const float *A, size_t nrows, size_t ncols)
{
	assert(A && "Can't A be null");

	float sum = 0.0;
	for (size_t i = 0; i < nrows; ++i) {
		for (size_t j = 0; j < ncols; ++j) {
			sum += A[i * ncols + j];
		}
	}
	
	return sum;
}

/* Matf32_transpose: Transposes an nrows x ncols matrix A into B (B = A^T)  */
void Matf32_transpose(const float *A, float *B, size_t nrows, size_t ncols)
{
	assert(A && "Can't A be null");
	assert(B && "Can't B be null");

	for (size_t i = 0; i < nrows; ++i) {
		for (size_t j = 0; j < ncols; ++j) {
			B[j * nrows + i] = A[i * ncols + j];
		}
	}
}


