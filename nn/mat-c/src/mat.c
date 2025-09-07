#include <assert.h>
#include <stddef.h>

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

/* Matf32_mul_scalar: multiply all elements of matrix `A` of `nrows` and `ncols` by scalar a */
void Matf32_mul_scalar(float *A, size_t nrows, size_t ncols, float a) {
	assert(A && "A can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		A[i] *= a;
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

/* Matf32_mul: element-wise (Hadamard) multiplication of matrices A and B, result in C (all of size nrows x ncols) */
void Matf32_mul(const float *A, const float *B, float *C, size_t nrows, size_t ncols) {
	assert(A && "A can't be null");
	assert(B && "B can't be null");
	assert(C && "C can't be null");
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		C[i] = A[i] * B[i];
	}
}



