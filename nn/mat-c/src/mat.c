#include <assert.h>
#include <time.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

/* TODO: Implement the optional versions */


/* Matf32_rand_uniform: fill matrix A (nrows x ncols) with samples from U[min, max) */
void Matf32_rand_uniform(float *A, size_t nrows, size_t ncols, float min, float max) {
	assert(A && "A can't be null");
	srand((unsigned int) time(NULL));  // seed with current time
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i++) {
		float u = (float) rand() / (float) RAND_MAX; // uniform [0,1]
		A[i] = min + (max - min) * u;
	}
}

/* Matf32_rand_normal: fill matrix A (nrows x ncols) with samples from N(mean, stddev^2) */
void Matf32_rand_normal(float *A, size_t nrows, size_t ncols, float mean, float stddev) {
	assert(A && "A can't be null");
	srand((unsigned int) time(NULL));  // seed with current time
	size_t total = nrows * ncols;
	for (size_t i = 0; i < total; i += 2) {
		// Generate two uniform random numbers in (0,1)
		float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
		float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
		
		// Box–Muller transform
		float mag = stddev * sqrtf(-2.0f * logf(u1));
		float z0 = mag * cosf(2.0f * M_PI * u2) + mean;
		float z1 = mag * sinf(2.0f * M_PI * u2) + mean;
		
		// Store results (check bounds for odd-sized matrices)
		A[i] = z0;
		if (i + 1 < total) {
			A[i + 1] = z1;
		}
	}
}

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

// TODO: Add a unit test of this

/* Matf32_equal: Evaluates if A ~ B within epsilon tolerance */
bool Matf32_equal(const float *A, const float *B, size_t nrows, size_t ncols, float eps)
{
    assert(A && "A can't be null");
    assert(B && "B can't be null");

    size_t total = nrows * ncols;
    for (size_t i = 0; i < total; i++) {
	    if (fabs(A[i] - B[i]) > eps) {
		    return false;
	    }
    }

    return true;
}

