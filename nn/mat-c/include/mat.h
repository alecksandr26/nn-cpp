#ifndef MAT_INCLUDED
#define MAT_INCLUDED

#include <stdbool.h>
#include <stddef.h>

/* --- Mat 32 bit operations --- */

/* Matf32_rand_uniform: fill matrix A (nrows x ncols) with samples from U[min, max) */
extern void Matf32_rand_uniform(float *A, size_t nrows, size_t ncols, float min, float max);

/* Matf32_rand_normal: fill matrix A (nrows x ncols) with samples from N(mean, stddev^2) */
extern void Matf32_rand_normal(float *A, size_t nrows, size_t ncols, float mean, float stddev);
 
/* Matf32_fill: set all elements of matrix `A` of `nrows` and `ncols` to value a */
extern void Matf32_fill(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_add_scalar: add scalar a to all elements of matrix `A` of `nrows` and `ncols` */
extern void Matf32_add_scalar(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_sub_scalar: sub scalar a to all elements of matrix `A` of `nrows` and `ncols` */
extern void Matf32_sub_scalar(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_mul_scalar: multiply all elements of matrix `A` of `nrows` and `ncols` by scalar a */
extern void Matf32_mul_scalar(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_div_scalar: divide all elements of matrix `A` of `nrows` and `ncols` by scalar a */
extern void Matf32_div_scalar(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_add: element-wise addition of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf32_add(const float *A, const float *B, float *C, size_t nrows,
                       size_t ncols);

/* Matf32_sub: element-wise subtraction of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf32_sub(const float *A, const float *B, float *C, size_t nrows,
                       size_t ncols);

/* Matf32_mul: element-wise (Hadamard) multiplication of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf32_mul(const float *A, const float *B, float *C, size_t nrows,
                       size_t ncols);

/* Matf32_div: element-wise (Hadamard) division of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf32_div(const float *A, const float *B, float *C, size_t nrows,
		       size_t ncols);

/* Matf32_dot: matrix product C = A * B 
 * A is (nrowsA x ncolsA), B is (ncolsA x ncolsB), result C is (nrowsA x ncolsB)
 */
extern void Matf32_dot(const float *A, const float *B, float *C, size_t nrowsA,
                       size_t ncolsA, size_t ncolsB);

/* Matf32_copy: copy matrix src (nrows x ncols) into dst */
extern void Matf32_copy(const float *src, float *dst, size_t nrows,
                        size_t ncols);

/* Matf32_grand_sum: Compute and return the sum of all elements in the matrix */
extern float Matf32_grand_sum(const float *A, size_t nrows, size_t ncols);

/* Matf32_transpose: Transposes an nrows x ncols matrix A into B (B = A^T)  */
extern void Matf32_transpose(const float *A, float *B, size_t nrows,
                             size_t ncols);

/* Matf32_equal: Evaluates if A ~ B within epsilon tolerance */
extern bool Matf32_equal(const float *A, const float *B, size_t nrows, size_t ncols, float eps);

// TODO: Build the implemenations, tests and don't forget sub and div matrices, sub and div scalar, and grand sum

/* --- Mat 64 bit operations --- */

/* Matf64_fill: set all elements of matrix `A` of `nrows` and `ncols` to value a */
extern void Matf64_fill(double *A, size_t nrows, size_t ncols, double a);

/* Matf64_add_scalar: add scalar a to all elements of matrix `A` of `nrows` and `ncols` */
extern void Matf64_add_scalar(double *A, size_t nrows, size_t ncols, double a);

/* Matf64_mul_scalar: multiply all elements of matrix `A` of `nrows` and `ncols` by scalar a */
extern void Matf64_mul_scalar(double *A, size_t nrows, size_t ncols, double a);

/* Matf64_add: element-wise addition of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf64_add(const double *A, const double *B, double *C, size_t nrows, size_t ncols);

/* Matf64_mul: element-wise (Hadamard) multiplication of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf64_mul(const double *A, const double *B, double *C, size_t nrows, size_t ncols);

/* Matf64_dot: matrix product C = A * B 
 * A is (nrowsA x ncolsA), B is (ncolsA x ncolsB), result C is (nrowsA x ncolsB)
 */
extern void Matf64_dot(const double *A, const double *B, double *C,
                       size_t nrowsA, size_t ncolsA, size_t ncolsB);

/* Matf64_copy: copy matrix src (nrows x ncols) into dst */
extern void Matf64_copy(const double *src, double *dst, size_t nrows, size_t ncols);


#endif
