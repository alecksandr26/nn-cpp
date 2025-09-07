#ifndef MAT_INCLUDED
#define MAT_INCLUDED

#include <stddef.h>

/* Matf32_fill: set all elements of matrix `A` of `nrows` and `ncols` to value a */
extern void Matf32_fill(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_add_scalar: add scalar a to all elements of matrix `A` of `nrows` and `ncols` */
extern void Matf32_add_scalar(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_mul_scalar: multiply all elements of matrix `A` of `nrows` and `ncols` by scalar a */
extern void Matf32_mul_scalar(float *A, size_t nrows, size_t ncols, float a);

/* Matf32_add: element-wise addition of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf32_add(const float *A, const float *B, float *C, size_t nrows, size_t ncols);

/* Matf32_mul: element-wise (Hadamard) multiplication of matrices A and B, result in C (all of size nrows x ncols) */
extern void Matf32_mul(const float *A, const float *B, float *C, size_t nrows, size_t ncols);

/* Matf32_dot: matrix product C = A * B 
 * A is (nrowsA x ncolsA), B is (ncolsA x ncolsB), result C is (nrowsA x ncolsB)
 */
extern void Matf32_dot(const float *A, const float *B, float *C,
                       size_t nrowsA, size_t ncolsA, size_t ncolsB);

#endif
