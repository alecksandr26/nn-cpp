#include <assert.h>
#include <stddef.h>

/* Matf32_dot: matrix product C = A * B 
 * A is (nrowsA x ncolsA), B is (ncolsA x ncolsB), result C is (nrowsA x ncolsB)
 */
void Matf32_dot(const float *A, const float *B, float *C,
                size_t nrowsA, size_t ncolsA, size_t ncolsB) {
	assert(A && "A can't be null");
	assert(B && "B can't be null");
	assert(C && "C can't be null");
	for (size_t i = 0; i < nrowsA; i++) {
		for (size_t j = 0; j < ncolsB; j++) {
			float sum = 0.0f;
			for (size_t k = 0; k < ncolsA; k++) {
				sum += A[i * ncolsA + k] * B[k * ncolsB + j];
			}
			C[i * ncolsB + j] = sum;
		}
	}
}







