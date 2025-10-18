#ifndef NN_MAT_INCLUDED
#define NN_MAT_INCLUDED

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <unordered_map>

namespace nn::mathops {
	extern "C" {
#include "../mat-c/include/mat.h"
	}	
	
	class Shape {
	public:
		std::size_t rows, cols;
		
		Shape(void);
		Shape(const std::initializer_list<std::size_t> &l);
		Shape(std::size_t init_rows, std::size_t init_cols);
		Shape(const Shape &s);
		bool operator==(const Shape &s) const;
		bool operator!=(const Shape &s) const;
		void operator=(const Shape &s);
		
		friend std::ostream& operator<<(std::ostream& os, const Shape &shape) {
			os << "shape=(rows=" << shape.rows << ", cols=" << shape.cols << ")";
			return os;
		}
	};

	
	class MatDispatchOps {
	public:
		// TODO: Lets add the double version
		
		// --- Inline static dispatch operations ---
		inline static void Mat_rand_uniform(float *A, const Shape &shape, float min_val, float max_val) {
			Matf32_rand_uniform(A, shape.rows, shape.cols, min_val, max_val);
		}

		inline static void Mat_rand_normal(float *A, const Shape &shape, float mean, float stddev) {
			Matf32_rand_normal(A, shape.rows, shape.cols, mean, stddev);
		}
		
		inline static void Mat_fill(float* A, const Shape &shape, float a) {
			Matf32_fill(A, shape.rows, shape.cols, a);
		}

		inline static void Mat_add_scalar(float* A, const Shape &shape, float a) {
			Matf32_add_scalar(A, shape.rows, shape.cols, a);
		}

		inline static void Mat_sub_scalar(float* A, const Shape &shape, float a) {
			Matf32_sub_scalar(A, shape.rows, shape.cols, a);
		}

		inline static void Mat_mul_scalar(float* A, const Shape &shape, float a) {
			Matf32_mul_scalar(A, shape.rows, shape.cols, a);
		}

		inline static void Mat_div_scalar(float* A, const Shape &shape, float a) {
			Matf32_div_scalar(A, shape.rows, shape.cols, a);
		}

		inline static void Mat_add(const float* A, const float* B, float* C, const Shape &shape) {
			Matf32_add(A, B, C, shape.rows, shape.cols);
		}
		
		inline static void Mat_sub(const float* A, const float* B, float* C, const Shape &shape) {
			Matf32_sub(A, B, C, shape.rows, shape.cols);
		}

		inline static void Mat_mul(const float* A, const float* B, float* C, const Shape &shape) {
			Matf32_mul(A, B, C, shape.rows, shape.cols);
		}

		inline static void Mat_div(const float* A, const float* B, float* C, const Shape &shape) {
			Matf32_div(A, B, C, shape.rows, shape.cols);
		}

		inline static void Mat_dot(const float* A, const float* B, float* C, 
					       const Shape &shapeA, size_t ncolsB) {
			Matf32_dot(A, B, C, shapeA.rows, shapeA.cols, ncolsB);
		}

		inline static void Mat_copy(const float* src, float* dst, const Shape &shape) {
			Matf32_copy(src, dst, shape.rows, shape.cols);
		}

		inline static float Mat_grand_sum(const float *A, const Shape &shape) {
			return Matf32_grand_sum(A, shape.rows, shape.cols);
		}

		inline static void Mat_transposem(const float *A, float *B, const Shape &shape) {
			Matf32_transpose(A, B, shape.rows, shape.cols);
		}

		constexpr static double eq_tolerance = 1e-8;
		inline static bool Mat_equal(const float *A, const float *B, const Shape &shape) {
			return Matf32_equal(A, B, shape.rows, shape.cols, static_cast<float>(eq_tolerance));
		}
			
	};

	template<typename T>
	class Mat : private MatDispatchOps {
	public:
		Mat(void);
		Mat(std::size_t rows, std::size_t cols, T *mat = nullptr);
		Mat(const Shape &shape, T *mat = nullptr);
		Mat(const Mat<T> &A);           // copy constructor
		Mat(Mat<T> &&A);                // move constructor
		Mat(const std::initializer_list<std::initializer_list<T>> &A);
		~Mat(void);
		
		void operator=(const std::initializer_list<std::initializer_list<T>> &A);
		void operator=(Mat<T> &&A);              // move constructor
		void operator=(const Mat<T> &A);
		Mat<T> dot(const Mat<T> &A) const;
		Mat<T> &dot_and_assign(const Mat<T> &A);
		Mat<T> operator+(const Mat<T> &A) const;
		Mat<T> &operator+=(const Mat<T> &A);
		Mat<T> operator-(const Mat<T> &A) const;
		Mat<T> &operator-=(const Mat<T> &A);
		Mat<T> operator*(const Mat<T> &A) const;
		Mat<T> &operator*=(const Mat<T> &A);
		Mat<T> operator/(const Mat<T> &A) const;
		Mat<T> &operator/=(const Mat<T> &A);
		bool operator==(const Mat<T> &A) const;
		bool operator!=(const Mat<T> &A) const;
		
		Mat<T> operator+(T a) const;
		Mat<T> &operator+=(T a);
		Mat<T> operator-(T a) const;
		Mat<T> &operator-=(T a);
		Mat<T> operator*(T a) const;
		Mat<T> &operator*=(T a);
		Mat<T> operator/(T a) const;
		Mat<T> &operator/=(T a);

		Mat<T> &transpose(void);
		Mat<T> transpose_copy(void) const;
		Mat<T> &resize(const Shape &shape);
		Mat<T> &resize(std::size_t rows, std::size_t cols);
		Mat<T> &fill(T a);
		Mat<T> &rand_uniform(T min_val, T max_val);
		Mat<T> &rand_normal(T mean, T stddev);
		
		// Zero based index
		Mat<T> &get_row(std::size_t row); // shape ~ (1, cols)
		T grand_sum(void) const;
		const Shape &get_shape(void) const;
		Mat<T> &set_shape(const Shape &shape); // Risky method
		std::size_t rows(void) const;
		std::size_t cols(void) const;
		T *get_mat_raw(void) const;
		Mat<T> &set_mat_raw(T *mat); // Risky method
		

		// Zero based index for this matrices
		T &operator()(std::size_t row, std::size_t col);
		const T &operator()(std::size_t row, std::size_t col) const;
		
		friend std::ostream &operator<<(std::ostream &os, const Mat<T> &A)
		{
			os << "Mat=(\n";
			os << "[";
			for (std::size_t i = 0; i < A.shape_.rows; i++) {
				if (i > 0)
					os << " ";
				os << "[";
				for (std::size_t j = 0; j < A.shape_.cols; j++) {
					os << A.mat_[i * A.shape_.cols + j];
					if (j < A.shape_.cols - 1)
						os << "\t";
				}
				os << "]";
				if (i < A.shape_.rows - 1)
					os << "\n";
			}
			os << "],\n" << A.shape_ << ", "
			   << ((std::is_same<T, float>::value)
			       ? "float32"
			       : "float64") << ", "
			   << "addrs=" << (void *) &A << ")";
			
			return os;
		}
	private:
		std::unordered_map<std::size_t, Mat<T>> fetched_rows_;
		
		
		Shape shape_;
		bool mat_shared_mem_; // Simple boolean to know where comes the memory
		T *mat_;	// TODO: Try to use a unique pointer here or a shared pointer
		
		
	};
}

#endif
