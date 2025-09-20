#include <algorithm>
#include <stdexcept>

#include "../include/mat.hpp"

using namespace nn::mathops;


nn::mathops::Shape::Shape(void)
	: rows(0), cols(0)
{
}

nn::mathops::Shape::Shape(const std::initializer_list<std::size_t> &l)
{
	if (l.size() != 2)
		throw std::invalid_argument("invalid argument: Invalid list of initializer");
	rows = *(l.begin());
	cols = *(l.begin() + 1);
}

nn::mathops::Shape::Shape(std::size_t init_rows, std::size_t init_cols)
{
	rows = init_rows;
	cols = init_cols;
}

nn::mathops::Shape::Shape(const Shape &s)
{
	rows = s.rows;
	cols = s.cols;
}

bool nn::mathops::Shape::operator==(const Shape &s) const
{
	return rows == s.rows && cols == s.cols;
}

bool nn::mathops::Shape::operator!=(const Shape &s) const
{
	return rows != s.rows || cols != s.cols;
}

void nn::mathops::Shape::operator=(const Shape &s)
{
	rows = s.rows;
	cols = s.cols;
}

template <typename T>
nn::mathops::Mat<T>::Mat(void) : shape_(0, 0), mat_shared_mem_(false), mat_(nullptr)
{
}

template<typename T>
nn::mathops::Mat<T>::Mat(std::size_t rows, std::size_t cols, T *mat)
	: shape_(rows, cols), mat_(mat)
{
	if (rows == 0 || cols == 0)
		throw std::invalid_argument("invalid argument: invalid shape of matrix");

	mat_shared_mem_ = mat != nullptr;
	
	// TODO: Lets add a custom allocator
	if (!mat_shared_mem_) mat_ = new T[rows * cols];
}

template<typename T>
nn::mathops::Mat<T>::Mat(const Shape &shape, T *mat)
	: shape_(shape), mat_(mat)
{
	if (shape.rows == 0 || shape.cols == 0)
		throw std::invalid_argument("invalid argument: invalid shape of matrix");

	mat_shared_mem_ = mat != nullptr;
	
	// TODO: Lets add a custom allocator
	if (!mat_shared_mem_) mat_ = new T[shape.rows * shape.cols];
}

template<typename T>
nn::mathops::Mat<T>::Mat(const Mat<T> &A)
	: shape_(A.get_shape()), mat_shared_mem_(false)
{
	const Shape &shape = A.get_shape();
	const T *src = A.get_mat_raw();
	
	// TODO: Lets add a custom allocator
	mat_ = new T[shape.rows * shape.cols];
	Mat_copy(src, mat_, shape);
}

template<typename T>
nn::mathops::Mat<T>::Mat(Mat<T> &&A)
	: shape_(A.get_shape())
{
	mat_shared_mem_ = A.mat_shared_mem_;
	mat_ = A.mat_;
	A.mat_ = NULL;
}

template<typename T>
nn::mathops::Mat<T>::Mat(const std::initializer_list<std::initializer_list<T>> &A)
	: mat_shared_mem_(false)
{
	if (A.size() == 0) throw std::invalid_argument("invalid argument: Empty initializer structure");
	size_t n = A.begin()->size();
	for (size_t i = 1; i < A.size(); i++)
		if (n != (A.begin() + i)->size())
			throw std::invalid_argument("invalid argument: Invalid structure of the matrix");
	
	shape_.rows = A.size();
	shape_.cols = n;
	mat_ = new T[shape_.rows * shape_.cols];   // allocate directly
	
	for (size_t i = 0; i < shape_.rows; i++)
		for (size_t j = 0; j < shape_.cols; j++)
			mat_[i * n + j] = *((A.begin() + i)->begin() + j);
}

template<typename T>
nn::mathops::Mat<T>::~Mat(void)
{
	// TODO: Lets add a custom free
	if (!mat_shared_mem_) delete[] mat_;
	mat_ = NULL;
}

template<typename T>
void nn::mathops::Mat<T>::operator=(const std::initializer_list<std::initializer_list<T>> &A)
{
	if (A.size() == 0) throw std::invalid_argument("invalid argument: Empty initializer structure");
	size_t n = A.begin()->size();
	for (size_t i = 1; i < A.size(); i++)
		if (n != (A.begin() + i)->size())
			throw std::invalid_argument("invalid argument: Invalid structure of the matrix");
	
	resize(A.size(), n);
	
	for (size_t i = 0; i < shape_.rows; i++)
		for (size_t j = 0; j < shape_.cols; j++)
			mat_[i * n + j] = *((A.begin() + i)->begin() + j);
}

template<typename T>
void nn::mathops::Mat<T>::operator=(Mat<T> &&A)
{
	if (mat_ != NULL && !mat_shared_mem_)
		delete[] mat_;
	mat_shared_mem_ = A.mat_shared_mem_;
	mat_ = A.mat_;
	A.mat_ = NULL;
	shape_ = A.shape_;
}

template<typename T>
void nn::mathops::Mat<T>::operator=(const Mat<T> &A)
{
	if (this == &A) return;
	// TODO: add the custom allocator
	T *new_mat = new T[A.shape_.rows * A.shape_.cols];
	Mat_copy(A.mat_, new_mat, A.shape_);

	if (!mat_shared_mem_) delete[] mat_;
	mat_ = new_mat;
	shape_ = A.shape_;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::dot(const Mat<T> &A) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");

	const Shape &shapeA = this->shape_;   // shape of "this"
	const Shape &shapeB = A.get_shape();  // shape of argument

	// Validation: cols(this) == rows(A)
	if (shapeA.cols != shapeB.rows)
		throw std::invalid_argument("invalid argument: cols(this) != rows(A)");

	// Result shape: (rows(this) × cols(A))
	Mat<T> C(shapeA.rows, shapeB.cols);

	// Perform multiplication
	Mat_dot(mat_, A.get_mat_raw(), C.get_mat_raw(), shapeA, shapeB.cols);
	return C;
}


template <typename T>
Mat<T> &nn::mathops::Mat<T>::dot_and_assign(const Mat<T> &A)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	const Shape &shape = A.get_shape();
	if (shape_.rows != shape.cols)
		throw std::invalid_argument("invalid argument: invalid structure rows != cols");
	
	Mat<T> C(shape_.rows, shape.cols);
	Mat_dot(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_, shape.cols);

	*this = std::move(C);
	
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator+(const Mat<T> &A) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");

	Mat<T> C(shape_);
	Mat_add(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator+=(const Mat<T> &A)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");
	Mat_add(mat_, A.get_mat_raw(), mat_, shape_);
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator-(const Mat<T> &A) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");

	Mat<T> C(shape_);
	Mat_sub(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator-=(const Mat<T> &A)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");
	Mat_sub(mat_, A.get_mat_raw(), mat_, shape_);
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator*(const Mat<T> &A) const
{
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");

	Mat<T> C(shape_);
	Mat_mul(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator*=(const Mat<T> &A)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");
	Mat_mul(mat_, A.get_mat_raw(), mat_, shape_);
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator/(const Mat<T> &A) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");
	
	Mat<T> C(shape_);
	Mat_div(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator/=(const Mat<T> &A)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	if (A.get_mat_raw() == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `A`");
	if (A.get_shape() != shape_)
		throw std::invalid_argument("invalid argument: invalid structure `this.shape` != `A.shape`");
	Mat_div(mat_, A.get_mat_raw(), mat_, shape_);
	return *this;
}

template <typename T>
bool nn::mathops::Mat<T>::operator==(const Mat<T> &A) const
{
	if (shape_ != A.get_shape())
		return false;
	return Mat_equal(mat_, A.get_mat_raw(), shape_);
}

template <typename T>
bool nn::mathops::Mat<T>::operator!=(const Mat<T> &A) const
{
	if (shape_ != A.get_shape())
		return true;
	return !Mat_equal(mat_, A.get_mat_raw(), shape_);
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator+(T a) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat<T> C(*this);
	Mat_add_scalar(C.get_mat_raw(), shape_, a);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator+=(T a)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat_add_scalar(mat_, shape_, a);
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator-(T a) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat<T> C(*this);
	Mat_sub_scalar(C.get_mat_raw(), shape_, a);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator-=(T a)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat_sub_scalar(mat_, shape_, a);
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator*(T a) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat<T> C(*this);
	Mat_mul_scalar(C.get_mat_raw(), shape_, a);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator*=(T a)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat_mul_scalar(mat_, shape_, a);
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::operator/(T a) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat<T> C(*this);
	Mat_div_scalar(C.get_mat_raw(), shape_, a);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::operator/=(T a)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat_div_scalar(mat_, shape_, a);
	return *this;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::resize(const Shape &shape)
{
	// TODO: Lets add a custom  re allocator and free
	if (shape.rows == 0 or shape.cols == 0)
		throw std::invalid_argument("invalid argument: Invalid structure of the matrix");

	T *new_resized_mat = new T[shape.rows * shape.cols];
	if (mat_ != NULL) {
		Mat_copy(mat_, new_resized_mat, Shape(std::min(shape.rows, shape_.rows), std::min(shape.cols, shape_.cols)));
		if (!mat_shared_mem_) delete[] mat_;
	}
	mat_ = new_resized_mat;
	shape_.rows = shape.rows;
	shape_.cols = shape.cols;
	
	return *this;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::resize(std::size_t rows, std::size_t cols)
{
	// TODO: Lets add a custom  re allocator and free

	if (rows == 0 or cols == 0)
		throw std::invalid_argument("invalid argument: Invalid structure of the matrix");
	
	T *new_resized_mat = new T[rows * cols];
	if (mat_ != NULL) {
		Mat_copy(mat_, new_resized_mat, Shape(std::min(rows, shape_.rows), std::min(cols, shape_.cols)));
		if (!mat_shared_mem_) delete[] mat_;
	}
	
	mat_ = new_resized_mat;
	shape_.rows = rows;
        shape_.cols = cols;
	
	return *this;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::transpose(void)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat<T> C(shape_.cols, shape_.rows);
	Mat_transposem(mat_, C.get_mat_raw(), shape_);
	*this = std::move(C);
	return *this;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::transpose_copy(void) const
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat<T> C(shape_.cols, shape_.rows);
	Mat_transposem(mat_, C.get_mat_raw(), shape_);
	return C;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::fill(T a)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat_fill(mat_, shape_, a);
	return *this;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::rand_uniform(T min_val, T max_val)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat_rand_uniform(mat_, shape_, min_val, max_val);
	return *this;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::rand_normal(T mean, T stddev)
{
	if (mat_ == NULL)
		throw std::invalid_argument("invalid argument: Empty Matrix `this`");
	Mat_rand_normal(mat_, shape_, mean, stddev);
	return *this;
}

template<typename T>
const Shape &nn::mathops::Mat<T>::get_shape(void) const
{
	return shape_;
}

template<typename T>
Mat<T> &nn::mathops::Mat<T>::set_shape(const Shape &shape)
{
	shape_ = shape;
	return *this;
}

template <typename T>
std::size_t nn::mathops::Mat<T>::rows(void) const
{
	return shape_.rows;
}

template <typename T>
std::size_t nn::mathops::Mat<T>::cols(void) const
{
	return shape_.cols;
}

template<typename T>
T *nn::mathops::Mat<T>::get_mat_raw(void) const
{
	return mat_;
}

template <typename T>
Mat<T> &nn::mathops::Mat<T>::get_row(std::size_t row)
{
	if (fetched_rows_.count(row) > 0) {
		return fetched_rows_[row];
	}
	// the constructor where we share the allocation of the original matrix
	// 	fetched_rows_[row] = std::move(Mat<T>(1, cols(), mat_ + cols() * row));
	fetched_rows_.insert({row, std::move(Mat<T>(1, cols(), mat_ + cols() * row))});
	return fetched_rows_[row];
}

template <typename T>
T nn::mathops::Mat<T>::grand_sum(void) const
{
	return Mat_grand_sum(mat_, shape_);
}

template <typename T>
T &nn::mathops::Mat<T>::operator()(std::size_t row, std::size_t col)
{
	return mat_[row * shape_.cols + col];
}

template <typename T>
const T &nn::mathops::Mat<T>::operator()(std::size_t row, std::size_t col) const
{
	return mat_[row * shape_.cols + col];
}


template <typename T>
Mat<T> &nn::mathops::Mat<T>::set_mat_raw(T *mat)
{
	mat_ = mat;
	return *this;
}

template class nn::mathops::Mat<float>;
// template class nn::mathops::Mat<double>;


