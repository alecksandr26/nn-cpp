#include <algorithm>
#include <stdexcept>

#include "../include/mat.hpp"

using namespace nn::mathops;

nn::mathops::Shape::Shape(const std::initializer_list<std::size_t> &l)
{
	if (l.size() > 2)
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

template<typename T>
nn::mathops::Mat<T>::Mat(std::size_t rows, std::size_t cols) : shape_(rows, cols)
{
	if (rows == 0 || cols == 0)
		throw std::invalid_argument("invalid argument: invalid shape of matrix");
	
	// TODO: Lets add a custom allocator
	mat_ = new T[rows * cols];
}

template<typename T>
nn::mathops::Mat<T>::Mat(const Shape &shape) : shape_(shape)
{
	if (shape.rows == 0 || shape.cols == 0)
		throw std::invalid_argument("invalid argument: invalid shape of matrix");
	
	// TODO: Lets add a custom allocator
	mat_ = new T[shape.rows * shape.cols];
}

template<typename T>
nn::mathops::Mat<T>::Mat(const Mat<T> &A) : shape_(A.get_shape())
{
	const Shape &shape = A.get_shape();
	const T *src = A.get_mat_raw();
	
	// TODO: Lets add a custom allocator
	mat_ = new T[shape.rows * shape.cols];
	Mat_copy(src, mat_, shape);
}

template<typename T>
nn::mathops::Mat<T>::Mat(Mat<T> &&A) : shape_(A.get_shape())
{
	mat_ = A.mat_;
	A.mat_ = NULL;
}

template<typename T>
nn::mathops::Mat<T>::Mat(const std::initializer_list<std::initializer_list<T>> &A)
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
nn::mathops::Mat<T>::~Mat(void)
{
	// TODO: Lets add a custom free
	delete[] mat_;
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
	if (mat_ != NULL)
		delete[] mat_;
	mat_ = A.mat_;
	A.mat_ = NULL;
	shape_ = A.shape_;
}

template <typename T>
Mat<T> nn::mathops::Mat<T>::dot(const Mat<T> &A) const
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
	Mat<T> C(shape_);
	Mat_add(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	*this = std::move(C);
	
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
	Mat<T> C(shape_);
	Mat_sub(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	*this = std::move(C);
	
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
	Mat<T> C(shape_);
	Mat_mul(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	*this = std::move(C);
	
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
	Mat<T> C(shape_);
	Mat_div(mat_, A.get_mat_raw(), C.get_mat_raw(), shape_);
	*this = std::move(C);
	
	return *this;
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
		delete[] mat_;
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
		delete[] mat_;
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

template<typename T>
const Shape &nn::mathops::Mat<T>::get_shape(void) const
{
	return shape_;
}

template<typename T>
T *nn::mathops::Mat<T>::get_mat_raw(void) const
{
	return mat_;
}


template <typename T>
T nn::mathops::Mat<T>::grand_sum(void) const
{
	return Mat_grand_sum(mat_, shape_);
}

template <typename T>
T &nn::mathops::Mat<T>::operator()(std::size_t row, std::size_t col) const
{
	return mat_[row * shape_.cols + col];
}

template class nn::mathops::Mat<float>;
// template class nn::mathops::Mat<double>;


