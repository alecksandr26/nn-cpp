

#include "../include/activation_func.hpp"

using namespace nn::activation_funcs;

template <typename T>
nn::activation_funcs::ActivationFunc<T>::ActivationFunc(std::string name)
	: Layer<T>(false, Shape{0, 0}, Shape{0, 0}, std::move(name))
{
}

template class nn::activation_funcs::ActivationFunc<float>;
// template class nn::activation_funcs::ActivationFunc<double>;


template <typename T>
nn::activation_funcs::StepFunc<T>::StepFunc(void)
	: ActivationFunc<T>("StepFunc")
{
}

template <typename T>
Mat<T> nn::activation_funcs::StepFunc<T>::operator()(const Mat<T> &X)
{
	// TODO: find the way to optimize this code or at least make it smaller
	Mat<T> C(X.get_shape());
	for (std::size_t i = 0; i < X.get_shape().rows; i++) {
		for (std::size_t j = 0; j < X.get_shape().cols; j++) {
			C(i, j) = X(i, j) >= 0 ? 1 : 0;
		}
	}
	
	return C;
}


template <typename T>
Mat<T> nn::activation_funcs::StepFunc<T>::derivate(const Mat<T> &X)
{
	return Mat<T>(X.get_shape()).fill(0);
}

template <typename T>
StepFunc<T> &nn::activation_funcs::StepFunc<T>::build(const Shape &input_shape, const Shape &output_shape)
{
	((void) input_shape);
	((void) output_shape);
	return *this;
}


template class nn::activation_funcs::StepFunc<float>;
// template class nn::activation_funcs::StepFunc<double>;










