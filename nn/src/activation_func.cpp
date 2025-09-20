#include "../include/activation_func.hpp"

using namespace nn::activation_funcs;

nn::activation_funcs::ActivationFunc::ActivationFunc(std::string name)
	: Layer(Shape{0, 0}, Shape{0, 0}, false, std::move(name))
{
}

// NOTE: Creating a foo deconstructor to avoid linker problems
nn::activation_funcs::ActivationFunc::~ActivationFunc(void)
{
}

template <typename T>
nn::activation_funcs::StepFunc<T>::StepFunc(void)
	: ActivationFunc("StepFunc")
{
}

template <typename T>
StepFunc<T> &nn::activation_funcs::StepFunc<T>::build(const Shape &input_shape, const Shape &output_shape)
{
	((void) input_shape);
	((void) output_shape);

	register_funcs();

	return *this;
}

template <typename T>
StepFunc<T> &nn::activation_funcs::StepFunc<T>::build(std::size_t input_size, std::size_t output_size)
{
	((void) input_size);
	((void) output_size);

	register_funcs();
	
	return *this;
}

template <typename T>
StepFunc<T> &nn::activation_funcs::StepFunc<T>::build(void)
{
	register_funcs();
	
	return *this;
}


template <typename T>
StepFunc<T> &nn::activation_funcs::StepFunc<T>::register_funcs(void)
{
	register_func<Mat<T>, const Mat<T> &>
		("feedforward", [this](const Mat<T> &X) -> Mat<T> {
			// TODO: find the way to optimize this code or at least make it smaller
			Mat<T> C(X.get_shape());
			for (std::size_t i = 0; i < X.rows(); i++) {
				for (std::size_t j = 0; j < X.cols(); j++) {
					C(i, j) = X(i, j) >= 0 ? 1.0f : 0.0f;
				}
			}
			return C;
		});

	// Notice that derivate of the absolute value is not defined in zero
	// but for programming we return zero either way
	register_func<Mat<T>, const Mat<T> &>
		("gradient", [this](const Mat<T> &X) -> Mat<T> {
			return Mat<T>(X.get_shape()).fill(0.0f);
		});


	register_func<Mat<T>, const Mat<T> &>
		("jacobian", [this](const Mat<T> &X) -> Mat<T> {
			// Supposing that X ~ (n, 1) -> jacobian -> (n, n)
			return Mat<T>(X.rows(), X.rows()).fill(0.0f);
		});
	
	return *this;
}

template class nn::activation_funcs::StepFunc<float>;
// template class nn::activation_funcs::StepFunc<double>;



