#include "../include/optimizer.hpp"

using namespace nn::optimizers;

nn::optimizers::Optimizer::Optimizer(std::string name, double learning_rate)
	: name_(name), learning_rate_(learning_rate)
{
}

nn::optimizers::Optimizer::Optimizer(double learning_rate)
	: name_("Optimizer"), learning_rate_(learning_rate)
{
}

nn::optimizers::Optimizer::Optimizer(std::string name, float learning_rate)
	: name_(name), learning_rate_(learning_rate)
{
}

nn::optimizers::Optimizer::Optimizer(float learning_rate)
	: name_("Optimizer"), learning_rate_(static_cast<double>(learning_rate))
{
}

// NOTE: Creating a foo deconstructor to avoid linker problems
nn::optimizers::Optimizer::~Optimizer(void)
{
}

Optimizer &nn::optimizers::Optimizer::set_name(std::string name)
{
	name_ = std::move(name);
	return *this;
}

Optimizer &nn::optimizers::Optimizer::set_learning_rate(double learning_rate)
{
	learning_rate_ = learning_rate;
	return *this;
}

const std::string &nn::optimizers::Optimizer::get_name(void) const
{
	return name_;
}

double nn::optimizers::Optimizer::get_learning_rate(void) const
{
	return learning_rate_;
}



template <typename T>
nn::optimizers::PerceptronOptimizer<T>::PerceptronOptimizer(T learning_rate)
	: Optimizer("PerceptronOptimizer", learning_rate)
{
	register_funcs();
}


template <typename T>
PerceptronOptimizer<T> &nn::optimizers::PerceptronOptimizer<T>::register_funcs(void)
{
	register_func<void, Mat<T> &, const Mat<T> &, const Mat<T> &>
		("update", [this](Mat<T> &weights, const Mat<T> &error, const Mat<T> &input) -> void {
			// error = e = d - y

			// weights += weights
			// Since the error has a shape (n, 1)
			for (std::size_t i = 0; i < weights.rows(); i++) {
				Mat<T> &row = weights.get_row(i);
				row += input.transpose_copy() * (static_cast<T>(learning_rate_) * error(i, 0)); // Element wise add + scalar mul
			}
		});
	
	return *this;
}

template class nn::optimizers::PerceptronOptimizer<float>;
// template class nn::optimizers::PerceptronOptimizer<double>;


template <typename T>
nn::optimizers::GradientDescentOptimizer<T>::GradientDescentOptimizer(T learning_rate)
	: Optimizer("GradientDescentOptimizer", learning_rate)
	  
{
	register_funcs();
}

template <typename T>
GradientDescentOptimizer<T> &nn::optimizers::GradientDescentOptimizer<T>::register_funcs(void)
{
	register_func<void, Mat<T> &, const Mat<T> &, const Mat<T> &>
		("update", [this](Mat<T> &weights, const Mat<T> &grad, const Mat<T> &input) -> void {
			// Here is the gradient of the backpropagation
			// grad = g = dL/dW = dL/dY * dY/dZ * dZ/dW
			// where: L = is the loss function
			//        Y = f(Z) = is the activation function
			//        Z = W * X
			// and the dL/dW \in R^{m, n}
			// where m is the output and n is the size of the vector of the input
			// w_{k + 1} = w_{k} + l * dL/dW
			((void) input);

			// --- Validation section ---
			// 1. Shape check
			if (weights.get_shape() != grad.get_shape()) {
				throw std::invalid_argument("[update] Shape mismatch: weights(" + std::to_string(weights.rows()) + "x" + std::to_string(weights.cols()) + ") vs grad(" + std::to_string(grad.rows()) + "x" + std::to_string(grad.cols()) + ")");
			}
			
			weights += grad * this->learning_rate_;
		});
	
	return *this;
}

template class nn::optimizers::GradientDescentOptimizer<float>;
// template class nn::optimizers::GradientDescentOptimizer<float>;
