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
