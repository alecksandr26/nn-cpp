#include "../include/optimizer.hpp"

using namespace nn::optimizers;

template <typename T>
nn::optimizers::Optimizer<T>::Optimizer(std::string name, double learning_rate)
	: name_(name), learning_rate_(learning_rate)
{
}

template <typename T>
nn::optimizers::Optimizer<T>::Optimizer(double learning_rate)
	: name_("Optimizer"), learning_rate_(learning_rate)
{
}

template <typename T>
nn::optimizers::Optimizer<T>::Optimizer(std::string name, float learning_rate)
	: name_(name), learning_rate_(learning_rate)
{
}

template <typename T>
nn::optimizers::Optimizer<T>::Optimizer(float learning_rate)
	: name_("Optimizer"), learning_rate_(learning_rate)
{
}

template <typename T>
Optimizer<T> &nn::optimizers::Optimizer<T>::set_name(std::string name)
{
	name_ = std::move(name);
	return *this;
}

template <typename T>
Optimizer<T> &nn::optimizers::Optimizer<T>::set_learning_rate(double learning_rate)
{
	learning_rate_ = learning_rate;
	return *this;
}

template <typename T>
const std::string &nn::optimizers::Optimizer<T>::get_name(void) const
{
	return name_;
}

template <typename T>
double nn::optimizers::Optimizer<T>::get_learning_rate(void) const
{
	return learning_rate_;
}


template class nn::optimizers::Optimizer<float>;
template class nn::optimizers::Optimizer<double>;

template <typename T>
nn::optimizers::PerceptronOptimizer<T>::PerceptronOptimizer(double learning_rate)
	: Optimizer<T>("PerceptronOptimizer", learning_rate)
{
}

template <typename T>
nn::optimizers::PerceptronOptimizer<T>::PerceptronOptimizer(float learning_rate)
	: Optimizer<T>("PerceptronOptimizer", learning_rate)
{
}

template <typename T>
void nn::optimizers::PerceptronOptimizer<T>::update(Mat<T> &weights, const Mat<T> &error, const Mat<T> &input)
{
	weights += (error * Optimizer<T>::learning_rate_).dot(input.transpose_copy());
}

template class nn::optimizers::PerceptronOptimizer<float>;
// template class nn::optimizers::PerceptronOptimizer<double>;


