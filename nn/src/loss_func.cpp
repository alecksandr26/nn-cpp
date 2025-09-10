#include "../include/loss_func.hpp"
using namespace nn::loss_funcs;

template <typename T>
nn::loss_funcs::Loss<T>::Loss(const std::vector<Mat<T>> &inputs,
                              const std::vector<Mat<T>> &outputs)
    : inputs_(inputs), outputs_(outputs), model_(nullptr), name_("Loss")
{
	if (inputs.empty() || outputs.empty())
		throw std::invalid_argument("Inputs and outputs cannot be empty.");
	
	if (inputs.size() != outputs.size())
		throw std::invalid_argument("Inputs and outputs must have the same number of examples.");
	input_shape_ = inputs[0].get_shape();
	output_shape_ = outputs[0].get_shape();
}


template <typename T>
nn::loss_funcs::Loss<T>::Loss(const std::vector<Mat<T>> &inputs,
                              const std::vector<Mat<T>> &outputs,
                              std::string name)
    : inputs_(inputs), outputs_(outputs), model_(nullptr), name_(std::move(name))
{
	if (inputs.empty() || outputs.empty())
		throw std::invalid_argument("Inputs and outputs cannot be empty.");

	if (inputs.size() != outputs.size())
		throw std::invalid_argument("Inputs and outputs must have the same number of examples.");
	
	input_shape_ = inputs[0].get_shape();
	output_shape_ = outputs[0].get_shape();
}


// Pure virtual destructor
template <typename T>
nn::loss_funcs::Loss<T>::~Loss(void) {}

// Setters
template <typename T>
nn::loss_funcs::Loss<T> &nn::loss_funcs::Loss<T>::set_name(std::string name)
{
    name_ = std::move(name);
    return *this;
}

template <typename T>
nn::loss_funcs::Loss<T> &nn::loss_funcs::Loss<T>::set_model(Model<T> &model)
{
    model_ = &model;
    return *this;
}

// Getters
template <typename T>
const std::vector<Mat<T>> &nn::loss_funcs::Loss<T>::get_inputs(void) const
{
	return inputs_;
}

template <typename T>
const std::vector<Mat<T>> &nn::loss_funcs::Loss<T>::get_outputs(void) const
{
	return outputs_;
}

template <typename T>
const std::vector<Mat<T>> &nn::loss_funcs::Loss<T>::get_predictions(void) const
{
	return predictions_;
}

template <typename T>
const std::string &nn::loss_funcs::Loss<T>::get_name(void) const
{
	return name_;
}

template <typename T>
const Shape &nn::loss_funcs::Loss<T>::get_input_shape(void) const
{
	return input_shape_;
}

template <typename T>
const Shape &nn::loss_funcs::Loss<T>::get_output_shape(void) const
{
	return output_shape_;
}

template <typename T>
const Mat<T> &nn::loss_funcs::Loss<T>::get_last_loss(void) const
{
	return last_loss_;
}

// Normalized loss based on predictions_
template <typename T>
T nn::loss_funcs::Loss<T>::get_normalized_loss(void) const
{
	if (predictions_.empty() || last_loss_.get_shape().rows == 0 || last_loss_.get_shape().cols == 0)
		return static_cast<T>(0);

	// Find min/max in predictions_
	T min_val = predictions_[0](0,0);
	T max_val = predictions_[0](0,0);

	for (const auto &pred : predictions_) {
		for (std::size_t i = 0; i < pred.get_shape().rows; ++i) {
			for (std::size_t j = 0; j < pred.get_shape().cols; ++j) {
				min_val = std::min(min_val, pred(i,j));
				max_val = std::max(max_val, pred(i,j));
			}
		}
	}

	T range = max_val - min_val;
	if (range == 0)
		return static_cast<T>(0); // avoid division by zero

	// Normalize last_loss_
	T sum = 0;
	for (std::size_t i = 0; i < last_loss_.get_shape().rows; ++i)
		for (std::size_t j = 0; j < last_loss_.get_shape().cols; ++j)
			sum += (last_loss_(i,j) - min_val) / range;

	return sum / (last_loss_.get_shape().rows * last_loss_.get_shape().cols);
}

template class nn::loss_funcs::Loss<float>;
// template class nn::loss_funcs::Loss<double>;



template <typename T>
nn::loss_funcs::MeanAbsoluteError<T>::MeanAbsoluteError(const std::vector<Mat<T>> &inputs,
							const std::vector<Mat<T>> &outputs)
	: Loss<T>(inputs, outputs, "MeanAbsoluteError")
{
}

// TODO: Find the way to optimize this thing, like create a few mat functions

// Evaluate MAE on all stored inputs/outputs
template <typename T>
Mat<T> nn::loss_funcs::MeanAbsoluteError<T>::operator()(void)
{
	if (!this->model_)
		throw std::runtime_error("Model pointer not set in Loss function.");

	Loss<T>::predictions_.clear();
	Loss<T>::last_loss_.resize(Loss<T>::output_shape_);

	for (std::size_t i = 0; i < this->inputs_.size(); ++i) {
		Mat<T> y_pred = (*this->model_)(this->inputs_[i]);
		Loss<T>::predictions_.push_back(y_pred);

		Mat<T> diff = y_pred - this->outputs_[i];
		for (std::size_t i = 0; i < y_pred.get_shape().rows; ++i) {
			for (std::size_t j = 0; j < y_pred.get_shape().cols; ++j) {
				Loss<T>::last_loss_(i, j) += std::abs(diff(i, j));
			}
		}
	}

	return Loss<T>::last_loss_ / static_cast<T>(Loss<T>::inputs_.size());
}

// Evaluate MAE on a batch of input-output pairs
template <typename T>
Mat<T> nn::loss_funcs::MeanAbsoluteError<T>::operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch)
{
	if (!this->model_)
		throw std::runtime_error("Model pointer not set in Loss function.");

	Loss<T>::predictions_.clear();
	Loss<T>::last_loss_.resize(Loss<T>::output_shape_);

	for (std::size_t i = 0; i < batch.size(); ++i) {
		const auto &[x, y_true] = batch[i];
		Mat<T> y_pred = (*this->model_)(x);
		Loss<T>::predictions_.push_back(y_pred);

		Mat<T> diff = y_pred - y_true;
		for (std::size_t i = 0; i < y_pred.get_shape().rows; ++i) {
			for (std::size_t j = 0; j < y_pred.get_shape().cols; ++j) {
				Loss<T>::last_loss_(i, j) += std::abs(diff(i, j));
			}
		}
	}

	return Loss<T>::last_loss_ / static_cast<T>(batch.size());
}

// Evaluate MAE on a single example
template <typename T>
Mat<T> nn::loss_funcs::MeanAbsoluteError<T>::operator()(const std::pair<Mat<T>, Mat<T>> &example)
{
	if (!this->model_)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*this->model_)(x);
	Loss<T>::predictions_.clear();
	Loss<T>::predictions_.push_back(y_pred);

	Mat<T> diff = y_pred - y_true;
	Loss<T>::last_loss_.resize(Loss<T>::output_shape_);
	for (std::size_t i = 0; i < y_pred.get_shape().rows; ++i) {
		for (std::size_t j = 0; j < y_pred.get_shape().cols; ++j) {
			Loss<T>::last_loss_(i, j) = std::abs(diff(i, j));
		}
	}

	return Loss<T>::last_loss_;
}

// Derivative of MAE for a single example: sign(y_pred - y_true)
template <typename T>
Mat<T> nn::loss_funcs::MeanAbsoluteError<T>::derivate(const std::pair<Mat<T>, Mat<T>> &example)
{
	if (!this->model_)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*this->model_)(x);

	Mat<T> grad(Loss<T>::output_shape_);
	for (std::size_t i = 0; i < y_pred.get_shape().rows; ++i) {
		for (std::size_t j = 0; j < y_pred.get_shape().cols; ++j) {
			T val = y_pred(i, j) - y_true(i, j);
			grad(i, j) = (val > 0) ? 1 : ((val < 0) ? -1 : 0);
		}
	}
	
	return grad;
}


template class nn::loss_funcs::MeanAbsoluteError<float>;
// template class nn::loss_funcs::MeanAbsoluteError<double>;
