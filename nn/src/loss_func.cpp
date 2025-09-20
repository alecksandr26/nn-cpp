#include "../include/loss_func.hpp"
#include <stdexcept>
#include <algorithm>

using namespace nn::loss_funcs;

// ===================== LOSS IMPLEMENTATION =====================

template <typename T>
Loss<T>::Loss(std::shared_ptr<std::vector<Mat<T>>> inputs,
              std::shared_ptr<std::vector<Mat<T>>> outputs,
              std::string name)
	: inputs_(inputs),
	  outputs_(outputs),
	  model_(nullptr),
	  name_(std::move(name))
{
	if (inputs_ != nullptr && outputs_ != nullptr) {
		if (inputs_->size() != outputs_->size())
			throw std::invalid_argument("Inputs and outputs must have the same number of examples.");
		input_shape_ = (*inputs_)[0].get_shape();
		output_shape_ = (*outputs_)[0].get_shape();
	}
}

// Pure virtual destructor
template <typename T>
Loss<T>::~Loss() = default;

// --------------------- SETTERS ---------------------

template <typename T>
Loss<T> &Loss<T>::set_name(std::string name)
{
	name_ = std::move(name);
	return *this;
}

template <typename T>
Loss<T> &Loss<T>::set_model(Model &model)
{
	model_ = std::shared_ptr<Model>(&model, [](Model*){}); // Just store the reference
	return *this;
}

template <typename T>
Loss<T> &Loss<T>::set_inputs(std::shared_ptr<std::vector<Mat<T>>> inputs)
{
	if (inputs == nullptr || !inputs->size())
		throw std::invalid_argument("Inputs cannot be empty.");
	inputs_ = inputs;
	input_shape_ = (*inputs)[0].get_shape();
	return *this;
}

template <typename T>
Loss<T> &Loss<T>::set_outputs(std::shared_ptr<std::vector<Mat<T>>> outputs)
{
	if (outputs == nullptr || !outputs->size())
		throw std::invalid_argument("Outputs cannot be empty.");
	outputs_ = outputs;
	output_shape_ = (*outputs)[0].get_shape();
	return *this;
}

// --------------------- GETTERS ---------------------

template <typename T>
std::shared_ptr<std::vector<Mat<T>>> Loss<T>::get_inputs(void) const
{
	return inputs_;
}

template <typename T>
std::shared_ptr<std::vector<Mat<T>>> Loss<T>::get_outputs(void) const
{
	return outputs_;
}

template <typename T>
const std::vector<Mat<T>> &Loss<T>::get_predictions(void) const
{
	return predictions_;
}

template <typename T>
const std::string &Loss<T>::get_name(void) const
{
	return name_;
}

template <typename T>
const Shape &Loss<T>::get_input_shape(void) const
{
	return input_shape_;
}

template <typename T>
const Shape &Loss<T>::get_output_shape(void) const
{
	return output_shape_;
}

template <typename T>
const Mat<T> &Loss<T>::get_last_loss(void) const
{
	return last_loss_;
}


// TODO: Refactor this
template <typename T>
T Loss<T>::get_normalized_loss(void) const
{
	if (!predictions_.size() || last_loss_.get_shape().rows == 0 || last_loss_.get_shape().cols == 0)
		return static_cast<T>(0);

	T min_val = predictions_[0](0, 0);
	T max_val = predictions_[0](0, 0);

	for (const auto &pred : predictions_) {
		for (std::size_t i = 0; i < pred.get_shape().rows; ++i)
			for (std::size_t j = 0; j < pred.get_shape().cols; ++j) {
				min_val = std::min(min_val, pred(i, j));
				max_val = std::max(max_val, pred(i, j));
			}
	}

	T range = max_val - min_val;
	if (range == 0) return static_cast<T>(0);

	T sum = 0;
	for (std::size_t i = 0; i < last_loss_.get_shape().rows; ++i)
		for (std::size_t j = 0; j < last_loss_.get_shape().cols; ++j)
			sum += (last_loss_(i, j) - min_val) / range;

	return sum / (last_loss_.get_shape().rows * last_loss_.get_shape().cols);
}

// Explicit template instantiation for Loss
template class nn::loss_funcs::Loss<float>;
// template class Loss<double>;

// ===================== MEAN ABSOLUTE ERROR IMPLEMENTATION =====================

template <typename T>
MeanAbsoluteError<T>::MeanAbsoluteError(std::shared_ptr<std::vector<Mat<T>>> inputs,
                                        std::shared_ptr<std::vector<Mat<T>>> outputs)
	: Loss<T>(inputs, outputs, "MeanAbsoluteError")
{
}

// Evaluate MAE on all stored inputs/outputs
template <typename T>
Mat<T> MeanAbsoluteError<T>::operator()(void)
{
	if (this->model_ == nullptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));

	for (std::size_t i = 0; i < this->inputs_->size(); ++i) {
		Mat<T> y_pred = (*this->model_)((*this->inputs_)[i]);
		this->predictions_.push_back(y_pred);

		Mat<T> diff = y_pred - (*this->outputs_)[i];
		for (std::size_t r = 0; r < y_pred.rows(); ++r)
			for (std::size_t c = 0; c < y_pred.cols(); ++c)
				this->last_loss_(r, c) += std::abs(diff(r, c));
	}

	this->last_loss_ /= static_cast<T>(this->inputs_->size());
	return this->last_loss_;
}

// Evaluate MAE on a batch
template <typename T>
Mat<T> MeanAbsoluteError<T>::operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch)
{
	if (this->model_ == nullptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));

	for (const auto &ex : batch) {
		const auto &[x, y_true] = ex;
		Mat<T> y_pred = (*this->model_)(x);
		this->predictions_.push_back(y_pred);

		Mat<T> diff = y_pred - y_true;
		for (std::size_t r = 0; r < y_pred.rows(); ++r)
			for (std::size_t c = 0; c < y_pred.cols(); ++c)
				this->last_loss_(r, c) += std::abs(diff(r, c));
	}

	return this->last_loss_ / static_cast<T>(batch.size());
}

// Evaluate MAE on a single example
template <typename T>
Mat<T> MeanAbsoluteError<T>::operator()(const std::pair<Mat<T>, Mat<T>> &example)
{
	if (!this->model_) throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*this->model_)(x);

	this->predictions_.clear();
	this->predictions_.push_back(y_pred);

	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0f));
	for (std::size_t r = 0; r < y_pred.rows(); ++r)
		for (std::size_t c = 0; c < y_pred.cols(); ++c)
			this->last_loss_(r, c) = std::abs(y_pred(r, c) - y_true(r, c));

	return this->last_loss_;
}

// Derivative of MAE
template <typename T>
Mat<T> MeanAbsoluteError<T>::derivate(const std::pair<Mat<T>, Mat<T>> &example)
{
	if (!this->model_) throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*this->model_)(x);

	Mat<T> grad(this->output_shape_);
	for (std::size_t r = 0; r < y_pred.rows(); ++r)
		for (std::size_t c = 0; c < y_pred.cols(); ++c) {
			T val = y_pred(r, c) - y_true(r, c);
			grad(r, c) = (val > 0) ? 1 : ((val < 0) ? -1 : 0);
		}

	return grad;
}

// Explicit template instantiation for MAE
template class nn::loss_funcs::MeanAbsoluteError<float>;
// template class MeanAbsoluteError<double>
