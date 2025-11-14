#include "../include/loss_func.hpp"
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <utility>

using namespace nn::loss_funcs;

// ===================== LOSS IMPLEMENTATION =====================

template <typename T>
Loss<T>::Loss(std::shared_ptr<std::vector<Mat<T>>> inputs,
              std::shared_ptr<std::vector<Mat<T>>> outputs,
              std::string name)
	: inputs_(inputs),
	  outputs_(outputs),
	  model_(),  // Default construct weak_ptr (empty state)
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
Loss<T> &Loss<T>::set_model(std::shared_ptr<Model> model)
{
	model_ = model;
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


// gradient of MAE for all the inputs and outputs
template <typename T>
Mat<T> Loss<T>::gradient(void)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");
	if (!this->inputs_ || !this->outputs_)
		throw std::runtime_error("Not set input and output");

	Mat<T> grad(this->output_shape_);
	for (std::size_t i = 0; i < this->inputs_->size(); i++) {
		Mat<T> g = this->gradient(std::make_pair((*this->inputs_)[i], (*this->outputs_)[i]));
		grad += g;
	}

	return grad;
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


// TODO: we should refactor these functions since it has a lot of repetate code 
// Evaluate MAE on all stored inputs/outputs
template <typename T>
Mat<T> MeanAbsoluteError<T>::operator()(void)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");
	
	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));

	for (std::size_t i = 0; i < this->inputs_->size(); ++i) {
		Mat<T> y_pred = (*model_ptr)((*this->inputs_)[i]);
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
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));

	for (const auto &ex : batch) {
		const auto &[x, y_true] = ex;
		Mat<T> y_pred = (*model_ptr)(x);
		this->predictions_.push_back(y_pred);

		Mat<T> diff = y_pred - y_true;
		for (std::size_t r = 0; r < y_pred.rows(); ++r)
			for (std::size_t c = 0; c < y_pred.cols(); ++c)
				this->last_loss_(r, c) += std::abs(diff(r, c));
	}

	this->last_loss_ /= static_cast<T>(this->inputs_->size());
	return this->last_loss_;
}

// Evaluate MAE on a single example
template <typename T>
Mat<T> MeanAbsoluteError<T>::operator()(const std::pair<Mat<T>, Mat<T>> &example)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);

	this->predictions_.clear();
	this->predictions_.push_back(y_pred);

	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0f));
	for (std::size_t r = 0; r < y_pred.rows(); ++r)
		for (std::size_t c = 0; c < y_pred.cols(); ++c)
			this->last_loss_(r, c) = std::abs(y_pred(r, c) - y_true(r, c));

	this->last_loss_ /= static_cast<T>(this->inputs_->size());
	return this->last_loss_;
}

// gradient of MAE
template <typename T>
Mat<T> MeanAbsoluteError<T>::gradient(const std::pair<Mat<T>, Mat<T>> &example)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);
	
	Mat<T> grad(this->output_shape_);
	for (std::size_t r = 0; r < y_pred.rows(); ++r)
		for (std::size_t c = 0; c < y_pred.cols(); ++c) {
			T val = y_pred(r, c) - y_true(r, c);
			grad(r, c) = (val > 0) ? 1 : ((val < 0) ? -1 : 0);
		}

	return grad;
}


// jacobian
template <typename T>
Mat<T> MeanAbsoluteError<T>::jacobian(const std::pair<Mat<T>, Mat<T>> &example)
{
	// This is supposing that, F:  X \in R^{n, 1} -> Y \in R^{n, 1}
	// Such that its Jacobian should be: J(F) \in R^{n, n}
	
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);
	
	Mat<T> jaco(this->output_shape_.rows, this->input_shape_.rows);
	jaco.fill(static_cast<T>(0.0));
	for (std::size_t i = 0; i < this->output_shape_.rows; i++) {
		T val = y_pred(i, 0) - y_true(i, 0);
		jaco(i, i) = (val > 0) ? 1 : ((val < 0) ? -1 : 0);
	}
	
	return jaco;
}



// Explicit template instantiation for MAE
template class nn::loss_funcs::MeanAbsoluteError<float>;
// template class MeanAbsoluteError<double>


// ===================== Cross Entropy IMPLEMENTATION =====================


template <typename T>
CrossEntropy<T>::CrossEntropy(std::shared_ptr<std::vector<Mat<T>>> inputs,
			      std::shared_ptr<std::vector<Mat<T>>> outputs)
	: Loss<T>(inputs, outputs, "CrossEntropy")
{
}


template <typename T>
Mat<T> CrossEntropy<T>::operator()(void)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");
		
	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));

	// sum_{i = 1}^n - [y_i * log(s(x_i^T * \theta)) + (1 - y_i) * log(1 - s(1 - s(x_i^T * \theta)))]
	for (std::size_t i = 0; i < this->inputs_->size(); i++) {
		Mat<T> y_pred = (*model_ptr)((*this->inputs_)[i]);
		this->predictions_.push_back(y_pred);

		// Compute the log
		Mat<T> term1(y_pred.rows(), 1);
		Mat<T> term2(y_pred.rows(), 1);
		
		for (std::size_t i = 0; i < y_pred.rows(); i++) {
			term1(i, 0) = std::log(y_pred(i, 0) + 1e-8);
			term2(i, 0) = std::log(1 - y_pred(i, 0) + 1e-8);
		}

		// Create ones 
		Mat<T> ones = Mat<T>(y_pred.rows(), 1).fill(static_cast<T>(1.0));
		this->last_loss_ += ((*this->outputs_)[i] * term1 + ((*this->outputs_)[i] * (-1) + ones) * term2) * (-1);
	}

	return this->last_loss_;
}



template <typename T>
Mat<T> CrossEntropy<T>::operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));
	for (const auto &ex : batch) {
		const auto &[x, y_true] = ex;
		Mat<T> y_pred = (*model_ptr)(x);
		this->predictions_.push_back(y_pred);

		Mat<T> term1(y_pred.rows(), 1);
		Mat<T> term2(y_pred.rows(), 1);
		
		for (std::size_t i = 0; i < y_pred.rows(); i++) {
			term1(i, 0) = std::log(y_pred(i, 0) + 1e-8);
			term2(i, 0) = std::log(1 - y_pred(i, 0) + 1e-8);
		}

		// Create ones 
		Mat<T> ones = Mat<T>(y_pred.rows(), 1).fill(static_cast<T>(1.0));
		this->last_loss_ += (y_true * term1 + (y_true * (-1) + ones) * term2) * (-1);
	}

	return this->last_loss_;
}


template <typename T>
Mat<T> CrossEntropy<T>::operator()(const std::pair<Mat<T>, Mat<T>> &example)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	this->predictions_.clear();
	
	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);
	this->predictions_.push_back(y_pred);

	Mat<T> term1(y_pred.rows(), 1);
	Mat<T> term2(y_pred.rows(), 1);

	
	for (std::size_t i = 0; i < y_pred.rows(); i++) {
		term1(i, 0) = std::log(y_pred(i, 0) + 1e-8);
		term2(i, 0) = std::log(1 - y_pred(i, 0) + 1e-8);
	}
	
	// Create ones 
	Mat<T> ones = Mat<T>(y_pred.rows(), 1).fill(static_cast<T>(1.0));
	this->last_loss_ = (y_true * term1 + (y_true * (-1) + ones) * term2) * (-1);
	
	return this->last_loss_;
}


template <typename T>
Mat<T> CrossEntropy<T>::gradient(const std::pair<Mat<T>, Mat<T>> &example)
{
	/*
	  L(a) = - (y * log(a) + (1 - y) * log(1 - a))
	  dL/da = - (y * 1/a) + (1 - y) * 1/(1 - a)
	        = - (a - y)/(a * ( 1 - a))
	 */
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);
	Mat<T> grad(this->output_shape_);
	for (std::size_t r = 0; r < y_pred.rows(); ++r)
		for (std::size_t c = 0; c < y_pred.cols(); ++c)
			grad(r, c) = (y_pred(r, c) - y_true(r, c)) / ((y_pred(r, c) + 1e-8) * (1 - y_pred(r, c) + 1e-8));

	return grad;
}


template <typename T>
Mat<T> CrossEntropy<T>::jacobian(const std::pair<Mat<T>, Mat<T>> &example)
{
	// This is supposing that, F:  X \in R^{n, 1} -> Y \in R^{n, 1}
	// Such that its Jacobian should be: J(F) \in R^{n, n}
	
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);

	Mat<T> jaco(this->output_shape_.rows, this->input_shape_.rows);
	jaco.fill(static_cast<T>(0.0));
	for (std::size_t i = 0; i < this->output_shape_.rows; i++)
		jaco(i, i) = - (y_true(i, 0) / (y_pred(i, 0) + 1e-8)) + ((1 - y_true(i, 0)) / (1 - y_pred(i, 0) + 1e-8));
	return jaco;
}


template class nn::loss_funcs::CrossEntropy<float>;
// template class nn::loss_funcs::CrossEntropy<double>;


// ===================== MEAN SQUARED ERROR IMPLEMENTATION =====================

template <typename T>
MeanSquaredError<T>::MeanSquaredError(std::shared_ptr<std::vector<Mat<T>>> inputs,
                                      std::shared_ptr<std::vector<Mat<T>>> outputs)
    : Loss<T>(inputs, outputs, "MeanSquaredError")
{
}

// Evaluate MSE on all stored inputs/outputs
template <typename T>
Mat<T> MeanSquaredError<T>::operator()(void)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));

	for (std::size_t i = 0; i < this->inputs_->size(); ++i) {
		Mat<T> y_pred = (*model_ptr)((*this->inputs_)[i]);
		this->predictions_.push_back(y_pred);

		Mat<T> diff = y_pred - (*this->outputs_)[i];
		for (std::size_t r = 0; r < y_pred.rows(); ++r)
			for (std::size_t c = 0; c < y_pred.cols(); ++c)
				this->last_loss_(r, c) += diff(r, c) * diff(r, c);
	}

	this->last_loss_ /= static_cast<T>(this->inputs_->size());
	return this->last_loss_;
}

// Evaluate MSE on a batch
template <typename T>
Mat<T> MeanSquaredError<T>::operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	this->predictions_.clear();
	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));

	for (const auto &ex : batch) {
		const auto &[x, y_true] = ex;
		Mat<T> y_pred = (*model_ptr)(x);
		this->predictions_.push_back(y_pred);

		Mat<T> diff = y_pred - y_true;
		for (std::size_t r = 0; r < y_pred.rows(); ++r)
			for (std::size_t c = 0; c < y_pred.cols(); ++c)
				this->last_loss_(r, c) += diff(r, c) * diff(r, c);
	}

	this->last_loss_ /= static_cast<T>(batch.size());
	return this->last_loss_;
}

// Evaluate MSE on a single example
template <typename T>
Mat<T> MeanSquaredError<T>::operator()(const std::pair<Mat<T>, Mat<T>> &example)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);

	this->predictions_.clear();
	this->predictions_.push_back(y_pred);

	this->last_loss_.resize(this->output_shape_).fill(static_cast<T>(0.0));
	for (std::size_t r = 0; r < y_pred.rows(); ++r)
		for (std::size_t c = 0; c < y_pred.cols(); ++c) {
			T diff = y_pred(r, c) - y_true(r, c);
			this->last_loss_(r, c) = diff * diff;
		}

	this->last_loss_ /= static_cast<T>(this->inputs_->size());
	return this->last_loss_;
}

// Gradient of MSE
template <typename T>
Mat<T> MeanSquaredError<T>::gradient(const std::pair<Mat<T>, Mat<T>> &example)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);

	Mat<T> grad(this->output_shape_);
	for (std::size_t r = 0; r < y_pred.rows(); ++r)
		for (std::size_t c = 0; c < y_pred.cols(); ++c)
			grad(r, c) = 2 * (y_pred(r, c) - y_true(r, c));

	return grad;
}

// Jacobian of MSE (diagonal)
template <typename T>
Mat<T> MeanSquaredError<T>::jacobian(const std::pair<Mat<T>, Mat<T>> &example)
{
	auto model_ptr = this->model_.lock();
	if (!model_ptr)
		throw std::runtime_error("Model pointer not set in Loss function.");

	const auto &[x, y_true] = example;
	Mat<T> y_pred = (*model_ptr)(x);

	Mat<T> jaco(this->output_shape_.rows, this->input_shape_.rows);
	jaco.fill(static_cast<T>(0.0));

	for (std::size_t i = 0; i < this->output_shape_.rows; ++i)
		jaco(i, i) = 2 * (y_pred(i, 0) - y_true(i, 0));

	return jaco;
}

// Explicit template instantiation for MSE
template class nn::loss_funcs::MeanSquaredError<float>;
// template class nn::loss_funcs::MeanSquaredError<double>
