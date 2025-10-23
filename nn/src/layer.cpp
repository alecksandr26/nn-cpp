#include "../include/layer.hpp"
#include <assert.h>
#include <cstddef>
#include <memory>
#include <stdexcept>

using namespace nn::mathops;
using namespace nn::layers;

nn::layers::Layer::Layer(void)
	: trainable_(false), name_("Layer")
{
}

nn::layers::Layer::Layer(const Shape &input_shape, const Shape &output_shape, bool trainable, std::string name)
	: input_shape_(input_shape), output_shape_(output_shape),
	  trainable_(trainable), built_(false), name_(name)
{
}

nn::layers::Layer::Layer(std::size_t input_size, std::size_t output_size, bool trainable, std::string name)
	: input_shape_({input_size, 1}), output_shape_(output_size > 0 ? Shape{output_size, 1} : Shape()),
	  trainable_(trainable), built_(false),
	  name_(name)
{
}

nn::layers::Layer::~Layer(void) = default;

bool nn::layers::Layer::is_trainable(void)
{
	return trainable_;
}

bool nn::layers::Layer::is_built(void)
{
	return built_;
}

Layer &nn::layers::Layer::set_input_shape(const Shape &input_shape)
{
	input_shape_ = input_shape;
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

Layer &nn::layers::Layer::set_input_size(std::size_t input_size)
{
	input_shape_ = {input_size, input_shape_.cols};
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

Layer &nn::layers::Layer::set_output_shape(const Shape &output_shape)
{
	output_shape_ = output_shape;
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

Layer &nn::layers::Layer::set_output_size(std::size_t output_size)
{
	output_shape_ = {output_size, output_shape_.cols};
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

Layer &nn::layers::Layer::set_name(std::string name)
{
	name_ = std::move(name);
	return *this;
}

const Shape &nn::layers::Layer::get_input_shape(void) const
{
	return input_shape_;
}

const std::size_t &nn::layers::Layer::get_input_size(void) const
{
	return input_shape_.rows;
}

const Shape &nn::layers::Layer::get_output_shape(void) const
{
	return output_shape_;
}

const std::size_t &nn::layers::Layer::get_output_size(void) const
{
	return output_shape_.rows;
}

const std::string &nn::layers::Layer::get_name(void) const
{
	return name_;
}

nn::layers::WeightedLayer::~WeightedLayer(void) = default;

WeightedLayer &nn::layers::WeightedLayer::set_optimizer(std::shared_ptr<Optimizer> optimizer)
{
	optimizer_ = optimizer;	
	return *this;
}

std::shared_ptr<Optimizer> nn::layers::WeightedLayer::get_optimizer(void) const
{
	return optimizer_;
}

// TODO: Use here for this alloc weights methods a custom allocator
// TODO: Add custom ways to create this weights like the randomize way,
// different rand functions
template <typename T>
std::unique_ptr<Mat<T>> nn::layers::WeightedLayer::add_weights(const Shape &shape, std::shared_ptr<RandInitializer> rand_init) const
{
	std::unique_ptr<Mat<T>> weights = std::make_unique<Mat<T>>(shape);
	if (rand_init != nullptr) {
		(*rand_init)(*weights);
	} else {
		weights->fill(static_cast<T>(0.0f));
	}
	
	return weights;
}

template <typename T>
std::unique_ptr<Mat<T>> nn::layers::WeightedLayer::add_weights(std::size_t input_size, std::shared_ptr<RandInitializer> rand_init) const
{
	std::unique_ptr<Mat<T>> weights = std::make_unique<Mat<T>>(Shape{input_size, 1});
	if (rand_init != nullptr) {
		(*rand_init)(*weights);
	} else {
		weights->fill(static_cast<T>(0.0f));
	}
	
	return weights;
}

template std::unique_ptr<Mat<float>> nn::layers::WeightedLayer::add_weights(const Shape &shape, std::shared_ptr<RandInitializer> rand_init) const;
// template std::unique_ptr<Mat<double>> nn::layers::WeightedLayer::add_weights(const Shape &shape) const;
template std::unique_ptr<Mat<float>> nn::layers::WeightedLayer::add_weights(std::size_t input_size, std::shared_ptr<RandInitializer> rand_init) const;
// template std::unique_ptr<Mat<double>> nn::layers::WeightedLayer::add_weights(std::size_t input_size) const;

template <typename T>
nn::layers::Dense<T>::Dense(const Shape &input_shape, const Shape &output_shape, std::shared_ptr<Layer> activation_func, std::shared_ptr<RandInitializer> rand_init)
	: WeightedLayer(input_shape, output_shape, true, "Dense"), activation_func_(activation_func), rand_init_(rand_init)
{
}

template <typename T>
nn::layers::Dense<T>::Dense(std::size_t input_size, std::size_t output_size, std::shared_ptr<Layer> activation_func, std::shared_ptr<RandInitializer> rand_init)
	: WeightedLayer(input_size, output_size, true, "Dense"), activation_func_(activation_func), rand_init_(rand_init)
{
}

template <typename T>
nn::layers::Dense<T>::Dense(const Shape &input_shape, std::shared_ptr<Layer> activation_func, std::shared_ptr<RandInitializer> rand_init)
	: WeightedLayer(input_shape, Shape(), true, "Dense"), activation_func_(activation_func), rand_init_(rand_init)
{
}

template <typename T>
nn::layers::Dense<T>::Dense(std::size_t input_size, std::shared_ptr<Layer> activation_func, std::shared_ptr<RandInitializer> rand_init)
	: WeightedLayer(input_size, 0, true, "Dense"), activation_func_(activation_func), rand_init_(rand_init)
{
}

template <typename T>
Mat<T> &nn::layers::Dense<T>::get_weights(void) const
{
	if (weights_ == nullptr) {
		throw std::invalid_argument("Dense layer not built yet" );
	}
	return *weights_;
}

template <typename T>
Mat<T> &nn::layers::Dense<T>::get_bias(void) const
{
	if (bias_ == nullptr) {
		throw std::invalid_argument("Dense layer not built yet" );
	}
	return *bias_;
}


template <typename T>
bool nn::layers::Dense<T>::has_activation_func(void) const
{
	return this->activation_func_ != nullptr;
}

template <typename T>
std::shared_ptr<Layer> nn::layers::Dense<T>::get_activation_func(void) const
{
	return activation_func_;
}

template <typename T>
Dense<T> &nn::layers::Dense<T>::build(const Shape &input_shape, const Shape &output_shape)
{
	if (input_shape.rows == 0 || input_shape.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + name_);
	}
	
	if (output_shape.cols == 0 || output_shape.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + name_);
	}

	input_shape_ = input_shape;
	output_shape_ = output_shape;

	// TODO: free the previous memory of the weights
	weights_ = add_weights<T>(Shape{output_shape.rows, input_shape.rows}, rand_init_);
	bias_ = add_weights<T>(output_shape.rows, rand_init_);

	register_funcs();

	
	if (activation_func_ != nullptr) {
		// TODO: This needs to be verify 
		activation_func_.get()->build(output_shape, output_shape);
	}

	built_ = true;
	return *this;
}

template <typename T>
Dense<T> &nn::layers::Dense<T>::build(std::size_t input_size, std::size_t output_size)
{
	if (input_size == 0) {
		throw std::invalid_argument("Invalid input size of the layer: " + name_);
	}
	
	if (input_size == 0) {
		throw std::invalid_argument("Invalid output size of the layer: " + name_);
	}

	input_shape_ = Shape{input_size, 1};
	output_shape_ = Shape{output_size, 1};

	// TODO: free the previous memory of the weights
	weights_ = add_weights<T>(Shape{output_size, input_size}, rand_init_);
	bias_ = add_weights<T>(output_size, rand_init_);

	register_funcs();
	
	if (activation_func_ != nullptr) {
		// TODO: This needs to be verify 
		activation_func_.get()->build(output_size, output_size);
	}

	built_ = true;
	return *this;
}

template <typename T>
Dense<T> &nn::layers::Dense<T>::build(void)
{
	if (input_shape_.rows == 0 || input_shape_.cols == 0) {
		throw std::invalid_argument("Invalid input shape of the layer: " + name_);
	}
	
	if (output_shape_.cols == 0 || output_shape_.cols == 0) {
		throw std::invalid_argument("Invalid output shape of the layer: " + name_);
	}

	// TODO: free the previous memory of the weights
	weights_ = add_weights<T>(Shape{output_shape_.rows, input_shape_.rows}, rand_init_);
	bias_ = add_weights<T>(output_shape_.rows, rand_init_);

	register_funcs();


	if (activation_func_ != nullptr) {
		// TODO: This needs to be verify 
		activation_func_.get()->build();
	}

	built_ = true;
	return *this;
}

template <typename T>
Dense<T> &nn::layers::Dense<T>::register_funcs(void)
{
	// Register the functions
	register_func<Mat<T>, const Mat<T> &>
		("feedforward", [this](const Mat<T> &X) -> Mat<T> {
			if (activation_func_ != nullptr) {
				// f(W . X + B), where 'f' is an activation function
				Mat<T> Z = weights_->dot(X) + *bias_;
				return (*activation_func_)(Z);
			}
			
			// W . X + B
			return weights_->dot(X) + *bias_;
		});

	register_func<Mat<T>, const Mat<T> &>
		("gradient", [this](const Mat<T> &X) -> Mat<T> {
			// Z = W . X + B
			// g_z(X) = W^T . [1]_m = [ \sum_j^m w_{i, j} ]_n
			if (activation_func_ != nullptr) {
				// g_f(X) = g_f(Z) * g_z(X) = g_f(Z) * (W^T . [1]_m)
				Mat<T> Z = weights_->dot(X) + *bias_;
				return activation_func_->gradient(Z)	\
					* ((*weights_).transpose_copy()	\
					   .dot(Mat<T>(output_shape_).fill(1.0f)));
			}
			return weights_->transpose_copy().dot(Mat<T>(output_shape_).fill(1.0f));
		});
	
	register_func<Mat<T>, const Mat<T> &>
		("jacobian", [this](const Mat<T> &X) -> Mat<T> {
			// Z = W . X + B
			// J_z(X) = W, Jacobian
			if (activation_func_ != nullptr) {
				// J_f(X) = J_f(Z) . J_z(X) = J_f(Z) . W
				Mat<T> Z = weights_->dot(X) + *(bias_.get());
				// J_f(Z) ~ (m, m) x (m, n) = (m, n)
				return activation_func_->jacobian(Z).dot(*weights_);
			}
			return *weights_;
		});


	register_func<void, const Mat<T> &, const Mat<T> &>
		("fit", [this](const Mat<T> &signal_update, const Mat<T> &input) -> void {
			optimizer_.get()->update(*weights_, signal_update, input);
			optimizer_.get()->update(*bias_, signal_update, Mat<T>(1, 1).fill(static_cast<T>(1.0f)));
		});
	
	return *this;
}

template class nn::layers::Dense<float>;
// template class nn::layers::Dense<double>;

