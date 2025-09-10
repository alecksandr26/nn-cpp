#include "../include/layer.hpp"
#include <cstddef>
#include <memory>

using namespace nn::mathops;
using namespace nn::layers;

template <typename T>
nn::layers::Layer<T>::Layer(bool trainable, const Shape &input_shape,
                            const Shape &output_shape)
	: input_shape_(input_shape), output_shape_(output_shape),
      trainable_(trainable), built_(false), name_("Layer")
{
	
}

template <typename T>
nn::layers::Layer<T>::Layer(bool trainable, const Shape &input_shape, const Shape &output_shape, std::string name)
	: input_shape_(input_shape), output_shape_(output_shape),
	  trainable_(trainable), built_(false), name_(name)
{
}

template <typename T>
nn::layers::Layer<T>::Layer(bool trainable, std::size_t input_size, std::size_t output_size)
	: input_shape_({input_size, 1}), output_shape_({output_size, 1}),
	  trainable_(trainable), built_(false), name_("Layer")
{
}

template <typename T>
nn::layers::Layer<T>::Layer(bool trainable, std::size_t input_size, std::size_t output_size, std::string name)
	: input_shape_({input_size, 1}), output_shape_({output_size, 1}),
	  trainable_(trainable), built_(false),
	  name_(name)
{
}

// NOTE: Creating a foo deconstructor to avoid linker problems
template <typename T>
nn::layers::Layer<T>::~Layer(void)
{
}

template <typename T>
bool nn::layers::Layer<T>::is_trainable(void)
{
	return trainable_;
}

template <typename T>
bool nn::layers::Layer<T>::is_built(void)
{
	return built_;
}

template <typename T>
Layer<T> &nn::layers::Layer<T>::set_input_shape(const Shape &input_shape)
{
	input_shape_ = input_shape;
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

template <typename T>
Layer<T> &nn::layers::Layer<T>::set_input_size(std::size_t input_size)
{
	input_shape_ = {input_size, input_shape_.cols};
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

template <typename T>
Layer<T> &nn::layers::Layer<T>::set_output_shape(const Shape &output_shape)
{
	output_shape_ = output_shape;
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

template <typename T>
Layer<T> &nn::layers::Layer<T>::set_output_size(std::size_t output_size)
{
	output_shape_ = {output_size, output_shape_.cols};
	built_ = false;  // If shape changes, the layer probably needs rebuild
	return *this;
}

template <typename T>
Layer<T> &nn::layers::Layer<T>::set_name(std::string name)
{
	name_ = std::move(name);
	return *this;
}

template <typename T>
const Shape &nn::layers::Layer<T>::get_input_shape(void) const
{
	return input_shape_;
}

template <typename T>
const std::size_t &nn::layers::Layer<T>::get_input_size(void) const
{
	return input_shape_.rows;
}

template <typename T>
const Shape &nn::layers::Layer<T>::get_output_shape(void) const
{
	return output_shape_;
}

template <typename T>
const std::size_t &nn::layers::Layer<T>::get_output_size(void) const
{
	return output_shape_.rows;
}

template <typename T>
const std::string &nn::layers::Layer<T>::get_name(void) const
{
	return name_;
}


template class nn::layers::Layer<float>;
template class nn::layers::Layer<double>;


template <typename T>
nn::layers::WeightedleLayer<T>::WeightedleLayer(const Shape &input_shape, const Shape &output_shape)
	: Layer<T>(true, input_shape, output_shape)
{
}

template <typename T>
nn::layers::WeightedleLayer<T>::WeightedleLayer(const Shape &input_shape, const Shape &output_shape, std::string name)
	: Layer<T>(true, input_shape, output_shape, std::move(name))
{
}


template <typename T>
nn::layers::WeightedleLayer<T>::WeightedleLayer(std::size_t input_size, std::size_t output_size)
	: Layer<T>(true, input_size, output_size)
{
}


template <typename T>
nn::layers::WeightedleLayer<T>::WeightedleLayer(std::size_t input_size, std::size_t output_size, std::string name)
	: Layer<T>(true, input_size, output_size, std::move(name))
{
}

// NOTE: Foo deconstructor just to avoid linker problems
template <typename T>
nn::layers::WeightedleLayer<T>::~WeightedleLayer(void)
{
}

template <typename T>
WeightedleLayer<T> &nn::layers::WeightedleLayer<T>::set_optimizer(Optimizer<T> &optimizer)
{
	optimizer_ = std::shared_ptr<Optimizer<T>>(&optimizer);
	return *this;
}

template <typename T>
std::shared_ptr<Optimizer<T>> nn::layers::WeightedleLayer<T>::get_optimizer(void) const
{
	return optimizer_;
}

// TODO: Use here for this alloc weights methods a custom allocator
// TODO: Add custom ways to create this weights like the randomize way, different rand functions
template <typename T>
std::unique_ptr<Mat<T>> nn::layers::WeightedleLayer<T>::add_weights(const Shape &shape) const
{
	return std::make_unique<Mat<T>>(shape);
}

template <typename T>
std::unique_ptr<Mat<T>> nn::layers::WeightedleLayer<T>::add_weights(std::size_t input_size) const
{
	return std::make_unique<Mat<T>>(Shape{input_size, 1});
}


template class nn::layers::WeightedleLayer<float>;
// template class nn::layers::WeightedleLayer<double>;


template <typename T>
nn::layers::Dense<T>::Dense(const Shape &input_shape, const Shape &output_shape)
	: WeightedleLayer<T>(input_shape, output_shape, "Dense")
{
}

template <typename T>
nn::layers::Dense<T>::Dense(std::size_t input_size, std::size_t output_size)
	: WeightedleLayer<T>(input_size, output_size, "Dense")
{
}

template <typename T>
Mat<T> nn::layers::Dense<T>::operator()(const Mat<T> &X)
{
	return X.dot(*weights_.get()) + *bias_.get();
}

template <typename T>
Mat<T> nn::layers::Dense<T>::derivate(const Mat<T> &X)
{
	return X;
}

template <typename T>
Dense<T> &nn::layers::Dense<T>::build(const Shape &input_shape, const Shape &output_shape)
{
	// TODO: free the previous memory of the weights
	weights_ = WeightedleLayer<T>::add_weights(Shape{output_shape.rows, input_shape.rows});
	bias_ = WeightedleLayer<T>::add_weights(output_shape.rows);
	return *this;
}

template <typename T>
Dense<T> &nn::layers::Dense<T>::fit(const Mat<T> &signal_update, const Mat<T> &input)
{
	// TODO: Validate that the shared pointer of the optimizer is not nullptr
	WeightedleLayer<T>::optimizer_.get()->update(*weights_.get(), signal_update, input);
	return *this;
}

template class nn::layers::Dense<float>;
// template class nn::layers::Dense<double>;

