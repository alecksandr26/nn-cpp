#include "../include/layer.hpp"
#include <cstddef>

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


