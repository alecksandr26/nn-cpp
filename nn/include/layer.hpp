#ifndef NN_LAYER_INCLUDED
#define NN_LAYER_INCLUDED

#include "mat.hpp"
#include "optimizer.hpp"

namespace nn::layers {
	using namespace mathops;
	using namespace optimizers;

	template <typename T>
	class Layer {
	public:
		Layer(bool trainable, const Shape &input_shape, const Shape &output_shape);
		Layer(bool trainable, std::size_t input_size, std::size_t output_size);
		Layer(bool trainable, const Shape &input_shape, const Shape &output_shape, std::string name);
		Layer(bool trainable, std::size_t input_size, std::size_t output_size, std::string name);
		virtual ~Layer(void) = default;
		
		bool is_trainable(void);
		bool is_built(void);
		Layer &set_input_shape(const Shape &input_shape);
		Layer &set_input_size(std::size_t input_size);
		Layer &set_output_shape(const Shape &output_shape);
		Layer &set_output_size(std::size_t output_size);
		Layer &set_name(std::string name);
		const Shape &get_input_shape(void) const;
		const Shape &get_output_shape(void) const;
		const std::size_t &get_input_size(void) const;
		const std::size_t &get_output_size(void) const;
		const std::string &get_name(void) const;
		
		
		virtual Mat<T> operator()(const Mat<T> &X) = 0;
		/* derivate: Compute the derivative of the layer's output with respect to its input. */
		virtual Mat<T> derivate(const Mat<T> &x) = 0;
		virtual Layer &build(const Shape &input_shape) = 0;
		
        protected:
		Shape input_shape_;
		Shape output_shape_;
		bool trainable_;
		bool built_;
		std::string name_;
	};

	template <typename T>
	class TrainableLayer : public Layer<T> {
	public:
		
		virtual ~TrainableLayer(void) = default;
		virtual void fit(Mat<T> &weights, const Mat<T> &signal_update) = 0;
	protected:
		Mat<T> add_weights(const Shape &shape);
		Optimizer<T> &optimizer;
	};
}

#endif

