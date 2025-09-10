#ifndef NN_LAYER_INCLUDED
#define NN_LAYER_INCLUDED

#include "mat.hpp"
#include "optimizer.hpp"
#include <memory>

namespace nn::layers {
	using namespace mathops;
	using namespace optimizers;

	
	// TODO: There is a way to stop having this class thing, by creating manually our virtual tables
	// https://chatgpt.com/share/68c0b394-3688-800e-813d-952ec1979b52
	
	template <typename T>
	class Layer {
	public:
		Layer(bool trainable, const Shape &input_shape, const Shape &output_shape);
		Layer(bool trainable, std::size_t input_size, std::size_t output_size);
		Layer(bool trainable, const Shape &input_shape, const Shape &output_shape, std::string name);
		Layer(bool trainable, std::size_t input_size, std::size_t output_size, std::string name);
		virtual ~Layer(void) = 0;
		
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
		virtual Layer &build(const Shape &input_shape, const Shape &output_shape) = 0;
		
        protected:
		Shape input_shape_;
		Shape output_shape_;
		bool trainable_;
		bool built_;
		std::string name_;
	};

	// TODO: Add the unit tests for a weighted layer
	
	template <typename T>
	class WeightedleLayer : public Layer<T> {
	public:
		WeightedleLayer(const Shape &input_shape, const Shape &output_shape);
		WeightedleLayer(std::size_t input_size, std::size_t output_size);
		WeightedleLayer(const Shape &input_shape, const Shape &output_shape, std::string name);
		WeightedleLayer(std::size_t input_size, std::size_t output_size, std::string name);
		virtual ~WeightedleLayer(void) = 0;
		
		// A weighted layer needs an optimizer to optimize the weights
		WeightedleLayer &set_optimizer(Optimizer<T> &optimizer);
		std::shared_ptr<Optimizer<T>> get_optimizer(void) const;
		
		// the signal update could be any error, or gradient to be used by the optimizer
		virtual WeightedleLayer &fit(const Mat<T> &signal_update, const Mat<T> &input) = 0;
	protected:
		// add_weights: Is an allocator method to alloc new weights
		std::unique_ptr<Mat<T>> add_weights(const Shape &shape) const;
		// for the input_size will alloc a row vector column
		std::unique_ptr<Mat<T>> add_weights(std::size_t input_size) const;
		std::shared_ptr<Optimizer<T>> optimizer_;
	};


	// A dense layer of neurons
	template <typename T>
	class Dense : public WeightedleLayer<T> {
	public:
		Dense(const Shape &input_shape, const Shape &output_shape);
		Dense(std::size_t input_size, std::size_t output_size);
		~Dense(void) override = default;
		
		Mat<T> operator()(const Mat<T> &X) override;
		Mat<T> derivate(const Mat<T> &X) override;
		Dense &build(const Shape &input_shape, const Shape &output_shape) override;
		Dense &fit(const Mat<T> &signal_update, const Mat<T> &input) override;
	private:
		std::unique_ptr<Mat<T>> weights_;
		std::unique_ptr<Mat<T>> bias_;
	};
}

#endif

