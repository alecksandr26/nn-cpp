#ifndef NN_LAYER_INCLUDED
#define NN_LAYER_INCLUDED

#include <memory>

#include "mat.hpp"
#include "optimizer.hpp"
#include "model.hpp"
#include "rand.hpp"

namespace nn::layers {
	using namespace mathops;
	using namespace optimizers;
	using namespace models;
	using namespace rand;

	class Layer : public Model {
	public:
		using Model::Model;
		
		Layer(void);
		Layer(const Shape &input_shape, const Shape &output_shape = Shape(), bool trainable = false, std::string name = "Layer");
		Layer(std::size_t input_size, std::size_t output_size = 0, bool trainable = false, std::string name = "Layer");
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

		// NOTE: Needs to override these functions
		template <typename T>
		Mat<T> operator()(const Mat<T> &X)
		{
			return get_func<Mat<T>, const Mat<T> &>("feedforward", __FILE__, __LINE__)(X);
		}
		
		/* jacobian: Compute the jacobian of the layer's output with respect to its input. */
		template <typename T>
		Mat<T> jacobian(const Mat<T> &X)
		{
			return get_func<Mat<T>, const Mat<T> &>("jacobian", __FILE__, __LINE__)(X);
		}

		/* gradient: Compute the gradient of the layer's output with respect to its input. */
		template <typename T>
		Mat<T> gradient(const Mat<T> &X)
		{
			return get_func<Mat<T>, const Mat<T> &>("gradient", __FILE__, __LINE__)(X);
		}
		
		virtual Layer &build(const Shape &input_shape, const Shape &output_shape) = 0;
		virtual Layer &build(std::size_t input_size, std::size_t output_size) = 0;
		virtual Layer &build(void) = 0;
		
        protected:
		Shape input_shape_;
		Shape output_shape_;
		bool trainable_;
		bool built_;
		std::string name_;
	};
	
	class WeightedLayer : public Layer {
	public:
		using Layer::Layer;
		
		virtual ~WeightedLayer(void) = 0;
		
		// A weighted layer needs an optimizer to optimize the weights
		WeightedLayer &set_optimizer(std::shared_ptr<Optimizer> optimizer);
		std::shared_ptr<Optimizer> get_optimizer(void) const;
		
		// the signal update could be any error, or gradient to be used by the optimizer
		template <typename T>
		WeightedLayer &fit(const Mat<T> &signal_update, const Mat<T> &input)
		{
			get_func<void, const Mat<T> &, const Mat<T> &>("fit", __FILE__, __LINE__)(signal_update, input);
			return *this;
		}
		
	protected:
		
		// add_weights: Is an allocator method to alloc new weights
		template <typename T>
		std::unique_ptr<Mat<T>> add_weights(const Shape &shape, std::shared_ptr<RandInitializer> rand_init = nullptr) const;
		
		// for the input_size will alloc a row vector column
		template <typename T>
		std::unique_ptr<Mat<T>> add_weights(std::size_t input_size, std::shared_ptr<RandInitializer> rand_init = nullptr) const;
		std::shared_ptr<Optimizer> optimizer_;
	};

	// A dense layer of neurons
	template <typename T>
	class Dense : public WeightedLayer {
	public:
		using WeightedLayer::WeightedLayer;
		
		Dense(const Shape &input_shape, const Shape &output_shape, std::shared_ptr<Layer> activation_func = nullptr, std::shared_ptr<RandInitializer> rand_init = nullptr);
		Dense(std::size_t input_size, std::size_t output_size, std::shared_ptr<Layer> activation_func = nullptr, std::shared_ptr<RandInitializer> rand_init = nullptr);
		Dense(const Shape &input_shape, std::shared_ptr<Layer> activation_func = nullptr, std::shared_ptr<RandInitializer> rand_init = nullptr);
		Dense(std::size_t input_size, std::shared_ptr<Layer> activation_func = nullptr, std::shared_ptr<RandInitializer> rand_init = nullptr);
		
		~Dense(void) override = default;
		
		Mat<T> &get_weights(void) const;
		Mat<T> &get_bias(void) const;
		bool has_activation_func(void) const;
		Layer get_activation_func(void) const;
		
		Dense &build(const Shape &input_shape, const Shape &output_shape) override;
		Dense &build(std::size_t input_size, std::size_t output_size) override;
		Dense &build(void) override;
	private:
		Dense &register_funcs(void) override;
		
		std::shared_ptr<Layer> activation_func_;
		std::shared_ptr<RandInitializer> rand_init_;
		std::unique_ptr<Mat<T>> weights_;
		std::unique_ptr<Mat<T>> bias_;
	};
}

#endif

