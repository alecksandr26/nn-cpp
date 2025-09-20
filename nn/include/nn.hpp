#ifndef NN_INCLUDED
#define NN_INCLUDED

#include "mat.hpp"
#include "layer.hpp"
#include "loss_func.hpp"
#include <memory>


// TODO: Build a module to get graphs over the layer and mat objects to create computable grpahs

namespace nn::models {
	using namespace layers;
	using namespace optimizers;
	using namespace loss_funcs;

	template <typename T>
	class WeightedModel : public WeightedLayer {
	public:
		using WeightedLayer::WeightedLayer;
		virtual ~WeightedModel(void) = 0;

		// TODO: Add the settters and getters and also lets add the X_train_ shared pointers
		virtual WeightedModel &fit(const std::shared_ptr<std::vector<Mat<T>>> X_train, const std::shared_ptr<std::vector<Mat<T>>> Y_train, std::size_t nepochs = 100, std::size_t batch_size = 1) = 0;
		
		// TODO: Add matrics object
		Mat<T> test(const std::shared_ptr<std::vector<Mat<T>>> X_test, const std::shared_ptr<std::vector<Mat<T>>> Y_test);

		WeightedModel &set_loss(std::shared_ptr<Loss<T>> loss);
		const std::shared_ptr<Loss<T>> get_loss(void) const;

		
	protected:
		std::shared_ptr<Loss<T>> loss_;
		std::size_t nepochs_;
		std::size_t batch_size_;
	};
	
	template <typename T>
	class Perceptron : public WeightedModel<T> {
	public:
		using WeightedModel<T>::WeightedModel;
		
		// In case of having multiple nested percetrons
		Perceptron(const Shape &input_shape, const Shape &output_shape, std::shared_ptr<RandInitializer> rand_init = nullptr);
		Perceptron(const Shape &input_shape, std::shared_ptr<RandInitializer> rand_init = nullptr);
		Perceptron(std::size_t input_size, std::size_t output_size, std::shared_ptr<RandInitializer> rand_init = nullptr);
		Perceptron(std::size_t input_size, std::shared_ptr<RandInitializer> rand_init = nullptr);
		~Perceptron(void) = default;

		Mat<T> &get_weights(void) const;

		Mat<T> &get_bias(void) const;
		
		Perceptron &build(const Shape &input_shape, const Shape &output_shape) override;
		Perceptron &build(std::size_t input_size, std::size_t output_size) override;
		Perceptron &build(void) override;
		Perceptron &fit(const std::shared_ptr<std::vector<Mat<T>>> X_train, const std::shared_ptr<std::vector<Mat<T>>> Y_train, std::size_t nepochs = 100, std::size_t batch_size = 1) override;
	private:
		Perceptron &register_funcs(void) override;
		
		std::unique_ptr<Dense<T>> dense_;
	};
}


#endif
