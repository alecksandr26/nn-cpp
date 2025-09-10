#ifndef NN_OPTIMIZER_INCLUDED
#define NN_OPTIMIZER_INCLUDED

#include "mat.hpp"

namespace nn::optimizers {
	using namespace mathops;
	
	template <typename T>
	class Optimizer {
	public:
		Optimizer(std::string name, double learning_rate);
		Optimizer(double learning_rate);
		Optimizer(std::string name, float learning_rate);
		Optimizer(float learning_rate);
		virtual ~Optimizer(void) = 0;
		
		Optimizer &set_name(std::string name);
		Optimizer &set_learning_rate(double learning_rate);
		const std::string &get_name(void) const;
		double get_learning_rate(void) const;

		/**
		 * Update the weights of the layer.
		 * 
		 * @param weights        The current weight matrix of the layer.
		 * @param signal_update  The "update signal" for the weights. 
		 *                       This can be:
		 *                         - The gradient of the loss with respect to the layer's output (from backpropagation),
		 *                         - The derivative of the output of the current layer with respect to its inputs,
		 *                         - Or a simple error/correction signal like (y_true - y_pred) in a perceptron.
		 * @param input          The input that produced the current output (or the relevant activations from the previous layer),
		 *                       used to compute weight updates in most learning rules.
		 */
		virtual void update(Mat<T> &weights, const Mat<T> &signal_update, const Mat<T> &input) = 0;
		// virtual Mat<T> get_initial_signal_update(void);
	protected:
		std::string name_;
		double learning_rate_;
	};

	
	template <typename T>
	class PerceptronOptimizer : public Optimizer<T> {
	public:
		PerceptronOptimizer(double learning_rate);
		PerceptronOptimizer(float learning_rate);
		~PerceptronOptimizer(void) = default;
		
	public:
		void update(Mat<T> &weights, const Mat<T> &error, const Mat<T> &input) override;
	};
}

#endif



