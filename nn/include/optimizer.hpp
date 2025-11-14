#ifndef NN_OPTIMIZER_INCLUDED
#define NN_OPTIMIZER_INCLUDED

#include "mat.hpp"
#include "utils.hpp"

namespace nn::optimizers {
	using namespace mathops;
	using namespace utils;

	class Optimizer : public GenericVTable {
	public:
		using GenericVTable::GenericVTable;
		
		Optimizer(std::string name, double learning_rate);
		Optimizer(double learning_rate);
		Optimizer(float learning_rate);
		Optimizer(std::string name, float learning_rate);
		virtual ~Optimizer(void) = 0;
		
		Optimizer &set_name(std::string name);
		Optimizer &set_learning_rate(double learning_rate);
		const std::string &get_name(void) const;
		double get_learning_rate(void) const;

				/**
		 * @brief Update the weights of the layer.
		 *
		 * Performs a parameter update on the layer’s weight matrix
		 * using the provided gradient signal and input activations.
		 *
		 * @tparam T  Numeric type of the matrix (e.g., float or double)
		 * @param weights        The weight matrix to be updated.
		 * @param signal_update  The gradient signal (e.g., dL/dZ).
		 * @param input          The input activations (from the previous layer).
		 */
		template <typename T>
		void update(Mat<T> &weights, const Mat<T> &signal_update, const Mat<T> &input)
		{
			get_func<void, Mat<T> &, const Mat<T> &, const Mat<T> &>
				("update", __FILE__, __LINE__)(weights, signal_update, input);
		}

		/**
		 * @brief Update the bias vector of the layer.
		 *
		 * This overload is used for biases, where the update does not depend
		 * on the input activations. The update rule typically follows:
		 *     b ← b - η * dL/db
		 *
		 * @tparam T  Numeric type of the matrix (e.g., float or double)
		 * @param bias           The bias vector to be updated.
		 * @param signal_update  The gradient signal (e.g., dL/dZ).
		 */
		template <typename T>
		void update(Mat<T> &bias, const Mat<T> &signal_update)
		{
			get_func<void, Mat<T> &, const Mat<T> &>
				("update_bias", __FILE__, __LINE__)(bias, signal_update);
		}
	protected:
		std::string name_;
		double learning_rate_;
	};

	template <typename T>
	class PerceptronOptimizer : public Optimizer {
	public:
		using Optimizer::Optimizer;
		
		PerceptronOptimizer(T learning_rate = 0.01);
		~PerceptronOptimizer(void) override = default;

	private:
		PerceptronOptimizer &register_funcs(void) override;
	};


	// TODO: Batch Gradient descent we need a function called
	// fit probably not sure ?
	template <typename T>
	class GradientDescentOptimizer : public Optimizer {
	public:
		using Optimizer::Optimizer;

		GradientDescentOptimizer(T learning_rate = 0.01);
		~GradientDescentOptimizer(void) override = default;
	private:
		GradientDescentOptimizer &register_funcs(void) override;
	};
}

#endif



