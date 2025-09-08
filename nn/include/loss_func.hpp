#ifndef NN_LOSS_FUNC_INCLUDED
#define NN_LOSS_FUNC_INCLUDED

#include <vector>
#include "model.hpp"
#include "mat.hpp"


namespace nn::loss_funcs {
	using namespace mathops;
	using namespace models;
	
	template <typename T>
	class Loss {
	public:
		// Constructors
		Loss(const std::vector<Mat<T>> &inputs,
		     const std::vector<Mat<T>> &outputs);
		Loss(const std::vector<Mat<T>> &inputs,
		     const std::vector<Mat<T>> &outputs,
		     std::string name);

		virtual ~Loss(void) = 0;

		// Setters
		Loss &set_name(std::string name);
		Loss &set_model(Model<T> &model);

		// Getters
		const std::vector<Mat<T>> &get_inputs(void) const;
		const std::vector<Mat<T>> &get_outputs(void) const;
		const std::vector<Mat<T>> &get_predictions(void) const; // Last predicted outputs
		const std::string &get_name(void) const;

		// Input/Output shapes and sizes
		const Shape &get_input_shape(void) const;  
		// Shape of the inputs to the model.
		// Needed to know the expected input dimensions for the model.

		const Shape &get_output_shape(void) const; 
		// Shape of the outputs from the model.
		// Needed to compute the derivative of the loss w.r.t. the model output.

		// Get last computed loss
		const Mat<T> &get_last_loss(void) const;

		// Get normalized version of the last loss
		T get_normalized_loss(void) const;

		// Evaluate the loss using currently set inputs/outputs
		virtual Mat<T> operator()(void) = 0;

		// Evaluate the loss on a batch of input-output pairs
		virtual Mat<T> operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch) = 0;
		//   ^-- pair.first  = input
		//   ^-- pair.second = expected output

		// Evaluate the loss on a single input-output pair
		virtual Mat<T> operator()(const std::pair<Mat<T>, Mat<T>> &example) = 0;
		//   ^-- example.first  = input
		//   ^-- example.second = expected output

		// Compute derivative of loss w.r.t. model output for a single example
		virtual Mat<T> derivate(const std::pair<Mat<T>, Mat<T>> &example) = 0;

	protected:
		const std::vector<Mat<T>> &inputs_;        // Stored input matrices
		const std::vector<Mat<T>> &outputs_;       // Stored expected outputs
		std::vector<Mat<T>> predictions_;   // Stores predicted outputs after evaluation
		Model<T> *model_;                   // Pointer to the model to run predictions
		std::string name_;                  // Name of the loss function

		// Shape/size info
		Shape input_shape_;   // Shape of inputs to the model
		Shape output_shape_;  // Shape of outputs from the model

		// Stores last computed loss for batch or single example
		Mat<T> last_loss_;
	};



	template <typename T>
	class MeanAbsoluteError : public Loss<T> {
	public:
		// Constructors forwarding to base class
		MeanAbsoluteError(const std::vector<Mat<T>> &inputs,
				  const std::vector<Mat<T>> &outputs);
		~MeanAbsoluteError(void) override = default;
		
		// Override operators for evaluation
		Mat<T> operator()(void) override;
		Mat<T> operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch) override;
		Mat<T> operator()(const std::pair<Mat<T>, Mat<T>> &example) override;
		
		// Override derivative computation
		Mat<T> derivate(const std::pair<Mat<T>, Mat<T>> &example) override;
	};
}

#endif


