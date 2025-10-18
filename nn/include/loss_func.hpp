#ifndef NN_LOSS_FUNC_INCLUDED
#define NN_LOSS_FUNC_INCLUDED

#include <vector>
#include <memory>
#include <string>

#include "model.hpp"
#include "mat.hpp"

namespace nn::loss_funcs {
	using namespace mathops;
	using namespace models;

	template <typename T>
	class Loss {
	public:
		// Constructors
		Loss(std::shared_ptr<std::vector<Mat<T>>> inputs = nullptr,
		     std::shared_ptr<std::vector<Mat<T>>> outputs = nullptr,
		     std::string name = "Loss");

		virtual ~Loss(void) = 0;

		// Setters
		Loss &set_name(std::string name);
		Loss &set_model(Model &model);
		Loss &set_inputs(std::shared_ptr<std::vector<Mat<T>>> inputs);
		Loss &set_outputs(std::shared_ptr<std::vector<Mat<T>>> outputs);

		// Getters
		std::shared_ptr<std::vector<Mat<T>>> get_inputs(void) const;
		std::shared_ptr<std::vector<Mat<T>>> get_outputs(void) const;
		const std::vector<Mat<T>> &get_predictions(void) const;
		const std::string &get_name(void) const;

		// Input/Output shapes and sizes
		const Shape &get_input_shape(void) const;  
		const Shape &get_output_shape(void) const;

		// Get last computed loss
		const Mat<T> &get_last_loss(void) const;

		// Get normalized version of the last loss
		T get_normalized_loss(void) const;

		// Evaluate the loss
		virtual Mat<T> operator()(void) = 0;
		virtual Mat<T> operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch) = 0;
		virtual Mat<T> operator()(const std::pair<Mat<T>, Mat<T>> &example) = 0;

		// Compute gradient & Jacobian
		virtual Mat<T> gradient(const std::pair<Mat<T>, Mat<T>> &example) = 0;
		virtual Mat<T> jacobian(const std::pair<Mat<T>, Mat<T>> &example) = 0;

	protected:
		std::shared_ptr<std::vector<Mat<T>>> inputs_;
		std::shared_ptr<std::vector<Mat<T>>> outputs_;
		std::vector<Mat<T>> predictions_;
		std::shared_ptr<Model> model_;
		std::string name_;

		Shape input_shape_;
		Shape output_shape_;
		Mat<T> last_loss_;
	};

	template <typename T>
	class MeanAbsoluteError : public Loss<T> {
	public:
		using Loss<T>::Loss;

		MeanAbsoluteError(std::shared_ptr<std::vector<Mat<T>>> inputs = nullptr,
				  std::shared_ptr<std::vector<Mat<T>>> outputs = nullptr);

		~MeanAbsoluteError(void) override = default;

		Mat<T> operator()(void) override;
		Mat<T> operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch) override;
		Mat<T> operator()(const std::pair<Mat<T>, Mat<T>> &example) override;

		Mat<T> gradient(const std::pair<Mat<T>, Mat<T>> &example) override;
		Mat<T> jacobian(const std::pair<Mat<T>, Mat<T>> &example) override;
	};


	template <typename T>
	class CrossEntropy : public Loss<T> {
	public:
		using Loss<T>::Loss;
		
		CrossEntropy(std::shared_ptr<std::vector<Mat<T>>> inputs = nullptr,
			     std::shared_ptr<std::vector<Mat<T>>> outputs = nullptr);
		~CrossEntropy(void) override = default;

		Mat<T> operator()(void) override;
		Mat<T> operator()(const std::vector<std::pair<Mat<T>, Mat<T>>> &batch) override;
		Mat<T> operator()(const std::pair<Mat<T>, Mat<T>> &example) override;

		Mat<T> gradient(const std::pair<Mat<T>, Mat<T>> &example) override;
		Mat<T> jacobian(const std::pair<Mat<T>, Mat<T>> &example) override;
	};
}

#endif
