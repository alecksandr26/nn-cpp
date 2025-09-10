#ifndef NN_ACTIVATION_INCLUDED
#define NN_ACTIVATION_INCLUDED

#include "layer.hpp"



namespace nn::activation_funcs {
	using namespace layers;

	template <typename T>
	class ActivationFunc : public Layer<T> {
	public:
		ActivationFunc(std::string name);
	protected:
	};

	// TODO: Write a few unit tests for this activation functions
	template <typename T>
	class StepFunc : public ActivationFunc<T> {
	public:
		StepFunc(void);
		~StepFunc(void) override = default;
		
		Mat<T> operator()(const Mat<T> &X) override;
		Mat<T> derivate(const Mat<T> &X) override;
		StepFunc &build(const Shape &input_shape, const Shape &output_shape) override;
	protected:
	};
}


#endif



