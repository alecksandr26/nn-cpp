#ifndef NN_ACTIVATION_INCLUDED
#define NN_ACTIVATION_INCLUDED

#include "layer.hpp"

namespace nn::activation_funcs {
	using namespace layers;

	class ActivationFunc : public Layer {
	public:
		using Layer::Layer;
		
		ActivationFunc(std::string name = "ActivationFunc");
		virtual ~ActivationFunc(void) = 0;
	};

	// TODO: Write a few unit tests for these activation functions

	// ======================
	// Step Function
	// ======================
	template <typename T>
	class StepFunc : public ActivationFunc {
	public:
		using ActivationFunc::ActivationFunc;
		
		StepFunc(void);
		~StepFunc(void) override = default;
		
		StepFunc &build(const Shape &input_shape, const Shape &output_shape) override;
		StepFunc &build(std::size_t input_size, std::size_t output_size) override;
		StepFunc &build(void) override;

	private:
		StepFunc &register_funcs(void) override;
	};


	// ======================
	// Sigmoid Function
	// ======================
	template <typename T>
	class SigmoidFunc : public ActivationFunc {
	public:
		using ActivationFunc::ActivationFunc;

		SigmoidFunc(void);
		~SigmoidFunc(void) override = default;

		SigmoidFunc &build(const Shape &input_shape, const Shape &output_shape) override;
		SigmoidFunc &build(std::size_t input_size, std::size_t output_size) override;
		SigmoidFunc &build(void) override;
		
	private:
		SigmoidFunc &register_funcs(void) override;
	};


	// ======================
	// Hyperbolic Tangent Function
	// ======================
	template <typename T>
	class TanhFunc : public ActivationFunc {
	public:
		using ActivationFunc::ActivationFunc;

		TanhFunc(void);
		~TanhFunc(void) override = default;

		TanhFunc &build(const Shape &input_shape, const Shape &output_shape) override;
		TanhFunc &build(std::size_t input_size, std::size_t output_size) override;
		TanhFunc &build(void) override;

	private:
		TanhFunc &register_funcs(void) override;
	};


	// ======================
	// ReLU (Rectified Linear Unit)
	// ======================
	template <typename T>
	class ReluFunc : public ActivationFunc {
	public:
		using ActivationFunc::ActivationFunc;

		ReluFunc(void);
		~ReluFunc(void) override = default;

		ReluFunc &build(const Shape &input_shape, const Shape &output_shape) override;
		ReluFunc &build(std::size_t input_size, std::size_t output_size) override;
		ReluFunc &build(void) override;

	private:
		ReluFunc &register_funcs(void) override;
	};

}

#endif

