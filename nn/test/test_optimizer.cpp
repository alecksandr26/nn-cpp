#include "../include/optimizer.hpp"
#include <gtest/gtest.h>
#include <string>

using namespace nn::mathops;
using namespace nn::optimizers;

// Dummy concrete optimizer for testing
template <typename T>
class FooOptimizer : public Optimizer {
public:
	// Forward to base constructors
	FooOptimizer(std::string name, double lr)
		: nn::optimizers::Optimizer(std::move(name), lr)
	{
		register_funcs();
	}

	FooOptimizer(double lr)
		: nn::optimizers::Optimizer(lr)
	{
		register_funcs();
	}

	FooOptimizer &register_funcs(void) override
	{
		register_func<void, Mat<T> &, const Mat<T> &, const Mat<T> &>
			("update", [this](Mat<T> &, const Mat<T> &, const Mat<T> &) -> void {
				
			});
		return *this;
	}
};

TEST(OptimizerTest, ConstructorWithName) {
	FooOptimizer<float> opt("TestOpt", 0.01);
	
	EXPECT_EQ(opt.get_name(), "TestOpt");
	EXPECT_DOUBLE_EQ(opt.get_learning_rate(), 0.01);
}

TEST(OptimizerTest, ConstructorWithoutName) {
	FooOptimizer<float> opt(0.1);

	// Default name is empty string
	EXPECT_EQ(opt.get_name(), "Optimizer");
	EXPECT_DOUBLE_EQ(opt.get_learning_rate(), 0.1);
}

TEST(OptimizerTest, SettersWork) {
	FooOptimizer<float> opt("Init", 0.5);

	opt.set_name("UpdatedName").set_learning_rate(0.9);
	EXPECT_EQ(opt.get_name(), "UpdatedName");
	EXPECT_DOUBLE_EQ(opt.get_learning_rate(), 0.9);
}

TEST(OptimizerTest, CanCallUpdate) {
	FooOptimizer<float> opt("Test", 0.1);
	Mat<float> weights, update_signal, X;

	// Just check it compiles and runs
	opt.update(weights, update_signal, X);
}


TEST(PerceptronOptimizerTest, UpdateWeights) {
	// mat, supposing two neurons as output
	Mat<float> weights = {{0.0f, 0.0f},
			      {0.0f, 0.0f}}; 
	Mat<float> input   = {{1.0f},
			      {2.0f}};   // column vector
	Mat<float> error   = {{0.5f},
			      {-0.5f}}; // column vector
	
	PerceptronOptimizer<float> opt(0.1f);
	opt.update(weights, error, input);
	
	EXPECT_FLOAT_EQ(weights(0, 0), 0.05f);
	EXPECT_FLOAT_EQ(weights(0, 1), 0.1f);
	EXPECT_FLOAT_EQ(weights(1, 0), -0.05f);
	EXPECT_FLOAT_EQ(weights(1, 1), -0.1f);
}
