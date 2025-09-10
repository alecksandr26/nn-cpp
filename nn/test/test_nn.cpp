#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <vector>
#include "../include/layer.hpp"
#include "../include/activation_func.hpp"
#include "../include/optimizer.hpp"

using namespace nn::layers;
using namespace nn::activation_funcs;
using namespace nn::optimizers;

TEST(NNTest, Perceptron) {
	Dense<float> dense(2, 1);
	StepFunc<float> step;
	PerceptronOptimizer<float> opt(0.001f);


	
	dense.build(Shape{10, 1}, Shape{1, 1});
	dense.set_optimizer(opt);
	
	

	std::vector<std::pair<Mat<float>, Mat<float>>> and_data = {
		{
			{{0.0f}, {0.0f}}, {{0.0f}}
		},
		{
			{{0.0f}, {1.0f}}, {{0.0f}}
		},
		{
			{{1.0f}, {0.0f}}, {{0.0f}}
		},
		{
			{{1.0f}, {1.0f}}, {{1.0f}}
		},
	};

	GTEST_LOG_(INFO) << "Hello";

	
}



