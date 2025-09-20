#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "../include/nn.hpp"

using namespace nn::models;


TEST(NNTest, PerceptronAndGate) {
	// And - Data
	std::vector<Mat<float>> X_data = {
		{
			{0.0f},
			{0.0f}
		},
		{
			{0.0f},
			{1.0f}
		},
		{
			{1.0f},
			{0.0f}
		},
		{
			{1.0f},
			{1.0f}
		},
	};

	std::vector<Mat<float>> Y_data = {
		{{0.0f}},
		{{0.0f}},
		{{0.0f}},
		{{1.0f}},
	};

	
	Perceptron<float> model(2, 1, std::make_shared<RandUniformInitializer<float>>());
	
	model.set_optimizer(std::make_shared<PerceptronOptimizer<float>>(0.1f));
	model.set_loss(std::make_shared<MeanAbsoluteError<float>>());
	model.build();

	auto X_ptr = std::make_shared<std::vector<Mat<float>>>(X_data);
	auto Y_ptr = std::make_shared<std::vector<Mat<float>>>(Y_data);
	
	model.test(X_ptr, Y_ptr);
	GTEST_LOG_(INFO) << "Before Training MAE: " << model.get_loss()->get_last_loss()(0, 0);
	GTEST_LOG_(INFO) << "Before Training MAE normalized: " << model.get_loss()->get_normalized_loss();
	GTEST_LOG_(INFO) << "Before Training Weights: ";
	std::cout << model.get_weights() << std::endl;
	std::cout << model.get_bias() << std::endl;
	
	model.fit(X_ptr, Y_ptr, 200);
	model.test(X_ptr, Y_ptr);
	
	GTEST_LOG_(INFO) << "Before Training MAE: " << model.get_loss()->get_last_loss()(0, 0);
	GTEST_LOG_(INFO) << "After Training MAE normalized: " << model.get_loss()->get_normalized_loss();
	GTEST_LOG_(INFO) << "After Training Weights: ";
	std::cout << model.get_weights() << std::endl;
	std::cout << model.get_bias() << std::endl;
	
	EXPECT_NEAR(model.get_loss()->get_last_loss()(0, 0), 0.0f, 0.1f);
}
