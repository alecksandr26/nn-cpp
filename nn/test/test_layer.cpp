#include <gtest/gtest.h>

#include "../include/layer.hpp"
#include "../include/optimizer.hpp"

using namespace nn::layers;
using namespace nn::optimizers;

// ---- Mock Optimizer ----

template <typename T>
class MockOptimizer : public Optimizer {
public:
	MockOptimizer(double learning_rate = 0.01)
		: Optimizer("MockOptimizer", learning_rate) {}

	~MockOptimizer(void) override = default;


	MockOptimizer &register_funcs(void) override
	{
		register_func<void, Mat<T> &, const Mat<T> &, const Mat<T> &>
			("update", [this](Mat<T> &weights, const Mat<T> &signal_update, const Mat<T> &input) -> void {
				((void) weights);
				((void) signal_update);
				((void) input);
			});
		
		return *this;
	}
};


// ---- Trivial concrete subclass of Layer for testing ----
class FooLayer : public Layer {
public:
	using Layer::Layer;
	
	~FooLayer(void) override {};

	FooLayer &build(const Shape &input_shape, const Shape &output_shape) override
	{
		this->set_input_shape(input_shape);
		this->set_output_shape(output_shape);
		
		return *this;
	}

	FooLayer &build(std::size_t input_size, std::size_t output_size) override
	{
		this->set_input_shape(Shape{input_size, 1});
		this->set_output_shape(Shape{output_size, 1});
		
		return *this;
	}

	FooLayer &build(void) override
	{
		return *this;
	}
	
	FooLayer &register_funcs(void) override
	{
		// Register trivial functions
		register_func<Mat<float>, const Mat<float>&>
			("feedforward", [](const Mat<float>& X) {
				return X;
			});
		register_func<Mat<float>, const Mat<float>&>
			("derivate", [](const Mat<float>& X) {
				return X;
			});

		return *this;
	}
};

// ======================================================
//                  Layer Tests
// ======================================================

TEST(LayerTest, TrainableFlag) {
	FooLayer t_layer(3, 2, true);
	EXPECT_TRUE(t_layer.is_trainable());

	FooLayer f_layer(3, 2, false);
	EXPECT_FALSE(f_layer.is_trainable());
}

TEST(LayerTest, BuiltFlagDefault) {
	FooLayer layer;
	EXPECT_FALSE(layer.is_built());
}

TEST(LayerTest, ShapeSetGet) {
	FooLayer layer;
	Shape input_shape{2, 3};
	Shape output_shape{4, 5};

	layer.set_input_shape(input_shape);
	layer.set_output_shape(output_shape);
	
	EXPECT_EQ(layer.get_input_shape(), input_shape);
	EXPECT_EQ(layer.get_output_shape(), output_shape);
}

TEST(LayerTest, SizeSetGet) {
	FooLayer layer;
	layer.set_input_size(7);
	layer.set_output_size(9);

	EXPECT_EQ(layer.get_input_size(), 7);
	EXPECT_EQ(layer.get_output_size(), 9);
}

TEST(LayerTest, BuildSetsInputShape) {
	FooLayer layer;
	Shape new_shape{10, 20};
	layer.build(new_shape, new_shape);
	EXPECT_EQ(layer.get_input_shape(), new_shape);
}

TEST(LayerTest, ConstructorsWithName) {
	FooLayer l1;
	l1.set_name("hidden1");
	EXPECT_EQ(l1.get_name(), "hidden1");

	FooLayer l2(3, 2, true, "output");
	EXPECT_EQ(l2.get_name(), "output");
}

TEST(LayerTest, SetGetName) {
	FooLayer layer;
	EXPECT_EQ(layer.get_name(), "Layer");

	layer.set_name("hidden2");
	EXPECT_EQ(layer.get_name(), "hidden2");

	std::string new_name = "hidden3";
	layer.set_name(new_name);
	EXPECT_EQ(layer.get_name(), "hidden3");
}

TEST(LayerTest, ChainingSetName) {
	FooLayer layer;
	layer.set_name("first")
		.set_input_size(5)
		.set_output_size(10);
	
	EXPECT_EQ(layer.get_name(), "first");
	EXPECT_EQ(layer.get_input_size(), 5);
	EXPECT_EQ(layer.get_output_size(), 10);
}

// ======================================================
//               WeightedLayer Tests
// ======================================================

class DummyWeighted : public WeightedLayer {
public:
	using WeightedLayer::WeightedLayer;
	
	~DummyWeighted() override = default;

	DummyWeighted &build(const Shape &input_shape, const Shape &output_shape) override
	{
		set_input_shape(input_shape);
		set_output_shape(output_shape);

		register_funcs();
		
		return *this;
	}

	DummyWeighted &build(std::size_t input_size, std::size_t output_size) override
	{
		set_input_shape(Shape{input_size, 1});
		set_output_shape(Shape{output_size, 1});

		register_funcs();
		
		return *this;
	}

	DummyWeighted &build(void) override
	{
		register_funcs();
		
		return *this;
	}
	
	
	DummyWeighted &register_funcs(void) override
	{
		// register dummy fit function
		register_func<void, const Mat<float>&, const Mat<float>&>
			("fit", [](const Mat<float>&, const Mat<float>&) {
				
			});

		return *this;
	}
};

TEST(WeightedLayerTest, OptimizerSetGet) {
	DummyWeighted wl(2, 2);
	auto opt = std::make_shared<MockOptimizer<float>>(0.1);
	wl.set_optimizer(opt);
	EXPECT_EQ(wl.get_optimizer(), opt);
}

TEST(WeightedLayerTest, FitFunctionCalls) {
	DummyWeighted wl(2, 2);
	wl.build();
	Mat<float> X(2,2), Y(2,2);
	EXPECT_NO_THROW(wl.fit(X, Y)); // calls registered fit
}

// ======================================================
//                Dense Layer Tests
// ======================================================

TEST(DenseLayerTest, Construction) {
	Dense<float> d(3, 2);
	EXPECT_EQ(d.get_input_size(), 3);
	EXPECT_EQ(d.get_output_size(), 2);
}

TEST(DenseLayerTest, FeedForwardAndDerivate) {
	Dense<float> d(2, 2);
	Mat<float> X = {
		{1.0f},
		{1.0f}
	};
	
	d.build();
	
	auto out = d(X);           // operator() → feedforward
	auto grad = d.gradient(X);
	
	EXPECT_EQ(out.rows(), 2);
	EXPECT_EQ(out.cols(), 1);
	EXPECT_EQ(grad.rows(), 2);
	EXPECT_EQ(grad.cols(), 1);
}
