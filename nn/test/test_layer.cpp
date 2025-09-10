#include <gtest/gtest.h>
#include "../include/layer.hpp"

using namespace nn::layers;

// Trivial concrete subclass to instantiate Layer
template <typename T>
class FooLayer : public Layer<T> {
public:
	FooLayer(bool trainable = true, std::size_t in_size = 1, std::size_t out_size = 1)
		: Layer<T>(trainable, in_size, out_size) {}

	FooLayer(bool trainable, std::size_t in_size, std::size_t out_size, std::string name)
		: Layer<T>(trainable, in_size, out_size, std::move(name)) {}
	FooLayer(bool trainable, Shape input_shape, Shape output_shape, std::string name)
		: Layer<T>(trainable, input_shape, output_shape, std::move(name)) {}
	~FooLayer(void) override = default;
	

	Mat<T> operator()(const Mat<T> &X) override {
		return X; // trivial
	}

	Mat<T> derivate(const Mat<T> &x) override {
		return x; // trivial
	}

	Layer<T>& build(const Shape &input_shape, const Shape &output_shape) override {
		this->set_input_shape(input_shape);
		this->set_output_shape(output_shape);
		return *this;
	}
};

// Test trainable flag
TEST(LayerTest, TrainableFlag) {
	FooLayer<float> t_layer(true, 3, 2);
	EXPECT_TRUE(t_layer.is_trainable());

	FooLayer<float> f_layer(false, 3, 2);
	EXPECT_FALSE(f_layer.is_trainable());
}

// Test built flag defaults to false
TEST(LayerTest, BuiltFlagDefault) {
	FooLayer<float> layer;
	EXPECT_FALSE(layer.is_built());
}

// Test input/output shape setters and getters
TEST(LayerTest, ShapeSetGet) {
	FooLayer<float> layer;
	Shape input_shape{2, 3};
	Shape output_shape{4, 5};

	layer.set_input_shape(input_shape);
	layer.set_output_shape(output_shape);

	EXPECT_EQ(layer.get_input_shape(), input_shape);
	EXPECT_EQ(layer.get_output_shape(), output_shape);
}

// Test input/output size setters and getters
TEST(LayerTest, SizeSetGet) {
	FooLayer<float> layer;
	layer.set_input_size(7);
	layer.set_output_size(9);

	EXPECT_EQ(layer.get_input_size(), 7);
	EXPECT_EQ(layer.get_output_size(), 9);
}

// Test build method sets input shape
TEST(LayerTest, BuildSetsInputShape) {
	FooLayer<float> layer;
	Shape new_shape{10, 20};
	layer.build(new_shape, new_shape);
	EXPECT_EQ(layer.get_input_shape(), new_shape);
}


// Test constructors with name
TEST(LayerTest, ConstructorsWithName) {
	FooLayer<float> l1(true, 3, 2, "hidden1");
	EXPECT_EQ(l1.get_name(), "hidden1");

	FooLayer<float> l2(true, Shape{2, 3}, Shape{4, 5}, "output");
	EXPECT_EQ(l2.get_name(), "output");
}

// Test setter and getter
TEST(LayerTest, SetGetName) {
	FooLayer<float> layer;
	EXPECT_EQ(layer.get_name(), "Layer"); 

	layer.set_name("hidden2");
	EXPECT_EQ(layer.get_name(), "hidden2");

	std::string new_name = "hidden3";
	layer.set_name(new_name);
	EXPECT_EQ(layer.get_name(), "hidden3");
}

// Test chaining with set_name
TEST(LayerTest, ChainingSetName) {
	FooLayer<float> layer;
	layer.set_name("first").set_input_size(5).set_output_size(10);
	EXPECT_EQ(layer.get_name(), "first");
	EXPECT_EQ(layer.get_input_size(), 5);
	EXPECT_EQ(layer.get_output_size(), 10);
}
