
#include <gtest/gtest.h>
#include "../include/utils.hpp"   // Your header file

using namespace nn::utils;

// A dummy class inheriting GenericVTable for testing
class DummyLayer : public GenericVTable {
public:
	using GenericVTable::GenericVTable;


	DummyLayer(void) {
		register_funcs();
	}
	
	~DummyLayer(void) override = default;
	
	DummyLayer &register_funcs(void) override
	{
		// Register a simple function: int f(int)
		register_func<int, int>("double_plus_one", [this](int x) { return x * 2 + 1; });

		// Register another: std::string f(const std::string&)
		register_func<std::string, std::string>("echo", [this](const std::string& s) { return "echo: " + s; });

		// Register a function with multiple args
		register_func<int, int, int>("sum", [this](int a, int b) { return a + b; });
		
		return *this;
	}
};

// ------------------- TESTS ------------------- //

TEST(GenericVTableTest, RegisteredFunctionWorks) {
	DummyLayer layer;
	auto f = layer.get_func<int, int>("double_plus_one");
	EXPECT_EQ(f(3), 7);   // (3 * 2 + 1) = 7
}

TEST(GenericVTableTest, WorksWithStrings) {
	DummyLayer layer;
	auto f = layer.get_func<std::string, std::string>("echo");
	EXPECT_EQ(f("hello"), "echo: hello");
}

TEST(GenericVTableTest, WorksWithMultipleArgs) {
	DummyLayer layer;
	auto f = layer.get_func<int, int, int>("sum");
	EXPECT_EQ(f(2, 5), 7);
}

TEST(GenericVTableTest, ThrowsOnUnregisteredFunction) {
	DummyLayer layer;
	EXPECT_THROW(
		     layer.get_func<void>("non_existent_func", __FILE__, __LINE__),
		     std::runtime_error
		     );
}

TEST(GenericVTableTest, DifferentSignaturesAreDistinct) {
	DummyLayer layer;
	// Register two funcs with same name but different signatures
	layer.register_func<int, int>("same_name", [](int x) { return x + 1; });
	layer.register_func<double, double>("same_name", [](double x) { return x * 0.5; });

	auto f1 = layer.get_func<int, int>("same_name");
	auto f2 = layer.get_func<double, double>("same_name");

	EXPECT_EQ(f1(4), 5);
	EXPECT_DOUBLE_EQ(f2(4.0), 2.0);
}




