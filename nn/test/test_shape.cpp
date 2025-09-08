#include <gtest/gtest.h>
#include "../include/mat.hpp"  // replace with your actual header path

using namespace nn::mathops;


// Test initializer_list constructor
TEST(ShapeTest, InitializerListConstructor) {
	Shape s({3, 4});
	EXPECT_EQ(s.rows, 3);
	EXPECT_EQ(s.cols, 4);
}

// Test (rows, cols) constructor
TEST(ShapeTest, RowsColsConstructor) {
	Shape s(5, 6);
	EXPECT_EQ(s.rows, 5);
	EXPECT_EQ(s.cols, 6);
}

// Test Shape constructor
TEST(ShapeTest, ShapeConstructor) {
	Shape s1(5, 6);
	Shape s2(s1);
	EXPECT_EQ(s2.rows, s1.rows);
	EXPECT_EQ(s2.cols, s2.cols);
}

// Test equality operator
TEST(ShapeTest, EqualityOperator) {
	Shape s1(2, 3);
	Shape s2({2, 3});
	Shape s3(3, 2);

	EXPECT_TRUE(s1 == s2);
	EXPECT_FALSE(s1 == s3);
}

// Test ostream output (optional)
TEST(ShapeTest, OstreamOutput) {
	Shape s(2, 3);
	std::ostringstream oss;
	oss << s;
	EXPECT_EQ(oss.str(), "shape=(rows=2, cols=3)");
}

TEST(ShapeTest, NotEqualDifferentRows) {
	Shape s1(2, 3);
	Shape s2(4, 3);
	EXPECT_TRUE(s1 != s2);
}

TEST(ShapeTest, NotEqualDifferentCols) {
	Shape s1(2, 3);
	Shape s2(2, 5);
	EXPECT_TRUE(s1 != s2);
}

TEST(ShapeTest, EqualShapes) {
	Shape s1(2, 3);
	Shape s2(2, 3);
	EXPECT_FALSE(s1 != s2);  // they are equal
}
