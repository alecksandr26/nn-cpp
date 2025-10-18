#include <gtest/gtest.h>
#include "../include/activation_func.hpp"

using namespace nn::activation_funcs;

class SigmoidTest : public ::testing::Test {
protected:
	void SetUp() override {
		sigmoid.build();
	}

	SigmoidFunc<float> sigmoid;
};

// Test para la función feedforward (evaluación directa)
TEST_F(SigmoidTest, Feedforward) {
	// Test con valores positivos
	Mat<float> X_pos(2, 1);
	X_pos(0, 0) = 1.0f;
	X_pos(1, 0) = 2.0f;
    
	Mat<float> result_pos = sigmoid(X_pos);
    
	EXPECT_NEAR(result_pos(0, 0), 0.7310585786f, 1e-6f);
	EXPECT_NEAR(result_pos(1, 0), 0.8807970780f, 1e-6f);
    
	// Test con valores negativos
	Mat<float> X_neg(2, 1);
	X_neg(0, 0) = -1.0f;
	X_neg(1, 0) = -2.0f;
    
	Mat<float> result_neg = sigmoid(X_neg);
    
	EXPECT_NEAR(result_neg(0, 0), 0.2689414214f, 1e-6f);
	EXPECT_NEAR(result_neg(1, 0), 0.1192029220f, 1e-6f);
    
	// Test con cero
	Mat<float> X_zero(1, 1);
	X_zero(0, 0) = 0.0f;
    
	Mat<float> result_zero = sigmoid(X_zero);
	EXPECT_NEAR(result_zero(0, 0), 0.5f, 1e-6f);
}

// Test para el gradiente (derivada)
TEST_F(SigmoidTest, Gradient) {
	// Test con valores varios
	Mat<float> X(3, 1);
	X(0, 0) = 0.0f;
	X(1, 0) = 1.0f;
	X(2, 0) = -1.0f;
    
	Mat<float> grad = sigmoid.gradient(X);
    
	// Para x=0: sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
	EXPECT_NEAR(grad(0, 0), 0.25f, 1e-6f);
    
	// Para x=1: sigmoid(1) ≈ 0.731, sigmoid'(1) ≈ 0.731 * (1 - 0.731) ≈ 0.1966
	EXPECT_NEAR(grad(1, 0), 0.1966119332f, 1e-6f);
    
	// Para x=-1: sigmoid(-1) ≈ 0.269, sigmoid'(-1) ≈ 0.269 * (1 - 0.269) ≈ 0.1966
	EXPECT_NEAR(grad(2, 0), 0.1966119332f, 1e-6f);
}

// Test para el jacobiano
TEST_F(SigmoidTest, Jacobian) {
	// Test con vector de entrada
	Mat<float> X(2, 1);
	X(0, 0) = 1.0f;
	X(1, 0) = -1.0f;
    
	Mat<float> jac = sigmoid.jacobian(X);
    
	// El jacobiano debería ser una matriz diagonal 2x2
	EXPECT_EQ(jac.rows(), 2);
	EXPECT_EQ(jac.cols(), 2);
    
	// Elementos diagonales
	float expected_diag_0 = 0.1966119332f; // sigmoid'(1)
	float expected_diag_1 = 0.1966119332f; // sigmoid'(-1)
    
	EXPECT_NEAR(jac(0, 0), expected_diag_0, 1e-6f);
	EXPECT_NEAR(jac(1, 1), expected_diag_1, 1e-6f);
    
	// Elementos no diagonales deberían ser cero
	EXPECT_NEAR(jac(0, 1), 0.0f, 1e-6f);
	EXPECT_NEAR(jac(1, 0), 0.0f, 1e-6f);
}

// Test para valores extremos
TEST_F(SigmoidTest, ExtremeValues) {
	// Valores muy grandes positivos
	Mat<float> X_large_pos(2, 1);
	X_large_pos(0, 0) = 10.0f;
	X_large_pos(1, 0) = 100.0f;
    
	Mat<float> result_large = sigmoid(X_large_pos);
	EXPECT_NEAR(result_large(0, 0), 0.9999546021f, 1e-6f);
	EXPECT_NEAR(result_large(1, 0), 1.0f, 1e-6f);
    
	// Valores muy grandes negativos
	Mat<float> X_large_neg(2, 1);
	X_large_neg(0, 0) = -10.0f;
	X_large_neg(1, 0) = -100.0f;
    
	Mat<float> result_small = sigmoid(X_large_neg);
	EXPECT_NEAR(result_small(0, 0), 4.539786870e-5f, 1e-9f);
	EXPECT_NEAR(result_small(1, 0), 0.0f, 1e-9f);
}

// Test para matrices con múltiples filas y columnas
TEST_F(SigmoidTest, MultiDimensional) {
	Mat<float> X(2, 3);
	X(0, 0) = 0.0f; X(0, 1) = 1.0f; X(0, 2) = -1.0f;
	X(1, 0) = 2.0f; X(1, 1) = -2.0f; X(1, 2) = 0.5f;
    
	Mat<float> result = sigmoid(X);
    
	// Verificar que mantiene la forma
	EXPECT_EQ(result.rows(), 2);
	EXPECT_EQ(result.cols(), 3);
    
	// Verificar algunos valores
	EXPECT_NEAR(result(0, 0), 0.5f, 1e-6f);
	EXPECT_NEAR(result(0, 1), 0.7310585786f, 1e-6f);
	EXPECT_NEAR(result(0, 2), 0.2689414214f, 1e-6f);
	EXPECT_NEAR(result(1, 0), 0.8807970780f, 1e-6f);
	EXPECT_NEAR(result(1, 1), 0.1192029220f, 1e-6f);
	EXPECT_NEAR(result(1, 2), 0.6224593312f, 1e-6f);
}

// Test de propiedades matemáticas
TEST_F(SigmoidTest, MathematicalProperties) {
	// Propiedad: sigmoid(x) + sigmoid(-x) = 1
	Mat<float> X(3, 1);
	X(0, 0) = 1.0f;
	X(1, 0) = 2.0f;
	X(2, 0) = 3.0f;
    
	Mat<float> X_neg(3, 1);
	X_neg(0, 0) = -1.0f;
	X_neg(1, 0) = -2.0f;
	X_neg(2, 0) = -3.0f;
    
	Mat<float> result = sigmoid(X);
	Mat<float> result_neg = sigmoid(X_neg);
    
	for (std::size_t i = 0; i < result.rows(); ++i) {
		EXPECT_NEAR(result(i, 0) + result_neg(i, 0), 1.0f, 1e-6f);
	}
}

// Test para verificar que se puede construir correctamente
TEST_F(SigmoidTest, Construction) {
	EXPECT_FALSE(sigmoid.is_built());
	EXPECT_FALSE(sigmoid.is_trainable());
	EXPECT_EQ(sigmoid.get_name(), "SigmoidFunc");
}
