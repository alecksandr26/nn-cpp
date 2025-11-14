#include "trainer.hpp"
#include <cstddef>
#include <QDebug>
#include <QThread>
#include "activation_func.hpp"
#include "layer.hpp"

using namespace nn::models;
using namespace nn::layers;
using namespace nn::activation_funcs;
using namespace nn::mathops;

void Trainer::stop(void)
{
	qDebug() << "Trainer: stop() called";
	stopped_ = true;
}

Mat<float> Trainer::generateContourPredictions(void)
{
	// Crear matriz de predicciones (contour_resolution_ x contour_resolution_)
	Mat<float> predictions(contour_resolution_, contour_resolution_);
	
	// Rango: 0 a 10 para ambos ejes (X1 y X2)
	for (std::size_t i = 0; i < contour_resolution_; i++) {
		for (std::size_t j = 0; j < contour_resolution_; j++) {
			// Mapear índices a coordenadas [0, 10]
			float x1 = (static_cast<float>(j) / static_cast<float>(contour_resolution_ - 1)) * 10.0f;
			float x2 = (static_cast<float>(i) / static_cast<float>(contour_resolution_ - 1)) * 10.0f;
			
			// Crear punto de entrada
			Mat<float> input_point(2, 1);
			input_point(0, 0) = x1;
			input_point(1, 0) = x2;
			
			// Hacer predicción con el modelo
			Mat<float> pred = (*model_)(input_point);
			
			// Guardar predicción (asumiendo salida (1, 1))
			predictions(i, j) = pred(0, 0);
		}
	}
	
	return predictions;
}

float Trainer::calculateAccuracy(void)
{
	if (!X_ptr_ || !Y_ptr_ || X_ptr_->empty()) {
		return 0.0f;
	}
	
	std::size_t correct = 0;
	std::size_t total = X_ptr_->size();
	
	for (std::size_t i = 0; i < total; i++) {
		Mat<float> pred = (*model_)((*X_ptr_)[i]);
		
		// Clasificar: > 0.5 = clase 1, <= 0.5 = clase 0
		float predicted_class = (pred(0, 0) > 0.5f) ? 1.0f : 0.0f;
		float true_class = (*Y_ptr_)[i](0, 0);
		
		if (std::abs(predicted_class - true_class) < 0.1f) {
			correct++;
		}
	}
	
	return static_cast<float>(correct) / static_cast<float>(total);
}

void Trainer::train(void)
{
	stopped_ = false;
	emit startingTraining();
	
	// Preparar el modelo para entrenamiento
	model_->get_loss()->set_inputs(X_ptr_);
	model_->get_loss()->set_outputs(Y_ptr_);
	model_->get_loss()->set_model(model_);
	
	for (std::size_t e = 0; e < nepochs_; e++) {
		if (stopped_)
			break;
		
		// Entrenar una época completa
		for (std::size_t i = 0; i < X_ptr_->size(); i++) {
			if (stopped_)
				break;
			
			const Mat<float> &x = (*X_ptr_)[i];
			const Mat<float> &y = (*Y_ptr_)[i];
			
			// Calcular gradiente (esto ya lo hace tu Sequential internamente)
			Mat<float> grad = model_->get_loss()->gradient(std::make_pair(x, y));
			
			// Hacer backpropagation
			model_->WeightedLayer::fit(grad, x);
		}
		
		if (stopped_)
			break;
		
		// Después de cada época, actualizar métricas y contour plot
		
		// 1. Calcular cross entropy
		float crossEntropy = model_->test(X_ptr_, Y_ptr_)(0, 0);
		emit updateCrossEntropy(crossEntropy);
		
		// 2. Calcular accuracy
		float accuracy = calculateAccuracy();
		emit updateAccuracy(accuracy);
		
		// 3. Actualizar número de época
		emit updateEpoch(static_cast<int>(e + 1));
		
		// 4. Generar y emitir contour plot (cada N épocas para no saturar la UI)
		// Ajusta este valor según necesites: 1 = cada época, 10 = cada 10 épocas
		if ((e + 1) % 5 == 0 || e == 0 || e == nepochs_ - 1) {
			Mat<float> contourPredictions = generateContourPredictions();
			emit updateContourPlot(contourPredictions);
		}
		
		qDebug() << "Epoch:" << (e + 1) << "/" << nepochs_ 
		         << "| CrossEntropy:" << crossEntropy 
		         << "| Accuracy:" << (accuracy * 100.0f) << "%";
		
		// Pequeña pausa para no saturar la UI
		// if ((e + 1) % 10 == 0) {
		// 	QThread::msleep(1);
		// }
	}
	
	// Generar contour plot final
	if (!stopped_) {
		Mat<float> finalContour = generateContourPredictions();
		emit updateContourPlot(finalContour);
	}
	
	emit finishTraining();
}

void Trainer::setNEpochs(std::size_t nepochs)
{
	nepochs_ = nepochs;
}

void Trainer::setContourResolution(std::size_t resolution)
{
	contour_resolution_ = resolution;
}

void Trainer::setData(std::shared_ptr<std::vector<Mat<float>>> X_ptr, 
                      std::shared_ptr<std::vector<Mat<float>>> Y_ptr)
{
	X_ptr_ = X_ptr;
	Y_ptr_ = Y_ptr;
}

void Trainer::setModel(std::shared_ptr<Sequential<float>> model)
{
	model_ = model;
	
	// Configurar resolución por defecto del contour plot
	if (contour_resolution_ == 0) {
		contour_resolution_ = 100;  // 100x100 por defecto
	}
}
