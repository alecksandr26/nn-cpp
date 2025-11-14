#ifndef INCLUDED_TRAINER
#define INCLUDED_TRAINER

#include <QObject>
#include <vector>
#include <atomic>
#include "../../../nn/include/mat.hpp"
#include "../../../nn/include/nn.hpp"

using namespace nn::models;
using namespace nn::mathops;

class Trainer : public QObject {
	Q_OBJECT
public:
	explicit Trainer(QObject *parent = nullptr) : QObject(parent) {}
	
	void setModel(std::shared_ptr<Sequential<float>> model);
	void setData(std::shared_ptr<std::vector<Mat<float>>> X_ptr, 
	             std::shared_ptr<std::vector<Mat<float>>> Y_ptr);
	void setNEpochs(std::size_t nepochs);
	void setContourResolution(std::size_t resolution);  // Default 100x100
					    
public slots:
	void train(void);
	void stop(void);
	
signals:
	// Emitir matriz de predicciones para el contour plot (100x100 por defecto)
	void updateContourPlot(const Mat<float> &predictions);
	
	// Emitir cross entropy actual
	void updateCrossEntropy(float crossEntropy);
	
	// Emitir accuracy
	void updateAccuracy(float accuracy);
	
	// Emitir época actual
	void updateEpoch(int epoch);
	
	// Señales de estado
	void finishTraining(void);
	void startingTraining(void);
	
private:
	// Generar la matriz de predicciones para el contour plot
	Mat<float> generateContourPredictions(void);
	
	// Calcular accuracy
	float calculateAccuracy(void);
	
	std::shared_ptr<Sequential<float>> model_;
	std::size_t nepochs_;
	std::size_t contour_resolution_;  // 100x100 por defecto
	
	std::shared_ptr<std::vector<Mat<float>>> X_ptr_;
	std::shared_ptr<std::vector<Mat<float>>> Y_ptr_;
	
	std::atomic<bool> stopped_{false};
};

#endif // INCLUDED_TRAINER
