#ifndef INCLUDED_TRAINER
#define INCLUDED_TRAINER

#include <QObject>
#include <vector>
#include "../../../nn/include/mat.hpp"
#include "../../../nn/include/nn.hpp"

using namespace nn::models;

class Trainer : public QObject {
	Q_OBJECT
public:
	explicit Trainer(QObject *parent = nullptr) : QObject(parent) {}
        void setModel(Adeline<float> *model);
	void setData(std::shared_ptr<std::vector<Mat<float>>> X_ptr, std::shared_ptr<std::vector<Mat<float>>> Y_ptr);
	void setNEpochs(std::size_t nepochs);
					    
public slots:
	void train(void);
	void stop(void);
	
signals:
	void updateWeights(float w0, float w1, float b);
	void updateCrossEntropy(float crossEntropy);
	void finishTraining(void);
	void startingTraining(void);

private:
	Adeline<float> *model_;
	std::size_t nepochs_;
	std::shared_ptr<std::vector<Mat<float>>> X_ptr_;
	std::shared_ptr<std::vector<Mat<float>>> Y_ptr_;
	std::atomic<bool> stopped_{false}; // thread-safe flag
};


#endif
