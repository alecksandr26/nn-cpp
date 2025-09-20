#include "trainer.hpp"
#include <cstddef>
#include <QDebug>
#include <QThread>

using namespace nn::models;

void Trainer::train(void)
{
	float last_mae = -1.0f;
	emit startingTraining();
	for (std::size_t e = 0; e < this->nepochs_; e++) {
		for (std::size_t i = 0; i < X_ptr_->size(); i++) {
			Mat<float> Y_pred = (*model_)((*X_ptr_)[i]);
			if (Y_pred != (*Y_ptr_)[i]) {
				model_->WeightedLayer::fit((*Y_ptr_)[i] - Y_pred, (*X_ptr_)[i]);

				Mat<float> &W = model_->get_weights();
				Mat<float> &B = model_->get_bias();
				emit updateWeights(W(0, 0), W(0, 1), B(0, 0));
				float mae = model_->test(X_ptr_, Y_ptr_)(0, 0);
				qDebug() << "epoch: " << e << " mae: " << mae;
				emit updateMAE(mae);
				if (last_mae != mae) {
					last_mae = mae;
					// Sleep a few miliseconds
					QThread::msleep(100);
				}
			}
		}
	}
	emit finishTraining();
}


void Trainer::setNEpochs(std::size_t nepochs)
{
	nepochs_ = nepochs;
}

void Trainer::setData(std::shared_ptr<std::vector<Mat<float>>> X_ptr, std::shared_ptr<std::vector<Mat<float>>> Y_ptr)
{
	X_ptr_ = X_ptr;
	Y_ptr_ = Y_ptr;
}



void Trainer::setModel(Perceptron<float> *model)
{
	model_ = model;
}


