#include "trainer.hpp"
#include <cstddef>
#include <QDebug>
#include <QThread>

#include "../../../nn/include/activation_func.hpp"

using namespace nn::models;
using namespace nn::activation_funcs;

void Trainer::stop(void)
{
	qDebug() << "Trainer: stop() called";
	stopped_ = true;
}

void Trainer::train(void)
{
	stopped_ = false;
	emit startingTraining();
	float last_cross_entropy = -1.0f;
	SigmoidFunc<float> sigmoid;
	sigmoid.build();
	model_->get_loss()->set_inputs(X_ptr_);
	model_->get_loss()->set_outputs(Y_ptr_);
	for (std::size_t e = 0; e < this->nepochs_; e++) {
		if (stopped_)
			break;
		for (std::size_t i = 0; i < X_ptr_->size(); i++) {
			if (stopped_)
				break;
			Mat<float> &W = model_->get_weights();
			Mat<float> &B = model_->get_bias();
			Mat<float> Z = W.dot((*X_ptr_)[i]) + B;				
			Mat<float> grad_Y_Z = sigmoid.gradient(Z);
			Mat<float> grad_L_Y = model_->get_loss()->gradient({(*X_ptr_)[i], (*Y_ptr_)[i]});
			Mat<float> grad_L_Z = grad_L_Y * grad_Y_Z;
			
			model_->WeightedLayer::fit(grad_L_Z, (*X_ptr_)[i]);
			
			emit updateWeights(W(0, 0), W(0, 1), B(0, 0));
			float crossEntropy = model_->test(X_ptr_, Y_ptr_)(0, 0);
			qDebug() << "epoch: " << e << " CrossEntropy: " << crossEntropy;
			emit updateCrossEntropy(crossEntropy);
			// if (last_cross_entropy != crossEntropy) {
			// 	last_cross_entropy = crossEntropy;
			// 	// Sleep a few miliseconds
			// 	QThread::msleep(1);
			// }
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



void Trainer::setModel(Adeline<float> *model)
{
	model_ = model;
}


