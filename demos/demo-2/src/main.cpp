#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QHBoxLayout>
#include <QDebug>
#include <QThread>

#include <memory>
#include <unistd.h>

#include "chart.hpp"
#include "controls.hpp"
#include "trainer.hpp"

// include your perceptron header if you want to call it here
#include "../../../nn/include/nn.hpp"

using namespace nn::models;

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QMainWindow window;
	QWidget *central = new QWidget();
	window.setCentralWidget(central);

	QHBoxLayout* mainLayout = new QHBoxLayout(central);

	std::size_t nepochs = 25;
	ControlsWidget *controls = new ControlsWidget(nepochs);
	controls->setMinimumWidth(300); // left column

	ChartWidget *chart = new ChartWidget();
	// Let chart expand to the rest of the window
	chart->setMinimumSize(500, 500);

	mainLayout->addWidget(controls, 0);   // left (fixed-ish)
	mainLayout->addWidget(chart, 1);      // right (expands)

	// Set the data set
	std::vector<Mat<float>> X_data;
	std::vector<Mat<float>> Y_data;
	
	// Create the model
	Adeline<float> model(2, 1);
	model.set_loss(std::make_shared<MeanSquaredError<float>>());
	model.set_optimizer(std::make_shared<GradientDescentOptimizer<float>>(0.1f));
	model.build();


	// when user clicks points on chart, store them
	QObject::connect(chart, &ChartWidget::pointAdded,
			 [&](double x1, double x2, int label){
				 qDebug() << "pointAdded:" << x1 << x2 << "label" << label;
				 X_data.push_back({
						 {static_cast<float>(x1)},
						 {static_cast<float>(x2)}
					 });
				 Y_data.push_back({{static_cast<float>(label)}});
			 });

	QObject::connect(controls, &ControlsWidget::onChangeW1,
			 [&](double w1) {
				 qDebug() << "w1 changed" << w1;
				 Mat<float> &W = model.get_weights();
				 W(0, 0) = static_cast<float>(w1);
			 });

	QObject::connect(controls, &ControlsWidget::onChangeW2,
			 [&](double w2) {
				 qDebug() << "w2 changed" << w2;
				 Mat<float> &W = model.get_weights();
				 W(0, 1) = static_cast<float>(w2);
			 });

	QObject::connect(controls, &ControlsWidget::onChangeB,
			 [&](double b) {
				 qDebug() << "b changed" << b;
				 Mat<float> &B = model.get_bias();
				 B(0, 0) = static_cast<float>(b);
			 });

	QObject::connect(controls, &ControlsWidget::onChangeLr,
			 [&](double lr) {
				 qDebug() << "lr changed" << lr;
				 model.get_optimizer()->set_learning_rate(lr);
			 });

	QObject::connect(controls, &ControlsWidget::onChangeEpochs,
			 [&](int epochs) {
				 qDebug() << "epochs changed" << epochs;
				 nepochs = static_cast<int>(epochs);
			 });

	QObject::connect(controls, &ControlsWidget::weightsRandomized,
			 [&](double w1, double w2, double b) {
				 Mat<float> &W = model.get_weights();
				 W(0, 0) = static_cast<float>(w1);
				 W(0, 1) = static_cast<float>(w2);
				 Mat<float> &B = model.get_bias();
				 B(0, 0) = static_cast<float>(b);
			 });

	

	// clear points request
	QObject::connect(controls, &ControlsWidget::requestClear, chart, &ChartWidget::clearPoints);
	QObject::connect(controls, &ControlsWidget::requestClear,
			 [&]() {
				 X_data.clear();
				 Y_data.clear();
			 });
	
	// start training - you should connect this to your perceptron/train code
	QObject::connect(controls, &ControlsWidget::startTraining,
			 [&](void) {
				 Trainer *trainer = new Trainer();
				 trainer->setModel(&model);
				 trainer->setNEpochs(nepochs);
				 
				 // Connect signals back to GUI safely
				 // trainer and thread already created
				 
				 QObject::connect(trainer, &Trainer::updateWeights,
						  controls, &ControlsWidget::setWeights);				 
				 QObject::connect(trainer, &Trainer::updateWeights,
						  chart, &ChartWidget::setLineFromWeights);
				 QObject::connect(trainer, &Trainer::updateCrossEntropy,
						  controls, &ControlsWidget::setMSE);
				 QObject::connect(trainer, &Trainer::startingTraining, controls,
						  [=]() {
							  controls->setStatus("Training");
						  });
				 
				 QObject::connect(trainer, &Trainer::finishTraining, controls,
						  [=]() {
							  controls->setStatus("Finished");
						  });
				 
				 // Create the shared pointers  to set the data
				 trainer->setData(std::make_shared<std::vector<Mat<float>>>(X_data), std::make_shared<std::vector<Mat<float>>>(Y_data));
				 
				 QThread *thread = new QThread();
				 trainer->moveToThread(thread);
				 // Connect the thread with the trainer
				 QObject::connect(thread, &QThread::started, trainer, &Trainer::train);				 
				 
				 // Clean up when done
				 QObject::connect(trainer, &Trainer::finishTraining, thread, &QThread::quit);
				 QObject::connect(thread, &QThread::finished, trainer, &QObject::deleteLater);
				 QObject::connect(thread, &QThread::finished, thread, &QObject::deleteLater);
				 
				 thread->start();
			 });
	
	window.resize(1100, 700);
	window.setWindowTitle("Perceptron UI Training");
	window.show();

	return app.exec();
}
