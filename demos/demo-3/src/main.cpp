#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QHBoxLayout>
#include <QDebug>
#include <QThread>

#include <memory>

#include "chart.hpp"
#include "controls.hpp"
#include "trainer.hpp"

// Include neural network headers
#include "../../../nn/include/nn.hpp"
#include "../../../nn/include/layer.hpp"
#include "../../../nn/include/activation_func.hpp"
#include "../../../nn/include/optimizer.hpp"
#include "../../../nn/include/loss_func.hpp"
#include "../../../nn/include/rand.hpp"

using namespace nn::models;
using namespace nn::layers;
using namespace nn::activation_funcs;
using namespace nn::optimizers;
using namespace nn::loss_funcs;
using namespace nn::rand;
using namespace nn::mathops;

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QMainWindow window;
	QWidget *central = new QWidget();
	window.setCentralWidget(central);

	QHBoxLayout* mainLayout = new QHBoxLayout(central);

	std::size_t nepochs = 3000;
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
	
	// Create the Sequential model with shared_ptr
	auto model = std::make_shared<Sequential<float>>(
							 std::initializer_list<std::unique_ptr<Layer>>{
								 std::make_unique<Dense<float>>(2, 8, 
				std::make_shared<SigmoidFunc<float>>(), 
				std::make_shared<RandNormalInitializer<float>>(0.0f, 1.0f)),
			std::make_unique<Dense<float>>(8, 1, 
				std::make_shared<SigmoidFunc<float>>(), 
				std::make_shared<RandNormalInitializer<float>>(0.0f, 1.0f)),
		}
	);
	
	// Configure model
	model->set_optimizer(std::make_shared<GradientDescentOptimizer<float>>(0.01f));
	model->set_loss(std::make_shared<CrossEntropy<float>>());
	model->build();

	qDebug() << "Model created and built successfully";

	// Lambda to randomize weights of all Dense layers
	auto randomizeWeights = [&model]() {
		qDebug() << "Randomizing weights...";
		
		// Create a randomizer
		RandNormalInitializer<float> randomizer(0.0f, 1.0f);
		
		// Iterate through all layers
		const auto& layers = model->get_layers();
		for (const auto& layer_ptr : layers) {
			// Check if layer is trainable (WeightedLayer)
			if (layer_ptr->is_trainable()) {
				// Cast to Dense to access weights
				Dense<float>* dense_layer = dynamic_cast<Dense<float>*>(layer_ptr.get());
				
				if (dense_layer) {
					// Randomize weights
					Mat<float>& weights = dense_layer->get_weights();
					randomizer(weights);
					
					// Randomize bias
					Mat<float>& bias = dense_layer->get_bias();
					randomizer(bias);
					
					qDebug() << "Layer randomized - Weights shape:" 
					         << weights.rows() << "x" << weights.cols();
				}
			}
		}
		
		qDebug() << "All weights randomized!";
	};

	// When user clicks points on chart, store them
	QObject::connect(chart, &ChartWidget::pointAdded,
		[&](double x1, double x2, int label) {
			qDebug() << "pointAdded:" << x1 << x2 << "label" << label;
			X_data.push_back({
				{static_cast<float>(x1)},
				{static_cast<float>(x2)}
			});
			Y_data.push_back({{static_cast<float>(label)}});
		});

	// Connect learning rate change
	QObject::connect(controls, &ControlsWidget::onChangeLr,
		[&](double lr) {
			qDebug() << "Learning rate changed to:" << lr;
			model->get_optimizer()->set_learning_rate(static_cast<float>(lr));
		});

	// Connect epochs change
	QObject::connect(controls, &ControlsWidget::onChangeEpochs,
		[&](int epochs) {
			qDebug() << "Epochs changed to:" << epochs;
			nepochs = static_cast<std::size_t>(epochs);
		});

	// Connect randomize weights signal
	QObject::connect(controls, &ControlsWidget::requestRandomize,
		[&]() {
			randomizeWeights();
			controls->setStatus("Weights randomized");
			
			// Generate initial contour plot after randomization
			if (!X_data.empty()) {
				// Create a temporary trainer just to generate the contour
				Mat<float> predictions(100, 100);
				for (std::size_t i = 0; i < 100; i++) {
					for (std::size_t j = 0; j < 100; j++) {
						float x1 = (static_cast<float>(j) / 99.0f) * 10.0f;
						float x2 = (static_cast<float>(i) / 99.0f) * 10.0f;
						
						Mat<float> input_point(2, 1);
						input_point(0, 0) = x1;
						input_point(1, 0) = x2;
						
						Mat<float> pred = (*model)(input_point);
						predictions(i, j) = pred(0, 0);
					}
				}
				chart->updateContourPlot(predictions);
			}
		});

	// Clear points request
	QObject::connect(controls, &ControlsWidget::requestClear, 
		chart, &ChartWidget::clearPoints);
	
	QObject::connect(controls, &ControlsWidget::requestClear,
		[&]() {
			X_data.clear();
			Y_data.clear();
			qDebug() << "Data cleared";
		});
	
	// Start training
	QObject::connect(controls, &ControlsWidget::startTraining,
		[&]() {
			if (X_data.empty()) {
				qDebug() << "No data to train!";
				controls->setStatus("No data - add points first");
				return;
			}
			
			qDebug() << "Starting training with" << X_data.size() << "points";
			
			// Create trainer
			Trainer *trainer = new Trainer();
			trainer->setModel(model);
			trainer->setNEpochs(nepochs);
			trainer->setContourResolution(100);  // 100x100 contour plot
			
			// Create shared pointers for data
			auto X_ptr = std::make_shared<std::vector<Mat<float>>>(X_data);
			auto Y_ptr = std::make_shared<std::vector<Mat<float>>>(Y_data);
			trainer->setData(X_ptr, Y_ptr);
			
			// Connect signals from trainer to GUI
			QObject::connect(trainer, &Trainer::updateContourPlot,
				chart, &ChartWidget::updateContourPlot);
			
			QObject::connect(trainer, &Trainer::updateCrossEntropy,
				controls, &ControlsWidget::setCrossEntropy);
			
			QObject::connect(trainer, &Trainer::updateAccuracy,
				controls, &ControlsWidget::setAccuracy);
			
			QObject::connect(trainer, &Trainer::updateEpoch,
				controls, &ControlsWidget::setCurrentEpoch);
			
			QObject::connect(trainer, &Trainer::startingTraining,
				controls, [=]() {
					controls->setStatus("Training...");
				});
			
			QObject::connect(trainer, &Trainer::finishTraining,
				controls, [=]() {
					controls->setStatus("Finished");
				});
			
			// Create thread for training
			QThread *thread = new QThread();
			trainer->moveToThread(thread);
			
			// Connect thread start to training
			QObject::connect(thread, &QThread::started, 
				trainer, &Trainer::train);
			
			// Clean up when done
			QObject::connect(trainer, &Trainer::finishTraining, 
				thread, &QThread::quit);
			QObject::connect(thread, &QThread::finished, 
				trainer, &QObject::deleteLater);
			QObject::connect(thread, &QThread::finished, 
				thread, &QObject::deleteLater);
			
			// Start the thread
			thread->start();
		});
	
	// Randomize weights initially
	randomizeWeights();
	
	window.resize(1100, 700);
	window.setWindowTitle("Neural Network - Contour Plot Training");
	window.show();

	return app.exec();
}
