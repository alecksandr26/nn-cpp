#include "controls.hpp"
#include <QFormLayout>
#include <QVBoxLayout>
#include <QGroupBox>

ControlsWidget::ControlsWidget(std::size_t nepochs, QWidget* parent)
	: QWidget(parent)
{
	// === Sección de Hiperparámetros ===
	auto *hyperBox = new QGroupBox("Hyperparameters");
	auto *hyperForm = new QFormLayout;
	
	// Learning Rate
	lrSpin = new QDoubleSpinBox;
	lrSpin->setRange(1e-6, 10.0);
	lrSpin->setDecimals(6);
	lrSpin->setValue(1.0);
	lrSpin->setSingleStep(0.001);
	connect(lrSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
		this, &ControlsWidget::onChangeLr);
	
	// Epochs
	epochsSpin = new QSpinBox;
	epochsSpin->setRange(1, 100000);
	epochsSpin->setValue(nepochs);
	connect(epochsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
		this, &ControlsWidget::onChangeEpochs);
	
	hyperForm->addRow("Learning Rate:", lrSpin);
	hyperForm->addRow("Epochs:", epochsSpin);
	hyperBox->setLayout(hyperForm);
	
	// === Botones de Acción ===
	randomButton = new QPushButton("🎲 Randomize Weights");
	startButton = new QPushButton("▶ Start Training");

	startButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }");
	clearButton = new QPushButton("🗑 Clear Points");
	
	// === Sección de Métricas ===
	auto *metricsBox = new QGroupBox("Training Metrics");
	auto *metricsLayout = new QVBoxLayout;
	
	epochLabel = new QLabel("Epoch: 0 / 0");
	crossEntropyLabel = new QLabel("CrossEntropy: N/A");
	accuracyLabel = new QLabel("Accuracy: N/A");
	statusLabel = new QLabel("Status: Idle");
	
	metricsLayout->addWidget(epochLabel);
	metricsLayout->addWidget(crossEntropyLabel);
	metricsLayout->addWidget(accuracyLabel);
	metricsLayout->addWidget(statusLabel);
	metricsBox->setLayout(metricsLayout);
	
	// === Layout Principal ===
	QVBoxLayout* main = new QVBoxLayout;
	main->addWidget(hyperBox);
	main->addSpacing(10);
	main->addWidget(randomButton);
	main->addWidget(startButton);
	main->addWidget(clearButton);
	main->addSpacing(10);
	main->addWidget(metricsBox);
	main->addStretch(1);
	
	setLayout(main);
	
	// === Conexiones de Botones ===
	connect(startButton, &QPushButton::clicked, this, [=]() {
		emit startTraining();
	});
	
	connect(clearButton, &QPushButton::clicked, this, [=]() {
		emit requestClear();
	});
	
	// Randomize solo emite la señal, el modelo se encarga de todo
	connect(randomButton, &QPushButton::clicked, this, [=]() {
		emit requestRandomize();
	});
}

void ControlsWidget::setCrossEntropy(double cross_entropy)
{
	crossEntropyLabel->setText(QString("CrossEntropy: %1").arg(cross_entropy, 0, 'f', 6));
}

void ControlsWidget::setAccuracy(double accuracy)
{
	accuracyLabel->setText(QString("Accuracy: %1%").arg(accuracy * 100.0, 0, 'f', 2));
}

void ControlsWidget::setStatus(const QString &text)
{
	statusLabel->setText("Status: " + text);
}

void ControlsWidget::setCurrentEpoch(int epoch)
{
	int totalEpochs = epochsSpin->value();
	epochLabel->setText(QString("Epoch: %1 / %2").arg(epoch).arg(totalEpochs));
}
