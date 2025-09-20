#include "controls.hpp"
#include <QFormLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>

ControlsWidget::ControlsWidget(QWidget* parent)
	: QWidget(parent)
{
	auto *form = new QFormLayout;

	w1Spin = new QDoubleSpinBox;
	w1Spin->setRange(-1000, 1000);
	w1Spin->setDecimals(4);
	w1Spin->setValue(0.0);
	connect(w1Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
		this, &ControlsWidget::onChangeW1);

	w2Spin = new QDoubleSpinBox;
	w2Spin->setRange(-1000, 1000);
	w2Spin->setDecimals(4);
	w2Spin->setValue(0.0);
	connect(w2Spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
		this, &ControlsWidget::onChangeW2);

	bSpin = new QDoubleSpinBox;
	bSpin->setRange(-1000, 1000);
	bSpin->setDecimals(4);
	bSpin->setValue(0.0);
	connect(bSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
		this, &ControlsWidget::onChangeB);

	lrSpin = new QDoubleSpinBox;
	lrSpin->setRange(1e-6, 100.0);
	lrSpin->setDecimals(6);
	lrSpin->setValue(0.1);
	connect(lrSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
		this, &ControlsWidget::onChangeLr);

	epochsSpin = new QSpinBox;
	epochsSpin->setRange(1, 100000);
	epochsSpin->setValue(100);
	connect(epochsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
		this, &ControlsWidget::onChangeEpochs);

	form->addRow("w1:", w1Spin);
	form->addRow("w2:", w2Spin);
	form->addRow("b:", bSpin);
	form->addRow("learning rate:", lrSpin);
	form->addRow("epochs:", epochsSpin);

	randomButton = new QPushButton("Randomize Weights");
	startButton = new QPushButton("Start Training");
	clearButton = new QPushButton("Clear Points");
	weightsLabel = new QLabel("Weights: w1=0.000 w2=0.000 b=0.000");
	maeLabel = new QLabel("MAE: N/A");
	statusLabel = new QLabel("Status: Idle");

	QVBoxLayout* main = new QVBoxLayout;
	main->addLayout(form);
	main->addWidget(randomButton);
	main->addWidget(startButton);
	main->addWidget(clearButton);
	main->addWidget(weightsLabel);
	main->addWidget(maeLabel);
	main->addWidget(statusLabel);
	main->addStretch(1);
	setLayout(main);

	connect(startButton, &QPushButton::clicked, this, [=]() {
		emit startTraining();
	});

	connect(clearButton, &QPushButton::clicked, this, [=]() {
		emit requestClear();
	});

	connect(randomButton, &QPushButton::clicked, this, [=]() {
		double w1 = static_cast<double>(rand()) / RAND_MAX; // [0,1]
		double w2 = static_cast<double>(rand()) / RAND_MAX; // [0,1]
		double b  = static_cast<double>(rand()) / RAND_MAX; // [0,1]

		w1Spin->setValue(w1);
		w2Spin->setValue(w2);
		bSpin->setValue(b);
		
		// optional: emit a signal if you want the model to react immediately
		emit weightsRandomized(w1, w2, b);
	});
}

void ControlsWidget::setWeights(double w1, double w2, double b)
{
	weightsLabel->setText(QString("Weights: w1=%1 w2=%2 b=%3")
			      .arg(w1, 0, 'f', 4)
			      .arg(w2, 0, 'f', 4)
			      .arg(b, 0, 'f', 4));
}

void ControlsWidget::setMAE(double mae)
{
	maeLabel->setText(QString("MAE: %1").arg(mae, 0, 'f', 6));
}

void ControlsWidget::setStatus(const QString &text)
{
	statusLabel->setText("Status: " + text);
}

