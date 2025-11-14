#ifndef INCLUDED_CONTROLS
#define INCLUDED_CONTROLS

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QSpinBox>

class ControlsWidget : public QWidget {
	Q_OBJECT
public:
	explicit ControlsWidget(std::size_t nepochs = 100, QWidget* parent = nullptr);

signals:
	// Emitido cuando el usuario hace click en Start Training
	void startTraining(void);
	
	// Emitido cuando el usuario hace click en Clear Points
	void requestClear(void);
	
	// Emitido cuando el usuario hace click en Randomize Weights
	void requestRandomize(void);
	
	// Emitido cuando cambia el learning rate
	void onChangeLr(double lr);
	
	// Emitido cuando cambian las épocas
	void onChangeEpochs(int epochs);
	
public slots:
	// Actualizar el Cross Entropy en la UI (llamar desde el thread de entrenamiento)
	void setCrossEntropy(double cross_entropy);
	
	// Actualizar accuracy en la UI
	void setAccuracy(double accuracy);
	
	// Actualizar el status
	void setStatus(const QString &text);
	
	// Actualizar la época actual
	void setCurrentEpoch(int epoch);

private:
	QDoubleSpinBox* lrSpin;
	QSpinBox* epochsSpin;
	
	QLabel* crossEntropyLabel;
	QLabel* accuracyLabel;
	QLabel* epochLabel;
	QLabel* statusLabel;
	
	QPushButton* startButton;
	QPushButton* clearButton;
	QPushButton* randomButton;
};

#endif // INCLUDED_CONTROLS
