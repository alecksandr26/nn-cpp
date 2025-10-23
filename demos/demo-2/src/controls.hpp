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
	// emitted when user clicks start
	void startTraining(void);
	void requestClear(void);
	void onChangeW1(double w1);
	void onChangeW2(double w2);
	void onChangeB(double b);
	void onChangeLr(double lr);	
	void onChangeEpochs(int epochs);
	void weightsRandomized(double w1, double w2, double b);
			     
public slots:
	// update UI labels (call from training thread / main after update)
	void setWeights(double w1, double w2, double b);
	void setMSE(double mse);
	void setStatus(const QString &text);

private:
	QDoubleSpinBox* w1Spin;
	QDoubleSpinBox* w2Spin;
	QDoubleSpinBox* bSpin;
	QDoubleSpinBox* lrSpin;
	QSpinBox* epochsSpin;
	QLabel* weightsLabel;
	QLabel* mseLabel;
	QLabel* statusLabel;
	QPushButton* startButton;
	QPushButton* clearButton;
	QPushButton* randomButton;
};

#endif // INCLUDED_CONTROLS



