#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>
#include <QMouseEvent>
#include <QImage>
#include "mat.hpp"  // Biblioteca de matrices

class ChartWidget : public QChartView {
	Q_OBJECT
public:
	explicit ChartWidget(QWidget* parent = nullptr);

	// Actualiza el contorno con la matriz de predicciones (valores 0..1)
	void updateContourPlot(const nn::mathops::Mat<float>& predictions);

	void clearPoints(void);

signals:
	void pointAdded(double x, double y, int label);

protected:
	void mousePressEvent(QMouseEvent* event) override;

private:
	QChart* chart;
	QScatterSeries* pointSeriesClassA;
	QScatterSeries* pointSeriesClassB;
	QValueAxis* axisX;
	QValueAxis* axisY;

	// Convierte predicción [0..1] a color RGBA
	QColor predictionToColor(float prediction);
};
