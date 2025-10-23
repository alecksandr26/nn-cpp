#ifndef INCLUDED_CHART
#define INCLUDED_CHART

#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>
#include <QMouseEvent>

class ChartWidget : public QChartView {
	Q_OBJECT

public:
	explicit ChartWidget(QWidget* parent = nullptr);
	void setLineFromWeights(double w1, double w2, double b);
	void clearPoints(void);
	
signals:
	void pointAdded(double x, double y, int label);
	
protected:
	void mousePressEvent(QMouseEvent* event) override;

private:
	QChart* chart;
	QLineSeries* lineSeries;
	QScatterSeries* pointSeriesClassA;
	QScatterSeries* pointSeriesClassB;
	QValueAxis* axisX;
	QValueAxis* axisY;
};

#endif
