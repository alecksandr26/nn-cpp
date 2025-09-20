#include <QtCharts/QValueAxis>
#include <QColor>

#include "chart.hpp"

ChartWidget::ChartWidget(QWidget* parent)
	: QChartView(parent),
	  chart(new QChart()),
	  lineSeries(new QLineSeries()),
	  pointSeriesClassA(new QScatterSeries()),
	  pointSeriesClassB(new QScatterSeries()),
	  axisX(new QValueAxis()),
	  axisY(new QValueAxis())
{
	// Configure points
	pointSeriesClassA->setMarkerSize(10);
	pointSeriesClassA->setColor(QColor("#2f8fff"));
	pointSeriesClassB->setMarkerSize(10);
	pointSeriesClassB->setColor(QColor("#7fcf5f"));

	// Add series to chart
	chart->addSeries(lineSeries);
	chart->addSeries(pointSeriesClassA);
	chart->addSeries(pointSeriesClassB);

	// Configure axes
	axisX->setRange(0, 10);
	axisX->setTitleText("X1");
	axisX->setTickCount(11);
	axisX->setGridLineVisible(true);

	axisY->setRange(0, 10);
	axisY->setTitleText("X2");
	axisY->setTickCount(11);
	axisY->setGridLineVisible(true);

	// Attach axes to series
	chart->setAxisX(axisX, lineSeries);
	chart->setAxisY(axisY, lineSeries);
	chart->setAxisX(axisX, pointSeriesClassA);
	chart->setAxisY(axisY, pointSeriesClassA);
	chart->setAxisX(axisX, pointSeriesClassB);
	chart->setAxisY(axisY, pointSeriesClassB);

	// style the decision line
	// QPen pen;
	// pen.setWidth(2);
	// pen.setColor(QColor("#2aa1f7"));
	// lineSeries->setPen(pen);

	chart->setTitle("Perceptron Chart");
	setChart(chart);
	setRenderHint(QPainter::Antialiasing);
}

void ChartWidget::mousePressEvent(QMouseEvent* event)
{
	// map screen pos -> chart value coordinates
	QPointF chartPt = chart->mapToValue(event->position());
	double x = chartPt.x();
	double y = chartPt.y();

	if (event->button() == Qt::LeftButton) {
		pointSeriesClassA->append(x, y);
		emit pointAdded(x, y, 0);
	} else if (event->button() == Qt::RightButton) {
		pointSeriesClassB->append(x, y);
		emit pointAdded(x, y, 1);
	}

	QChartView::mousePressEvent(event);
}

void ChartWidget::setLineFromWeights(double w1, double w2, double b)
{
	lineSeries->clear();

	// Decision boundary: w1*x + w2*y + b = 0
	// if w2 != 0: y = -(w1/w2)*x - b/w2
	// draw line across x=[0..10]
	if (qFuzzyIsNull(w2)) {
		// vertical line x = -b/w1
		if (!qFuzzyIsNull(w1)) {
			double x = -b / w1;
			lineSeries->append(x, 0);
			lineSeries->append(x, 10);
		}
	} else {
		double x1 = 0;
		double y1 = (qIsFinite(-b / w2)) ? (-(w1 / w2) * x1 - b / w2) : 0;
		double x2 = 10;
		double y2 = (qIsFinite(-(w1 / w2) * x2 - b / w2)) ? (-(w1 / w2) * x2 - b / w2) : 10;
		lineSeries->append(x1, y1);
		lineSeries->append(x2, y2);
	}
}

void ChartWidget::clearPoints(void)
{
	pointSeriesClassA->clear();
	pointSeriesClassB->clear();
}
