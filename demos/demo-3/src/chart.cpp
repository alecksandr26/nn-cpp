#include "chart.hpp"
#include <iostream>

ChartWidget::ChartWidget(QWidget* parent)
	: QChartView(parent),
	  chart(new QChart()),
	  pointSeriesClassA(new QScatterSeries()),
	  pointSeriesClassB(new QScatterSeries()),
	  axisX(new QValueAxis()),
	  axisY(new QValueAxis())
{
	// Configurar series de puntos
	pointSeriesClassA->setMarkerSize(12);
	pointSeriesClassA->setColor(QColor("#2f8fff"));  // Azul
	pointSeriesClassA->setBorderColor(Qt::white);
	
	pointSeriesClassB->setMarkerSize(12);
	pointSeriesClassB->setColor(QColor("#7fcf5f"));  // Verde
	pointSeriesClassB->setBorderColor(Qt::white);
	
	// Añadir series al gráfico
	chart->addSeries(pointSeriesClassA);
	chart->addSeries(pointSeriesClassB);
	
	// Configurar ejes (0-10)
	axisX->setRange(0, 10);
	axisX->setTitleText("X1");
	axisX->setTickCount(11);
	axisX->setGridLineVisible(true);
	
	axisY->setRange(0, 10);
	axisY->setTitleText("X2");
	axisY->setTickCount(11);
	axisY->setGridLineVisible(true);
	
	// Adjuntar ejes a las series
	chart->setAxisX(axisX, pointSeriesClassA);
	chart->setAxisY(axisY, pointSeriesClassA);
	chart->setAxisX(axisX, pointSeriesClassB);
	chart->setAxisY(axisY, pointSeriesClassB);
	
	// Fondo blanco del gráfico
	chart->setBackgroundVisible(true);
	chart->setBackgroundBrush(QBrush(Qt::white));
	chart->setTitle("Red Neuronal - Contour Plot");
	setChart(chart);
	setRenderHint(QPainter::Antialiasing);
	setBackgroundBrush(QBrush(Qt::white));
}

void ChartWidget::mousePressEvent(QMouseEvent* event)
{
	// Convertir coordenadas de clic a valores del gráfico
	QPointF chartPt = chart->mapToValue(event->position());
	double x = chartPt.x();
	double y = chartPt.y();
	
	if (event->button() == Qt::LeftButton) {
		pointSeriesClassA->append(x, y);
		emit pointAdded(x, y, 0);
	} 
	else if (event->button() == Qt::RightButton) {
		pointSeriesClassB->append(x, y);
		emit pointAdded(x, y, 1);
	}
	
	QChartView::mousePressEvent(event);
}

QColor ChartWidget::predictionToColor(float prediction)
{
    prediction = qBound(0.0f, prediction, 1.0f);

    // 0 → morado (bajo), 1 → rojo (alto)
    int r = static_cast<int>(150 + (255 - 150) * prediction);
    int g = static_cast<int>(0   + (50  - 0)   * prediction);
    int b = static_cast<int>(150 - (150 - 50)  * prediction);
    return QColor(r, g, b, 255);
}


void ChartWidget::updateContourPlot(const nn::mathops::Mat<float>& predictions)
{
	
	// Generar imagen de contorno del tamaño del plotArea
	int rows = predictions.rows();
	int cols = predictions.cols();
	if (rows <= 0 || cols <= 0) return;

	QRectF plotArea = chart->plotArea();
	if (plotArea.width() <= 0 || plotArea.height() <= 0) return;

	QImage contourImage(
			    static_cast<int>(plotArea.width()),
			    static_cast<int>(plotArea.height()),
			    QImage::Format_ARGB32
			    );

	contourImage.fill(Qt::transparent);

	// Pintar cada celda con QPainter
	qreal pixelWidth = plotArea.width() / static_cast<qreal>(cols);
	qreal pixelHeight = plotArea.height() / static_cast<qreal>(rows);

	QPainter painter(&contourImage);
	painter.setRenderHint(QPainter::Antialiasing, false);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			float pred = predictions(i, j);
			QColor color = predictionToColor(pred);
			color.setAlphaF(0.6); // 0.0 = totalmente transparente, 1.0 = opaco

			qreal x = j * pixelWidth;
			qreal y = i * pixelHeight; // invertimos Y

			painter.fillRect(QRectF(x, y, pixelWidth, pixelHeight), color);
		}
	}

	painter.end();

	// Crear el QBrush y alinear correctamente la imagen
	QBrush brush(contourImage);
	QTransform transform;
	transform.translate(plotArea.left(), plotArea.top());
	brush.setTransform(transform);
	brush.setStyle(Qt::TexturePattern);

	// Asignar como fondo del área de trazado
	chart->setPlotAreaBackgroundBrush(brush);
	chart->setPlotAreaBackgroundVisible(true);

	// Refrescar visualización
	chart->update();
	if (chart->scene())
		chart->scene()->update();

	viewport()->update();

}


void ChartWidget::clearPoints(void)
{
	pointSeriesClassA->clear();
	pointSeriesClassB->clear();
}
