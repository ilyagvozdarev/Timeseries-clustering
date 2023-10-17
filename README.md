# Timeseries-clustering

Кластеризация временных рядов, отображающих значения параметров устройств аппаратного обеспечения, измеренных во времени. 
Кластеризация производится несколькими алгоритмами кластеризации временных рядов - N2D, IDEC, K-Shape.
Качество кластеризации оценивается по метрикам: коэффициент силуэта, Calinski-Harabasz Index, Davies-Bouldin Index.
Оптимальное количество кластеров определяется по метрикам: коэффициент силуэта, Локтевой метод (Elbow method), статистика разрыва (gap statistic).

## Оглавление

- [Данные](#данные)
- [Learning hints](#learning-hints)
- [Extras](#extras)
- [What's next?](#whats-next)

## Данные

Временные ряды - значения параметров устройств аппаратного обеспечения, измеренных во времени.
Формат названия - monitoringMetric$*идентификатор*_*категория периода*
Категория периода - название периода, за который хранятся значения во временном ряду. Возможные значения - DAY, WEEK, MONTH, HALF_YEAR, INFINITE.

Пример:

ряд 'monitoringMetric$20221722__WEEK':

<img src="./img/readme/ts_example.jpg" style="max-width: 100%; margin-left: auto; margin-right: auto;" />
