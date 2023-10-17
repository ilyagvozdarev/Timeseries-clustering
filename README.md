# Timeseries-clustering

Кластеризация временных рядов, отображающих значения параметров устройств аппаратного обеспечения, измеренных во времени. 
Кластеризация производится несколькими алгоритмами кластеризации временных рядов - N2D, IDEC, K-Shape.
Качество кластеризации оценивается по метрикам: коэффициент силуэта, Calinski-Harabasz Index, Davies-Bouldin Index.
Оптимальное количество кластеров определяется по метрикам: коэффициент силуэта, Локтевой метод (Elbow method), статистика разрыва (gap statistic).

## Оглавление

- [Данные](#данные)
- [Предобработка рядов](#предобработка-рядов)
- [Extras](#extras)
- [What's next?](#whats-next)

## Данные

Временные ряды - значения параметров устройств аппаратного обеспечения, измеренных во времени.<br><br>
Формат названия - monitoringMetric$*идентификатор*_*категория периода*<br><br>
Категория периода - название периода, за который хранятся значения во временном ряду. <br>Возможные значения - DAY, WEEK, MONTH, HALF_YEAR, INFINITE.

Пример:

ряд 'monitoringMetric$20221722__WEEK':

<p align="center">
<img src="./img/readme/ts_example.jpg" />
</p>

## Предобработка рядов

- усечение рядов периода до заданных границ (левая граница = медиана начального времени по всем рядам периода + 10мин<br>
  правая граница = медиана последнего времени по всем рядам периода - 10мин)<br>
- ресемплирование с заданной частотой<br>
- интерполирование получаемых после ресемплилования пустых значений<br>
- удаление пустых рядов, которые могли появиться снова после усечения<br>
- удаление константных рядов<br>
- minmax-масштабирование<br>

**Downsampling** (уменьшение разрешения, децимация) - уменьшение частоты дискретизации временного ряда.<br>
**Upsampling** (увеличение разрешения) - увеличение частоты дискретизации временного ряда.<br>
**Ресемплирование** - Downsampling с последующим Upsampling. Появляющиеся в результате отсутствующие значения заполняются, например, интерполяцией.<br>
Получаемый в результате временной ряд будет с более низким разрешением, что ускоряет вычисления.
