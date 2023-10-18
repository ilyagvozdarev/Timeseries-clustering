# Timeseries-clustering

Кластеризация временных рядов, отображающих значения параметров устройств аппаратного обеспечения, измеренных во времени.<br> 
Кластеризация производится несколькими алгоритмами кластеризации временных рядов - N2D, IDEC, K-Shape.<br> 
Качество кластеризации оценивается по метрикам: коэффициент силуэта, Calinski-Harabasz Index, Davies-Bouldin Index.<br> 
Оптимальное количество кластеров определяется по метрикам: коэффициент силуэта, Локтевой метод (Elbow method), статистика разрыва (gap statistic).

## Оглавление

- [Данные](#данные)
- [Предобработка рядов](#предобработка-рядов)
- [Метрики](#метрики)
- [Методы кластеризации](#методы-кластеризации)
- [Оценка кластеризации и визуализация](#оценка-кластеризации-и-визуализация)<br><br>

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
- интерполирование получаемых после ресемплирования пустых значений<br>
- удаление пустых рядов, которые могли появиться снова после усечения<br>
- удаление константных рядов<br>
- minmax-масштабирование<br>

**Downsampling** (уменьшение разрешения, децимация) - уменьшение частоты дискретизации временного ряда.<br>
**Upsampling** (увеличение разрешения) - увеличение частоты дискретизации временного ряда.<br>
**Ресемплирование** - Downsampling с последующим Upsampling. Появляющиеся в результате отсутствующие значения заполняются, например, интерполяцией.<br>
Получаемый в результате временной ряд будет с более низким разрешением, что ускоряет вычисления.<br><br>

## Метрики

### Неконтролируемые метрики качества (метки неизвестны)
- **коэффициент силуэта**
- **Calinski-Harabasz Index**<br>
  Отношение суммы междукластерных дисперсий к внутрикластерным дисперсиям для всех кластеров (внутрекластерная дисперсия - сумма квадратов расстояний от точек до центров их кластеров, междукластерная дисперсия - сумма квадратов расстояний от центров кластеров до центра всех точек выборки, умноженных на количество точек в кластере).
- **Davies-Bouldin Index**<br>
  Среднее «сходство» между кластерами, где сходство — это мера, которая сравнивает расстояние между кластерами с размером самих   кластеров.

### Метрики для выбора количества кластеров
- **коэффициент силуэта**
- **Локтевой метод (Elbow method)**<br>
  Строится график зависимости суммы квадратов внутрикластерных расстояний от количества кластеров.<br>
  Изгиб указывает на то, что дополнительные кластеры после определенного не имеют большого значения
- **статистика разрыва (gap statistic)**<br>
  Статистика разрыва сравнивает сумму внутрикластерных расстояний с их ожидаемыми значениями при нулевом (как правило 
  равномерном) эталонном распределении данных.<br>
  Делается кластеризация выборки из нулевого распределения.
  Считается логарифм отношения суммы внутрикластерных расстояний для выборки из нулевого распределения к сумме внутрикластерных   расстояний нашей кластеризации.<br>
  Оптимальное количество кластеров - значение, при котором статистика разрыва максимальна. Это означает, что структура кластеризации далека от случайного равномерного распределения точек.<br><br>

## Методы кластеризации

- [N2D](#n2d)
- [IDEC](#idec)
- [k-Shape](#k-shape)<br><br>  


### N2D
- Обучаем и применяем **автоэнкодер** к необработанным данным, чтобы узнать начальные представления точек - эмбеддинги. <br>Функция потерь - MSE
- Дообучаем полученные эмбеддинги с помощью поиска более кластеризируемых представлений  методом **UMAP** - методом обучения многообразиям, который сохраняет локальные расстояния.
<br>UMAP - алгоритм, похожий на t-SNE, но с более сильным математическим обоснованием. При снижении размерности UMAP сначала выполняет построение взвешенного графа, соединяя ребрами только те объекты, которые являются ближайшими соседями. Множество из ребер графа — это нечёткое множество с функцией принадлежности, она определяется как вероятность существования ребра между двумя вершинами. Затем алгоритм создает граф в низкоразмерном пространстве и приближает его к исходному, минимизируя сумму дивергенций Кульбака-Лейблера для каждого ребра из множеств
- На полученных более кластеризуемых эмбеддингах мы применяем окончательный простой алгоритм кластеризации (например **GMM (GaussianMixture)**), чтобы обнаружить кластеры.<br><br>


### IDEC
<p align="center">
<img src="img/idec.jpg" width=500>
</p>

Обучаем и применяем **автоэнкодер** к необработанным данным, чтобы узнать начальные представления точек - эмбеддинги.

Одновременно к выходу энкодера автоэнкодера присоединяется **слой кластеризации**, который определяет сходство эмбеддинга с каждым центром кластера.

Вычисляется $q_{ij}$ - сходство между эмбеддингом точки $z_{i}$ и центром кластера $\mu_{j}$ с помощью t-распределения Стьюдента (также как в t-SNE алгоритме).
Полученную величину можно интерпретировать как вероятность кластера j для точки i:

<p align="center">
<img src="img/qij.jpg" width=330>
</p>
  
Вычисляется $p_{ij}$ - вероятность вложенной точки $z_{i}$ в кластере $\mu_{j}$ (p называют целевым распределением):

<p align="center">
<img src="img/pij.jpg" width=280>
</p>
  
Центры кластеров обновляются оптимизацией функции потерь - Дивергенции Кульбака-Лейблера (KL). <br>На первом шаге можно инициализировать центры любым алгоритмом, например k-means:

<p align="center">
<img src="img/KL.jpg" width=430>
</p>
  
В течении обратного распространения вычисляются градиенты L по $z_{i}$ (передаются для обновления $f_W$ (весов автоэнкодера)) и $\mu_{j}$ (используются для обновления центров кластеров $\mu_{j}$).
кластер объекта $i = argmax_j(q_{ij})$

общая функция потерь:

<p align="center">
<img src="img/idec_loss.jpg" width=200>
</p>

$L_{r}$ - функция потерь автоэнкодера - потеря реконструкции (MSE)
  
$L_{c}$ - потеря кластеризации (KL)

$\gamma > 0$ - коэффициент контролирующий степень искажения вложенного пространства. Чем больше значение, тем эмбеддинги точек будут более кластеризируемыми и менее надежными (хуже представляют исходные ряды)
<br>
<br>
### k-Shape
k-Shape основан на итерационной процедуре уточнения, аналогичной той, что используется в алгоритме k-means, но со значительными отличиями. В частности, k-Shape использует другую меру расстояния и метод вычисления центроида (центр кластера), отличный от методов k-means. Мера расстояния k-Shape пытается сохранить формы временных рядов при их сравнении. Для этого k-Shape использует меру расстояния, инвариантную к масштабированию (изменение всех значений и промежутков времени в одинаковое число раз не влияет на меру расстояния) и смещению (изменение сдвига одного ряда относительно другого не влияет на меру расстояния) - Shape-based distance (SBD).
<br>**Shape-based distance (SBD)** - мера которая тем меньше чем больше нормированное по автокорреляциям рядов максимальное (по всем сдвигам одного ряда относительно другого) значение кросс-корреляции рядов:
<p align="center">
<img src="img/SBD.jpg">
</p>
В k-means центроид (центр кластера) вычисляется как средняя последовательность из набора последовательностей — вычисление каждой координаты средней последовательности как среднее арифметическое соответствующих координат всех последовательностей.
<br>
В k-Shape вычисление центроида является задачей оптимизации, цель которой состоит в том, чтобы найти такую последовательность, что сумма квадратов расстояний (по мере SBD) до всех других последовательностей временных рядов будет максимальной:
<p align="center">
<img src="img/centr.jpg" width=600>
</p>
<br>
Алгоритм кластеризации:

На каждой итерации k-Shape выполняет два шага: 
- на этапе назначения алгоритм для каждого ряда обновляет принадлежность кластеру, сравнивая каждый временной ряд со всеми вычисленными центроидами и назначая каждый временной ряд кластеру ближайшего центроида, используется мера расстояния SBD для определения принадлежности кластеру 
- на этапе уточнения центроиды кластеров обновляются, чтобы отразить изменения в составе кластеров на предыдущем этапе.

Алгоритм повторяет эти два шага до тех пор, пока либо не произойдет никаких изменений в составе кластеров, либо не будет достигнуто максимально допустимое количество итераций.<br><br>


## Оценка кластеризации и визуализация


- [График коэффициента силуэта](#график-коэффициента-силуэта-и-визуализация-кластеров-в-2-мерном-пространстве)
- [Визуализация кластеров в 2-мерном пространстве](#график-коэффициента-силуэта-и-визуализация-кластеров-в-2-мерном-пространстве)
- [Графики метрик качества кластеризации](#графики-метрик-качества-кластеризации)
- [Визуализация кластеров в 2-мерном пространстве](#визуализация-кластеров-в-2-мерном-пространстве)
- [Сгруппированные по кластерам графики рядов](#сгруппированные-по-кластерам-графики-рядов)<br><br>

### График коэффициента силуэта и визуализация кластеров в 2-мерном пространстве
Для каждой кластеризации (по количеству кластеров) строится график значения коэффициента силуэта для всех точек каждого кластера: точки кластера расположены по вертикальной оси, правый край отрезка показывает значение коэффициента силуэта для точки (левый график).

Пунктирная линия - среднее (по всей выборке) значение коэффициента силуэта.

Точки в 2-мерном пространстве (размерность эмбеддингов рядов) окрашенные в цвет своего кластера (правый график)

<p align="center">
<img src="img/plot_silhouette.jpg">
</p>


### Графики метрик качества кластеризации
Графики метрик качества кластеризации (коэффициент силуэта, calinski harabasz score, davies bouldin score) для заданной частоты в зависимости от количества кластеров: 
<br><br>
<p align="center">
<img src="img/cluster_count_metrics.jpg">
</p>


### Визуализация кластеров в 2-мерном пространстве
Визуализация кластеров в 2-мерном пространстве (размерность эмбеддингов рядов) для определенной частоты и кластеризации:
<br><br>
<p align="center">
<img src="img/visualization.jpg">
</p>


### Сгруппированные по кластерам графики рядов
Графики рядов, сгруппированных по кластерам (столбцам), для заданной кластеризации (число кластеров), модели, частоты и периода:
<br><br>
<p align="center">
<img src="img/clusters.jpg">
</p>
