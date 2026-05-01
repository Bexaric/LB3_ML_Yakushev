# Методы искусственного интеллекта в мехатронике и робототехнике

**Номер лабораторной:** 3 
**Вариант:** 20  
**Студент:** Якушев Никита Евгеньевич  
**Группа:** 8EM51  
**Преподаватель:** Александр Павловский
**Версия Python:** Python 3.10.10   

## 1. Цель и задачи работы

### Цель работы
- Приобретение практических навыков в решении одной из актуальных задач в построении интеллектуальной СУ автономной робототехнической системы, функционирующей в условиях физически неоднородной среды;
- Практическое изучение особенностей и способов решения задачи несбалансированной классификации, а также расширение инструментария разработчика в части работы с методами машинного обучения;
- Развитие исследовательских навыков работы с методом машинного обучения на примере прикладной задачи и реальной выборке данных;
- Развитие навыков программирования систем вычислительного интеллекта на Python.

### Задание
>Построить **нейросетевой классификатор**, протестированный и более-менее устойчивый к новым данным, а также приобрести опыт и понимание работы нейросетевого алгоритма в части влияния различных его параметров (числа слоев, нейронов в слое, числа итераций, функции активации, способа предобработки данных и т.п.) на качество получаемой модели.

## 2. Описание робота

Описание подмножеств переменных:
{N} = {N1, N2, N3} – показания энкодеров: обороты 3-х двигателей (об/мин);
{ω} = {ω1, ω2, ω3} – реальные скорости оборотов 3-х двигателей;
{I} = {I1, I2, I3} – токи потребления 3-х двигателей (А);
{g} = {gx, gy, gz} – показания гироскопа: угловая скорость по 3-м осям (рад/с);
{a} = {ax, ay, az} – ускорение по 3-м осям (м/с).

{V1} = { {N1, N2, N3}, {ω1, ω2, ω3}, {I1, I2, I3}, {gx, gy, gz}, {ax, ay, az} } – значения переменных, полученные непосредственно сенсорной системой робота;
{V2} = {Vx, Vy, Ω, Ix, Iy, I, I𝛴} – множество абсолютных параметров, полученных путем математических преобразований непосредственно измеренных переменных;
{V3} = {Tx, Ty, T, Tz} – множество относительных параметров, полученных путем математических преобразований подмножеств {V1} и {V2}:

Ниже приведены формулы для кинематики робота:
![Кинематика_робота](images/robot_kin.jpg)

<table border="1" cellpadding="4" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center;">
  <thead>
    <tr>
      <th>№</th>
      <th>Обработка данных</th>
      <th>Архитектура скрытых слоёв</th>
      <th>Алгоритм оптимизации</th>
      <th>Функция активации</th>
      <th>Макс. число итераций</th>
      <th>Коэфф. L2-регуляризации</th>
      <th>Accuracy ± std</th>
      <th>F1-score ± std</th>
    </tr>
  </thead>
  <tbody>
    <!-- Без дополнительной обработки -->
    <tr>
      <td>1</td>
      <td rowspan="3">Без дополнительной обработки</td>
      <td>(50,)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8182 ± 0.0000</td>
      <td>0.5250 ± 0.0433</td>
    </tr>
    <tr>
      <td>2</td>
      <td>(50, 20)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8295 ± 0.0254</td>
      <td>0.6055 ± 0.0134</td>
    </tr>
    <tr>
      <td>3</td>
      <td>(32, 32, 16)</td>
      <td>adam</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8523 ± 0.0254</td>
      <td>0.5917 ± 0.0759</td>
    </tr>
    <!-- Сортировка относительно целевого класса -->
    <tr>
      <td>4</td>
      <td rowspan="3">Сортировка относительно целевого класса</td>
      <td>(50,)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8182 ± 0.0000</td>
      <td>0.5250 ± 0.0433</td>
    </tr>
    <tr>
      <td>5</td>
      <td>(50, 20)</td>
      <td>adam</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8409 ± 0.0161</td>
      <td>0.5893 ± 0.0246</td>
    </tr>
    <tr>
      <td>6</td>
      <td>(64, 64, 32)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8409 ± 0.0161</td>
      <td>0.6213 ± 0.0406</td>
    </tr>
    <!-- С нормализацией MinMaxScaler -->
    <tr>
      <td>7</td>
      <td rowspan="3">С нормализацией <i>MinMaxScaler</i></td>
      <td>(50,)</td>
      <td>lbfgs</td>
      <td>tanh</td>
      <td>500</td>
      <td>0.0001</td>
      <td>0.8409 ± 0.0161</td>
      <td>0.6307 ± 0.0280</td>
    </tr>
    <tr>
      <td>8</td>
      <td>(50, 20)</td>
      <td>adam</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8466 ± 0.0188</td>
      <td>0.6412 ± 0.0278</td>
    </tr>
    <tr>
      <td>9</td>
      <td>(30, 30, 20)</td>
      <td>adam</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.8693 ± 0.0188</td>
      <td>0.6671 ± 0.0389</td>
    </tr>
    <!-- С нормализацией MinMaxScaler и балансировкой SMOTE -->
    <tr>
      <td><strong>10</strong></td>
      <td rowspan="3">С нормализацией <i>MinMaxScaler</i> и балансировкой <i>SMOTE</i></td>
      <td><strong>(150,)</strong></td>
      <td><strong>lbfgs</strong></td>
      <td><strong>tanh</strong></td>
      <td><strong>1000</strong></td>
      <td><strong>0.0001</strong></td>
      <td><strong>0.9536 ± 0.0118</strong></td>
      <td><strong>0.9558 ± 0.0109</strong></td>
    </tr>
    <tr>
      <td>11</td>
      <td>(40, 30)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.9357 ± 0.0071</td>
      <td>0.9370 ± 0.0075</td>
    </tr>
    <tr>
      <td>12</td>
      <td>(48, 48, 32)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.9393 ± 0.0186</td>
      <td>0.9426 ± 0.0171</td>
    </tr>
    <!-- С нормализацией MinMaxScaler и балансировкой ADASYN -->
    <tr>
      <td><strong>13</strong></td>
      <td rowspan="3">С нормализацией <i>MinMaxScaler</i> и балансировкой <i>ADASYN</i></td>
      <td><strong>(200,)</strong></td>
      <td><strong>lbfgs</strong></td>
      <td><strong>relu</strong></td>
      <td><strong>1000</strong></td>
      <td>0.0001</td>
      <td><strong>0.9522 ± 0.0191</strong></td>
      <td><strong>0.9532 ± 0.0176</strong></td>
    </tr>
    <tr>
      <td>14</td>
      <td>(40, 30)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.9412 ± 0.0104</td>
      <td>0.9418 ± 0.0091</td>
    </tr>
    <tr>
      <td>15</td>
      <td>(48, 48, 32)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>1000</td>
      <td>0.0001</td>
      <td>0.9265 ± 0.0104</td>
      <td>0.9286 ± 0.0101</td>
    </tr>
    <!-- С нормализацией MinMaxScaler и балансировкой SMOTE + Optuna -->
    <tr>
      <td><strong>16</strong></td>
      <td rowspan="3">С нормализацией <i>MinMaxScaler</i> и балансировкой <i>SMOTE</i> + <i>Optuna</i></td>
      <td><strong>(63,)</strong></td>
      <td><strong>lbfgs</strong></td>
      <td><strong>relu</strong></td>
      <td><strong>470</strong></td>
      <td><strong>0.0006169778</strong></td>
      <td><strong>0.9607 ± 0.0234</strong></td>
      <td><strong>0.9627 ± 0.0222</strong></td>
    </tr>
    <tr>
      <td>17</td>
      <td>(30, 19)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>600</td>
      <td>0.0035987656</td>
      <td>0.9536 ± 0.0186</td>
      <td>0.9559 ± 0.0173</td>
    </tr>
    <tr>
      <td><strong>18</strong></td>
      <td><strong>(30, 51, 49)</strong></td>
      <td><strong>lbfgs</strong></td>
      <td><strong>relu</strong></td>
      <td><strong>670</strong></td>
      <td><strong>0.0126914484</strong></td>
      <td><strong>0.9571 ± 0.0202</strong></td>
      <td><strong>0.9593 ± 0.0186</strong></td>
    </tr>
    <!-- С нормализацией MinMaxScaler и балансировкой ADASYN + Optuna -->
    <tr>
      <td><strong>19</strong></td>
      <td rowspan="3">С нормализацией <i>MinMaxScaler</i> и балансировкой <i>ADASYN</i> + <i>Optuna</i></td>
      <td><strong>(66,)</strong></td>
      <td><strong>lbfgs</strong></td>
      <td><strong>relu</strong></td>
      <td><strong>230</strong></td>
      <td><strong>0.0032129369</strong></td>
      <td><strong>0.9596 ± 0.0241</strong></td>
      <td><strong>0.9606 ± 0.0234</strong></td>
    </tr>
    <tr>
      <td>20</td>
      <td>(57, 30)</td>
      <td>lbfgs</td>
      <td>relu</td>
      <td>190</td>
      <td>0.0320593885</td>
      <td>0.9485 ± 0.0164</td>
      <td>0.9499 ± 0.0153</td>
    </tr>
    <tr>
      <td><strong>21</strong></td>
      <td><strong>(60, 60, 56)</strong></td>
      <td><strong>lbfgs</strong></td>
      <td><strong>tanh</strong></td>
      <td><strong>540</strong></td>
      <td><strong>0.0013197827</strong></td>
      <td><strong>0.9559 ± 0.0180</strong></td>
      <td><strong>0.9568 ± 0.0167</strong></td>
    </tr>
  </tbody>
</table>