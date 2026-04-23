# Лабораторная работа №5
## Регрессия с применением Scikit-Learn — Предсказание цен на недвижимость

**Выполнил:** Лесницкий Александр Маркович P3121

---

## 1. Описание задачи

Задача — построить модель машинного обучения для прогнозирования стоимости домов в **округе Кинг (штат Вашингтон, США)** на основе их характеристик. Это задача **регрессии**: на выходе модели — непрерывное числовое значение (цена дома).

**Датасет:** [Kaggle — House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction)  
Период данных: май 2014 — май 2015. Всего **21 613 наблюдений**, 21 переменная.

Доля обучающей выборки — **70%** (15 129 объектов), тестовой — **30%** (6 484 объекта).

**Признаки датасета:**

| Столбец | Описание |
|---|---|
| Целевая.Цена | **Целевая переменная:** цена продажи дома |
| Жилая площадь | Общая жилая площадь (кв. фут) |
| Оценка риелтора | Оценка состояния объекта риелтором |
| Широта | Широта расположения объекта |
| Спальни | Количество спален |
| Количество этажей | Этажность дома |
| Год реновации | Год последней реновации (0 = не проводилась) |
| ... | 20 признаков итого |

---

## 2. Ход работы

### 2.1. Загрузка и анализ данных

Данные загружены из файлов Excel с помощью `pd.read_excel()`.

```python
training_data = pd.read_excel('predict_house_price_training_data.xlsx')
test_data = pd.read_excel('predict_house_price_test_data.xlsx')
```

Метод `info()` подтвердил отсутствие пропусков: все 15 129 строк заполнены. Типы данных — `int64` и `float64`.

### 2.2. Предобработка данных

Целевая переменная отделена от признаков:

```python
target_variable_name = 'Целевая.Цена'
training_values = training_data[target_variable_name]
training_points = training_data.drop([target_variable_name], axis=1)

test_values = test_data[target_variable_name]
test_points = test_data.drop(target_variable_name, axis=1)
```

### 2.3. Обучение базовых моделей

Обучены две модели из библиотеки `scikit-learn`:

**Линейная регрессия:**
```python
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(training_points, training_values)
```

**Случайный лес:**
```python
random_forest_model = ensemble.RandomForestRegressor()
random_forest_model.fit(training_points, training_values)
```

### 2.4. Оценка базовых моделей

Метрики качества для задачи регрессии:
- **MAE** (Mean Absolute Error) — средняя абсолютная ошибка
- **RMSE** (Root Mean Squared Error) — корень из средней квадратичной ошибки; сильнее штрафует за большие ошибки

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_lr  = mean_absolute_error(test_values, test_predictions_linear)
rmse_lr = mean_squared_error(test_values, test_predictions_linear) ** 0.5

mae_rf  = mean_absolute_error(test_values, test_predictions_forest)
rmse_rf = mean_squared_error(test_values, test_predictions_forest) ** 0.5
```

**Результаты:**

| Модель | MAE | RMSE |
|---|---|---|
| Линейная регрессия | ~130 000 | ~200 000 |
| Random Forest (по умолчанию) | ~75 000 | ~130 000 |

Случайный лес значительно точнее линейной регрессии. На scatter-графике точки модели случайного леса плотнее прилегают к диагонали «идеального предсказания».

---

## 3. Анализ важности признаков

Обученная модель Random Forest позволяет оценить вклад каждого признака через `feature_importances_`:

```python
feature_importance = pd.DataFrame({
    'Название признака': training_points.keys(),
    'Важность признака': random_forest_model.feature_importances_
})
feature_importance.sort_values(by='Важность признака', ascending=False)
```

**Топ-5 наиболее важных признаков:**
1. Оценка риелтора (~31%)
2. Жилая площадь (~29%)
3. Широта (~17%)
4. Год постройки (~7%)
5. Количество ванных комнат (~3%)

**Наименее значимые признаки:**
- Год реновации — содержит 14 490 нулей из 15 129 строк, фактически неинформативен
- Количество этажей
- Спальни

Широта попала в топ, потому что центр Сиэтла расположен на севере округа — чем выше широта, тем ближе к центру города и дороже недвижимость.

---

## 4. Самостоятельная работа

### 4.1. Удаление незначащих признаков

Из обучающей и тестовой выборок исключены три наименее важных признака: `Год реновации`, `Спальни`, `Количество этажей`.

```python
drop_cols = ['Год реновации', 'Спальни', 'Количество этажей']

train_points = training_data.drop([target_variable_name] + drop_cols, axis=1)
train_values = training_data[target_variable_name]

test_points = test_data.drop([target_variable_name] + drop_cols, axis=1)
test_values  = test_data[target_variable_name]
```

После удаления признаков качество **незначительно улучшилось** для Random Forest — модель перестала обращать внимание на «шумовые» столбцы. Линейная регрессия практически не изменила результат, поскольку незначащие коэффициенты и без того стремились к нулю.

### 4.2. Подбор параметров Random Forest

Исследованы ключевые гиперпараметры модели:

- `n_estimators` — количество деревьев (больше = точнее, но медленнее)
- `max_depth` — максимальная глубина дерева
- `min_samples_leaf` — минимальное число объектов в листе (регуляризация)
- `max_features` — доля признаков при построении каждого дерева

```python
model_forest = ensemble.RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_leaf=1,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)
model_forest.fit(train_points, train_values)
pred_forest = model_forest.predict(test_points)

RMSE = mean_squared_error(test_values, pred_forest) ** 0.5
MAE  = mean_absolute_error(test_values, pred_forest)
print(f"RandomForest (tuned) — MAE: {MAE:.2f}, RMSE: {RMSE:.2f}")
```

Увеличение `n_estimators` до 1000 и подбор `max_features=0.7` снизили RMSE примерно на **8–12%** относительно модели с параметрами по умолчанию.

### 4.3. Исследование дополнительных моделей

#### Gradient Boosting Regressor (sklearn)

```python
from sklearn.ensemble import GradientBoostingRegressor

model_gb = GradientBoostingRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    max_features=0.3,
    min_samples_leaf=5,
    random_state=42
)
model_gb.fit(train_points, train_values)
pred_gb = model_gb.predict(test_points)
```

#### MLP Neural Network

Нейронная сеть требует предварительного масштабирования признаков:

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_points)
test_scaled  = scaler.transform(test_points)

model_nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model_nn.fit(train_scaled, train_values)
pred_nn = model_nn.predict(test_scaled)
```

#### ElasticNet

Регуляризованная линейная регрессия, объединяющая Lasso и Ridge:

```python
from sklearn.linear_model import ElasticNet

model_elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
model_elastic.fit(train_points, train_values)
pred_elastic = model_elastic.predict(test_points)
```

#### XGBoost

```python
from xgboost import XGBRegressor

model_xgb = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42
)
model_xgb.fit(train_points, train_values)
pred_xgb = model_xgb.predict(test_points)
```

### 4.4. Итоговое сравнение всех моделей

| Модель | MAE | RMSE | Примечание |
|---|---|---|---|
| Линейная регрессия | ~130 000 | ~200 000 | Базовый ориентир |
| Random Forest (по умолчанию) | ~75 000 | ~130 000 | Стандартные параметры |
| **Random Forest (tuned)** | **~68 000** | **~118 000** | n_estimators=1000, max_features=0.7 |
| Gradient Boosting | ~70 000 | ~120 000 | learning_rate=0.01 |
| MLP Neural Network | ~90 000 | ~145 000 | Хуже на небольших данных |
| ElasticNet | ~125 000 | ~195 000 | Слабее Random Forest |
| **XGBoost** | **~65 000** | **~112 000** | Лучший результат |

**Вывод:** Лучшую точность показал XGBoost. Настроенный Random Forest также превзошёл базовую версию. Нейронная сеть уступает ансамблевым методам на данном объёме данных. ElasticNet показал результат, близкий к обычной линейной регрессии.

Ключевые факторы улучшения качества:
1. Удаление неинформативных признаков (`Год реновации`)
2. Увеличение числа деревьев (`n_estimators=1000`)
3. Применение градиентного бустинга вместо простого усреднения

---

## 5. Интеграция модели с веб-сервисом (FastAPI)

Пошаговый алгоритм развёртывания обученной регрессионной модели в виде REST API:

**Шаг 1. Сохранение модели и препроцессора**

После обучения сериализуем объекты с помощью `joblib`:

```python
import joblib
joblib.dump(model_xgb, 'house_price_model.pkl')
# если нужна нормализация:
joblib.dump(scaler, 'scaler.pkl')
```

**Шаг 2. Описание схемы входных данных (Pydantic)**

```python
from pydantic import BaseModel

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    zipcode: int
    lat: float
    long: float
```

**Шаг 3. Создание FastAPI-приложения**

```python
from fastapi import FastAPI
import joblib, numpy as np

app = FastAPI()
model = joblib.load('house_price_model.pkl')

@app.post("/predict")
def predict_price(data: HouseFeatures):
    features = np.array([[
        data.bedrooms, data.bathrooms, data.sqft_living,
        data.sqft_lot, data.floors, data.waterfront,
        data.view, data.condition, data.grade,
        data.sqft_above, data.sqft_basement, data.yr_built,
        data.zipcode, data.lat, data.long
    ]])
    predicted_price = model.predict(features)[0]
    return {
        "predicted_price_usd": round(float(predicted_price), 2)
    }
```

**Шаг 4. Запуск сервера**

```bash
pip install fastapi uvicorn joblib xgboost
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Шаг 5. Проверка через Swagger UI**

FastAPI автоматически генерирует интерактивную документацию по адресу `http://localhost:8000/docs` — можно тестировать запросы прямо в браузере.

**Шаг 6. Контейнеризация (Docker)**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Шаг 7. Деплой в облако**

Готовый Docker-образ загружается в облачный реестр (GCP Artifact Registry, Yandex Container Registry) и запускается в виде сервиса (Cloud Run, Yandex Serverless Containers). При необходимости настраивается периодическое переобучение по актуальным данным.

---

## 6. Выводы

- Задача предсказания цены на недвижимость является задачей **регрессии**; основные метрики — MAE и RMSE.
- Random Forest значительно точнее линейной регрессии за счёт нелинейного моделирования зависимостей.
- Анализ важности признаков выявил, что ключевыми предикторами являются **оценка риелтора**, **жилая площадь** и **географическое положение (широта)**.
- Удаление неинформативных признаков (`Год реновации`) дало небольшое улучшение качества.
- Настройка гиперпараметров Random Forest (`n_estimators=1000`, `max_features=0.7`) снизила RMSE на ~10%.
- **XGBoost** показал наилучший результат среди всех исследованных моделей.
- Обученная модель легко оборачивается в REST API с помощью FastAPI и может быть развёрнута в облаке.

---

## Ссылки

- [GitHub](https://github.com/fafakaj/Python_labs_II_sem/tree/main/Pylab5)
- [Ноутбук на Google Colab](https://colab.research.google.com/drive/1W9i7C_JfV5mrgqikVs0KaR3-L1d7gXy9?usp=sharing)
- [Kaggle — House Sales in King County](https://www.kaggle.com/harlfoxem/housesalesprediction)
- [Habr — Обзор моделей регрессии](https://habr.com/ru/company/mailru/blog/513842/)
- [Habr — Продвинутые методы регрессии](https://habr.com/ru/companies/ods/articles/645887/)
- [Документация scikit-learn](https://scikit-learn.org/)
- [XGBoost документация](https://xgboost.readthedocs.io/)