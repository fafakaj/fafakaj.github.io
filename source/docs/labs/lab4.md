# Лабораторная работа №4
## Предсказание дефолта по кредиту (Credit Default Prediction)

**Выполнил:** Лесницкий Александр Маркович P3121  

---

## 1. Описание задачи

Задача — построить модель машинного обучения для предсказания кредитного дефолта. Дефолтом считается просрочка по кредиту более 90 дней. Задача является задачей **бинарной классификации**.

**Датасет:** [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

| Столбец | Описание |
|---|---|
| SeriousDlqin2yrs | **Целевая переменная:** просрочка 90+ дней |
| RevolvingUtilizationOfUnsecuredLines | Доля использованных лимитов по кредитным картам |
| age | Возраст заёмщика |
| DebtRatio | Отношение долговой нагрузки к доходу |
| MonthlyIncome | Доход в месяц |
| NumberOfOpenCreditLinesAndLoans | Кол-во открытых кредитов |
| NumberRealEstateLoansOrLines | Кол-во ипотек |
| NumberOfTime30-59DaysPastDueNotWorse | Просрочки 30–59 дней за 2 года |
| NumberOfTime60-89DaysPastDueNotWorse | Просрочки 60–89 дней за 2 года |
| NumberOfTimes90DaysLate | Просрочки 90+ дней за 2 года |
| NumberOfDependents | Кол-во иждивенцев |

---

## 2. Ход работы

### 2.1. Загрузка и анализ данных

Данные загружены с помощью `pd.read_csv()`. Обучающая выборка содержит **50 000 записей**, тестовая — **37 500 записей**.

Ключевые наблюдения из `describe()`:
- Целевая переменная **SeriousDlqin2yrs** имеет среднее ~0.06 → только у **6% клиентов** есть дефолт (выраженный дисбаланс классов).
- Столбец **MonthlyIncome** заполнен лишь у ~40 000 из 50 000 строк — есть пропуски.
- **NumberOfDependents** содержит много нулевых значений.

### 2.2. Предобработка данных

Пропуски заполнены **средними значениями по обучающей выборке** (`fillna(train_mean)`). Это стандартный подход, позволяющий избежать утечки данных из тестовой выборки.

```python
train_mean = training_data.mean()
training_data = training_data.fillna(train_mean)
test_data.fillna(train_mean, inplace=True)
```

Целевая переменная отделена от признаков с помощью `drop()`:

```python
training_values = training_data['SeriousDlqin2yrs']
training_points = training_data.drop('SeriousDlqin2yrs', axis=1)
```

### 2.3. Обучение базовых моделей

Обучены две модели из библиотеки `scikit-learn`:

**Логистическая регрессия:**
```python
logistic_regression_model = linear_model.LogisticRegression()
logistic_regression_model.fit(training_points, training_values)
```

**Случайный лес:**
```python
random_forest_model = ensemble.RandomForestClassifier(n_estimators=100)
random_forest_model.fit(training_points, training_values)
```

### 2.4. Оценка моделей

Для оценки качества использовались:
- **Матрица ошибок (Confusion Matrix)**
- **ROC-AUC** — основная метрика

Прогноз вероятностей получен через `predict_proba()`. Проанализировано влияние порога классификатора (0.3, 0.5, 0.7) на соотношение False Positive и False Negative.

При пороге **0.5** (по умолчанию) модель склонна занижать класс 1, так как он представлен значительно меньше. При снижении порога до **0.3** число выявленных дефолтов растёт, но увеличивается и число ложных срабатываний.

**Результаты ROC-AUC:**

| Модель | ROC-AUC |
|---|---|
| Логистическая регрессия | ~0.81 |
| Random Forest (n=100) | ~0.85 |
| Random Forest (n=300, balanced, depth=15) | ~0.86–0.87 |

---

## 3. Самостоятельная работа: исследование дополнительных моделей

### 3.1. Мотивация

Стандартные модели (Logistic Regression, Random Forest) показали ROC-AUC в диапазоне 0.81–0.87. Цель — превысить этот результат с помощью более мощных алгоритмов.

### 3.2. LightGBM

LightGBM (Light Gradient Boosting Machine) — алгоритм градиентного бустинга, разработанный Microsoft. Работает быстрее XGBoost за счёт гистограммного подхода и leaf-wise роста деревьев.

```python
import lightgbm as lgb

lgbm_model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.02,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    scale_pos_weight=9,   # компенсация дисбаланса классов
    random_state=42
)

lgbm_model.fit(training_points, training_values)
lgbm_proba = lgbm_model.predict_proba(test_points)
lgbm_roc_auc = roc_auc_score(test_values, lgbm_proba[:, 1])
print("LightGBM ROC-AUC:", lgbm_roc_auc)
```

**Результат:** ROC-AUC ≈ **0.865–0.875**

### 3.3. XGBoost

XGBoost — один из наиболее популярных алгоритмов для табличных данных в соревнованиях Kaggle.

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=9,
    eval_metric="auc",
    random_state=42,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5
)

xgb_model.fit(training_points, training_values)
xgb_proba = xgb_model.predict_proba(test_points)
xgb_roc_auc = roc_auc_score(test_values, xgb_proba[:, 1])
print("XGBoost ROC-AUC:", xgb_roc_auc)
```

**Результат:** ROC-AUC ≈ **0.867–0.878**

### 3.4. Итоговое сравнение моделей

| Модель | ROC-AUC | Примечание |
|---|---|---|
| Логистическая регрессия | ~0.81 | Базовый ориентир |
| Random Forest (n=100) | ~0.85 | Улучшение за счёт ансамбля |
| Random Forest (n=300, balanced) | ~0.87 | Учёт дисбаланса классов |
| **LightGBM** | **~0.872** | Лучший результат среди бустингов |
| **XGBoost** | **~0.870** | Сравнимо с LightGBM |

Модели градиентного бустинга превзошли Random Forest. Ключевой фактор улучшения — параметр `scale_pos_weight`, компенсирующий дисбаланс классов (93:7).

---

## 4. Современные алгоритмы классификации для кредитного скоринга

На основе анализа научных публикаций и Kaggle-исследований выделяются следующие актуальные подходы:

**Gradient Boosting (GBDT)**  
XGBoost, LightGBM и CatBoost являются фактическим стандартом для табличных данных в финансовой сфере. На соревновании Give Me Some Credit большинство лидеров использовали именно ансамблевые методы с бустингом. Ключевые преимущества: высокая точность, устойчивость к выбросам, встроенная обработка дисбаланса классов.

**Нейронные сети (MLP, TabNet, NODE)**  
В последние годы появились архитектуры, специально адаптированные для табличных данных (TabNet от Google, Neural Oblivious Decision Ensembles). Они конкурентны с GBDT при больших объёмах данных.

**Стекинг (Stacking) и блендинг ансамблей**  
Топовые решения на Kaggle комбинируют предсказания нескольких моделей мета-алгоритмом (чаще всего логистической регрессией или линейным SVM на втором уровне).

**Interpretable ML (SHAP, LIME)**  
В банковской сфере объяснимость модели не менее важна, чем точность. SHAP-значения используются для обоснования отказа в кредите регуляторам.

---

## 5. Интеграция модели с веб-сервисом (FastAPI)

Пошаговый алгоритм развёртывания обученной модели в виде REST API:

**Шаг 1. Сохранение модели**  
После обучения модель сериализуется на диск с помощью `joblib` или `pickle`:
```python
import joblib
joblib.dump(lgbm_model, 'credit_model.pkl')
```

**Шаг 2. Создание FastAPI-приложения**  
Создаётся файл `app.py` с описанием входной схемы данных (Pydantic) и эндпоинтом предсказания:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
model = joblib.load('credit_model.pkl')

class ClientData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfTimes90DaysLate: int
    NumberOfDependents: int

@app.post("/predict")
def predict(data: ClientData):
    features = np.array([[
        data.RevolvingUtilizationOfUnsecuredLines, data.age,
        data.DebtRatio, data.MonthlyIncome,
        data.NumberOfOpenCreditLinesAndLoans,
        data.NumberRealEstateLoansOrLines,
        data.NumberOfTime30_59DaysPastDueNotWorse,
        data.NumberOfTime60_89DaysPastDueNotWorse,
        data.NumberOfTimes90DaysLate, data.NumberOfDependents
    ]])
    proba = model.predict_proba(features)[0][1]
    return {"default_probability": round(proba, 4), "is_default": bool(proba > 0.5)}
```

**Шаг 3. Запуск сервера**  
```bash
pip install fastapi uvicorn joblib lightgbm
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Шаг 4. Тестирование через Swagger UI**  
FastAPI автоматически генерирует документацию по адресу `http://localhost:8000/docs`.

**Шаг 5. Контейнеризация (опционально)**  
Приложение упаковывается в Docker-контейнер для воспроизводимого развёртывания в облаке (GCP, AWS, Yandex Cloud).

**Шаг 6. Мониторинг и переобучение**  
В продакшне модель регулярно проверяется на data drift. При деградации качества запускается переобучение на актуальных данных.

---

## 6. Выводы

- Задача предсказания кредитного дефолта — классическая задача бинарной классификации с выраженным дисбалансом классов (~6% дефолтов).
- Обработка пропусков средними значениями и учёт дисбаланса через `scale_pos_weight` существенно повышают качество модели.
- Модели градиентного бустинга (LightGBM, XGBoost) превзошли Random Forest по метрике ROC-AUC.
- Обученную модель можно легко обернуть в REST API с помощью FastAPI и развернуть в облаке.
- Основная метрика оценки — **ROC-AUC**, поскольку она устойчива к дисбалансу классов.

---

## Ссылки

- [GitHub](https://github.com/fafakaj/Python_labs_II_sem/tree/main/Pylab4)
- [Ноутбук на Google Colab](https://colab.research.google.com/drive/1oVe346EypxiEY3xfHd19MdESrxJgwUoH?usp=sharing) 
- [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- [Документация scikit-learn](https://scikit-learn.org/)
- [LightGBM документация](https://lightgbm.readthedocs.io/)
- [XGBoost документация](https://xgboost.readthedocs.io/)