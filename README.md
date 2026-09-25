# Titanic — Survival Prediction (учебный ML-проект)

Учебный проект по классическому датасету [Kaggle: Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Цель — предсказать, выжил ли пассажир, на основе демографических и билетных признаков, и пройти полный ML-пайплайн: от EDA до подбора гиперпараметров и анализа ошибок.

## Содержание

- [О проекте](#о-проекте)
- [Структура репозитория](#структура-репозитория)
- [Как запустить](#как-запустить)
- [Пайплайн проекта](#пайплайн-проекта)
- [Результаты](#результаты)
- [Стек](#стек)

## О проекте

В ноутбуке пошагово разобраны:

1. **EDA** — распределения признаков, зависимость выживаемости от класса, пола, порта посадки, корреляции числовых признаков.
2. **Feature engineering** — извлечение титула из имени (`Title`), признак наличия семьи на борту (`HasFamily`).
3. **Препроцессинг** — `ColumnTransformer` с импутацией и стандартизацией числовых признаков, `OneHotEncoder` для категориальных.
4. **Baseline-сравнение моделей** — `LogisticRegression`, `SVC`, `DecisionTree`, `RandomForest`, `KNN`, `Dummy` через кросс-валидацию (`StratifiedKFold`), сравнение с/без масштабирования признаков.
5. **Подбор гиперпараметров** — `GridSearchCV` и `RandomizedSearchCV` по нескольким моделям.
6. **Ансамблирование** — перебор комбинаций моделей в `VotingClassifier` (soft/hard voting).
7. **Финальное сравнение моделей** на отложенной выборке (accuracy, precision, recall, ROC-AUC, log loss).
8. **Анализ ошибок** — confusion matrix, разбор ошибочных предсказаний по полу, классу, возрастной группе, самые "уверенные" ошибки модели.
9. **Feature importance** — permutation importance для лучшей модели.
10. **Предсказание на тестовой выборке Kaggle** и сохранение submission-файлов для нескольких моделей.

## Структура репозитория

Предлагаемая структура (замените под свою, если она отличается от той, что у вас на скрине):

```
titanic-ml/
├── data/
│   ├── train.csv
│   └── test.csv
├── submissions/
│   ├── submission_svc.csv
│   ├── submission_rf.csv
│   └── submission_vot.csv
├── notebooks/
│   └── titanic_survival_prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Как запустить

```bash
git clone <ссылка-на-репозиторий>
cd titanic-ml
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/titanic_survival_prediction.ipynb
```

Данные `train.csv` / `test.csv` нужно скачать со страницы [соревнования на Kaggle](https://www.kaggle.com/competitions/titanic/data) и положить в `data/`.

## Пайплайн проекта

```
Сырые данные → Feature engineering (Title, HasFamily)
             → ColumnTransformer (impute + scale / one-hot)
             → Сравнение baseline-моделей (CV)
             → GridSearch / RandomizedSearch
             → VotingClassifier (перебор комбинаций)
             → Финальная модель → анализ ошибок → submission.csv
```

## Результаты

Метрика на отложенной выборке (ROC-AUC) и итоговый public-score на Kaggle:

| Модель | ROC-AUC (test) | Accuracy (test) | Kaggle score |
|---|---|---|---|
| SVC | 0.8747 | 0.8747 | 0.76794 |
| Random Forest | 0.8492 | 0.8492 | 0.77511 |
| Voting Ensemble (RF + DT) | 0.7345 | 0.7455 | 0.76555 |

Ключевые выводы:
- Лучшая модель по ROC-AUC — **SVC**; по accuracy на отложенной выборке — **Random Forest**.
- `RandomForest` и `DecisionTree` сильнее всего склонны к переобучению; наиболее стабильна `LogisticRegression`.
- Масштабирование признаков критично для `SVC` и `KNN`.
- Ансамбль (`VotingClassifier`) не улучшил результат лучшей одиночной модели.
- Модель чаще всего ошибается на пассажирах 3 класса в возрасте 0–12 лет.
- Самый важный признак — `Sex`, за ним — `Pclass` и `Age`.

## Стек

`Python`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

*Учебный проект, выполнен в рамках практики по машинному обучению.*
