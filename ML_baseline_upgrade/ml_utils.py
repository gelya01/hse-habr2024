from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import random
from typing import Dict, List, Union, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
random.seed(42)
np.random.seed(42)
pd.options.mode.chained_assignment = None


# Функция для предсказания метрик precision, recall, f1-score и hamming-loss
def calculate_metrics(
    y_test: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    average: str = '',
    zero_division: int = 0
) -> Dict[str, float]:
    """
    Вычисление метрик precision, recall, f1-score и hamming-loss

    Параметры:
        y_test: Истинные значения
        y_pred: Предсказанные значения
        average: Метод усреднения для метрик precision, recall и f1-score
                       Может быть 'micro', 'macro'или 'weighted'
        zero_division: Обработка деления на ноль в метриках (0 или 1).

    Возвращает:
        Dict: Словарь с метриками
    """
    precision = round(precision_score(y_test, y_pred, average=average,
                                      zero_division=zero_division), 4)
    recall = round(recall_score(y_test, y_pred, average=average, zero_division=zero_division), 4)
    f1 = round(f1_score(y_test, y_pred, average=average, zero_division=zero_division), 4)
    hamming = round(hamming_loss(y_test, y_pred), 4)

    metrics = {'Precision': precision, 'Recall': recall,
               'F1-Score': f1, 'Hamming Loss': hamming}
    return metrics


# Функция предобработка датасета
def df_preprocess(df: pd.DataFrame) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    VarianceThreshold,
    Dict[int, str],
    MultiLabelBinarizer,
    pd.Series,
    Dict[int, str]
]:
    """
    Предобработка датасета: фильтрация, объединение текстовых токенов,
    кодировка меток(хабов) и преобразование рейтингов.

    Параметры:
        df (pd.DataFrame): Исходный DataFrame.

    Возвращает:
        Tuple[
            pd.DataFrame,                # Предобработанный DataFrame
            pd.DataFrame,                # Матрица многометочных данных с сокращенными метками
            VarianceThreshold,           # Объект для отбора по порогу дисперсии
            Dict[int, str],              # Отображение индексов в названия меток
            MultiLabelBinarizer,         # Объект MultiLabelBinarizer
            pd.Series,                   # Преобразованные рейтинги в формате Series
            Dict[int, str]               # Обратное отображение для рейтингов
        ]
    """
    # Уберем слишком короткие (неинформативные) статьи
    df = df[df['text_length'] > 100].copy()

    # Объединяем текстовые токены в единую строку
    df['text_combined'] = df['text_tokens'] + ' ' + df['title_tokens'] + ' ' + df['tags_tokens']
    df = df.drop(columns=['tags_tokens', 'title_tokens', 'text_tokens']).copy()

    # Извлекаем уникальные метки из тегов
    unique_labels = set()
    df['hubs'].str.split(', ').apply(unique_labels.update)  # Собираем уникальные метки

    # Маппинг и обратный маппинг хабов
    label_to_index = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    index_to_label = {v: k for k, v in label_to_index.items()}

    # Преобразуем строки в списки индексов
    df['hubs_encoded'] = df['hubs'].apply(
        lambda x: [label_to_index[label] for label in x.split(', ')])

    # Преобразование hubs_ecoded в формат матрицы с уникальными метками хабов
    mlb = MultiLabelBinarizer()
    y_multi = mlb.fit_transform(df['hubs_encoded'])

    # Удаляем метки, которые встречаются в менее чем 1% случаев
    selector = VarianceThreshold(threshold=0.01)
    y_multi_reduced = selector.fit_transform(y_multi)

    # Маппинг рейтинга
    y_rating = df['rating_level']
    rating_mapping = {'very negative': -2, 'negative': -1, 'neutral': 0,
                      'positive': 1, 'very positive': 2}

    y_rating = y_rating.map(rating_mapping)
    inverse_rating_mapping = {value: key for key, value in rating_mapping.items()}

    return (df, y_multi_reduced, selector,
            index_to_label, mlb,
            y_rating, inverse_rating_mapping)


# Функция предикта
def predict_func(
    df: pd.DataFrame,
    selector: SelectorMixin,
    index_to_label: Dict[int, str],
    mlb: MultiLabelBinarizer,
    inverse_rating_mapping: Dict[int, str],
    model_hubs: Pipeline,
    model_rating: Pipeline
) -> pd.DataFrame:
    """
    Функция предсказания меток (хабов) и рейтингов на основе текстовых данных статьи.

    Параметры:
        df (pd.DataFrame): Исходный DataFrame с текстовыми данными.
        selector (SelectorMixin): Объект для выбора значимых признаков (маска).
        index_to_label (Dict[int, str]): Словарь для преобразования индексов в метки.
        mlb (MultiLabelBinarizer): Объект для обратного преобразования многометочных предсказаний.
        inverse_rating_mapping (Dict[int, str]): Словарь для обратного преобразования рейтингов.
        model_hubs (Any): Модель для предсказания хабов.
        model_rating (Any): Модель для предсказания рейтингов.

    Возвращает:
        pd.DataFrame: DataFrame с исходными данными и предсказанными значениями хабов и рейтингов.
    """
    # Хабы
    X = df['text_combined']
    y_pred_h = model_hubs.predict(X)

    # Маска выбранных признаков
    mask = selector.get_support()

    # Создание матрицы исходной формы и возвращение убранных столбцов
    y_pred_restored = np.zeros((y_pred_h.shape[0], len(mask)), dtype=int)
    y_pred_restored[:, mask] = y_pred_h

    # Восстанавливаем сами списки меток
    original_hubs = mlb.inverse_transform(y_pred_restored)

    predicted_hubs = []
    for sample_indexes in original_hubs:
        hubs = [index_to_label[idx] for idx in sample_indexes]
        predicted_hubs.append(hubs)

    # Рейтинг
    y_pred_r = model_rating.predict(X)
    pred_text = [inverse_rating_mapping[label] for label in y_pred_r]

    # Преобразование предиктов хабов и рейтинга в DataFrame
    hubs_df = pd.DataFrame({'predicted_hubs':
                            [item[0] if item else '-' for item in predicted_hubs]}).reset_index()
    rating_df = pd.DataFrame(pred_text, columns=['predicted_rating']).reset_index()
    df_fin = pd.concat([df[['url', 'text_combined', 'hubs', 'rating_level']].reset_index().copy(),
                        hubs_df, rating_df], axis=1)

    return df_fin


# Функция выделения категории рейтинга
def rating_func(row: Series, pos_d: Series, neg_d: Series) -> str:
    """
    Определяет категорию рейтинга на основе квартилей положительных и отрицательных значений.

    Параметры:
        row: Строка DataFrame с полем 'rating_new'
        pos_d: Статистика описательных данных для положительных значений рейтинга
        neg_d: Статистика описательных данных для отрицательных значений рейтинга
    Возвращает:
        Категория рейтинга в виде строки
    """
    rate = row['rating_new']

    if pos_d['25%'] >= rate >= neg_d['75%']:
        return 'neutral'
    elif pos_d['75%'] >= rate > pos_d['25%']:
        return 'positive'
    elif rate > pos_d['75%']:
        return 'very positive'
    elif neg_d['75%'] > rate >= neg_d['25%']:
        return 'negative'
    elif neg_d['25%'] > rate:
        return 'very negative'


# Функция классификации рейтинга на категории
def categorize_ratings(sample_df: DataFrame) -> Series:
    """
    Классифицирует рейтинги в DataFrame на категории на основе квартилей положительных и отрицательных значений

    Параметры:
        sample_df: DataFrame со столбцом 'rating_new'

    Возвращает:
        Series с категориями рейтинга
    """
    neg_d = sample_df[sample_df['rating_new'] < 0]['rating_new'].describe()
    pos_d = sample_df[sample_df['rating_new'] > 0]['rating_new'].describe()

    return sample_df.apply(lambda row: rating_func(row, pos_d, neg_d), axis=1)


# Функция построения кривых обучения
def plot_learning_curve(estimator: BaseEstimator,
                        X: pd.DataFrame,
                        y: np.ndarray,
                        cv: int,
                        scoring: str):
    """
    Строит график кривой обучения для заданной модели

    Функция вычисляет значения метрик на обучающих и валидационных выборках с использованием
    кросс-валидации, усредняет полученные значения и строит график, показывающий зависимость
    метрики от числа обучающих примеров.

    Параметры:
    estimator : Модель
    X : Матрица признаков обучающей выборки.
    y : Вектор целевых значений.
    cv : Количество фолдов для кросс-валидации
    scoring : Метод оценки, применяемый для вычисления метрик (например, 'accuracy', 'r2', и т.д.).

    Возвращает:
        Функция строит и отображает график, не возвращая значения
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=7, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

    # Усредняем метрики
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
