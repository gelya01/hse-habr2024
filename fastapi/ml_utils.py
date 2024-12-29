import numpy as np
import pandas as pd
import random
from typing import Dict, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectorMixin
random.seed(42)
np.random.seed(42)
pd.options.mode.chained_assignment = None


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
                            [item[0] if item else '-' for item in predicted_hubs]})
    rating_df = pd.DataFrame(pred_text, columns=['predicted_rating']).reset_index(drop=True)
    df_fin = pd.concat([df[['url', 'hubs', 'rating_level']].reset_index(drop=True),
                        hubs_df.reset_index(drop=True), rating_df.reset_index(drop=True)], axis=1)

    return df_fin
