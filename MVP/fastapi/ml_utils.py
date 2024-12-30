import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectorMixin
from sklearn.model_selection import learning_curve
from joblib import cpu_count
from sklearn.model_selection import BaseCrossValidator
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
            pd.DataFrame,           # Предобработанный DataFrame
            pd.DataFrame,           # Матрица c сокращенными метками
            VarianceThreshold,      # Объект для отбора по порогу дисперсии
            Dict[int, str],         # Отображение индексов в названия меток
            MultiLabelBinarizer,    # Объект MultiLabelBinarizer
            pd.Series,              # Преобразованные рейтинги в формате Series
            Dict[int, str]          # Обратное отображение для рейтингов
        ]
    """
    # Уберем слишком короткие (неинформативные) статьи
    df = df[df['text_length'] > 100].copy()

    # Объединяем текстовые токены в единую строку
    df['text_combined'] = (df['text_tokens'] + ' '
                           + df['title_tokens'] + ' '
                           + df['tags_tokens'])
    df = df.drop(columns=['tags_tokens', 'title_tokens', 'text_tokens']).copy()

    # Извлекаем уникальные метки из тегов
    unique_labels = set()
    # Собираем уникальные метки
    df['hubs'].str.split(', ').apply(unique_labels.update)

    # Маппинг и обратный маппинг хабов
    label_to_index = {label: idx for idx,
                      label in enumerate(sorted(unique_labels))}
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
    inverse_rating_mapping = {value: key for key,
                              value in rating_mapping.items()}

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
    Функция предсказания меток (хабов) и рейтингов на основе текста статьи.

    Параметры:
        df: Исходный DataFrame с текстовыми данными.
        selector: Объект для выбора значимых признаков (маска).
        index_to_label: Словарь для преобразования индексов в метки.
        mlb: Объект для обратного преобразования многометочных предсказаний.
        inverse_rating_mapping: Словарь для обратного преобразования рейтингов.
        model_hubs: Модель для предсказания хабов.
        model_rating Модель для предсказания рейтингов.

    Возвращает:
        pd.DataFrame: DataFrame с исходными данными
                      и предсказанными значениями хабов и рейтингов.
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
                            [item[0] if item else '-'
                             for item in predicted_hubs]})
    rating_df = (pd.DataFrame(pred_text, columns=['predicted_rating'])
                 .reset_index(drop=True))
    df_fin = (pd.concat([df[['url', 'hubs', 'rating_level']]
                        .reset_index(drop=True),
                        hubs_df.reset_index(drop=True),
                        rating_df.reset_index(drop=True)], axis=1))

    return df_fin


def plot_learning_curve(
    estimator: BaseCrossValidator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    scoring: str,
    save_path: str,
    model_id: str,
    curve_type: str
) -> str:
    """
    Построение и сохранение графика кривой обучения для заданной модели.

    Параметры:
        estimator: Стратегия кросс-валидации (KFold или StratifiedKFold).
        scoring: Метрика оценки качества ('accuracy', 'f1').
        save_path: Путь для сохранения файла с графиком.
        model_id:Уникальный идентификатор модели.
        curve_type: Тип кривой ('validation', 'training').

    Возвращает:
        Путь к сохраненному файлу с графиком.
        Обучаемая модель (например, объект класса из sklearn).

    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=cpu_count() - 1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )

    # Усредняем метрики
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, "o-", color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g",
             label="Cross-validation score")

    plt.title(f"Learning Curve ({curve_type.capitalize()}) for {model_id}")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid()

    # Сохранение графика в файл
    filename = f"learning_curve_{model_id}_{curve_type}.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, format="png")
    plt.close()  # Закрываем фигуру, чтобы не копились лишние
    return filename
