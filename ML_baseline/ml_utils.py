from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from typing import Dict, List, Union
import numpy as np


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
