import multiprocessing as mp
import os
from fastapi import FastAPI, HTTPException, UploadFile
import uvicorn
import pandas as pd
from typing import Dict, List, Any, Union
from pydantic import BaseModel
import joblib
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
import ml_utils as mu
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing.managers
from fastapi.staticfiles import StaticFiles

# Создаём папку logs, если её нет
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Логирование с ротацией
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ротация логов (максимальный размер 1MB, до 2 бэкапов)
log_file = os.path.join(log_dir, "app.log")
handler = RotatingFileHandler(
    filename=log_file,
    maxBytes=1_000_000,
    backupCount=2,
    encoding="utf-8"
)
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

# Создаём папку curves, если её нет (для графиков)
# И монтируем как статическую
curves_dir = "curves"
os.makedirs(curves_dir, exist_ok=True)
app.mount("/static/curves", StaticFiles(directory=curves_dir), name="curves")

logger.info("Старт приложения и запуск логирования")

# Глобальные хранилища:
# models - словарь, хранящий обученные модели (пайплайны) по их id
# current_model_id - id модели, которая в данный момент загружена для предикта
# предобработанные датасеты
models = {}
current_model_id = {'rating': None, 'hubs': None}
data_stor = {}

# Загрузка обученных моделей при старте приложения
# Предсказание хабов
try:
    with open("pipeline_h.pkl", "rb") as hubs_model:
        fit_pipeline_h = joblib.load(hubs_model)

    # Предсказание рейтинга
    with open("pipeline_r.pkl", "rb") as rate_model:
        fit_pipeline_r = joblib.load(rate_model)

    # Добавляем предобученные модели
    models["pretrained_hubs"] = {"type": "hubs", "model": fit_pipeline_h}
    models["pretrained_rating"] = {"type": "rating", "model": fit_pipeline_r}

    logger.info("Предобученные модели успешно загружены")

except Exception as e:
    logger.error(f"Не удалось загрузить предобученные модели: {e}")


# Необходимые классы
# Конфигурация для TfidfVectorizer
class TfidfConfig(BaseModel):
    params: Dict[str, Any]  # гиперпараметры для TfidfVectorizer


# Конфигурация для классификатора
class ClassifierConfig(BaseModel):
    type: str  # тип модели классификатора (LogisticRegression или SVC)
    params: Dict[str, Any]  # гиперпараметры классификатора


# Конфигурация для пайплайна
class PipelineConfig(BaseModel):
    id: str  # id пайплайна
    tfidf: TfidfConfig  # конфиг TfidfVectorizer
    classifier: ClassifierConfig  # конфиг классификатора


class PreprocessResponse(BaseModel):
    message: str


class FitResponse(BaseModel):
    message: str
    trained_models: List[str]


class ModelItem(BaseModel):
    id: str  # id модели
    type: str  # тип модели


class ModelListResponse(BaseModel):
    # Ответ при запросе списка моделей, возвращает массив объектов ModelItem
    models: List[ModelItem]


# Запрос на предикт
class PredictRequest(BaseModel):
    id: str  # id модели


# Ответ на предикт
class PredictionResponse(BaseModel):
    predictions: List[dict]


# Функция для создания пайплайна
def create_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Функция для создания пайплайнов из конфига
    """
    # Tfidf
    tfidf = TfidfVectorizer(**config.tfidf.params)

    # Выбор классификатора
    if config.classifier.type == "LogisticRegression":
        classifier = LogisticRegression(**config.classifier.params)
    elif config.classifier.type == "SVC":
        classifier = SVC(**config.classifier.params)
    else:
        raise ValueError(
            f"Неподдерживаемый тип классификатора: {config.classifier.type}")

    pipeline = Pipeline([('tfidf', tfidf),
                         ('clf', OneVsRestClassifier(classifier))])
    return pipeline


# Вспомогательная функция обучения
def _train_model_func(
    config_dict: Dict[str, Union[str, Dict]],
    X_hubs_list: List[str],
    X_rating_list: List[str],
    y_rating_list: List[Union[int, float]],
    y_hubs_list: List[List[int]],
    return_dict: multiprocessing.managers.DictProxy
) -> None:
    """
    Функция, которая запускается в отдельном процессе для обучения моделей
    Результат возвращается через return_dict
    """
    try:
        # Восстанавливаем Pydantic-модель из dict
        config_obj = PipelineConfig(**config_dict)

        logger.info(f"Процесс обучения модели '{config_obj.id}' начался")

        rating_pipeline = create_pipeline(config_obj)
        rating_pipeline.fit(X_rating_list, y_rating_list)

        hubs_pipeline = create_pipeline(config_obj)
        hubs_pipeline.fit(X_hubs_list, y_hubs_list)

        return_dict["rating_pipeline"] = rating_pipeline
        return_dict["hubs_pipeline"] = hubs_pipeline
        return_dict["error"] = None

    except Exception as e:
        logger.error(f"Ошибка в _train_model_func: {e}")
        return_dict["error"] = str(e)


# Проверка состояния сервера
@app.get("/")
def read_root():
    return {"message": "Мы работаем"}


# Предобработка данных
@app.post("/preprocess", response_model=PreprocessResponse,
          summary='Preprocess',
          description="Предобработка исходного датафрейма")
async def preprocess(file: UploadFile):
    logger.info("Вызов /preprocess")
    # Чтение содержимого файла
    try:
        contents = await file.read()
        df = pd.read_parquet(BytesIO(contents))
        logger.info(f"Датафрейм загружен, shape={df.shape}")
    except Exception as e:
        logger.error(f"Ошибка чтения файла в /preprocess: {e}")
        raise HTTPException(status_code=400,
                            detail=f"Ошибка при чтении файла: {e}")

    # Проверка на наличие необходимых столбцов
    required_columns = ['author', 'publication_date', 'hubs', 'comments', 'views', 'url',
                        'reading_time', 'individ/company', 'bookmarks_cnt', 'text_length',
                        'rating_new', 'text_pos_tags', 'tags_tokens', 'title_tokens',
                        'text_tokens']
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Не хватает столбцов в /preprocess: {missing_cols}")
        raise HTTPException(status_code=400,
                            detail=f"Нет столбцов: {missing_cols}")

    # Предобработка текстовых данных
    try:
        df['rating_level'] = mu.categorize_ratings(df)
        (df, y_multi_reduced, selector,
         index_to_label, mlb,
         y_rating, inverse_rating_mapping, indices) = mu.df_preprocess(df)
        logger.info("Данные успешно предобработаны")
    except Exception as e:
        logger.error(f"Ошибка при предобработке: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Ошибка при предобработке данных:{str(e)}")

    # Добавляеам предобработанные данные в хранилище
    data_stor["df"] = df
    data_stor["y_multi_reduced"] = y_multi_reduced
    data_stor["y_rating"] = y_rating
    data_stor["selector"] = selector
    data_stor["index_to_label"] = index_to_label
    data_stor["mlb"] = mlb
    data_stor["inverse_rating_mapping"] = inverse_rating_mapping
    data_stor['indices'] = indices

    return PreprocessResponse(message="Данные предобработаны")


# Fit
@app.post("/fit", response_model=FitResponse, summary="Fit",
          description="Обучает модель с переданными параметрами конфига.")
async def fit(config: str):
    """
    Если config.id == "pretrained", считаем, что пользователь хочет
    использовать уже загруженную модель и не обучаем ничего
    Иначе – обучаем новую модель
    """
    logger.info("Вызов /fit")
    # Преобразование конфигурации из строки JSON в объект Pydantic
    try:
        config_obj = PipelineConfig.model_validate_json(config)
        model_id = config_obj.id
        logger.info(f"Спарсили конфиг для модели: {model_id}")
        if model_id in models:
            logger.warning(f"Модель '{model_id}' уже существует")
            raise HTTPException(status_code=400,
                                detail=f"Модель '{model_id}' уже существует")
    except Exception as e:
        logger.error(f"Ошибка при парсинге конфига в /fit: {e}")
        raise HTTPException(status_code=400,
                            detail=f"Ошибка в конфигурации модели: {e}")

    # Извлечение X, y_rating и y_hubs
    df = data_stor['df']
    indices = data_stor['indices']
    X = df["text_combined"]
    X_hubs = X.iloc[indices].copy()
    X_rating = X.copy()
    y_rating = data_stor['y_rating']
    y_hubs = data_stor['y_multi_reduced']

    # Готовим dict для передачи в подпроцесс
    config_dict = config_obj.model_dump()

    # Используем Manager, чтобы вернуть результат
    manager = mp.Manager()
    return_dict = manager.dict()

    # Создаём процесс
    p = mp.Process(
        target=_train_model_func,
        args=(config_dict, list(X_hubs), list(X_rating),
              list(y_rating), list(y_hubs), return_dict)
    )

    p.start()
    p.join(timeout=10)  # Ждём 10 секунд

    if p.is_alive():
        logger.warning(
            f"Обучение модели прервалось '{model_id}' (дольше 10 секунд)")
        p.terminate()
        p.join()
        raise HTTPException(
            status_code=408,
            detail=(
                f"Обучение модели прервалось '{model_id}' (дольше 10 секунд)")
        )

    # Если процесс завершился до таймаута, проверяем ошибку
    if return_dict.get("error") is not None:
        logger.error(f"Ошибка при обучении: {return_dict['error']}")
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка при обучении: {return_dict['error']}"
        )

    # Сохраняем результаты
    rating_pipeline = return_dict["rating_pipeline"]
    hubs_pipeline = return_dict["hubs_pipeline"]

    model_id_rating = f"{model_id}_rating"
    model_id_hubs = f"{model_id}_hubs"

    models[model_id_rating] = {"type": "rating", "model": rating_pipeline}
    models[model_id_hubs] = {"type": "hubs", "model": hubs_pipeline}

    logger.info(
        f"Модели '{model_id_rating}' и '{model_id_hubs}' обучены и сохранены.")

    return FitResponse(
        message=f"Модели '{model_id}' успешно обучены и сохранены",
        trained_models=[model_id_rating, model_id_hubs]
    )


# Предикт
@app.post("/predict", response_model=PredictionResponse, summary="Predict",
          description="Выполняет предсказания на основе датасета")
async def predict(config: str):
    """
    Выполняет предсказания меток (хабов) и рейтингов
    Параметр config (JSON-строка) должен содержать 'id' (например,'pretrained')
    """
    logger.info("Вызов /predict")
    try:
        config_obj = PredictRequest.model_validate_json(config)
        logger.info(f"PredictRequest id={config_obj.id}")
    except Exception as e:
        logger.error(f"Ошибка при парсинге конфига в /predict: {e}")
        raise HTTPException(status_code=400,
                            detail=f"Ошибка в конфигурации предсказания: {e}")

    # Формируем идентификаторы моделей
    model_id = config_obj.id
    model_id_hubs = f"{model_id}_hubs"
    model_id_rating = f"{model_id}_rating"

    # Проверяем наличие обеих моделей
    if model_id_rating not in models:
        logger.error(f"Модель '{model_id_rating}' не найдена")
        raise HTTPException(status_code=404,
                            detail=(f"Модель '{model_id_rating}' не найдена"))

    if model_id_hubs not in models:
        logger.error(f"Модель '{model_id_hubs}' не найдена")
        raise HTTPException(status_code=404,
                            detail=f"Модель '{model_id_hubs}' не найдена")

    # Извлекаем модели и вспомогательные параметры
    model_info_hubs = models[model_id_hubs]
    model_info_rating = models[model_id_rating]

    model_hubs = model_info_hubs["model"]
    model_rating = model_info_rating["model"]

    df = data_stor['df']
    selector = data_stor['selector']
    index_to_label = data_stor['index_to_label']
    mlb = data_stor['mlb']
    irm = data_stor['inverse_rating_mapping']
    indices = data_stor['indices']

    if any(param is None for param in [selector, index_to_label,
                                       mlb, irm]):
        logger.error("Вспомогательные параметры отсутствуют в /predict")
        raise HTTPException(
            status_code=500,
            detail="Вспомогательные параметры отсутствуют в /predict")
    try:
        # Предикт
        result_df = mu.predict_func(model_id=model_id, df=df, selector=selector,
                                    index_to_label=index_to_label, mlb=mlb,
                                    inverse_rating_mapping=irm,
                                    indices=indices,
                                    model_hubs=model_hubs,
                                    model_rating=model_rating)
        logger.info("Предсказание успешно")
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка при выполнении предсказания: {str(e)}")

    # Преобразуем DataFrame в список словарей
    # (для корректной сериализации ответа в JSON)
    return PredictionResponse(predictions=result_df.to_dict(orient='records'))


# Список моделей
@app.get("/models", response_model=ModelListResponse, summary="List Models",
         description="Возвращает список всех обученных моделей.")
async def list_models():
    """
    Получение списка всех обученных моделей
    """
    logger.info("Вызов всех моделей в /models")
    model_list = [
        ModelItem(id=m_id, type=m_info["type"])
        for m_id, m_info in models.items()
    ]
    return ModelListResponse(models=model_list)


# Установка текущей модели
@app.post("/set",
          summary="Set active model",
          description="Устанавливает модель как текущую для предикта.")
async def set_model(model_id: str):
    """
    Устанавливает модель с данным id как активную (и для rating, и для hubs).
    """
    logger.info(f"Вызов /set с моделью {model_id}")
    rating_key = f"{model_id}_rating"
    hubs_key = f"{model_id}_hubs"

    if rating_key not in models:
        logger.error(f"{rating_key} не найден в /set")
        raise HTTPException(
            status_code=404,
            detail=f"Модель рейтинга '{rating_key}' не найдена.")

    if hubs_key not in models:
        logger.error(f"{hubs_key} не найден в /set")
        raise HTTPException(
            status_code=404,
            detail=f"Модель хабов '{hubs_key}' не найдена.")

    current_model_id["rating"] = rating_key
    current_model_id["hubs"] = hubs_key
    logger.info(f"Текущие модели для рейтинга={rating_key}, хабов={hubs_key}")

    return {"message": f"Текущая модель установлена на: {model_id}"}

if __name__ == "__main__":
    uvicorn.run("api_mvp:app", host="127.0.0.1", port=8000, reload=True)


# Построение кривых обучения
@app.post("/plot_learning_curve", summary="Plot Learning Curves",
          description="Генерирует и возвращает URL-адреса для графиков")
async def plt_curve(
    model_id: str,
    cv: int = 3,
    scoring: str = "f1_micro",
):
    """
    Генерирует кривую обучения сразу для двух пайплайнов модели (rating и hubs)
    и возвращает их в виде zip-архива.
    """
    logger.info(f"Вызов /plot_learning_curve для модели {model_id}")

    # Проверяем, что обе модели присутствуют
    model_rating_id = f"{model_id}_rating"
    model_hubs_id = f"{model_id}_hubs"
    if model_rating_id not in models or model_hubs_id not in models:
        err_msg = f"Модель '{model_id}' (rating/hubs) не найдена!"
        logger.error(err_msg)
        raise HTTPException(status_code=404, detail=err_msg)

    try:
        # Пайплайны
        model_rating_pipeline = models[model_rating_id]["model"]
        model_hubs_pipeline = models[model_hubs_id]["model"]

        # Данные
        df = data_stor["df"]
        indices = data_stor["indices"]
        X = df["text_combined"]
        X_hubs = X.iloc[indices].copy()
        X_rating = X.copy()
        y_rating = data_stor["y_rating"]
        y_hubs = data_stor["y_multi_reduced"]

        # Генерируем графики
        rating_filename = mu.plot_learning_curve(
            model_rating_pipeline, X_rating, y_rating, cv=cv, scoring=scoring,
            save_path=curves_dir, model_id=model_id, curve_type="rating"
        )
        hubs_filename = mu.plot_learning_curve(
            model_hubs_pipeline, X_hubs, y_hubs, cv=cv, scoring=scoring,
            save_path=curves_dir, model_id=model_id, curve_type="hubs"
        )

        # Формируем URL-адреса к графикам
        base_url = "http://127.0.0.1:8000"
        rating_url = f"{base_url}/static/curves/{rating_filename}"
        hubs_url = f"{base_url}/static/curves/{hubs_filename}"

        logger.info(f"Графики сохранены: {rating_filename}, {hubs_filename}")

        return {
            "rating_curve_url": rating_url,
            "hubs_curve_url": hubs_url
        }

    except Exception as e:
        err_msg = f"Ошибка при построении кривых обучения: {str(e)}"
        logger.error(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)

# Если запускаемся через скрипт main_mvp.py
# if __name__ == "__main__":
#    uvicorn.run("api_mvp:app", host="127.0.0.1", port=8000, reload=True)
