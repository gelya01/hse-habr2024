import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import requests
import json
import logging
from logging.handlers import RotatingFileHandler
import altair as alt
from nltk import FreqDist
from collections import Counter
import plotly.express as px
import itertools
from typing import List

# Логирование
# Создаём папку logs, если её нет
import os
os.makedirs("logs", exist_ok=True)

# Инициализируем логгер
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ротация логов (максимальный размер 1MB, до 5 бэкапов)
log_file = os.path.join("logs", "streamlit_app.log")
handler = RotatingFileHandler(
    filename=log_file,
    maxBytes=1_000_000,
    backupCount=5,
    encoding="utf-8"
)
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Настройка приложения
st.set_page_config(page_title="Habr ML App", layout="wide")
# Если запуск через скрипт main_mvp.py
# API_URL = "http://127.0.0.1:8000"

#  Если запуск через контейнер в Docker
API_URL = "http://fastapi:8000"


# Взаимодействие с API
# Предобработка данных
def preprocess_dataset(file_bytes: bytes):
    """
    Отправка датасета в эндпоинт /preprocess для предобработки.
    """
    endpoint = f"{API_URL}/preprocess"
    files = {"file": ("dataset.parquet", file_bytes,
                      "application/octet-stream")}
    response = requests.post(endpoint, files=files)
    return response


# Фит
def fit_model(config_json_str: str):
    """
    Обучение модели - отправляем JSON-строку в /fit.
    """
    endpoint = f"{API_URL}/fit"
    # Параметр config передаётся как обычная строка
    resp = requests.post(endpoint, params={"config": config_json_str})
    return resp


# Список моделей
def list_models():
    """
    Получение списка всех обученных моделей из /models.
    """
    endpoint = f"{API_URL}/models"
    resp = requests.get(endpoint)
    return resp


# Установка текущей модели
def set_model(model_id: str):
    """
    Установка указанной модели как активной (эндпоинт /set).
    """
    endpoint = f"{API_URL}/set"
    resp = requests.post(endpoint, params={"model_id": model_id})
    return resp


# Получение кривых обучения
def plot_curve_api(model_id: str, cv: int, scoring: str):
    url = f"{API_URL}/plot_learning_curve"
    params = {
        "model_id": model_id,
        "cv": cv,
        "scoring": scoring
    }
    return requests.post(url, params=params)


# Построение кривых обучения моделей
def show_learning_curves(model_id: str, cv: int, scoring: str):
    """
    Запрос к /plot_learning_curve, получение URL графиков и отображение.
    """
    try:
        response = plot_curve_api(model_id, cv, scoring)
        if response.status_code == 200:
            data = response.json()
            rating_url = data.get("rating_curve_url")
            hubs_url = data.get("hubs_curve_url")

            if rating_url and hubs_url:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Learning Curve - Rating")
                    st.image(rating_url, use_container_width=True)

                with col2:
                    st.subheader("Learning Curve - Hubs")
                    st.image(hubs_url, use_container_width=True)
            else:
                st.error("Не удалось получить URL графиков из ответа.")
        else:
            st.error(f"Ошибка: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке кривых: {str(e)}")


# Функция для формирования списка слов из всех токенов
def parse_tokens(row: pd.Series) -> List[str]:
    """
    Убираем лишние символы, такие как кавычки и перевод строк
    И формируем список слов
    """
    # Убираем лишние символы, такие как кавычки и перевод строк
    cleaned_row = row.replace('\n', '').replace("'", "").strip()
    return cleaned_row.strip("[]").split()


# Предикт
def predict_model(config_json_str: str):
    """
    Выполнение предикта с помощью выбранной модели.
    """
    endpoint = f"{API_URL}/predict"
    resp = requests.post(endpoint, params={"config": config_json_str})
    return resp


# Глобальные переменные
st.session_state.setdefault("df", None)


# Страница Upload
def page_upload_data():
    st.header("1. Загрузка и предобработка датасета")

    uploaded_file = st.file_uploader("Загрузите файл с данными (parquet)",
                                     type=["parquet"])
    if uploaded_file is not None:
        # Читаем файл в DF (локально, чтобы показать EDA)
        df = pd.read_parquet(uploaded_file)
        st.session_state["df"] = df  # Сохраняем в сессию

        st.write("**Пример данных (первые 5 строк):**")
        st.dataframe(df.head())

        st.write(f"**Форма датасета**: {df.shape}")

        # Кнопка отправки на предобработку
        if st.button("**Предобработать датасет**"):
            bytes_data = uploaded_file.getvalue()
            with st.spinner("Предобработка..."):
                response = preprocess_dataset(bytes_data)

            if response.ok:
                st.success("Данные успешно отправлены и предобработаны.")
                logger.info("Датасет отправлен на /preprocess и обработан.")
            else:
                st.error(f"Ошибка при отправке: {response.text}")
                logger.error(f"Ошибка при вызове /preprocess: {response.text}")
    else:
        st.info("Пожалуйста, загрузите parquet-файл.")


# Страница EDA
def page_eda():
    st.header("2. Exploratory Data Analysis")
    # Проверяем наличие датасета
    if st.session_state.df is None:
        st.warning(
            "Сначала загрузите датасет на предыдущей странице.")
        return
    df = st.session_state["df"]
    df['parsed_tokens'] = df['text_tokens'].apply(parse_tokens)
    df['parsed_tags'] = df['tags_tokens'].apply(parse_tokens)
    numeric_columns = (df.select_dtypes(include=["int64", "float64"])
                       .columns.tolist())
    # Выбор анализа
    analysis_type = st.selectbox(
        "Выберите тип анализа",
        [
            "Агрегированная информация",
            "Распределение рейтинга статей",
            "Топ-10 частотных слов",
            "Облако слов",
            "Распределение частей речи",
        ],
    )

    # Агрегированная информация
    if analysis_type == "Агрегированная информация":
        st.subheader("Агрегированная информация по числовым колонкам")
        agg_info = (df[numeric_columns].
                    agg(["mean", "median", "std", "max", "min"]))
        st.dataframe(
            agg_info.T.rename(
                columns={
                    "mean": "Среднее",
                    "median": "Медиана",
                    "std": "Стандартное отклонение",
                    "max": "Максимум",
                    "min": "Минимум",
                }), use_container_width=True)

    # Распределение рейтинга
    elif analysis_type == 'Распределение рейтинга статей':
        st.subheader("Распределение Рейтинга статей:")
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X("rating_level:N", axis=alt.Axis(
                labelAngle=0, labelFontSize=14)),
                y=alt.Y("count()", axis=alt.Axis(
                    labelFontSize=14, titleFontSize=16)), tooltip=["count()"])
            .properties(width=400, height=300))
        st.altair_chart(chart, use_container_width=True)

    # Частотное распределение
    elif analysis_type == 'Топ-10 частотных слов':
        all_tokens = [tok for toks in df['parsed_tokens'] for tok in toks]
        freq_dist = FreqDist(all_tokens)
        freq_df = pd.DataFrame(freq_dist.most_common(10),
                               columns=['Слово', 'Частота'])

        # График частотных слов
        st.header("Топ-10 наиболее частотных слов")
        freq_chart = alt.Chart(freq_df).mark_bar().encode(
            x=alt.X('Слово:O', sort='-y',
                    axis=alt.Axis(labelAngle=-45, labelFontSize=14)),
            y=alt.Y('Частота:Q', axis=alt.Axis(
                    labelFontSize=14, titleFontSize=16)),
            tooltip=['Слово', 'Частота']
        ).properties(
            width=700,
            height=400,
        )
        st.altair_chart(freq_chart, use_container_width=True)

    # Облако слов
    elif analysis_type == "Облако слов":
        st.subheader("Облако слов")
        text_column = st.selectbox(
            "Выберите текстовую колонку", list(['Text', 'Tags'])
        )
        if text_column == 'Text':
            all_tokens = [tok for toks in df['parsed_tokens'] for tok in toks]
        elif text_column == 'Tags':
            all_tokens = [tok for toks in df['parsed_tags'] for tok in toks]
        word_counts = Counter(all_tokens)
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_counts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # Распределение частей речи в текстах статей
    elif analysis_type == "Распределение частей речи":
        st.subheader("Распределение частей речи")
        all_pos = itertools.chain.from_iterable(df["text_pos_tags"])
        pos_counts = Counter(all_pos)
        pos_df = pd.DataFrame(
            pos_counts.items(), columns=["Часть речи", "Частота"]
            ).sort_values(by="Частота", ascending=False)
        fig = px.bar(
            pos_df, x="Часть речи", y="Частота",
            title="Распределение частей речи")
        st.plotly_chart(fig)


# Страница Train Model
def page_train_model():
    st.header("3. Создание и обучение новой модели")
    # Проверяем наличие датасета
    if st.session_state.df is None:
        st.warning(
            "Сначала загрузите и предобработайте датасет на первой странице.")
        return

    st.write("Настройте гиперпараметры и обучите новую модель.")

    with st.expander("Параметры TfidfVectorizer"):
        max_features = st.slider("max_features", min_value=500, max_value=5000,
                                 value=2500, step=100)
        max_df = st.slider("max_df", min_value=0.0,
                           max_value=1.0, value=0.9, step=0.01)
        min_df = st.number_input("min_df", min_value=1, value=1)
        # Можно добавить и другие параметры

    with st.expander("Параметры классификатора"):
        classifier_type = st.selectbox("Тип классификатора",
                                       ["LogisticRegression", "SVC"])
        if classifier_type == "LogisticRegression":
            C = st.number_input("C (регуляризация)",
                                min_value=0.0001, value=1.0)
            max_iter = st.number_input("max_iter", min_value=100, value=1000)
            solver = st.selectbox("solver", ["sag", "saga"])
            # Собираем в dict
            classifier_params = {
                "C": float(C),
                "max_iter": int(max_iter),
                "solver": solver
            }
        else:
            # SVC
            C = st.number_input("C (регуляризация)",
                                min_value=0.0001, value=1.0)
            kernel = st.selectbox("kernel", ["linear", "sigmoid"])
            classifier_params = {
                "C": float(C),
                "kernel": kernel
            }

    model_id = st.text_input("ID новой модели (любое уникальное название)",
                             value="my_new_model")

    if st.button("Обучить модель"):
        # Формируем конфиг в формате, который ожидает FastAPI (PipelineConfig)
        config_dict = {
            "id": model_id,
            "tfidf": {
                "params": {
                    "max_features": max_features,
                    "max_df": max_df,
                    "min_df": min_df,
                }
            },
            "classifier": {
                "type": classifier_type,
                "params": classifier_params
            }
        }
        # Превращаем dict в JSON-строку
        config_json_str = json.dumps(config_dict, ensure_ascii=False)

        st.write("Отправляем конфигурацию в /fit:")
        st.json(config_dict)

        with st.spinner("Обучение модели..."):
            try:
                response = fit_model(config_json_str)
                if response.ok:
                    resp_json = response.json()
                    st.success("Модель успешно обучена!")
                    st.json(resp_json)
                    logger.info(f"Модель '{model_id}' успешно обучена.")
                else:
                    st.error(f"Ошибка при обучении модели: {response.text}")
                    logger.error(f"Ошибка при вызове /fit: {response.text}")

            except requests.exceptions.ConnectionError as e:
                st.error("Обучение модели прервалось (дольше 10 секунд)")
                st.error(f"Ошибка соединения: {e}")
            except requests.exceptions.Timeout:
                st.error("Превышено время ожидания ответа от сервера.")


# Страница ModelInfo
def page_model_info():
    st.header("4. Список моделей и выбор активной")

    with st.spinner("Запрашиваем список моделей..."):
        response = list_models()
    if response.ok:
        models_data = response.json()
        models = models_data.get("models", [])
        if not models:
            st.warning("Пока нет обученных моделей.")
        else:
            st.write(f"Найдено моделей: {len(models)}")
            df_models = pd.DataFrame(models)
            st.dataframe(df_models)

            # Извлекаем уникальные model_id без суффиксов и активируем нужную
            model_ids = list(set([m["id"].replace("_rating", "")
                                  .replace("_hubs", "") for m in models]))
            selected_model_id = st.selectbox("Выберите модель (ID) для set",
                                             model_ids)
            if st.button("Активировать модель"):
                with st.spinner("Установка выбранной модели..."):
                    req_set = set_model(selected_model_id)
                if req_set.ok:
                    st.success(req_set.json().get("message", "Успешно."))
                    st.session_state['active_model_id'] = selected_model_id
                    logger.info(
                        f"Установлена активная модель: {selected_model_id}")
                else:
                    st.error(f"Ошибка при установке модели: {req_set.text}")
                    logger.error(f"Ошибка при вызове /set: {req_set.text}")

            # Проверяем наличие датасета
            if st.session_state.df is None:
                st.warning(
                    """Для построения кривых обучения загрузите
                       и предобработайте датасет на первой странице.""")
                return
            if "active_model_id" in st.session_state:
                st.subheader("Построение кривых обучения для активной модели")
                cv = st.number_input("Количество фолдов (cv)", min_value=2,
                                     max_value=10, value=5)
                scoring = st.selectbox("Метрика (scoring)",
                                       ["f1_micro", "f1_macro", "f1_weighted"])

                if st.button("Показать кривые обучения"):
                    with st.spinner("Генерация и загрузка кривых обучения..."):
                        show_learning_curves(
                            st.session_state['active_model_id'], cv, scoring)

    else:
        st.error(f"Ошибка при получении списка моделей: {response.text}")
        logger.error(f"Ошибка при вызове /models: {response.text}")


# Страница Predict
def page_inference():
    st.header("5. Инференс с использованием выбранной модели")
    # Проверяем наличие датасета
    if st.session_state.df is None:
        st.warning(
            "Сначала загрузите и предобработайте датасет на первой странице.")
        return

    # Инициализация ключа в session_state, если он отсутствует
    if "active_model_id" not in st.session_state:
        st.session_state.active_model_id = None

    # Проверяем наличие активной модели
    if not st.session_state['active_model_id']:
        st.warning(
            "Сначала выберите и активируйте модель на предыдущей странице.")
        return

    active_model_id = st.session_state['active_model_id']
    st.write(f"Используем активную модель: {active_model_id}")

    if st.button("Сделать предсказание"):

        # Формируем JSON для predict
        predict_req = {
            "id": active_model_id,
        }
        config_json_str = json.dumps(predict_req, ensure_ascii=False)

        with st.spinner("Выполняем предсказание..."):
            response = predict_model(config_json_str)

        if response.ok:
            resp_json = response.json()
            predictions = resp_json.get("predictions", [])
            if predictions:
                st.success("Предсказание успешно получено!")
                st.write("Пример результата (5 элементов):")
                predictions_df = pd.DataFrame(predictions)

                # Берём случайные 5 предсказаний
                if len(predictions_df) > 5:
                    sample_df = predictions_df.sample(5)
                    st.write("**Случайные 5 предсказаний**:")
                    st.dataframe(sample_df, use_container_width=True)
                else:
                    # Иначе выводим все, если их меньше или равно 5
                    st.dataframe(predictions_df, use_container_width=True)
            else:
                st.info("Предсказаний нет или список пуст.")
        else:
            st.error(f"Ошибка при выполнении предсказания: {response.text}")
            logger.error(f"Ошибка при вызове /predict: {response.text}")


# Sidebar
page_options = {
    "Upload Dataset": page_upload_data,
    "EDA": page_eda,
    "Train New Model": page_train_model,
    "Model Info": page_model_info,
    "Inference": page_inference
}

st.sidebar.title("Навигация")
selected_page = st.sidebar.radio("Перейти к странице:",
                                 list(page_options.keys()))
page_options[selected_page]()
