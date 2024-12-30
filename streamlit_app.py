import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import re

st.title("Milti-label classification MVP app")

# Шаг 1: Загрузка датасета
st.header("Шаг 1: Загрузка датасета")

uploaded_file = st.file_uploader("Загрузите файл (.parquet)", type=["parquet"])

if uploaded_file:
    try:
        data = pd.read_parquet(uploaded_file)
        st.success("Датасет успешно загружен!")
        st.dataframe(data.head())  # Показываем первые строки

        # Подготовка текстовых колонок
        processed_text_columns = {}
        for column in ["tags_tokens", "title_tokens", "text_tokens"]:
            if column in data.columns:
                processed_text_columns[column] = data[column].apply(
                    lambda x: (
                        re.sub(r"[^\w\s]", "", x).replace("\n", "").strip().split()
                        if isinstance(x, str)
                        else []
                    )
                )
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
else:
    st.warning("Для продолжения загрузите файл.")
    st.stop()

# Шаг 2: Выбор функционала
st.header("Шаг 2: Выберите функционал")
functional_choice = st.selectbox(
    "Что вы хотите сделать?",
    [
        "EDA",
        "Создание новой модели и выбор гиперпараметров",
        "Просмотр информации о модели и кривых обучения",
        "Инференс с использованием обученной модели",
    ],
)

if functional_choice == "EDA":
    st.subheader("EDA")
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Выбор анализа
    analysis_type = st.selectbox(
        "Выберите тип анализа",
        [
            "Агрегированная информация",
            "Распределение числовых данных",
            "Топ-10 частотных слов",
            "Облако слов",
            "Распределение частей речи",
        ],
    )

    # Агрегированная информация
    if analysis_type == "Агрегированная информация":
        st.subheader("Агрегированная информация по числовым колонкам")
        agg_info = data[numeric_columns].agg(["mean", "median", "std", "max", "min"])
        st.dataframe(
            agg_info.T.rename(
                columns={
                    "mean": "Среднее",
                    "median": "Медиана",
                    "std": "Стандартное отклонение",
                    "max": "Максимум",
                    "min": "Минимум",
                }
            )
        )

    # Распределение числовых данных
    elif analysis_type == "Распределение числовых данных":
        st.subheader("Распределение числовых колонок")
        column = st.selectbox("Выберите числовую колонку для анализа", numeric_columns)
        bins = st.slider(
            "Количество бинов для распределения", 5, 50, 20
        )  # Выбор количества бинов
        if column in data.columns:
            fig = px.histogram(
                data, x=column, nbins=bins, title=f"Распределение: {column}"
            )
            st.plotly_chart(fig)

    # Топ-10 частотных слов
    elif analysis_type == "Топ-10 частотных слов":
        st.subheader("Топ-10 наиболее частотных слов")
        text_column = st.selectbox(
            "Выберите текстовую колонку", list(processed_text_columns.keys())
        )
        if text_column in processed_text_columns:
            all_tokens = itertools.chain.from_iterable(
                processed_text_columns[text_column]
            )
            word_counts = Counter(all_tokens)
            top_10_words = word_counts.most_common(10)
            df_top_words = pd.DataFrame(top_10_words, columns=["Слово", "Частота"])
            st.write("Топ-10 слов:")
            st.dataframe(df_top_words)

    # Облако слов
    elif analysis_type == "Облако слов":
        st.subheader("Облако слов")
        text_column = st.selectbox(
            "Выберите текстовую колонку", list(processed_text_columns.keys())
        )
        if text_column in processed_text_columns:
            all_tokens = itertools.chain.from_iterable(
                processed_text_columns[text_column]
            )
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
        column = st.selectbox("Выберите колонку с частями речи", ["text_pos_tags"])
        if column in data.columns:
            all_pos = itertools.chain.from_iterable(data["text_pos_tags"])
            pos_counts = Counter(all_pos)
            pos_df = pd.DataFrame(
                pos_counts.items(), columns=["Часть речи", "Частота"]
            ).sort_values(by="Частота", ascending=False)
            fig = px.bar(
                pos_df, x="Часть речи", y="Частота", title="Распределение частей речи"
            )
            st.plotly_chart(fig)
