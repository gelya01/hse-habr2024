import streamlit as st
import requests
import pandas as pd
import ast

API_URL = "http://0.0.0.0:8000"

st.title("Multilabel Text Classification")

training_df = None

model_id = st.text_input("Введите model_id", "awesome_model")
model_type = st.text_input("Введите model_type", "logreg")
hyperparameters = st.text_input("Введите гиперпараметры", "{'max_iter': 1000}")

uploaded_file = st.file_uploader("Загрузите файл с данными", type=["parquet"])
if uploaded_file is not None:
    training_df = pd.read_parquet(uploaded_file)

if st.button("Обучить модель"):
    fit_request = {
        "model_info": {
            "model_id": model_id,
            "model_type": model_type,
            "hyperparameters": ast.literal_eval(hyperparameters),
        },
        "text": training_df["Content"].values.tolist(),
        "labels": training_df["Hubs"].str.split(", ").values.tolist(),
    }
    response = requests.post(f"{API_URL}/fit", json=[fit_request])
    if response.ok:
        st.write(response.json()[0]["message"])
    else:
        st.write(response.text)


response = requests.get(f"{API_URL}/list_models")
if len(response.json()) > 0:
    models = requests.get(f"{API_URL}/list_models").json()
    selected_model = st.selectbox(
        "Выберите модель", [model["model_id"] for model in models]
    )
    text = st.text_input("Введите текст для предсказания")

    if text is not None and st.button("Сделать предсказание"):
        data = {"model_id": selected_model, "text": "Ваш текст для классификации"}
        response = requests.post(f"{API_URL}/predict", json=[data])

        if response.ok:
            prediction = response.json()[0]
            st.write(f"Предсказанные метки: {', '.join(prediction['labels'])}")
        else:
            st.error("Ошибка при получении предсказания")


# Добавьте здесь другие элементы интерфейса:
# - Визуализация данных
# - Информация о модели
