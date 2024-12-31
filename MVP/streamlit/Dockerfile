FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы с зависимостями и устанавливаем их
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное
COPY . /app

# Открываем порт, на котором будет работать Streamlit
EXPOSE 8501

# Запускаем streamlit
CMD ["streamlit", "run", "streamlit_mvp.py", "--server.port=8501", "--server.address=0.0.0.0"]