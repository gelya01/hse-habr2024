FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы с зависимостями и устанавливаем их
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное в контейнер
COPY . /app

# Открываем порт
EXPOSE 8000

# Запускаем uvicorn
CMD ["uvicorn", "api_mvp:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]