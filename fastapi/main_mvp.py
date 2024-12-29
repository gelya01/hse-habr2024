import subprocess
import threading


def run_fastapi():
    subprocess.run(["uvicorn", "api_mvp:app", "--host", "127.0.0.1", "--port", "8000"])


def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_mvp.py"])


if __name__ == "__main__":
    # Запуск FastAPI в отдельном потоке
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Запуск Streamlit
    run_streamlit()
