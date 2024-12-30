import subprocess
import threading
import os


def run_fastapi():
    api_path = os.path.join("fastapi")
    subprocess.run(
        ["uvicorn", "api_mvp:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=api_path)


def run_streamlit():
    streamlit_path = os.path.join("streamlit")
    subprocess.run(["streamlit", "run", "streamlit_mvp.py"],
                   cwd=streamlit_path)


if __name__ == "__main__":
    # Запуск FastAPI в отдельном потоке
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    # Запуск Streamlit
    run_streamlit()
