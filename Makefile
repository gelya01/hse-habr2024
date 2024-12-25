CODE = streamlit_app

install:
	python3.10 -m pip install poetry
	poetry install

lint:
	poetry run pflake8 $(CODE)

format:
	poetry run black $(CODE)