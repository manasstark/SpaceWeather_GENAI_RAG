.PHONY: setup lint test app
setup:
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
lint:
	.venv/bin/ruff check src tests
test:
	.venv/bin/pytest
app:
	.venv/bin/streamlit run app/main.py
