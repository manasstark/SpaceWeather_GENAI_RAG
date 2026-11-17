.PHONY: setup lint test app data cards kb baseline generate evaluate notebooks

PY := .venv/bin/python

setup:
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e .
lint:
	.venv/bin/ruff check src tests
test:
	.venv/bin/pytest
app:
	.venv/bin/streamlit run app/main.py

# pipeline stages (python -m spacewx_rag --help)
data:            ## needs data/raw (scripts/download_data.py); slow
	$(PY) -m spacewx_rag interim && $(PY) -m spacewx_rag align && $(PY) -m spacewx_rag events
cards:
	$(PY) -m spacewx_rag cards
kb:
	$(PY) -m spacewx_rag kb
baseline:        ## offline, no model needed
	$(PY) -m spacewx_rag generate --provider baseline
generate:        ## local LLM through Ollama (ollama pull qwen2.5:7b-instruct)
	$(PY) -m spacewx_rag generate --provider ollama --run ollama_rag
	$(PY) -m spacewx_rag generate --provider ollama --no-retrieval --run ollama_no_retrieval
	$(PY) -m spacewx_rag generate --provider ollama --run ollama_consistency --events E216,E158,E039,E105,E221 --runs 3 --temperature 0.7
evaluate:
	$(PY) -m spacewx_rag evaluate
notebooks:
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 06_rag_generation.ipynb 07_evaluation.ipynb
