PYTHON ?= python

.PHONY: install generate-data ingest run-api test evaluate evaluate-official finalize compare lint

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .

generate-data:
	$(PYTHON) scripts/generate_sample_data.py --profile benchmark

ingest:
	$(PYTHON) scripts/ingest.py --profile benchmark --chunking-strategy section_semantic

run-api:
	$(PYTHON) scripts/run_api.py

test:
	$(PYTHON) -m pytest

evaluate:
	$(PYTHON) scripts/evaluate.py --skip-ragas --profile benchmark --tag local

evaluate-official:
	$(PYTHON) scripts/evaluate.py --ragas-mode official --profile benchmark --tag official

finalize:
	$(PYTHON) scripts/finalize_results.py --max-queries 60 --tag final

compare:
	$(PYTHON) scripts/compare_configs.py

lint:
	$(PYTHON) -m ruff check .
