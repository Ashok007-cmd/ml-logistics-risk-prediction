.PHONY: install train test lint format run-api run-app docker-build docker-run clean

install:
	pip install -e ".[dev]"

train:
	python -m src.train

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ app/ tests/

format:
	ruff format src/ app/ tests/

typecheck:
	mypy src/ app/

run-api:
	python -m uvicorn app.api:app --host 0.0.0.0 --port 8000

run-app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t logistics-risk .

docker-run:
	docker run -p 8501:8501 -p 8000:8000 logistics-risk

clean:
	rm -rf models/ data/raw/ data/processed/ .pytest_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
