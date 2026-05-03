# Project orchestration. Targets grow as modules land.
# On Windows without `make`, run the underlying commands directly (shown in each recipe).

PYTHON := python

.PHONY: install install-dev download-data test lint typecheck format clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements-dev.txt

download-data:
	$(PYTHON) -m churn.data.download

test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

typecheck:
	$(PYTHON) -m mypy

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info coverage.xml htmlcov
