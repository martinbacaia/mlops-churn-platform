# Project orchestration. Targets grow as modules land.
# On Windows without `make`, run the underlying commands directly (shown in each recipe).

PYTHON := python

.PHONY: install install-dev download-data train tune tune-logreg tune-xgboost tune-mlp promote serve drift-report inject-drift mlflow-ui test lint typecheck format clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements-dev.txt

download-data:
	$(PYTHON) -m churn.data.download

train:
	$(PYTHON) -m churn.training.train

tune:
	$(PYTHON) -m churn.training.tune --model all --n-trials 30 --cv-splits 5

tune-logreg:
	$(PYTHON) -m churn.training.tune --model logreg --n-trials 30

tune-xgboost:
	$(PYTHON) -m churn.training.tune --model xgboost --n-trials 30

tune-mlp:
	$(PYTHON) -m churn.training.tune --model tabular_mlp --n-trials 20

# Promote a registered model version to Production. Usage:
#   make promote VERSION=3                     # any model_type
#   make promote VERSION=3 MODEL=xgboost       # asserts model_type before transition
#   make promote VERSION=3 STAGE=Staging
promote:
	$(PYTHON) -m churn.training.promote --version $(VERSION) $(if $(MODEL),--model $(MODEL),) $(if $(STAGE),--stage $(STAGE),)

serve:
	$(PYTHON) -m uvicorn churn.serving.app:app --host 0.0.0.0 --port 8000

drift-report:
	$(PYTHON) scripts/detect_drift.py --baseline data/raw/telco.csv --current data/raw/telco_drifted.csv --out monitoring/reports/

inject-drift:
	$(PYTHON) scripts/inject_drift.py --out data/raw/telco_drifted.csv

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

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
