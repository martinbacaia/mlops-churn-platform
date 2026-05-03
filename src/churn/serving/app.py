"""FastAPI application that serves whichever model is currently in Production.

The serving contract is **model-agnostic**: this file does not import
``LogRegModel``, ``XGBoostModel``, or ``TabularMLPModel``. It loads
``models:/<settings.model_name>/<settings.model_stage>`` and treats the
result as an opaque ``Model``. Promoting a different version (or a different
runtime entirely) is invisible to the API surface.

Lifespan-loaded artifacts live on ``app.state.artifacts`` for the duration of
the process. ``/health`` reflects whether the load succeeded; orchestrators
should fail the readiness probe if the response is ``status="degraded"``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from churn.config import get_settings
from churn.data.ingest import load_raw, preprocess
from churn.features.pipeline import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from churn.logging_setup import configure_logging, get_logger
from churn.monitoring.report import build_drift_report, report_to_json
from churn.serving.loader import (
    NoActiveModelError,
    ProductionArtifacts,
    load_production_artifacts,
)
from churn.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    DriftReportRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)

DEFAULT_THRESHOLD = 0.5
_log = get_logger(__name__)


def _records_to_frame(records: list[CustomerFeatures]) -> pd.DataFrame:
    """Convert pydantic records to the dataframe shape the feature pipeline expects.

    Mirrors ``preprocess`` from :mod:`churn.data.ingest`: nullable ``TotalCharges``
    becomes 0.0 (semantically correct for tenure-0 customers).
    """
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)
    # ``to_numeric`` coerces None to NaN cleanly, avoiding the dtype-downcasting
    # warning ``fillna`` triggers on object columns from optional pydantic fields.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def _predict_one(
    artifacts: ProductionArtifacts,
    record: CustomerFeatures,
    threshold: float,
) -> PredictResponse:
    df = _records_to_frame([record])
    X_t = artifacts.feature_pipeline.transform(df)
    proba = artifacts.model.predict_proba(X_t)[0, 1]
    return PredictResponse(
        churn_probability=float(proba),
        prediction=1 if proba >= threshold else 0,
        threshold=threshold,
    )


def _predict_batch(
    artifacts: ProductionArtifacts,
    records: list[CustomerFeatures],
    threshold: float,
) -> list[PredictResponse]:
    df = _records_to_frame(records)
    X_t = artifacts.feature_pipeline.transform(df)
    proba_pos = artifacts.model.predict_proba(X_t)[:, 1]
    return [
        PredictResponse(
            churn_probability=float(p),
            prediction=1 if p >= threshold else 0,
            threshold=threshold,
        )
        for p in proba_pos
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load production artifacts at startup; record load failures on app.state."""
    configure_logging()
    try:
        app.state.artifacts = load_production_artifacts()
        app.state.load_error = None
        _log.info(
            "serving_started",
            model_type=app.state.artifacts.model_type,
            version=app.state.artifacts.model_version,
        )
    except NoActiveModelError as exc:
        # Don't crash — let /health report degraded so the operator sees what's wrong.
        app.state.artifacts = None
        app.state.load_error = str(exc)
        _log.error("serving_startup_no_model", error=str(exc))
    yield


def create_app() -> FastAPI:
    """Construct the FastAPI app. Factory pattern lets tests override lifespan."""
    app = FastAPI(
        title="churn-serving",
        version="0.1.0",
        description=(
            "Online churn prediction. Loads the version of `churn_classifier` "
            "currently in the Production stage of MLflow's Model Registry."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        artifacts: ProductionArtifacts | None = getattr(request.app.state, "artifacts", None)
        if artifacts is None:
            return HealthResponse(status="degraded", model_loaded=False)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_type=artifacts.model_type,
            model_version=artifacts.model_version,
        )

    @app.get("/model_info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        artifacts = _require_artifacts(request)
        return ModelInfoResponse(
            registered_model_name=artifacts.registered_model_name,
            model_type=artifacts.model_type,
            model_version=artifacts.model_version,
            stage=artifacts.stage,
            run_id=artifacts.run_id,
            feature_pipeline_version=artifacts.feature_pipeline_version,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: Request, payload: CustomerFeatures) -> PredictResponse:
        artifacts = _require_artifacts(request)
        return _predict_one(artifacts, payload, DEFAULT_THRESHOLD)

    @app.post("/predict_batch", response_model=BatchPredictResponse)
    def predict_batch(request: Request, payload: BatchPredictRequest) -> BatchPredictResponse:
        artifacts = _require_artifacts(request)
        preds = _predict_batch(artifacts, payload.records, DEFAULT_THRESHOLD)
        return BatchPredictResponse(predictions=preds)

    @app.post("/drift-report")
    def drift_report(request: Request, payload: DriftReportRequest) -> JSONResponse:
        """Compare the supplied production records against the training baseline.

        The baseline is the raw Telco dataset on disk (the same MD5-pinned file
        the active model trained on). PSI is computed per feature; prediction
        drift is computed by scoring both the baseline and the current records
        through the loaded production model.
        """
        artifacts = _require_artifacts(request)
        settings = get_settings()
        baseline_path = settings.data_dir / "raw" / "telco.csv"
        if not baseline_path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Baseline dataset not found at {baseline_path}. "
                    "Run `make download-data` on the API host."
                ),
            )

        baseline_df = preprocess(load_raw(baseline_path))
        baseline_features = baseline_df.drop(columns=["churn"], errors="ignore")
        current_df = _records_to_frame(payload.records)

        # Score baseline + current with the loaded production model so that
        # prediction drift is measured against the same model serving requests.
        base_scores = artifacts.model.predict_proba(
            artifacts.feature_pipeline.transform(baseline_features)
        )[:, 1]
        curr_scores = artifacts.model.predict_proba(
            artifacts.feature_pipeline.transform(current_df)
        )[:, 1]

        report = build_drift_report(
            baseline=baseline_features,
            current=current_df,
            numerical_columns=NUMERICAL_COLUMNS,
            categorical_columns=CATEGORICAL_COLUMNS,
            baseline_scores=base_scores,
            current_scores=curr_scores,
        )
        return JSONResponse(content=report_to_json(report))

    @app.exception_handler(NoActiveModelError)
    def _no_model_handler(_req: Request, exc: NoActiveModelError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


def _require_artifacts(request: Request) -> ProductionArtifacts:
    artifacts: ProductionArtifacts | None = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        load_error: str = getattr(request.app.state, "load_error", "Model not loaded.")
        raise HTTPException(status_code=503, detail=load_error)
    return artifacts


# Module-level instance for ``uvicorn churn.serving.app:app``.
app = create_app()
