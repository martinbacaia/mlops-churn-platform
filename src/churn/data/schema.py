"""Pandera schemas for the Telco dataset.

Two schemas, two contracts:

* :data:`RAW_SCHEMA` validates the dataframe **as the loader returns it** — after
  the small CSV-level cleanup (whitespace ``TotalCharges`` coerced to NaN) but
  before any modeling transformations. ``customerID`` is still present, ``Churn``
  is still ``"Yes"`` / ``"No"``.
* :data:`PROCESSED_SCHEMA` validates the dataframe **after preprocessing** —
  ``customerID`` dropped, ``TotalCharges`` non-null, target encoded as
  :data:`TARGET_COLUMN` ∈ {0, 1}, all categoricals still string-typed (one-hot /
  ordinal encoding lives in the feature pipeline, not here).

Schemas serve as runtime contracts: ingest validates against ``RAW_SCHEMA``
right after load, training validates against ``PROCESSED_SCHEMA`` right before
the feature pipeline runs. A schema mismatch fails fast and loudly.
"""

from __future__ import annotations

import pandera as pa

TARGET_COLUMN = "churn"

_BINARY = ["Yes", "No"]
_PHONE_OPT = ["Yes", "No", "No phone service"]
_INTERNET_OPT = ["Yes", "No", "No internet service"]
_INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
_CONTRACT = ["Month-to-month", "One year", "Two year"]
_PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_GENDER = ["Female", "Male"]


def _categorical(values: list[str]) -> pa.Column:
    return pa.Column(str, pa.Check.isin(values), nullable=False)


RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        "customerID": pa.Column(str, unique=True, nullable=False),
        "gender": _categorical(_GENDER),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": _categorical(_BINARY),
        "Dependents": _categorical(_BINARY),
        "tenure": pa.Column(int, pa.Check.in_range(0, 72)),
        "PhoneService": _categorical(_BINARY),
        "MultipleLines": _categorical(_PHONE_OPT),
        "InternetService": _categorical(_INTERNET_SERVICE),
        "OnlineSecurity": _categorical(_INTERNET_OPT),
        "OnlineBackup": _categorical(_INTERNET_OPT),
        "DeviceProtection": _categorical(_INTERNET_OPT),
        "TechSupport": _categorical(_INTERNET_OPT),
        "StreamingTV": _categorical(_INTERNET_OPT),
        "StreamingMovies": _categorical(_INTERNET_OPT),
        "Contract": _categorical(_CONTRACT),
        "PaperlessBilling": _categorical(_BINARY),
        "PaymentMethod": _categorical(_PAYMENT),
        "MonthlyCharges": pa.Column(float, pa.Check.gt(0), nullable=False),
        # TotalCharges is whitespace (' ') for ~11 brand-new customers in the
        # source CSV. The loader coerces those to NaN; the schema permits that.
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=True),
        "Churn": _categorical(_BINARY),
    },
    strict=True,
    coerce=True,
)


PROCESSED_SCHEMA = pa.DataFrameSchema(
    columns={
        "gender": _categorical(_GENDER),
        "SeniorCitizen": pa.Column(int, pa.Check.isin([0, 1])),
        "Partner": _categorical(_BINARY),
        "Dependents": _categorical(_BINARY),
        "tenure": pa.Column(int, pa.Check.in_range(0, 72)),
        "PhoneService": _categorical(_BINARY),
        "MultipleLines": _categorical(_PHONE_OPT),
        "InternetService": _categorical(_INTERNET_SERVICE),
        "OnlineSecurity": _categorical(_INTERNET_OPT),
        "OnlineBackup": _categorical(_INTERNET_OPT),
        "DeviceProtection": _categorical(_INTERNET_OPT),
        "TechSupport": _categorical(_INTERNET_OPT),
        "StreamingTV": _categorical(_INTERNET_OPT),
        "StreamingMovies": _categorical(_INTERNET_OPT),
        "Contract": _categorical(_CONTRACT),
        "PaperlessBilling": _categorical(_BINARY),
        "PaymentMethod": _categorical(_PAYMENT),
        "MonthlyCharges": pa.Column(float, pa.Check.gt(0), nullable=False),
        "TotalCharges": pa.Column(float, pa.Check.ge(0), nullable=False),
        TARGET_COLUMN: pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    strict=True,
    coerce=False,
)
