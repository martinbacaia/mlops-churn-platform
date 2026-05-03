"""Standalone drift-detection CLI.

Usage::

    python scripts/detect_drift.py \
        --baseline data/raw/telco.csv \
        --current  data/raw/telco_drifted.csv \
        --out monitoring/reports/

Produces ``drift_report.json`` and ``drift_report.html`` in ``--out``.
``--baseline`` and ``--current`` should both be raw Telco CSVs (the loader
applies the same coercion the training path uses).

If the API is running, the same logic is exposed at ``/drift-report``; this
script exists for offline runs (e.g. the GitHub Actions monthly retrain
workflow checks drift before it triggers training).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from churn.data.ingest import load_raw, preprocess
from churn.features.pipeline import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from churn.logging_setup import configure_logging, get_logger
from churn.monitoring.report import build_drift_report, write_reports

_log = get_logger(__name__)


def main(
    baseline_path: Path,
    current_path: Path,
    out_dir: Path,
    base_name: str = "drift_report",
) -> dict[str, Path]:
    configure_logging()
    baseline = preprocess(load_raw(baseline_path))
    current = preprocess(load_raw(current_path))

    report = build_drift_report(
        baseline=baseline,
        current=current,
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    paths = write_reports(report, out_dir=out_dir, base_name=base_name)
    _log.info(
        "drift_report_written",
        json=str(paths["json"]),
        html=str(paths["html"]),
        n_alerts=report.summary["n_total_alerts"],
        max_psi=report.summary["max_psi"],
    )
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a drift report.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("monitoring/reports/"))
    parser.add_argument("--name", default="drift_report")
    args = parser.parse_args()
    paths = main(args.baseline, args.current, args.out, args.name)
    print(paths["html"])
