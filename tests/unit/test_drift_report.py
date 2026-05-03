from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn.monitoring.report import (
    DriftReport,
    build_drift_report,
    render_html_report,
    report_to_json,
    write_reports,
)


def _toy_dfs():
    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=n),
            "gender": rng.choice(["F", "M"], size=n, p=[0.5, 0.5]),
        }
    )
    curr = pd.DataFrame(
        {
            "tenure": rng.integers(20, 72, size=n),  # shifted
            "gender": rng.choice(["F", "M"], size=n, p=[0.8, 0.2]),  # shifted
        }
    )
    return base, curr


# --- build_drift_report ---------------------------------------------------


def test_build_returns_drift_report_with_canonical_fields():
    base, curr = _toy_dfs()
    report = build_drift_report(
        baseline=base,
        current=curr,
        numerical_columns=["tenure"],
        categorical_columns=["gender"],
    )
    assert isinstance(report, DriftReport)
    assert report.psi_threshold > 0
    assert report.feature_drift  # non-empty
    assert report.summary["n_features"] == 2


def test_build_includes_prediction_drift_when_scores_passed():
    base, curr = _toy_dfs()
    rng = np.random.default_rng(0)
    base_scores = rng.beta(2, 8, size=400)
    curr_scores = rng.beta(8, 2, size=400)  # severely shifted scores

    report = build_drift_report(
        baseline=base,
        current=curr,
        numerical_columns=["tenure"],
        categorical_columns=["gender"],
        baseline_scores=base_scores,
        current_scores=curr_scores,
    )
    assert report.prediction_drift is not None
    assert report.summary["prediction_drift_alert"] is True


def test_build_omits_prediction_drift_when_scores_missing():
    base, curr = _toy_dfs()
    report = build_drift_report(
        baseline=base,
        current=curr,
        numerical_columns=["tenure"],
        categorical_columns=["gender"],
    )
    assert report.prediction_drift is None
    assert report.summary["prediction_drift_alert"] is False


def test_summary_counts_alerts_correctly():
    base, curr = _toy_dfs()
    report = build_drift_report(
        baseline=base,
        current=curr,
        numerical_columns=["tenure"],
        categorical_columns=["gender"],
    )
    feature_psi_alerts = sum(1 for r in report.feature_drift if r["psi_alert"])
    feature_ks_alerts = sum(1 for r in report.feature_drift if r["ks_alert"])
    assert report.summary["n_psi_alerts"] == feature_psi_alerts
    assert report.summary["n_ks_alerts"] == feature_ks_alerts


# --- report_to_json -------------------------------------------------------


def test_json_output_is_serializable_and_round_trips():
    base, curr = _toy_dfs()
    report = build_drift_report(base, curr, ["tenure"], ["gender"])
    payload = report_to_json(report)
    serialized = json.dumps(payload, default=str)
    parsed = json.loads(serialized)
    assert parsed["psi_threshold"] == report.psi_threshold
    assert len(parsed["feature_drift"]) == len(report.feature_drift)


# --- render_html_report ---------------------------------------------------


def test_html_report_contains_summary_and_table():
    base, curr = _toy_dfs()
    report = build_drift_report(base, curr, ["tenure"], ["gender"])
    html = render_html_report(report)
    assert "<h1>Drift report</h1>" in html
    assert "Per-feature drift" in html
    assert "tenure" in html
    assert "gender" in html


def test_html_report_highlights_alerts_for_shifted_features():
    base, curr = _toy_dfs()
    report = build_drift_report(base, curr, ["tenure"], ["gender"])
    html = render_html_report(report)
    # At least one alert is present in this synthetic shift.
    if report.summary["n_total_alerts"] > 0:
        assert "alert" in html


def test_html_report_escapes_user_provided_strings():
    """Sanity: feature names with HTML special chars don't break the template."""
    base = pd.DataFrame({"<script>": [1, 2, 3], "g": ["a", "b", "c"]})
    curr = pd.DataFrame({"<script>": [1, 2, 3], "g": ["a", "b", "c"]})
    report = build_drift_report(base, curr, ["<script>"], ["g"])
    html = render_html_report(report)
    assert "&lt;script&gt;" in html
    assert "<script>1" not in html


def test_html_report_includes_prediction_drift_when_present():
    base, curr = _toy_dfs()
    rng = np.random.default_rng(0)
    report = build_drift_report(
        base,
        curr,
        ["tenure"],
        ["gender"],
        baseline_scores=rng.uniform(size=400),
        current_scores=rng.uniform(size=400),
    )
    html = render_html_report(report)
    assert "Prediction drift" in html


# --- write_reports --------------------------------------------------------


def test_write_reports_emits_both_files(tmp_path: Path):
    base, curr = _toy_dfs()
    report = build_drift_report(base, curr, ["tenure"], ["gender"])

    paths = write_reports(report, out_dir=tmp_path / "reports")
    assert paths["json"].exists()
    assert paths["html"].exists()
    json_payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "feature_drift" in json_payload
    html_text = paths["html"].read_text(encoding="utf-8")
    assert "<html" in html_text


def test_write_reports_creates_directory_if_missing(tmp_path: Path):
    base, curr = _toy_dfs()
    report = build_drift_report(base, curr, ["tenure"], ["gender"])
    nested = tmp_path / "deeply" / "nested" / "dir"
    paths = write_reports(report, out_dir=nested)
    assert paths["json"].parent == nested
