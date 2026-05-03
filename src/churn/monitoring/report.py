"""Render drift findings as JSON or self-contained HTML.

The JSON form is the source of truth — easy to ship to dashboards, log to
MLflow as an artifact, or assert against in tests. The HTML form is for the
README screenshot and for ops folks who want to skim the result without
running ``jq``. Both views derive from the same :class:`DriftReport` payload.

The HTML is intentionally inline-styled and dependency-free so the file is
viewable directly from disk (``file://path/to/report.html``) — no asset
loading, no fetch errors when sent over Slack.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from churn.monitoring.drift import (
    DEFAULT_KS_ALPHA,
    DEFAULT_PSI_THRESHOLD,
    detect_drift,
    prediction_drift,
)


@dataclass
class DriftReport:
    """JSON-serializable summary of a drift run."""

    generated_at: str
    psi_threshold: float
    ks_alpha: float
    feature_drift: list[dict[str, Any]]
    prediction_drift: dict[str, float] | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def build_drift_report(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
    baseline_scores: Any | None = None,
    current_scores: Any | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    ks_alpha: float = DEFAULT_KS_ALPHA,
) -> DriftReport:
    """Run drift detection and bundle the results into a serializable report.

    Prediction drift is computed only when both ``baseline_scores`` and
    ``current_scores`` are provided — feature drift alone is still a useful
    standalone view (e.g. for upstream data validation pre-inference).
    """
    feature_df = detect_drift(
        baseline=baseline,
        current=current,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        psi_threshold=psi_threshold,
        ks_alpha=ks_alpha,
    )

    pred_drift: dict[str, float] | None = None
    if baseline_scores is not None and current_scores is not None:
        pred_drift = prediction_drift(baseline_scores, current_scores)

    n_alerts = int(feature_df["psi_alert"].sum() + feature_df["ks_alert"].sum())
    summary = {
        "n_features": int(len(feature_df)),
        "n_psi_alerts": int(feature_df["psi_alert"].sum()),
        "n_ks_alerts": int(feature_df["ks_alert"].sum()),
        "n_total_alerts": n_alerts,
        "max_psi": float(feature_df["psi"].max()) if len(feature_df) else 0.0,
        "prediction_drift_alert": bool(
            pred_drift is not None and pred_drift["psi"] >= psi_threshold
        ),
    }

    feature_records: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in feature_df.to_dict(orient="records")
    ]
    return DriftReport(
        generated_at=datetime.now(UTC).isoformat(),
        psi_threshold=psi_threshold,
        ks_alpha=ks_alpha,
        feature_drift=feature_records,
        prediction_drift=pred_drift,
        summary=summary,
    )


def _replace_nan_with_none(obj: Any) -> Any:
    """Recursively replace NaN floats with None so the result is JSON-compliant."""
    if isinstance(obj, dict):
        return {k: _replace_nan_with_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan_with_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def report_to_json(report: DriftReport) -> dict[str, Any]:
    """Convert a report to a JSON-serializable dict (NaN → None)."""
    out: dict[str, Any] = _replace_nan_with_none(asdict(report))
    return out


def _pill(label: str, fired: bool) -> str:
    css = "pill fired" if fired else "pill"
    return f'<span class="{css}">{escape(label)}</span>'


def render_html_report(report: DriftReport) -> str:
    """Render a self-contained HTML page for ``report``.

    Style is inlined; no external CSS or JS so the file can be opened from
    disk without a server. Alert rows are highlighted to draw the eye.
    """
    rows_html: list[str] = []
    for row in report.feature_drift:
        css_class = "alert" if row["psi_alert"] or row["ks_alert"] else ""
        ks_stat = f"{row['ks_statistic']:.4f}" if not pd.isna(row["ks_statistic"]) else "—"
        ks_pval = f"{row['ks_pvalue']:.4g}" if not pd.isna(row["ks_pvalue"]) else "—"
        rows_html.append(
            f"<tr class='{css_class}'>"
            f"<td>{escape(str(row['feature']))}</td>"
            f"<td>{escape(str(row['type']))}</td>"
            f"<td>{row['psi']:.4f}</td>"
            f"<td>{ks_stat}</td>"
            f"<td>{ks_pval}</td>"
            f"<td>{'⚠' if row['psi_alert'] else ''}</td>"
            f"<td>{'⚠' if row['ks_alert'] else ''}</td>"
            f"</tr>"
        )

    pred_html = ""
    if report.prediction_drift is not None:
        pd_obj = report.prediction_drift
        flagged = "alert" if pd_obj["psi"] >= report.psi_threshold else ""
        pred_html = (
            "<h2>Prediction drift</h2>"
            f"<table class='{flagged}'>"
            f"<tr><th>PSI</th><td>{pd_obj['psi']:.4f}</td></tr>"
            f"<tr><th>Baseline mean</th><td>{pd_obj['baseline_mean']:.4f}</td></tr>"
            f"<tr><th>Current mean</th><td>{pd_obj['current_mean']:.4f}</td></tr>"
            f"<tr><th>Baseline std</th><td>{pd_obj['baseline_std']:.4f}</td></tr>"
            f"<tr><th>Current std</th><td>{pd_obj['current_std']:.4f}</td></tr>"
            f"</table>"
        )

    summary = report.summary
    pred_pill_label = "Prediction drift: " + (
        "fired" if summary["prediction_drift_alert"] else "ok"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Drift report</title>
<style>
  body {{ font: 14px/1.5 -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 2rem; color: #222; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
  th, td {{ padding: 0.4rem 0.7rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f7f7f9; font-weight: 600; }}
  tr.alert td {{ background: #fff5f5; }}
  table.alert {{ background: #fff5f5; }}
  .pill {{ display: inline-block; padding: 0.1rem 0.5rem; border-radius: 1rem;
          background: #eee; margin-right: 0.4rem; font-size: 0.85em; }}
  .pill.fired {{ background: #fee; color: #b00; }}
</style>
</head>
<body>
<h1>Drift report</h1>
<div class="meta">Generated at {escape(report.generated_at)} ·
  PSI threshold {report.psi_threshold} · KS alpha {report.ks_alpha}</div>

<div>
  {_pill('PSI alerts: ' + str(summary['n_psi_alerts']), bool(summary['n_psi_alerts']))}
  {_pill('KS alerts: ' + str(summary['n_ks_alerts']), bool(summary['n_ks_alerts']))}
  {_pill(pred_pill_label, bool(summary['prediction_drift_alert']))}
  {_pill('Max PSI: ' + format(summary['max_psi'], '.4f'), False)}
</div>

<h2>Per-feature drift</h2>
<table>
  <thead><tr>
    <th>Feature</th><th>Type</th><th>PSI</th>
    <th>KS statistic</th><th>KS p-value</th>
    <th>PSI alert</th><th>KS alert</th>
  </tr></thead>
  <tbody>
    {"".join(rows_html)}
  </tbody>
</table>

{pred_html}
</body>
</html>
"""


def write_reports(
    report: DriftReport,
    out_dir: Path,
    base_name: str = "drift_report",
) -> dict[str, Path]:
    """Write JSON + HTML to ``out_dir/base_name.{json,html}``. Returns the paths."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{base_name}.json"
    html_path = out_dir / f"{base_name}.html"
    json_path.write_text(
        json.dumps(report_to_json(report), indent=2, default=str),
        encoding="utf-8",
    )
    html_path.write_text(render_html_report(report), encoding="utf-8")
    return {"json": json_path, "html": html_path}
