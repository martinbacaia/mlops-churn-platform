"""Generate a deliberately-perturbed slice of the Telco test set.

Used by the README's drift example: the script produces a CSV that *should*
trigger PSI / KS alerts on the features it touches, validating that the
detector works on actual shifts rather than only on synthetic-noise tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churn.data.download import download_telco
from churn.data.ingest import load_raw


def inject_drift(
    df: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """Apply a handful of realistic shifts to a copy of ``df``.

    Shifts:
      * ``MonthlyCharges`` is shifted up by ~25 % (price hike scenario).
      * ``Contract`` swings toward month-to-month (deteriorating retention).
      * ``PaymentMethod`` skews toward "Electronic check" (riskiest segment).

    Other columns are passed through untouched. The shifts are pronounced on
    purpose — the detector should fire even with the default thresholds.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["MonthlyCharges"] = out["MonthlyCharges"] * 1.25
    n = len(out)
    out["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.85, 0.10, 0.05],
    )
    out["PaymentMethod"] = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.70, 0.10, 0.10, 0.10],
    )
    return out


def main(out_path: Path, seed: int = 0) -> Path:
    raw_path = download_telco()
    df = load_raw(raw_path)
    perturbed = inject_drift(df, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    perturbed.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a perturbed Telco CSV for drift-detection demos."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/telco_drifted.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(main(out_path=args.out, seed=args.seed))
