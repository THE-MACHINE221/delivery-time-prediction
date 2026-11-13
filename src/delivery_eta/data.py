"""Loading, auditing and cleaning the food-delivery dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Delivery_Time"
ID_COL = "Order_ID"
NUMERIC = ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs"]
CATEGORICAL = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
FEATURES = NUMERIC + CATEGORICAL

def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the KAUST Academy CSV and check that the expected columns are there."""
    df = pd.read_csv(path)
    missing = [c for c in [ID_COL, TARGET, *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in {path}: {missing}")
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing count and share, distinct values."""
    out = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(1),
            "distinct": df.nunique(),
        }
    )
    return out.sort_values("missing", ascending=False)


def duplicate_report(df: pd.DataFrame) -> dict:
    """Count exact duplicate rows and repeated order IDs."""
    id_counts = df[ID_COL].value_counts()
    repeated_ids = id_counts[id_counts > 1]
    return {
        "rows": len(df),
        "unique_orders": int(df[ID_COL].nunique()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "rows_sharing_an_order_id": int(repeated_ids.sum() - len(repeated_ids)),
        "order_ids_repeated": int(len(repeated_ids)),
    }


def conflicting_copies(df: pd.DataFrame) -> int:
    """Number of order IDs whose copies disagree on a non-missing value.

    If this is 0, rows that share an Order_ID are copies of the same order
    (some with a value blanked out), so it is safe to keep one row per order.
    """
    conflicts = 0
    for _, group in df[df[ID_COL].duplicated(keep=False)].groupby(ID_COL):
        for col in group.columns:
            if group[col].dropna().nunique() > 1:
                conflicts += 1
                break
    return conflicts


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per order, then drop orders without a target.

    Within each Order_ID the copies are merged: every column takes its first
    non-missing value, so a value blanked out in one copy (target included) is
    recovered from another copy instead of being imputed or thrown away.
    """
    merged = df.groupby(ID_COL, as_index=False, sort=True).first()
    return merged.dropna(subset=[TARGET]).reset_index(drop=True)
