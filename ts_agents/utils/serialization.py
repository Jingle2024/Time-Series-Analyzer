from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd


def _df_to_records(df: pd.DataFrame, max_rows: int = 200) -> List[dict]:
    """Convert a DataFrame to JSON-serialisable records."""
    sub = df.head(max_rows).copy()
    sub = sub.reset_index()
    return json.loads(sub.to_json(orient="records", date_format="iso"))


def _safe(val):
    """Recursively convert numpy types to Python natives."""
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_safe(v) for v in val]
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, pd.Series):
        return val.tolist()
    if isinstance(val, pd.DataFrame):
        return val.to_dict(orient="records")
    if isinstance(val, (pd.Timestamp, pd.Timedelta)):
        return str(val)
    if val is None or val != val:
        return None
    return val
