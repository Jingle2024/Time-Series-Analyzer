from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from core.runtime import DEFAULT_MAX_CHART_POINTS


def _downsample_series(series: pd.Series, max_points: int = DEFAULT_MAX_CHART_POINTS) -> pd.Series:
    """Return an evenly spaced subset for large chart payloads."""
    if len(series) <= max_points:
        return series
    idx = np.linspace(0, len(series) - 1, num=max_points, dtype=int)
    return series.iloc[idx]


def _series_to_chart_payload(series: pd.Series, max_points: int = DEFAULT_MAX_CHART_POINTS) -> Dict[str, Any]:
    sampled = _downsample_series(series, max_points=max_points)
    return {
        "labels": [str(d)[:10] for d in sampled.index],
        "values": [round(float(v), 4) if not np.isnan(v) else None for v in sampled.values],
        "n_points_original": int(len(series)),
        "n_points_returned": int(len(sampled)),
        "downsampled": bool(len(sampled) < len(series)),
    }
