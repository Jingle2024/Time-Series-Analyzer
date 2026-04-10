from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

SESSIONS: Dict[str, Dict[str, Any]] = {}
DOWNLOADS: Dict[str, bytes] = {}

LARGE_DATASET_ROW_THRESHOLD = 100_000
LARGE_SERIES_THRESHOLD = 10_000
LARGE_GROUP_THRESHOLD = 2_000
DEFAULT_MAX_CHART_POINTS = 1_500


def _first_session_frame(sess: Dict[str, Any]) -> Optional[pd.DataFrame]:
    for key in ("treated_df", "imputed_df", "accumulated_df", "df"):
        if key in sess:
            candidate = sess[key]
            if isinstance(candidate, pd.DataFrame):
                return candidate
    return None


def _hierarchy_session_frame(sess: Dict[str, Any]) -> Optional[pd.DataFrame]:
    raw_df = sess.get("df")
    if not isinstance(raw_df, pd.DataFrame):
        return None

    hier_cols = [c for c in sess.get("hierarchy_cols", []) if c in raw_df.columns]
    if not hier_cols:
        return _first_session_frame(sess) or raw_df

    for key in ("treated_df", "imputed_df", "accumulated_df", "df"):
        candidate = sess.get(key)
        if isinstance(candidate, pd.DataFrame) and all(col in candidate.columns for col in hier_cols):
            return candidate

    # Treated/imputed frames currently drop hierarchy columns. When they still
    # align 1:1 with the raw frame, stitch the numeric values back onto the
    # raw hierarchy keys so hierarchy mode can still respect session cleanup.
    for key in ("treated_df", "imputed_df"):
        candidate = sess.get(key)
        if not isinstance(candidate, pd.DataFrame):
            continue
        if len(candidate) != len(raw_df) or not candidate.index.equals(raw_df.index):
            continue

        merged = raw_df.copy()
        overlay_cols = [c for c in candidate.columns if c in merged.columns and c not in hier_cols]
        if not overlay_cols:
            continue
        merged.loc[:, overlay_cols] = candidate.loc[:, overlay_cols].to_numpy()
        return merged

    return raw_df
