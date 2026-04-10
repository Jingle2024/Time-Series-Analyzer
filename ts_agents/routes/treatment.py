"""
routes/treatment.py  –  Data-cleaning treatment endpoints
==========================================================
Routes
------
POST /api/missing-values  – impute missing values
POST /api/outliers        – detect & treat outliers
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.missing_values_agent import MissingValuesAgent
from agents.outlier_detection_agent import OutlierDetectionAgent
from core.runtime import SESSIONS as _sessions
from utils.api_helper import _df_to_records, _safe

router = APIRouter()


# ── Missing values ─────────────────────────────────────────────────────────────

class MissingReq(BaseModel):
    token:           str
    method:          str  = "auto"
    period:          int  = 7
    zero_as_missing: bool = False


@router.post("/api/missing-values")
async def missing_values(body: MissingReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)
    work_df  = sess.get("accumulated_df", sess["df"])
    val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

    results:      dict = {}
    imputed_cols: dict = {}
    for col in val_cols:
        series = work_df[col]
        res    = MissingValuesAgent().execute(
            series=series,
            method=body.method,
            period=body.period,
            zero_as_missing=body.zero_as_missing,
        )
        if res.ok:
            imputed_cols[col] = res.data["imputed"]
            results[col]      = _safe(res.metadata["completeness"])

    imp_df = work_df.copy()
    for col, s in imputed_cols.items():
        imp_df[col] = s
    sess["imputed_df"] = imp_df

    return {"results": results, "preview": _df_to_records(imp_df.reset_index(), 10)}


# ── Outlier treatment ──────────────────────────────────────────────────────────

class OutlierReq(BaseModel):
    token:     str
    methods:   List[str] = ["iqr", "zscore", "isof"]
    treatment: str = "cap"   # cap | remove | keep


@router.post("/api/outliers")
async def outliers(body: OutlierReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)
    work_df  = sess.get("imputed_df", sess.get("accumulated_df", sess["df"]))
    val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

    summary_all:  dict = {}
    treated_cols: dict = {}
    for col in val_cols:
        series = work_df[col].dropna()
        res    = OutlierDetectionAgent().execute(
            series=series, methods=body.methods, contamination=0.05
        )
        if not res.ok:
            continue
        summary_all[col] = _safe(res.metadata["summary"])

        treated = series.copy()
        if body.treatment == "cap" and res.ok:
            fences = res.metadata["summary"]["fences"]
            lo, hi = fences["lower_1.5iqr"], fences["upper_1.5iqr"]
            treated = treated.clip(lower=lo, upper=hi)
        elif body.treatment == "remove" and res.ok:
            is_out  = res.data["is_outlier"]
            treated[is_out] = np.nan
        treated_cols[col] = treated

    treated_df = work_df.copy()
    for col, s in treated_cols.items():
        treated_df[col] = s
    sess["treated_df"] = treated_df

    return {
        "summary":           summary_all,
        "treatment_applied": body.treatment,
        "preview":           _df_to_records(treated_df.reset_index(), 10),
    }
