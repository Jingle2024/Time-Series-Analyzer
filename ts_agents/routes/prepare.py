"""
routes/prepare.py  –  Full data-prep pipeline endpoint
=======================================================
Routes
------
POST /api/prepare  – full data-prep pipeline → forecast-ready CSV/Excel
"""

from __future__ import annotations

import io
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.data_preparation_agent import DataPreparationAgent
from agents.decomposition_agent import DecompositionAgent
from agents.intermittency_agent import IntermittencyAgent
from agents.multi_variable_agent import ROLE_DEPENDENT
from core.runtime import DOWNLOADS as _downloads, SESSIONS as _sessions
from utils.api_helper import (
    _df_to_records,
    _first_session_frame,
    _recommend_model,
    _safe,
    _upgrade_model_for_exog,
)

router = APIRouter()


class PrepareReq(BaseModel):
    token:           str
    transform:       str       = "auto"
    scale_method:    str       = "minmax"
    rolling_windows: List[int] = [7, 14, 28]
    add_calendar:    bool      = True
    train_ratio:     float     = 0.70
    val_ratio:       float     = 0.15
    horizon:         int       = 1
    output_format:   str       = "csv"   # csv | excel
    hier_level:      Optional[str] = None


@router.post("/api/prepare")
async def prepare(body: PrepareReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)

    work_df    = _first_session_frame(sess)
    val_cols   = sess.get("value_cols",    work_df.select_dtypes("number").columns.tolist())
    dep_cols   = sess.get("dependent_cols", val_cols[:1])
    indep_cols = sess.get("independent_cols", [])
    event_cols = sess.get("event_cols", [])

    target_cols: List[str] = dep_cols if dep_cols else val_cols[:1]
    all_prepared: List[pd.DataFrame] = []

    for col in target_cols:
        if col not in work_df.columns:
            continue
        series = work_df[col].dropna()

        acf_lags = None
        dr = sess.get("decomp_result")
        if dr and dr.ok:
            acf_lags = dr.metadata.get("acf_significant_lags")

        interm_res  = IntermittencyAgent().execute(series=series)
        interm_cls  = interm_res.metadata.get("summary", {}).get("classification", "Smooth") if interm_res.ok else "Smooth"

        decomp_res  = DecompositionAgent().execute(series=series)
        Ft  = decomp_res.metadata.get("trend_strength_Ft",    0.0) if decomp_res.ok else 0.0
        Fs  = decomp_res.metadata.get("seasonal_strength_Fs", 0.0) if decomp_res.ok else 0.0
        d   = decomp_res.metadata.get("differencing_order",   0)   if decomp_res.ok else 0

        has_exog  = bool(indep_cols or event_cols)
        base_rec  = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d)
        model_rec = _upgrade_model_for_exog(base_rec, has_exog)

        prep_res = DataPreparationAgent().execute(
            series=series,
            target_col=col,
            transform=body.transform,
            scale_method=body.scale_method,
            rolling_windows=body.rolling_windows,
            add_calendar=body.add_calendar,
            train_ratio=body.train_ratio,
            val_ratio=body.val_ratio,
            horizon=body.horizon,
            acf_lags=acf_lags,
        )
        if not prep_res.ok:
            continue

        feat_df = prep_res.data["feature_matrix"].copy()

        for ic in indep_cols:
            if ic not in work_df.columns:
                continue
            indep_s = work_df[ic].reindex(feat_df.index)
            feat_df[f"indep__{ic}"]      = indep_s.values
            feat_df[f"indep_lag1__{ic}"] = indep_s.shift(1).values

        for ec in event_cols:
            if ec not in work_df.columns:
                continue
            ev_s = work_df[ec].reindex(feat_df.index).fillna(0)
            feat_df[f"event__{ec}"]       = ev_s.values
            feat_df[f"event_roll7__{ec}"] = ev_s.rolling(7, min_periods=1).sum().values

        feat_df["series_name"]             = col
        feat_df["variable_role"]           = ROLE_DEPENDENT
        feat_df["model_type_recommendation"]= model_rec
        feat_df["intermittency_class"]     = interm_cls
        feat_df["trend_strength_Ft"]       = round(Ft, 4)
        feat_df["seasonal_strength_Fs"]    = round(Fs, 4)
        feat_df["has_exogenous"]           = has_exog

        n       = len(feat_df)
        n_train = prep_res.data["X_train"].shape[0]
        n_val   = prep_res.data["X_val"].shape[0]
        split_labels = (
            ["train"]      * n_train +
            ["validation"] * n_val   +
            ["test"]       * (n - n_train - n_val)
        )
        feat_df["split"] = split_labels[:n]
        all_prepared.append(feat_df)

    if not all_prepared:
        raise HTTPException(500, "Data preparation failed for all columns")

    final_df = pd.concat(all_prepared, axis=0).reset_index()

    token_dl = str(uuid.uuid4())[:8]
    if body.output_format == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            final_df.to_excel(writer, index=False, sheet_name="PreparedData")
        _downloads[token_dl] = buf.getvalue()
        ext  = "xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        _downloads[token_dl] = final_df.to_csv(index=False).encode()
        ext  = "csv"
        mime = "text/csv"

    series_summaries: List[Dict[str, Any]] = []
    for col in target_cols:
        sub = final_df[final_df["series_name"] == col] if "series_name" in final_df.columns else final_df
        if len(sub):
            series_summaries.append({
                "series":        col,
                "n_rows":        len(sub),
                "n_features":    len(sub.columns),
                "model_type":    sub["model_type_recommendation"].iloc[0] if "model_type_recommendation" in sub.columns else "—",
                "intermittency": sub["intermittency_class"].iloc[0]       if "intermittency_class"        in sub.columns else "—",
                "Ft":            float(sub["trend_strength_Ft"].iloc[0])  if "trend_strength_Ft"          in sub.columns else 0,
                "Fs":            float(sub["seasonal_strength_Fs"].iloc[0])if "seasonal_strength_Fs"      in sub.columns else 0,
                "has_exogenous": bool(sub["has_exogenous"].iloc[0])       if "has_exogenous"              in sub.columns else False,
                "n_indep":       len(indep_cols),
                "n_events":      len(event_cols),
            })

    return {
        "download_token":  token_dl,
        "ext":             ext,
        "mime":            mime,
        "n_rows":          len(final_df),
        "n_features":      len(final_df.columns),
        "series_summaries":series_summaries,
        "preview":         _df_to_records(final_df, 15),
        "columns":         list(final_df.columns),
    }
