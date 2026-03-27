"""
server.py  –  TemporalMind FastAPI backend
==========================================
Run:  python server.py          (serves on http://localhost:8000)
Docs: http://localhost:8000/docs

Endpoints
---------
POST /api/upload            – ingest file, detect schema, return column info
POST /api/confirm-schema    – accept user column mapping
POST /api/analyze           – run full decomposition / outlier / intermittency per series
POST /api/interval-advice   – get interval recommendations
POST /api/accumulate        – resample to chosen interval
POST /api/hierarchy         – build hierarchy and aggregate
POST /api/missing-values    – impute missing values
POST /api/outliers          – detect & treat outliers
POST /api/prepare           – full data-prep pipeline → forecast-ready CSV/Excel
GET  /api/download/{token}  – stream the prepared file
GET  /                      – serve the SPA (index.html)
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# ── agent imports ─────────────────────────────────────────────────────────────
from agents.ingestion_agent import IngestionAgent
from agents.interval_advisor_agent import IntervalAdvisorAgent
from agents.accumulation_agent import AccumulationAgent
from agents.hierarchy_aggregation_agent import HierarchyAggregationAgent
from agents.decomposition_agent import DecompositionAgent
from agents.outlier_detection_agent import OutlierDetectionAgent
from agents.missing_values_agent import MissingValuesAgent
from agents.intermittency_agent import IntermittencyAgent
from agents.data_preparation_agent import DataPreparationAgent
from core.context_store import ContextStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")

# ── in-memory session store (keyed by upload token) ─────────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}
_downloads: Dict[str, bytes] = {}   # token → file bytes

# ── lazy FastAPI import ───────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import (HTMLResponse, JSONResponse,
                                    Response, StreamingResponse)
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    log.warning("FastAPI not installed – run: pip install fastapi uvicorn python-multipart")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_records(df: pd.DataFrame, max_rows: int = 200) -> List[dict]:
    """Convert a DataFrame to JSON-serialisable records."""
    sub = df.head(max_rows).copy()
    # make index a column
    sub = sub.reset_index()
    # convert everything to native Python types
    return json.loads(sub.to_json(orient="records", date_format="iso"))


def _series_stats(s: pd.Series) -> dict:
    s_clean = s.dropna()
    return {
        "n": int(len(s)),
        "n_missing": int(s.isna().sum()),
        "mean": round(float(s_clean.mean()), 4) if len(s_clean) else None,
        "std":  round(float(s_clean.std()),  4) if len(s_clean) > 1 else None,
        "min":  round(float(s_clean.min()),  4) if len(s_clean) else None,
        "max":  round(float(s_clean.max()),  4) if len(s_clean) else None,
        "pct_zero": round(float((s_clean == 0).mean() * 100), 2) if len(s_clean) else 0,
    }


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
    if val is None or val != val:   # NaN check
        return None
    return val


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def _recommend_model(
    classification: str,
    trend_strong: bool,
    seasonal_strong: bool,
    d_order: int,
) -> str:
    if classification == "Lumpy":
        return "TSB / Zero-Inflated"
    if classification == "Intermittent":
        return "Croston / SBA"
    if classification == "Erratic":
        return "ETS(M,N,N) / TBATS"
    # Smooth series
    if d_order >= 2:
        return "ARIMA(p,2,q)"
    if trend_strong and seasonal_strong:
        return "Holt-Winters / SARIMA"
    if trend_strong:
        return "Holt / ARIMA(p,1,0)"
    if seasonal_strong:
        return "ETS(A,N,A) / SARIMA"
    if d_order == 1:
        return "ARIMA(p,1,q)"
    return "ETS(A,N,N) / Naive"


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_FASTAPI:
    app = FastAPI(title="TemporalMind API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # ── Upload ────────────────────────────────────────────────────────────────

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            raw = await file.read()
            fname = file.filename or "upload"
            suffix = Path(fname).suffix.lower()

            if suffix in (".xlsx", ".xls"):
                df_raw = pd.read_excel(io.BytesIO(raw))
            elif suffix == ".csv":
                df_raw = pd.read_csv(io.BytesIO(raw))
            else:
                raise HTTPException(400, "Only CSV and Excel files are supported.")

            token = str(uuid.uuid4())[:8]
            _sessions[token] = {"raw_df": df_raw, "filename": fname}

            # heuristic column classification
            cols = list(df_raw.columns)
            dtypes = {c: str(df_raw[c].dtype) for c in cols}
            nunique = {c: int(df_raw[c].nunique()) for c in cols}
            n_rows = len(df_raw)

            ts_candidates, val_candidates, hier_candidates = [], [], []
            for c in cols:
                col_lower = c.lower()
                # try datetime parse
                try:
                    pd.to_datetime(df_raw[c].dropna().iloc[:10])
                    ts_candidates.append(c)
                    continue
                except Exception:
                    pass
                if any(h in col_lower for h in ["date","time","ts","period","month","week","year"]):
                    ts_candidates.append(c)
                elif pd.api.types.is_numeric_dtype(df_raw[c]):
                    val_candidates.append(c)
                elif df_raw[c].dtype == object and nunique[c] < n_rows * 0.3:
                    hier_candidates.append(c)

            # auto-decide: if hier_candidates exist it's likely a hierarchy
            is_hierarchy = len(hier_candidates) > 0
            needs_confirm = (len(ts_candidates) != 1 or
                             len(val_candidates) == 0 or
                             (is_hierarchy and len(hier_candidates) > 4))

            return {
                "token": token,
                "filename": fname,
                "n_rows": n_rows,
                "columns": cols,
                "dtypes": dtypes,
                "nunique": nunique,
                "ts_candidates": ts_candidates,
                "value_candidates": val_candidates,
                "hierarchy_candidates": hier_candidates,
                "is_hierarchy": is_hierarchy,
                "needs_confirm": needs_confirm,
                "preview": _df_to_records(df_raw, 8),
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))

    # ── Confirm schema ────────────────────────────────────────────────────────

    class SchemaConfirm(BaseModel):
        token: str
        timestamp_col: str
        value_cols: List[str]
        hierarchy_cols: List[str] = []

    @app.post("/api/confirm-schema")
    async def confirm_schema(body: SchemaConfirm):
        sess = _sessions.get(body.token)
        if not sess:
            raise HTTPException(404, "Session not found")
        df_raw = sess["raw_df"]

        agent = IngestionAgent()
        result = agent.execute(
            source=df_raw,
            timestamp_col=body.timestamp_col,
            value_cols=body.value_cols,
            hierarchy_cols=body.hierarchy_cols,
        )
        if not result.ok:
            raise HTTPException(400, result.errors[0] if result.errors else "Ingestion failed")

        df: pd.DataFrame = result.data
        sess["df"] = df
        sess["schema"] = result.metadata
        sess["value_cols"] = body.value_cols
        sess["hierarchy_cols"] = body.hierarchy_cols
        sess["detected_freq"] = result.metadata["detected_freq"]

        return {
            "ok": True,
            "schema": _safe(result.metadata["schema"]),
            "warnings": result.warnings,
            "preview": _df_to_records(df.reset_index(), 10),
        }

    # ── Interval advice ───────────────────────────────────────────────────────

    class IntervalReq(BaseModel):
        token: str
        top_n: int = 3

    @app.post("/api/interval-advice")
    async def interval_advice(body: IntervalReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404, "No ingested data for this token")
        df: pd.DataFrame = sess["df"]
        val_cols = sess.get("value_cols", [])
        col = val_cols[0] if val_cols else df.select_dtypes("number").columns[0]

        series = df[col].dropna()
        agent = IntervalAdvisorAgent()
        result = agent.execute(series=series, native_freq=sess.get("detected_freq"), top_n=body.top_n)
        if not result.ok:
            raise HTTPException(500, result.errors[0])

        recs = result.data
        return {
            "recommendations": [
                {
                    "alias": r["alias"],
                    "freq": r["freq"],
                    "score": r["score"],
                    "info_loss_pct": r["info_loss_pct"],
                    "rationale": r["rationale"],
                }
                for r in recs
            ],
            "best_freq": result.metadata["best_interval"],
            "best_alias": result.metadata["best_alias"],
            "summary": result.metadata["summary"],
            "native_step_days": result.metadata["native_step_days"],
        }

    # ── Accumulate ────────────────────────────────────────────────────────────

    class AccumReq(BaseModel):
        token: str
        target_freq: str
        method: str = "auto"
        quantity_type: str = "flow"

    @app.post("/api/accumulate")
    async def accumulate(body: AccumReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        df = sess["df"]
        val_cols = sess.get("value_cols")

        agent = AccumulationAgent()
        result = agent.execute(
            df=df, target_freq=body.target_freq,
            method=body.method, quantity_type=body.quantity_type,
            compare_freqs=None, value_cols=val_cols,
        )
        if not result.ok:
            raise HTTPException(500, result.errors[0])

        sess["accumulated_df"] = result.data
        sess["target_freq"] = body.target_freq

        comparison = {}
        for freq, stats in result.metadata.get("comparison", {}).items():
            for col, s in stats.items():
                comparison.setdefault(col, {})[freq] = _safe(s)

        return {
            "n_input": result.metadata["n_input"],
            "n_output": result.metadata["n_output"],
            "compression_ratio": result.metadata["compression_ratio"],
            "information_retention": _safe(result.metadata["information_retention"]),
            "comparison": comparison,
            "preview": _df_to_records(result.data.reset_index(), 12),
            "report": result.metadata.get("report", ""),
        }

    # ── Hierarchy ─────────────────────────────────────────────────────────────

    class HierarchyReq(BaseModel):
        token: str
        method: str = "sum"

    @app.post("/api/hierarchy")
    async def hierarchy(body: HierarchyReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        df = sess["df"]
        hier_cols = sess.get("hierarchy_cols", [])
        val_cols  = sess.get("value_cols", [])
        if not hier_cols:
            raise HTTPException(400, "No hierarchy columns in this session")

        agent = HierarchyAggregationAgent()
        result = agent.execute(df=df, hierarchy_cols=hier_cols,
                               value_cols=val_cols, method=body.method)
        if not result.ok:
            raise HTTPException(500, result.errors[0])

        sess["hierarchy_result"] = result

        level_info = {}
        for lk, series_dict_or_df in result.data.items():
            prefix = lk.split("__")[0]
            level_info.setdefault(prefix, []).append(lk)

        return {
            "n_total_series": result.metadata["n_total_series"],
            "level_meta": _safe(result.metadata["level_meta"]),
            "coherence": _safe(result.metadata["coherence"]),
            "level_effects": _safe(result.metadata["level_effects"]),
            "all_series_keys": list(result.data.keys()),
            "level_groups": {k: v for k, v in level_info.items()},
            "report": result.metadata.get("report", ""),
        }

    # ── Analyze ───────────────────────────────────────────────────────────────

    class AnalyzeReq(BaseModel):
        token: str
        series_key: Optional[str] = None   # None → first value col of accumulated df
        period: Optional[int] = None

    @app.post("/api/analyze")
    async def analyze(body: AnalyzeReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)

        # pick series
        work_df = sess.get("accumulated_df", sess["df"])
        val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

        if body.series_key and body.series_key in work_df.columns:
            series = work_df[body.series_key].dropna()
        else:
            col = val_cols[0] if val_cols else work_df.columns[0]
            series = work_df[col].dropna()

        ctx = ContextStore()

        # missing values
        mv_res = MissingValuesAgent(ctx).execute(series=series)
        clean = mv_res.data["imputed"] if mv_res.ok else series

        # decomposition
        decomp_res = DecompositionAgent(ctx).execute(series=clean, period=body.period)

        # outlier
        residual = decomp_res.data.get("residual") if decomp_res.ok else None
        out_res = OutlierDetectionAgent(ctx).execute(
            series=clean, residual=residual,
            methods=["iqr", "zscore", "isof"]
        )

        # intermittency
        interm_res = IntermittencyAgent(ctx).execute(series=clean)

        # build time series chart data (original + components)
        chart_data: Dict[str, Any] = {}
        ts_labels = [str(d)[:10] for d in clean.index]
        chart_data["labels"] = ts_labels
        chart_data["original"] = [round(float(v), 4) if not np.isnan(v) else None
                                   for v in clean.values]
        if decomp_res.ok:
            for comp in ("trend", "seasonal", "cycle", "residual"):
                arr = decomp_res.data[comp]
                if arr is not None:
                    chart_data[comp] = [round(float(v), 4) if not np.isnan(v) else None
                                        for v in arr.values]

        outlier_timestamps = []
        if out_res.ok and len(out_res.data["outlier_table"]) > 0:
            ot = out_res.data["outlier_table"]
            outlier_timestamps = [str(r["timestamp"])[:10] for _, r in ot.iterrows()]

        # stats summary
        summary = {
            "series_stats": _safe(_series_stats(clean)),
            "decomp": _safe({
                "trend_strength_Ft": decomp_res.metadata.get("trend_strength_Ft"),
                "seasonal_strength_Fs": decomp_res.metadata.get("seasonal_strength_Fs"),
                "period_used": decomp_res.metadata.get("period_used"),
                "differencing_order": decomp_res.metadata.get("differencing_order"),
                "stationarity": decomp_res.metadata.get("stationarity"),
                "trend_stats": decomp_res.metadata.get("trend_stats"),
                "interpretation": decomp_res.metadata.get("interpretation"),
                "acf_significant_lags": decomp_res.metadata.get("acf_significant_lags"),
            }) if decomp_res.ok else {},
            "outliers": _safe(out_res.metadata.get("summary", {})) if out_res.ok else {},
            "intermittency": _safe(interm_res.metadata.get("summary", {})) if interm_res.ok else {},
            "missing_values": _safe(mv_res.metadata.get("completeness", {})) if mv_res.ok else {},
        }

        # store clean series back
        col = series.name or val_cols[0]
        sess["clean_series"] = {col: clean}
        sess["decomp_result"] = decomp_res

        return {
            "chart_data": chart_data,
            "outlier_timestamps": outlier_timestamps,
            "summary": summary,
            "warnings": (decomp_res.warnings + out_res.warnings + interm_res.warnings),
        }

    # ── Missing values treatment ──────────────────────────────────────────────

    class MissingReq(BaseModel):
        token: str
        method: str = "auto"
        period: int = 7
        zero_as_missing: bool = False

    @app.post("/api/missing-values")
    async def missing_values(body: MissingReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        work_df = sess.get("accumulated_df", sess["df"])
        val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

        results = {}
        imputed_cols = {}
        for col in val_cols:
            series = work_df[col]
            res = MissingValuesAgent().execute(
                series=series, method=body.method,
                period=body.period, zero_as_missing=body.zero_as_missing,
            )
            if res.ok:
                imputed_cols[col] = res.data["imputed"]
                results[col] = _safe(res.metadata["completeness"])

        # update session df
        imp_df = work_df.copy()
        for col, s in imputed_cols.items():
            imp_df[col] = s
        sess["imputed_df"] = imp_df

        return {"results": results, "preview": _df_to_records(imp_df.reset_index(), 10)}

    # ── Outlier treatment ─────────────────────────────────────────────────────

    class OutlierReq(BaseModel):
        token: str
        methods: List[str] = ["iqr", "zscore", "isof"]
        treatment: str = "cap"   # cap | remove | keep

    @app.post("/api/outliers")
    async def outliers(body: OutlierReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        work_df = sess.get("imputed_df", sess.get("accumulated_df", sess["df"]))
        val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

        summary_all = {}
        treated_cols = {}
        for col in val_cols:
            series = work_df[col].dropna()
            res = OutlierDetectionAgent().execute(
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
                is_out = res.data["is_outlier"]
                treated[is_out] = np.nan
            treated_cols[col] = treated

        treated_df = work_df.copy()
        for col, s in treated_cols.items():
            treated_df[col] = s
        sess["treated_df"] = treated_df

        return {
            "summary": summary_all,
            "treatment_applied": body.treatment,
            "preview": _df_to_records(treated_df.reset_index(), 10),
        }

    # ── Prepare data ──────────────────────────────────────────────────────────

    class PrepareReq(BaseModel):
        token: str
        transform: str = "auto"
        scale_method: str = "minmax"
        rolling_windows: List[int] = [7, 14, 28]
        add_calendar: bool = True
        train_ratio: float = 0.70
        val_ratio: float = 0.15
        horizon: int = 1
        output_format: str = "csv"   # csv | excel
        hier_level: Optional[str] = None

    def _first_session_frame(sess: Dict[str, Any]) -> Optional[pd.DataFrame]:
        for key in ("treated_df", "imputed_df", "accumulated_df", "df"):
            if key in sess:
                candidate = sess[key]
                if isinstance(candidate, pd.DataFrame):
                    return candidate
        return None

    @app.post("/api/prepare")
    async def prepare(body: PrepareReq):
        try:
            sess = _sessions.get(body.token)
            if sess is None or "df" not in sess:
                raise HTTPException(
                    404,
                    "Session not found or schema not confirmed. Upload the file again and confirm the schema before preparing data.",
                )

            work_df = _first_session_frame(sess)
            if work_df is None:
                raise HTTPException(500, "No prepared working dataframe found in session.")

            val_cols = sess.get("value_cols")
            if val_cols is None:
                val_cols = work_df.select_dtypes("number").columns.tolist()

            all_prepared: List[pd.DataFrame] = []
            prep_failures: Dict[str, Any] = {}

            for col in val_cols:
                series = work_df[col].dropna()

                acf_lags = None
                dr = sess.get("decomp_result")
                if dr is not None and getattr(dr, "ok", False):
                    acf_lags = dr.metadata.get("acf_significant_lags")

                interm_res = IntermittencyAgent().execute(series=series)
                interm_cls = (
                    interm_res.metadata.get("summary", {}).get("classification", "Smooth")
                    if interm_res.ok else "Smooth"
                )

                decomp_res = DecompositionAgent().execute(series=series)
                Ft = decomp_res.metadata.get("trend_strength_Ft", 0.0) if decomp_res.ok else 0.0
                Fs = decomp_res.metadata.get("seasonal_strength_Fs", 0.0) if decomp_res.ok else 0.0
                d = decomp_res.metadata.get("differencing_order", 0) if decomp_res.ok else 0

                model_rec = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d)

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
                    prep_failures[col] = {
                        "errors": prep_res.errors,
                        "warnings": prep_res.warnings,
                    }
                    continue

                feat_df = prep_res.data["feature_matrix"].copy()
                feat_df["series_name"] = col
                feat_df["model_type_recommendation"] = model_rec
                feat_df["intermittency_class"] = interm_cls
                feat_df["trend_strength_Ft"] = round(Ft, 4)
                feat_df["seasonal_strength_Fs"] = round(Fs, 4)

                n = len(feat_df)
                n_train = prep_res.data["X_train"].shape[0]
                n_val = prep_res.data["X_val"].shape[0]
                split_labels = (
                    ["train"] * n_train +
                    ["validation"] * n_val +
                    ["test"] * (n - n_train - n_val)
                )
                feat_df["split"] = split_labels[:n]

                all_prepared.append(feat_df)

            if not all_prepared:
                raise HTTPException(
                    500,
                    _safe({
                        "message": "Data preparation failed for all columns",
                        "failures": prep_failures,
                    }),
                )

            final_df = pd.concat(all_prepared, axis=0).reset_index()

            token = str(uuid.uuid4())[:8]
            if body.output_format == "excel":
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    final_df.to_excel(writer, index=False, sheet_name="PreparedData")
                _downloads[token] = buf.getvalue()
                ext = "xlsx"
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                csv_bytes = final_df.to_csv(index=False).encode()
                _downloads[token] = csv_bytes
                ext = "csv"
                mime = "text/csv"

            series_summaries = []
            for col in val_cols:
                sub = final_df[final_df["series_name"] == col]
                if len(sub) > 0:
                    series_summaries.append({
                        "series": col,
                        "n_rows": len(sub),
                        "n_features": len(sub.columns),
                        "model_type": sub["model_type_recommendation"].iloc[0],
                        "intermittency": sub["intermittency_class"].iloc[0],
                        "Ft": float(sub["trend_strength_Ft"].iloc[0]),
                        "Fs": float(sub["seasonal_strength_Fs"].iloc[0]),
                    })

            return {
                "download_token": token,
                "ext": ext,
                "mime": mime,
                "n_rows": len(final_df),
                "n_features": len(final_df.columns),
                "series_summaries": series_summaries,
                "preview": _df_to_records(final_df, 15),
                "columns": list(final_df.columns),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                500,
                _safe({
                    "message": "Unexpected error in prepare",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }),
            )

    # ── Download ──────────────────────────────────────────────────────────────

    @app.get("/api/download/{token}")
    async def download(token: str, ext: str = "csv"):
        data = _downloads.get(token)
        if not data:
            raise HTTPException(404, "Download token expired or not found")
        mime = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if ext == "xlsx" else "text/csv")
        return Response(
            content=data,
            media_type=mime,
            headers={"Content-Disposition": f"attachment; filename=prepared_data.{ext}"},
        )

    # ── Serve SPA ─────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        base_dir = Path(__file__).parent
        candidate_paths = [
            base_dir / "ui" / "index.html",
            base_dir / "index.html",
        ]
        for html_path in candidate_paths:
            if html_path.exists():
                return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>TemporalMind UI not found. Place index.html in ./ui/ or project root.</h1>",
            status_code=404,
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
        log.info("Starting TemporalMind server on http://localhost:8000")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
    except ImportError:
        
        log.error("uvicorn not installed. Run: pip install uvicorn")
