"""
agents/orchestrator.py
───────────────────────
Master controller that:
  1. Accepts user intent + raw data source
  2. Builds a DAG of agent tasks based on data shape (single vs hierarchy)
  3. Dispatches agents in dependency order (parallel where possible)
  4. Collects results into a unified pipeline output
  5. Generates a final summary report

Usage
-----
    from agents.orchestrator import Orchestrator

    orch = Orchestrator()
    result = orch.run(
        source="data/sales.csv",
        timestamp_col="date",
        value_cols=["sales"],
        hierarchy_cols=["region", "store"],    # omit for single series
        target_freq="W",
        quantity_type="flow",
        horizon=13,
    )
    print(result.metadata["final_report"])
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

# ── import all specialized agents ────────────────────────────────────────────
from agents.ingestion_agent import IngestionAgent
from agents.interval_advisor_agent import IntervalAdvisorAgent
from agents.accumulation_agent import AccumulationAgent
from agents.hierarchy_aggregation_agent import HierarchyAggregationAgent
from agents.decomposition_agent import DecompositionAgent
from agents.outlier_detection_agent import OutlierDetectionAgent
from agents.missing_values_agent import MissingValuesAgent
from agents.intermittency_agent import IntermittencyAgent
from agents.data_preparation_agent import DataPreparationAgent

logger = logging.getLogger("orchestrator")


class Orchestrator:
    """
    ReAct (Reason → Act → Observe) style orchestration.

    The pipeline DAG:
      Ingest
        ├─ IntervalAdvisor
        ├─ (if hierarchy) HierarchyAggregation
        └─ Accumulation
               └─ [per series]
                    ├─ MissingValues
                    ├─ Outlier
                    ├─ Intermittency
                    ├─ Decomposition
                    └─ DataPreparation
    """

    def __init__(self, max_workers: int = 4):
        self.ctx = ContextStore()
        self.max_workers = max_workers

    # ── public ────────────────────────────────────────────────────────────────

    def run(
        self,
        source: Any,
        timestamp_col: Optional[str] = None,
        value_cols: Optional[List[str]] = None,
        hierarchy_cols: Optional[List[str]] = None,
        target_freq: Optional[str] = None,
        quantity_type: str = "flow",
        method: str = "auto",
        compare_freqs: Optional[List[str]] = None,
        transform: str = "auto",
        scale_method: str = "minmax",
        rolling_windows: Optional[List[int]] = None,
        horizon: int = 1,
        period: Optional[int] = None,
        run_intermittency: bool = True,
        **kwargs,
    ) -> AgentResult:
        """
        Full pipeline execution. Returns a consolidated AgentResult whose
        .data dict contains results from all sub-agents, keyed by agent name.
        """
        logger.info("══════════════════════════════════════")
        logger.info("Orchestrator: pipeline started")

        all_results: Dict[str, AgentResult] = {}
        pipeline_warnings: List[str] = []

        # ── STEP 1: INGEST ────────────────────────────────────────────────────
        logger.info("[1/8] Ingestion")
        ingest_result = IngestionAgent(self.ctx).execute(
            source=source,
            timestamp_col=timestamp_col,
            value_cols=value_cols,
            hierarchy_cols=hierarchy_cols,
        )
        all_results["ingestion"] = ingest_result
        if not ingest_result.ok:
            return self._fail("Ingestion failed.", all_results, pipeline_warnings)

        df: pd.DataFrame = ingest_result.data
        val_cols: List[str] = ingest_result.metadata["value_cols"]
        hier_cols: List[str] = ingest_result.metadata["hierarchy_cols"]
        detected_freq: str = ingest_result.metadata["detected_freq"]
        is_hierarchy = len(hier_cols) > 0
        pipeline_warnings += ingest_result.warnings

        # ── STEP 2: INTERVAL ADVISOR ──────────────────────────────────────────
        logger.info("[2/8] Interval Advisor")
        first_series = df[val_cols[0]] if not is_hierarchy else \
            df.groupby(hier_cols)[val_cols[0]].first().reset_index(drop=True).set_axis(df.index[:len(df.groupby(hier_cols)[val_cols[0]].first())])
        # use the full first column for interval advisory
        series_for_interval = df[val_cols[0]]
        interval_result = IntervalAdvisorAgent(self.ctx).execute(
            series=series_for_interval,
            native_freq=detected_freq,
            top_n=3,
        )
        all_results["interval_advisor"] = interval_result
        pipeline_warnings += interval_result.warnings

        # Use recommended freq if not explicitly provided
        if target_freq is None:
            if interval_result.ok:
                target_freq = interval_result.metadata.get("best_interval", detected_freq)
                pipeline_warnings.append(
                    f"Using advisor-recommended frequency: {target_freq}"
                )
            else:
                target_freq = detected_freq
                pipeline_warnings.append(f"Interval advisor failed; using detected freq: {target_freq}")

        compare_freqs = compare_freqs or self._default_compare_freqs(target_freq)

        # ── STEP 3A: HIERARCHY AGGREGATION (if applicable) ───────────────────
        if is_hierarchy:
            logger.info("[3a/8] Hierarchy Aggregation")
            hier_result = HierarchyAggregationAgent(self.ctx).execute(
                df=df,
                hierarchy_cols=hier_cols,
                value_cols=val_cols,
                method="sum",
            )
            all_results["hierarchy"] = hier_result
            pipeline_warnings += hier_result.warnings

        # ── STEP 3B: ACCUMULATION ─────────────────────────────────────────────
        logger.info("[3b/8] Temporal Accumulation")
        # Accumulate the flat version (if hierarchy, use total-level series)
        df_for_accum = df
        if is_hierarchy and all_results.get("hierarchy", AgentResult("", AgentStatus.FAILED)).ok:
            total_df = all_results["hierarchy"].data.get("total")
            if total_df is not None:
                df_for_accum = total_df

        accum_result = AccumulationAgent(self.ctx).execute(
            df=df_for_accum,
            target_freq=target_freq,
            method=method,
            quantity_type=quantity_type,
            compare_freqs=compare_freqs,
            value_cols=val_cols if not is_hierarchy else None,
        )
        all_results["accumulation"] = accum_result
        pipeline_warnings += accum_result.warnings

        # Working dataframe = accumulated data
        work_df: pd.DataFrame = accum_result.data if accum_result.ok else df_for_accum

        # ── STEPS 4-8: PER-SERIES ANALYSIS (parallel) ────────────────────────
        logger.info("[4-8/8] Per-series analysis (parallel)")
        series_results = self._run_series_analysis(
            work_df=work_df,
            val_cols=val_cols if val_cols[0] in work_df.columns else work_df.select_dtypes("number").columns.tolist(),
            period=period,
            transform=transform,
            scale_method=scale_method,
            rolling_windows=rolling_windows,
            horizon=horizon,
            run_intermittency=run_intermittency,
            pipeline_warnings=pipeline_warnings,
        )
        all_results.update(series_results)

        # ── FINAL REPORT ──────────────────────────────────────────────────────
        final_report = self._build_final_report(all_results, target_freq, is_hierarchy)
        logger.info("Orchestrator: pipeline complete")

        return AgentResult(
            agent_name="Orchestrator",
            status=AgentStatus.SUCCESS,
            data=all_results,
            metadata={
                "target_freq": target_freq,
                "is_hierarchy": is_hierarchy,
                "value_cols": val_cols,
                "hierarchy_cols": hier_cols,
                "final_report": final_report,
                "agent_statuses": {k: v.status.value for k, v in all_results.items()},
            },
            warnings=pipeline_warnings,
        )

    # ── per-series parallel analysis ──────────────────────────────────────────

    def _run_series_analysis(
        self,
        work_df: pd.DataFrame,
        val_cols: List[str],
        period: Optional[int],
        transform: str,
        scale_method: str,
        rolling_windows: Optional[List[int]],
        horizon: int,
        run_intermittency: bool,
        pipeline_warnings: List[str],
    ) -> Dict[str, AgentResult]:
        """Run missing values → outlier → intermittency → decomposition → prep per series."""

        results: Dict[str, AgentResult] = {}
        lock = threading.Lock()

        def analyse_one(col: str):
            series = work_df[col].dropna()
            col_results = {}

            # 4. Missing values
            mv_res = MissingValuesAgent(self.ctx).execute(series=series)
            col_results[f"missing_values_{col}"] = mv_res
            clean_series = mv_res.data["imputed"] if mv_res.ok else series

            # 5. Outlier detection
            out_res = OutlierDetectionAgent(self.ctx).execute(series=clean_series)
            col_results[f"outliers_{col}"] = out_res

            # 6. Decomposition
            decomp_res = DecompositionAgent(self.ctx).execute(
                series=clean_series, period=period
            )
            col_results[f"decomposition_{col}"] = decomp_res

            # Extract residual for outlier (post-decomposition)
            residual = None
            if decomp_res.ok:
                residual = decomp_res.data.get("residual")

            # Re-run outlier on residual (contextual)
            if residual is not None:
                ctx_out = OutlierDetectionAgent(self.ctx).execute(
                    series=clean_series, residual=residual, methods=["residual", "iqr"]
                )
                col_results[f"outliers_contextual_{col}"] = ctx_out

            # 7. Intermittency
            if run_intermittency:
                interm_res = IntermittencyAgent(self.ctx).execute(series=clean_series)
                col_results[f"intermittency_{col}"] = interm_res

            # 8. Data preparation
            acf_lags = None
            if decomp_res.ok:
                acf_lags = decomp_res.metadata.get("acf_significant_lags")

            prep_res = DataPreparationAgent(self.ctx).execute(
                series=clean_series,
                target_col=col,
                transform=transform,
                scale_method=scale_method,
                rolling_windows=rolling_windows,
                horizon=horizon,
                acf_lags=acf_lags,
            )
            col_results[f"data_prep_{col}"] = prep_res

            with lock:
                results.update(col_results)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(analyse_one, col): col for col in val_cols}
            for future in as_completed(futures):
                col = futures[future]
                try:
                    future.result()
                except Exception as e:
                    pipeline_warnings.append(f"Analysis of column '{col}' raised: {e}")
                    logger.exception("Column %s analysis failed", col)

        return results

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_compare_freqs(target_freq: str) -> List[str]:
        all_freqs = ["D", "W", "MS", "QS", "AS"]
        return [f for f in all_freqs if f != target_freq][:3]

    @staticmethod
    def _fail(msg: str, results: Dict, warnings: List[str]) -> AgentResult:
        return AgentResult(
            agent_name="Orchestrator",
            status=AgentStatus.FAILED,
            data=results,
            errors=[msg],
            warnings=warnings,
        )

    @staticmethod
    def _build_final_report(results: Dict[str, AgentResult], target_freq: str, is_hierarchy: bool) -> str:
        lines = [
            "╔══════════════════════════════════════════╗",
            "║     TEMPORALMIND — PIPELINE REPORT       ║",
            "╚══════════════════════════════════════════╝",
            "",
            f"  Target frequency : {target_freq}",
            f"  Hierarchy mode   : {'YES' if is_hierarchy else 'NO'}",
            "",
            "  Agent execution summary:",
        ]
        for agent_key, result in results.items():
            status_symbol = "✓" if result.ok else "✗"
            elapsed = getattr(result, "elapsed_seconds", 0)
            lines.append(f"    {status_symbol} {agent_key:45s}  [{result.status.value}]  {elapsed:.2f}s")

        # Collect and surface key insights
        lines += ["", "  Key insights:"]
        for key, result in results.items():
            if not result.ok:
                continue
            if key.startswith("decomposition_"):
                col = key.replace("decomposition_", "")
                meta = result.metadata
                lines.append(f"    [{col}] {meta.get('interpretation', '')}")
            if key.startswith("intermittency_"):
                col = key.replace("intermittency_", "")
                cls = result.metadata.get("summary", {}).get("classification", "?")
                adi = result.metadata.get("summary", {}).get("ADI", "?")
                lines.append(f"    [{col}] Intermittency: {cls}  ADI={adi}")
            if key.startswith("missing_values_"):
                col = key.replace("missing_values_", "")
                pct = result.metadata.get("completeness", {}).get("pct_missing_before", 0)
                if pct > 0:
                    lines.append(f"    [{col}] Missing: {pct}%  (imputed)")
            if key.startswith("outliers_") and "contextual" not in key:
                col = key.replace("outliers_", "")
                n_out = result.metadata.get("summary", {}).get("n_outliers", 0)
                if n_out > 0:
                    lines.append(f"    [{col}] Outliers detected: {n_out}")

        lines += ["", "  Warnings:"]
        for key, result in results.items():
            for w in result.warnings:
                lines.append(f"    ⚠  [{key}] {w}")

        return "\n".join(lines)
