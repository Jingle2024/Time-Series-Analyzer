"""
tests/test_pipeline.py
───────────────────────
End-to-end tests using synthetic data.
Run with:  python -m pytest tests/ -v
Or simply: python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(name)-30s %(levelname)s %(message)s")


# ── synthetic data generators ─────────────────────────────────────────────────

def make_single_series(n=365, freq="D", seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq=freq)
    trend = np.linspace(100, 150, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.normal(0, 5, n)
    # inject some outliers
    values = trend + seasonal + noise
    values[50] += 200
    values[min(200, n-1)] -= 150
    # inject some NaNs
    values[30:33] = np.nan
    values[180] = np.nan
    return pd.Series(values, index=dates, name="sales")


def make_hierarchy_df(seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=104, freq="W")  # 2 years weekly
    records = []
    regions = ["North", "South"]
    stores  = {"North": ["StoreA", "StoreB"], "South": ["StoreC", "StoreD"]}
    skus    = ["SKU-1", "SKU-2"]
    for region in regions:
        for store in stores[region]:
            for sku in skus:
                base = rng.uniform(50, 200)
                seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
                noise = rng.normal(0, 8, len(dates))
                # introduce zeros (intermittent demand)
                values = np.maximum(0, base + seasonal + noise)
                zero_idx = rng.choice(len(dates), size=10, replace=False)
                values[zero_idx] = 0
                for i, (dt, v) in enumerate(zip(dates, values)):
                    records.append({
                        "date": dt, "region": region,
                        "store": store, "sku": sku,
                        "demand": round(float(v), 2),
                    })
    return pd.DataFrame(records)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_ingestion():
    print("\n── Test: Ingestion ──")
    from agents.ingestion_agent import IngestionAgent
    series = make_single_series()
    df = series.reset_index()
    df.columns = ["date", "sales"]

    agent = IngestionAgent()
    result = agent.execute(source=df, timestamp_col="date", value_cols=["sales"])
    assert result.ok, f"Ingestion failed: {result.errors}"
    assert isinstance(result.data, pd.DataFrame)
    assert result.data.index.name == "timestamp"
    print(f"  ✓ Loaded {len(result.data)} rows, freq={result.metadata['detected_freq']}")


def test_interval_advisor():
    print("\n── Test: Interval Advisor ──")
    from agents.interval_advisor_agent import IntervalAdvisorAgent
    series = make_single_series(n=365)

    agent = IntervalAdvisorAgent()
    result = agent.execute(series=series)
    assert result.ok, f"IntervalAdvisor failed: {result.errors}"
    recs = result.data
    assert len(recs) > 0
    best = result.metadata["best_interval"]
    print(f"  ✓ Top recommendation: {result.metadata['best_alias']} ({best})")
    for r in recs:
        print(f"    {r['alias']:12s} score={r['score']:6.2f}  info_loss={r['info_loss_pct']}%  {r['rationale'][:60]}")


def test_accumulation():
    print("\n── Test: Accumulation ──")
    from agents.accumulation_agent import AccumulationAgent
    series = make_single_series(n=365)
    df = series.to_frame()

    agent = AccumulationAgent()
    result = agent.execute(
        df=df, target_freq="W", method="sum",
        compare_freqs=["MS", "QS"]
    )
    assert result.ok, f"Accumulation failed: {result.errors}"
    print(f"  ✓ Daily({len(df)}) → Weekly({len(result.data)})")
    print(f"    Compression: {result.metadata['compression_ratio']}x")
    print(f"    Variance retention: {result.metadata['information_retention']}")


def test_hierarchy():
    print("\n── Test: Hierarchy Aggregation ──")
    from agents.hierarchy_aggregation_agent import HierarchyAggregationAgent

    df = make_hierarchy_df()
    df = df.set_index("date")

    agent = HierarchyAggregationAgent()
    result = agent.execute(
        df=df,
        hierarchy_cols=["region", "store", "sku"],
        value_cols=["demand"],
        method="sum",
    )
    assert result.ok, f"Hierarchy failed: {result.errors}"
    print(f"  ✓ {result.metadata['n_total_series']} series across levels")
    for lvl, meta in result.metadata["level_meta"].items():
        print(f"    {lvl:15s}: {meta['n_series']} series")
    print(f"  Coherence: {result.metadata['coherence']}")
    print("  Level effects (avg CV per level):")
    for lvl, eff in result.metadata["level_effects"].items():
        print(f"    {lvl:10s}: avg_cv={eff['avg_cv']}")


def test_decomposition():
    print("\n── Test: Decomposition ──")
    from agents.decomposition_agent import DecompositionAgent
    series = make_single_series(n=365).fillna(0)

    agent = DecompositionAgent()
    result = agent.execute(series=series, period=7)
    assert result.ok, f"Decomposition failed: {result.errors}"
    meta = result.metadata
    print(f"  ✓ Ft={meta['trend_strength_Ft']:.3f}  Fs={meta['seasonal_strength_Fs']:.3f}")
    print(f"  Interpretation: {meta['interpretation']}")
    print(f"  Stationarity: {meta['stationarity'].get('ADF', {}).get('stationary', '?')}")


def test_outlier_detection():
    print("\n── Test: Outlier Detection ──")
    from agents.outlier_detection_agent import OutlierDetectionAgent
    series = make_single_series(n=200).fillna(0)

    agent = OutlierDetectionAgent()
    result = agent.execute(series=series, methods=["iqr", "zscore", "isof"])
    assert result.ok, f"Outlier failed: {result.errors}"
    s = result.metadata["summary"]
    print(f"  ✓ {s['n_outliers']} outliers ({s['pct_outliers']}%)")
    print(f"    HIGH={s['high_severity']}  MED={s['medium_severity']}  LOW={s['low_severity']}")
    if len(result.data["outlier_table"]) > 0:
        top = result.data["outlier_table"].head(3)
        for _, row in top.iterrows():
            print(f"    {str(row['timestamp'])[:10]}  val={row['value']:.2f}  {row['severity']}  → {row['treatment']}")


def test_missing_values():
    print("\n── Test: Missing Values ──")
    from agents.missing_values_agent import MissingValuesAgent
    series = make_single_series(n=200)  # has NaNs

    agent = MissingValuesAgent()
    result = agent.execute(series=series, method="spline")
    assert result.ok, f"MissingValues failed: {result.errors}"
    c = result.metadata["completeness"]
    print(f"  ✓ Before: {c['n_missing_before']} missing ({c['pct_missing_before']}%)")
    print(f"    After : {c['n_missing_after']} missing")
    print(f"    Pattern: {c['pattern']}  Method: {c['method_used']}")


def test_intermittency():
    print("\n── Test: Intermittency ──")
    from agents.intermittency_agent import IntermittencyAgent
    # create a lumpy series
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=104, freq="W")
    values = np.zeros(104)
    nonzero = rng.choice(104, size=30, replace=False)
    values[nonzero] = rng.exponential(scale=50, size=30)
    series = pd.Series(values, index=dates, name="sku_demand")

    agent = IntermittencyAgent()
    result = agent.execute(series=series)
    assert result.ok, f"Intermittency failed: {result.errors}"
    s = result.metadata["summary"]
    print(f"  ✓ ADI={s['ADI']}  CV²={s['CV2']}")
    print(f"    Classification: {s['classification']}")
    print(f"    Models: {s['model_recommendations']}")


def test_data_preparation():
    print("\n── Test: Data Preparation ──")
    from agents.data_preparation_agent import DataPreparationAgent
    series = make_single_series(n=365).ffill()

    agent = DataPreparationAgent()
    result = agent.execute(
        series=series,
        transform="log",
        scale_method="minmax",
        rolling_windows=[7, 14],
        horizon=13,
    )
    assert result.ok, f"DataPrep failed: {result.errors}"
    s = result.metadata["summary"]
    print(f"  ✓ Features: {s['n_features']}")
    print(f"    Lag features   : {s['lag_features']}")
    print(f"    Rolling features: {len(s['rolling_features'])}")
    print(f"    Calendar feats : {len(s['calendar_features'])}")
    print(f"    Train/Val/Test : {s['splits']}")


def test_full_pipeline_single():
    print("\n── Test: Full Pipeline (single series) ──")
    from agents.orchestrator import Orchestrator
    series = make_single_series(n=365)
    df = series.reset_index()
    df.columns = ["date", "sales"]

    orch = Orchestrator()
    result = orch.run(
        source=df,
        timestamp_col="date",
        value_cols=["sales"],
        target_freq="W",
        quantity_type="flow",
        horizon=13,
    )
    assert result.ok, f"Pipeline failed: {result.errors}"
    print(result.metadata["final_report"])


def test_full_pipeline_hierarchy():
    print("\n── Test: Full Pipeline (hierarchy) ──")
    from agents.orchestrator import Orchestrator
    df = make_hierarchy_df()

    orch = Orchestrator()
    result = orch.run(
        source=df,
        timestamp_col="date",
        value_cols=["demand"],
        hierarchy_cols=["region", "store", "sku"],
        target_freq="MS",
        quantity_type="flow",
        horizon=6,
    )
    assert result.ok, f"Hierarchy pipeline failed: {result.errors}"
    print(result.metadata["final_report"])


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_ingestion,
        test_interval_advisor,
        test_accumulation,
        test_hierarchy,
        test_decomposition,
        test_outlier_detection,
        test_missing_values,
        test_intermittency,
        test_data_preparation,
        test_full_pipeline_single,
        test_full_pipeline_hierarchy,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n  ✗ FAILED: {t.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'═'*50}")
    print(f"Results: {passed} passed / {failed} failed / {len(tests)} total")
