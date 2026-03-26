"""
example_usage.py
─────────────────
Demonstrates how to use each agent individually AND the full
orchestrated pipeline. Run from the ts_agents/ directory:

    python example_usage.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING)   # set INFO for verbose output


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Load from CSV (or simulate one)
# ═══════════════════════════════════════════════════════════════════════════════

def example_single_series_from_csv():
    """
    Simulates a CSV file and runs the full pipeline on a single time series.
    Replace csv_text with pd.read_csv("your_file.csv") for real data.
    """
    print("=" * 60)
    print("EXAMPLE 1: Single series from CSV-like data")
    print("=" * 60)

    # Simulate CSV content
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    trend   = np.linspace(200, 300, 500)
    weekly  = 30 * np.sin(2 * np.pi * np.arange(500) / 7)
    annual  = 50 * np.sin(2 * np.pi * np.arange(500) / 365)
    noise   = rng.normal(0, 10, 500)
    values  = trend + weekly + annual + noise
    # add missing values and a spike
    values[100:102] = np.nan
    values[300] += 500

    df = pd.DataFrame({"date": dates, "revenue": values})
    # Optionally: df.to_csv("revenue.csv", index=False)

    from agents.orchestrator import Orchestrator
    orch = Orchestrator()
    result = orch.run(
        source=df,
        timestamp_col="date",
        value_cols=["revenue"],
        # target_freq=None → auto-recommended by IntervalAdvisorAgent
        quantity_type="flow",
        horizon=30,
    )

    if result.ok:
        print(result.metadata["final_report"])

        # Access specific sub-results
        decomp = result.data.get("decomposition_revenue")
        if decomp and decomp.ok:
            print(f"\nDecomposition summary:")
            print(f"  Trend strength   : {decomp.metadata['trend_strength_Ft']:.3f}")
            print(f"  Seasonal strength: {decomp.metadata['seasonal_strength_Fs']:.3f}")

        prep = result.data.get("data_prep_revenue")
        if prep and prep.ok:
            X_train = prep.data["X_train"]
            y_train = prep.data["y_train"]
            print(f"\nForecast-ready data:")
            print(f"  X_train shape : {X_train.shape}")
            print(f"  y_train shape : {y_train.shape}")
            print(f"  Features      : {list(X_train.columns)[:6]} ...")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Hierarchical series (Excel-style multi-column)
# ═══════════════════════════════════════════════════════════════════════════════

def example_hierarchy():
    """
    Builds a 3-level hierarchy (Region → Store → SKU) and runs the pipeline.
    Replace the DataFrame construction with pd.read_excel("hierarchy.xlsx")
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Hierarchical time series")
    print("=" * 60)

    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=104, freq="W")
    rows = []
    for region in ["East", "West"]:
        for store in ["Store1", "Store2"]:
            for sku in ["A", "B", "C"]:
                base = rng.uniform(30, 100)
                y = base + 10 * np.sin(2*np.pi*np.arange(104)/52) + rng.normal(0, 5, 104)
                y = np.clip(y, 0, None)
                for dt, v in zip(dates, y):
                    rows.append({"date": dt, "region": region,
                                 "store": store, "sku": sku, "qty": round(v, 1)})
    df = pd.DataFrame(rows)

    from agents.orchestrator import Orchestrator
    orch = Orchestrator()
    result = orch.run(
        source=df,
        timestamp_col="date",
        value_cols=["qty"],
        hierarchy_cols=["region", "store", "sku"],
        target_freq="MS",
        quantity_type="flow",
        method="sum",
        horizon=6,
    )

    if result.ok:
        # Show hierarchy effects
        hier = result.data.get("hierarchy")
        if hier and hier.ok:
            print("\nHierarchy level effects:")
            for level, eff in hier.metadata["level_effects"].items():
                print(f"  {level:15s}: avg_cv={eff['avg_cv']}  ({eff['interpretation']})")

            print("\nAll generated series:")
            for key in sorted(hier.data.keys()):
                frame = hier.data[key]
                print(f"  {key:50s}: {len(frame)} periods")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Using agents individually (fine-grained control)
# ═══════════════════════════════════════════════════════════════════════════════

def example_individual_agents():
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Individual agents (fine-grained control)")
    print("=" * 60)

    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    y = 100 + np.cumsum(rng.normal(0, 2, 200))
    y[50] = np.nan; y[51] = np.nan
    y[120] += 300  # spike
    series = pd.Series(y, index=dates, name="price")

    # 1. Missing values
    from agents.missing_values_agent import MissingValuesAgent
    mv = MissingValuesAgent()
    mv_result = mv.execute(series=series, method="auto")
    clean = mv_result.data["imputed"]
    print(f"\n[MissingValues] pattern={mv_result.metadata['completeness']['pattern']}"
          f"  imputed {mv_result.metadata['completeness']['n_missing_before']} gaps")

    # 2. Outlier detection
    from agents.outlier_detection_agent import OutlierDetectionAgent
    od = OutlierDetectionAgent()
    od_result = od.execute(series=clean, methods=["iqr", "zscore", "isof"])
    print(f"[Outliers] {od_result.metadata['summary']['n_outliers']} found")

    # 3. Interval advisor
    from agents.interval_advisor_agent import IntervalAdvisorAgent
    ia = IntervalAdvisorAgent()
    ia_result = ia.execute(series=clean, top_n=3)
    print(f"[IntervalAdvisor] best={ia_result.metadata['best_alias']} ({ia_result.metadata['best_interval']})")

    # 4. Accumulate to weekly
    from agents.accumulation_agent import AccumulationAgent
    acc = AccumulationAgent()
    acc_result = acc.execute(df=clean.to_frame(), target_freq="W", method="last",
                              quantity_type="stock")
    weekly = acc_result.data
    print(f"[Accumulation] {len(clean)} daily → {len(weekly)} weekly (method=last)")

    # 5. Decomposition on weekly series
    from agents.decomposition_agent import DecompositionAgent
    dec = DecompositionAgent()
    dec_result = dec.execute(series=weekly.iloc[:, 0], period=52)
    if dec_result.ok:
        m = dec_result.metadata
        print(f"[Decomposition] Ft={m['trend_strength_Ft']:.3f}  Fs={m['seasonal_strength_Fs']:.3f}")
        print(f"  {m['interpretation']}")

    # 6. Data prep
    from agents.data_preparation_agent import DataPreparationAgent
    dp = DataPreparationAgent()
    dp_result = dp.execute(
        series=weekly.iloc[:, 0],
        transform="log",
        scale_method="robust",
        rolling_windows=[4, 8],
        acf_lags=dec_result.metadata.get("acf_significant_lags") if dec_result.ok else None,
    )
    if dp_result.ok:
        s = dp_result.metadata["summary"]
        print(f"[DataPrep] {s['n_features']} features, "
              f"train={s['splits']['train_idx']['n']} / "
              f"val={s['splits']['val_idx']['n']} / "
              f"test={s['splits']['test_idx']['n']}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: JSON from API (dict / list of records)
# ═══════════════════════════════════════════════════════════════════════════════

def example_json_source():
    print("\n" + "=" * 60)
    print("EXAMPLE 4: JSON source (API response format)")
    print("=" * 60)

    # Simulate an API response
    import json
    rng = np.random.default_rng(3)
    records = [
        {"ts": f"2023-{m:02d}-01", "value": round(float(100 + 20*np.sin(m/2) + rng.normal(0,5)), 2)}
        for m in range(1, 37)  # 3 years monthly
    ]
    # Could also be: requests.get(url).json()

    from agents.ingestion_agent import IngestionAgent
    from agents.interval_advisor_agent import IntervalAdvisorAgent

    ingest = IngestionAgent()
    result = ingest.execute(source=records, timestamp_col="ts", value_cols=["value"])
    if result.ok:
        print(f"[Ingestion] {len(result.data)} rows, freq={result.metadata['detected_freq']}")
        ia = IntervalAdvisorAgent()
        ia_r = ia.execute(series=result.data["value"])
        if ia_r.ok:
            print(f"[Interval] Best: {ia_r.metadata['best_alias']}")
            print(f"  {ia_r.metadata['summary']}")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    example_single_series_from_csv()
    example_hierarchy()
    example_individual_agents()
    example_json_source()
    print("\n✓ All examples complete.")
