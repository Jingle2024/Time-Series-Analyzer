import React, { useState, useEffect, useRef } from 'react';
import { useAppState } from '../../hooks/useAppState';
import { apiCall } from '../../services/api';
import { fmtNum, buildLineChartOptions } from '../../utils/helpers';
import { Alert, Loader, StatCard, StrengthBar, InfoRow, Divider, ModeToggle } from '../shared/UI';
import HierarchyNavigator from '../shared/HierarchyNavigator';

export default function AnalyzerTab({ toast }) {
  const { state } = useAppState();
  const [mode, setMode]         = useState('single');
  const [seriesKey, setSeriesKey] = useState('');
  const [period, setPeriod]     = useState('');
  const [nodePath, setNodePath] = useState({});
  const [aggMethod, setAggMethod] = useState('sum');
  const [nodePeriod, setNodePeriod] = useState('');
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const chartRefs = useRef({});

  useEffect(() => {
    if (state.valueCols?.length) setSeriesKey(state.valueCols[0]);
  }, [state.valueCols]);

  // Destroy & recreate charts when result changes
  useEffect(() => {
    if (!result) return;
    let Chart;
    import('chart.js/auto').then(m => {
      Chart = m.default;
      const cd = result.chart_data || {};
      renderChart(Chart, 'main-chart',     cd.labels, cd.observed,  '#3b82f6', 'Original');
      renderChart(Chart, 'trend-chart',    cd.labels, cd.trend,     '#10b981', 'Trend');
      renderChart(Chart, 'seasonal-chart', cd.labels, cd.seasonal,  '#8b5cf6', 'Seasonal');
      renderChart(Chart, 'cycle-chart',    cd.labels, cd.cycle,     '#f97316', 'Cycle');
      renderChart(Chart, 'residual-chart', cd.labels, cd.residual,  '#ef4444', 'Residual');
    });
    return () => { Object.values(chartRefs.current).forEach(c => c?.destroy()); chartRefs.current = {}; };
  }, [result]);

  const renderChart = (Chart, id, labels, data, color, label) => {
    const ctx = document.getElementById(id);
    if (!ctx || !labels || !data) return;
    if (chartRefs.current[id]) chartRefs.current[id].destroy();
    chartRefs.current[id] = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets: [{ label, data, borderColor: color, backgroundColor: color + '18', pointRadius: 0, borderWidth: 1.5, tension: 0.3, fill: true }] },
      options: buildLineChartOptions(),
    });
  };

  const runAnalysis = async () => {
    if (!state.token) { toast('No data loaded', 'err'); return; }
    setLoading(true); setResult(null);
    try {
      let r;
      if (mode === 'hierarchy') {
        if (!Object.keys(nodePath).length) { toast('Select a hierarchy node first', 'err'); setLoading(false); return; }
        r = await apiCall('/api/analyze-node', { token: state.token, node_path: nodePath, value_col: state.valueCols[0], period: parseInt(nodePeriod) || null, agg_method: aggMethod });
      } else {
        r = await apiCall('/api/analyze', { token: state.token, series_key: seriesKey, period: parseInt(period) || null });
      }
      setResult(r);
      toast('Analysis complete ✓', 'ok');
    } catch (e) { toast(`Analysis failed: ${e.message}`, 'err'); }
    finally { setLoading(false); }
  };

  if (!state.token) return (
    <div className="tab-panel-enter">
      <div className="section-h">Time Series <em>Analyzer</em></div>
      <Alert type="info">⬆ Upload and confirm a dataset in Tab 1 first.</Alert>
    </div>
  );

  const dc = result?.summary?.decomp || {};
  const st = result?.summary?.series_stats || {};
  const oc = result?.summary?.outliers || {};
  const ic = result?.summary?.intermittency || {};
  const mv = result?.summary?.missing_values || {};

  const Ft = dc.trend_strength_Ft || 0;
  const Fs = dc.seasonal_strength_Fs || 0;
  const modelRec = modelFromDecomp(dc, ic);

  return (
    <div className="tab-panel-enter">
      <div className="section-h">Time Series <em>Analyzer</em></div>
      <p className="section-sub">Select a series or hierarchy node, then run full decomposition, stationarity, outlier and intermittency analysis.</p>

      {/* Controls */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <ModeToggle
            modes={[{ value: 'single', label: 'Single Series' }, { value: 'hierarchy', label: 'Hierarchy Node' }]}
            active={mode}
            onChange={setMode}
          />
          {mode === 'single' && (
            <>
              <div className="form-group" style={{ flex: 1, margin: 0, minWidth: 140 }}>
                <label className="form-label">Value Column</label>
                <select className="form-control" value={seriesKey} onChange={e => setSeriesKey(e.target.value)} style={{ padding: '7px 10px' }}>
                  {state.valueCols.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group" style={{ flex: '0 0 120px', margin: 0 }}>
                <label className="form-label">Seasonal Period</label>
                <input type="number" className="form-control" value={period} onChange={e => setPeriod(e.target.value)} placeholder="Auto" min="2" max="365" style={{ padding: '7px 10px' }} />
              </div>
            </>
          )}
          {mode === 'hierarchy' && (
            <div style={{ flex: 1, fontSize: 11, color: 'var(--tx3)', letterSpacing: '.5px', textTransform: 'uppercase' }}>
              Select a node using the navigator below ↓
            </div>
          )}
          <button className="btn btn-primary" onClick={runAnalysis} disabled={loading}>▶ Run Analysis</button>
        </div>
      </div>

      {/* Hierarchy navigator */}
      {mode === 'hierarchy' && (
        <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 18, marginBottom: 18, alignItems: 'start' }}>
          <HierarchyNavigator idPrefix="az" onPathChange={setNodePath} />
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="card-title">⚙ Aggregation</div>
            <div className="form-group" style={{ marginBottom: 10 }}>
              <label className="form-label">Method (non-leaf nodes)</label>
              <select className="form-control" value={aggMethod} onChange={e => setAggMethod(e.target.value)}>
                <option value="sum">Sum — additive quantities</option>
                <option value="mean">Mean — stock / rate quantities</option>
              </select>
            </div>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Seasonal Period</label>
              <input type="number" className="form-control" value={nodePeriod} onChange={e => setNodePeriod(e.target.value)} placeholder="Auto-detect" min="2" max="365" />
            </div>
          </div>
        </div>
      )}

      {loading && <Loader message="Running analysis pipeline…" />}

      {result && (
        <>
          {/* Stat cards */}
          <div className="grid-4" style={{ marginBottom: 18 }}>
            <StatCard value={st.n || '—'} label="Observations" />
            <StatCard value={st.n_missing || 0} label="Missing" sub={`${fmtNum((st.n_missing || 0) / ((st.n || 1)) * 100, 1)}%`} color={st.n_missing ? 'var(--a2)' : 'var(--a3)'} />
            <StatCard value={oc.n_outliers || 0} label="Outliers" sub={`${fmtNum(oc.pct_outliers, 1)}%`} color={oc.n_outliers ? 'var(--a2)' : 'var(--a3)'} />
            <StatCard value={ic.classification || '—'} label="Demand Type" sub={`ADI=${fmtNum(ic.ADI, 2)}  CV²=${fmtNum(ic.CV2, 2)}`} color="var(--a4)" />
          </div>

          {/* Main chart + model rec */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 18, marginBottom: 18 }}>
            <div className="chart-wrap">
              <div className="chart-title">Original Series + Trend + Outliers</div>
              <canvas id="main-chart" height="200" />
            </div>
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="card-title">📊 Component Strengths</div>
              <StrengthBar label="Trend Strength (Ft)"    value={Ft} colorClass="fill-blue" />
              <StrengthBar label="Seasonal Strength (Fs)" value={Fs} colorClass="fill-violet" />
              <StrengthBar label="Outlier Rate" value={(oc.pct_outliers || 0) / 100} colorClass="fill-red" note={oc.pct_outliers > 10 ? 'High' : oc.pct_outliers > 3 ? 'Moderate' : 'Low'} />
              <Divider />
              <div className="card-title">🤖 Model Recommendation</div>
              <div style={{ background: 'rgba(59,130,246,.08)', border: '1px solid rgba(59,130,246,.25)', borderRadius: 8, padding: 12 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 16, color: 'var(--a1)', marginBottom: 5 }}>{modelRec}</div>
                <div style={{ fontSize: 12, color: 'var(--tx2)' }}>d={dc.differencing_order || 0} · period={dc.period_used || '?'}</div>
                <div style={{ fontSize: 12, color: 'var(--tx2)', marginTop: 5 }}>{dc.interpretation || ''}</div>
              </div>
            </div>
          </div>

          {/* Decomposition charts */}
          <div className="grid-2" style={{ marginBottom: 18 }}>
            <div className="chart-wrap"><div className="chart-title">Trend Component</div><canvas id="trend-chart" height="140" /></div>
            <div className="chart-wrap"><div className="chart-title">Seasonal Component</div><canvas id="seasonal-chart" height="140" /></div>
          </div>
          <div className="grid-2" style={{ marginBottom: 18 }}>
            <div className="chart-wrap"><div className="chart-title">Cycle (HP Filter)</div><canvas id="cycle-chart" height="140" /></div>
            <div className="chart-wrap"><div className="chart-title">Residual</div><canvas id="residual-chart" height="140" /></div>
          </div>

          {/* Detail cards */}
          <div className="grid-2" style={{ marginBottom: 18 }}>
            <div className="card">
              <div className="card-title">🔬 Stationarity Tests</div>
              <StationarityPanel tests={dc.stationarity || {}} />
            </div>
            <div className="card">
              <div className="card-title">⚡ Outlier Summary</div>
              <div className="grid-2" style={{ gap: 8 }}>
                <StatCard value={oc.n_outliers || 0} label="Outliers" color={oc.n_outliers ? 'var(--a2)' : 'var(--a3)'} />
                <StatCard value={`${fmtNum(oc.pct_outliers, 1)}%`} label="Rate" />
                <StatCard value={oc.high_severity || 0} label="HIGH" color="var(--a5)" />
                <StatCard value={oc.medium_severity || 0} label="MED" color="var(--a2)" />
              </div>
            </div>
          </div>
          <div className="grid-2">
            <div className="card">
              <div className="card-title">💤 Intermittency</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <InfoRow label="Class"  value={<span style={{ color: 'var(--a4)', fontWeight: 600 }}>{ic.classification || '—'}</span>} />
                <InfoRow label="ADI"    value={fmtNum(ic.ADI, 3)} />
                <InfoRow label="CV²"    value={fmtNum(ic.CV2, 3)} />
                <InfoRow label="% Zero" value={`${ic.pct_zero || 0}%`} />
              </div>
              <Divider />
              <div style={{ fontSize: 12, color: 'var(--tx2)' }}>
                <strong style={{ color: 'var(--tx)' }}>Recommended:</strong> {(ic.model_recommendations || []).slice(0, 3).join(', ') || '—'}
              </div>
            </div>
            <div className="card">
              <div className="card-title">🕳 Missing Values</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <InfoRow label="Missing"  value={`${mv.n_missing_before || 0} (${fmtNum(mv.pct_missing_before, 1)}%)`} />
                <InfoRow label="Pattern"  value={mv.pattern || '—'} />
                <InfoRow label="Max Gap"  value={`${(mv.gap_stats || {}).max_gap || 0} periods`} />
                <InfoRow label="Gaps"     value={`${(mv.gap_stats || {}).n_gaps || 0}`} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StationarityPanel({ tests }) {
  const entries = Object.entries(tests);
  if (!entries.length) return <div style={{ color: 'var(--tx3)', fontSize: 12, padding: 8 }}>No stationarity results</div>;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
      {entries.map(([test, res]) => (
        <div key={test} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '7px 11px', background: 'var(--bg2)', borderRadius: 6, border: '1px solid var(--border)' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{test}</span>
          <span style={{ fontSize: 11, color: 'var(--tx2)' }}>p={fmtNum(res.p_value, 3)}</span>
          <span className="td-tag" style={res.stationary ? { background: 'rgba(16,185,129,.15)', color: 'var(--a3)' } : { background: 'rgba(249,115,22,.15)', color: 'var(--a2)' }}>
            {res.stationary ? '✓ Stationary' : '✗ Non-stationary'}
          </span>
        </div>
      ))}
    </div>
  );
}

function modelFromDecomp(dc, ic) {
  const cls = (ic || {}).classification || 'Smooth';
  const Ft  = (dc || {}).trend_strength_Ft || 0;
  const Fs  = (dc || {}).seasonal_strength_Fs || 0;
  const d   = (dc || {}).differencing_order || 0;
  const map = { Lumpy: 'TSB / Zero-Inflated', Intermittent: 'Croston / SBA', Erratic: 'ETS(M,N,N) / TBATS' };
  if (map[cls]) return map[cls];
  if (Ft > 0.5 && Fs > 0.5) return 'Holt-Winters / SARIMA';
  if (Ft > 0.5) return 'Holt / ARIMA(p,1,0)';
  if (Fs > 0.5) return 'ETS(A,N,A) / SARIMA';
  if (d > 0) return 'ARIMA(p,d,q)';
  return 'ETS / ARIMA';
}
