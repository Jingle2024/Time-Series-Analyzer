import React, { useState, useEffect, useRef } from 'react';
import { useAppState } from '../../context/AppContext';
import { apiCall, getDownloadUrl } from '../../services/api';
import { fmtNum, buildLineChartOptions } from '../../utils/helpers';
import { Alert, Loader, StatCard, DataTable, Toggle } from '../shared/UI';
import HierarchyNavigator from '../shared/HierarchyNavigator';

const MODEL_PALETTE = ['#8b5cf6','#06b6d4','#f97316','#10b981','#ef4444','#f59e0b','#3b82f6','#ec4899'];

export default function PrepareDataTab({ toast }) {
  const { state, update } = useAppState();
  const [mode, setMode]           = useState('single');
  const [nodePath, setNodePath]   = useState({});
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [compareModels, setCompareModels] = useState([]);
  const [forecastOnly, setForecastOnly]   = useState(false);
  const chartRef = useRef(null);

  // Form state
  const [form, setForm] = useState({
    depCol: '', transform: 'auto', scale: 'minmax', intervalMode: 'session',
    targetFreq: '', qtyType: 'flow', accumMethod: 'auto',
    applyMissing: true, missingMethod: 'auto', missingPeriod: 7, zeroMissing: false,
    applyOutlier: true, outlierIqr: true, outlierZscore: true, outlierIsof: true, outlierLof: false,
    outlierTreatment: 'cap', windows: '7,14,28', calendar: true,
    negAllowed: false, comboModels: false,
    trainRatio: 0.70, valRatio: 0.15, horizon: 13, holdout: 0,
    outputFormat: 'excel',
  });

  useEffect(() => {
    const dep = (state.depCols?.length ? state.depCols : state.valueCols) || [];
    if (dep.length) setForm(f => ({ ...f, depCol: dep[0] }));
    if (state.isHierarchy) setMode('hierarchy'); else setMode('single');
  }, [state.depCols, state.valueCols, state.isHierarchy]);

  const setF = (key, val) => setForm(f => ({ ...f, [key]: val }));

  const getModelColor = (name, isSelected) => {
    const models = result?.available_models || [];
    const idx = Math.max(0, models.indexOf(name));
    const c = MODEL_PALETTE[idx % MODEL_PALETTE.length];
    return isSelected ? c : c + 'cc';
  };

  // Rebuild chart whenever selection/forecastOnly/result changes
  useEffect(() => {
    if (!result) return;
    let instance;
    import('chart.js/auto').then(m => {
      const Chart = m.default;
      const ctx = document.getElementById('prep-timeline-chart');
      if (!ctx) return;
      if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }

      const tl   = result.timeline || {};
      const mo   = result.model_outputs || {};
      const sel  = selectedModel || result.selected_model || result.best_model || null;
      const sOut = sel && mo[sel] ? mo[sel] : null;

      const allLabels = tl.labels || [];
      const allFutFc  = (sOut?.future_forecast) || tl.future_forecast || [];
      let sliceStart  = 0;

      if (forecastOnly) {
        const fi = allFutFc.findIndex(v => typeof v === 'number' && !isNaN(v));
        if (fi > 0) sliceStart = fi;
      }

      const slice = arr => forecastOnly ? (arr || []).slice(sliceStart) : (arr || []);
      const labels     = slice(allLabels);
      const train      = slice(tl.train);
      const validation = slice(tl.validation);
      const test       = slice(tl.test);
      const holdout    = slice(tl.holdout);
      const obs        = slice(tl.observed);
      const futFc      = slice(allFutFc);
      const futLo      = slice((sOut?.future_lower_95) || tl.future_lower_95);
      const futHi      = slice((sOut?.future_upper_95) || tl.future_upper_95);
      const modelFc    = slice((sOut?.forecast_path) || tl.model_forecast);

      const compareSets = compareModels
        .filter(n => n !== sel && mo[n])
        .map(name => ({
          label: `Compare · ${name}`,
          data: forecastOnly ? (mo[name]?.future_forecast || []).slice(sliceStart) : (mo[name]?.forecast_path || []),
          borderColor: getModelColor(name, false),
          backgroundColor: 'transparent',
          pointRadius: forecastOnly ? 3 : 0,
          borderWidth: forecastOnly ? 2 : 1.3,
          borderDash: [4, 4],
          tension: .2, fill: false,
        }));

      const datasets = [
        ...(forecastOnly ? [] : [
          { label: 'Train',      data: train,      borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.08)',  pointRadius: 0, borderWidth: 2, tension: .2, fill: false },
          { label: 'Validation', data: validation, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.08)',  pointRadius: 0, borderWidth: 2, tension: .2, fill: false },
          { label: 'Test',       data: test,       borderColor: '#14b8a6', backgroundColor: 'rgba(20,184,166,.08)',  pointRadius: 0, borderWidth: 2, tension: .2, fill: false },
          { label: 'Holdout (actual)', data: holdout, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,.08)', pointRadius: 2, borderWidth: 2, tension: .2, fill: false },
          { label: `Forecast · ${sel || ''}`, data: modelFc, borderColor: getModelColor(sel, true), backgroundColor: 'rgba(139,92,246,.08)', pointRadius: 0, borderWidth: 1.5, borderDash: [8, 5], tension: .2, fill: false },
        ]),
        { label: `Future · ${sel || ''}`, data: futFc, borderColor: getModelColor(sel, true), backgroundColor: 'rgba(124,58,237,.08)', pointRadius: forecastOnly ? 3 : 0, borderWidth: forecastOnly ? 3 : 2.5, tension: .2, fill: false },
        { label: '95% Lower', data: futLo, borderColor: 'rgba(139,92,246,.25)', pointRadius: 0, borderWidth: 1, tension: .2, fill: false },
        { label: '95% Upper', data: futHi, borderColor: 'rgba(139,92,246,.25)', pointRadius: 0, borderWidth: 1, tension: .2, fill: '-1', backgroundColor: 'rgba(139,92,246,.12)' },
        ...compareSets,
      ];

      chartRef.current = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: buildLineChartOptions({ plugins: { legend: { display: true, labels: { color: 'var(--tx2)', font: { size: 11 } } } } }),
      });
    });
    return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
  }, [result, selectedModel, compareModels, forecastOnly]);

  const run = async () => {
    if (!state.token || !form.depCol) { toast('Select a dependent series', 'err'); return; }
    const outlierMethods = [
      form.outlierIqr    && 'iqr',
      form.outlierZscore && 'zscore',
      form.outlierIsof   && 'isof',
      form.outlierLof    && 'lof',
    ].filter(Boolean);
    if (form.applyOutlier && !outlierMethods.length) { toast('Select at least one outlier method', 'err'); return; }
    if (mode === 'hierarchy' && !Object.keys(nodePath).length) { toast('Select a hierarchy path', 'err'); return; }

    setLoading(true); setResult(null);
    try {
      const r = await apiCall('/api/forecast-prepare', {
        token: state.token, mode, dep_col: form.depCol,
        node_path: mode === 'hierarchy' ? nodePath : {},
        transform: form.transform, scale_method: form.scale,
        interval_mode: form.intervalMode, target_freq: form.targetFreq || null,
        quantity_type: form.qtyType, accumulation_method: form.accumMethod,
        apply_missing_treatment: form.applyMissing, missing_method: form.missingMethod,
        missing_period: form.missingPeriod, missing_zero_as_missing: form.zeroMissing,
        apply_outlier_treatment: form.applyOutlier, outlier_methods: outlierMethods,
        outlier_treatment: form.outlierTreatment,
        rolling_windows: form.windows.split(',').map(v => parseInt(v.trim(), 10)).filter(v => !isNaN(v) && v > 0),
        add_calendar: form.calendar, train_ratio: form.trainRatio,
        val_ratio: form.valRatio, horizon: form.horizon, n_holdout: form.holdout,
        output_format: form.outputFormat,
        allow_negative_forecast: form.negAllowed,
        enable_combination_models: form.comboModels,
      });
      const avail = (r.available_models || []).filter(Boolean);
      const sel   = (avail.includes(r.selected_model) && r.selected_model) || (avail.includes(r.best_model) && r.best_model) || avail[0] || null;
      setSelectedModel(sel);
      setCompareModels([]);
      setResult(r);
      update({ forecastDownloadToken: r.download_token, forecastDownloadExt: r.download_ext || 'xlsx' });
      toast('Forecast preparation complete ✓', 'ok');
    } catch (e) { toast(`Forecast prep failed: ${e.message}`, 'err'); }
    finally { setLoading(false); }
  };

  const toggleCompare = (name) => {
    if (name === selectedModel) return;
    setCompareModels(prev => {
      const s = new Set(prev);
      if (s.has(name)) { s.delete(name); }
      else { if (s.size >= 3) { const f = [...s][0]; s.delete(f); } s.add(name); }
      return [...s];
    });
  };

  const download = () => {
    const { forecastDownloadToken, forecastDownloadExt } = state;
    if (!forecastDownloadToken) { toast('No export available', 'err'); return; }
    window.open(getDownloadUrl(forecastDownloadToken, forecastDownloadExt), '_blank');
  };

  const depCols = (state.depCols?.length ? state.depCols : state.valueCols) || [];

  if (!state.token) return (
    <div className="tab-panel-enter">
      <div className="section-h">Forecast <em>Preparation</em></div>
      <Alert type="info">⬆ Upload and confirm a dataset in Tab 1 first.</Alert>
    </div>
  );

  return (
    <div className="tab-panel-enter">
      <div className="section-h">Forecast <em>Preparation</em></div>
      <p className="section-sub">Build one model-ready series dataset with explicit train/validation/test/holdout/future splits.</p>

      {/* Mode */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Mode &amp; Series</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
          <button className={`btn ${mode === 'single' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('single')}>Single Series</button>
          {state.isHierarchy && <button className={`btn ${mode === 'hierarchy' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('hierarchy')}>Hierarchy Node</button>}
        </div>
        <div className="grid-2">
          <div className="form-group">
            <label className="form-label">Dependent Series</label>
            <select className="form-control" value={form.depCol} onChange={e => setF('depCol', e.target.value)}>
              {depCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>
        {mode === 'hierarchy' && state.isHierarchy && (
          <div style={{ marginTop: 14 }}>
            <HierarchyNavigator idPrefix="prep" onPathChange={setNodePath} />
          </div>
        )}
      </div>

      {/* Config */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Splits &amp; Horizon</div>
        <div className="grid-3">
          <div className="form-group"><label className="form-label">Train Ratio</label><input type="number" className="form-control" value={form.trainRatio} onChange={e => setF('trainRatio', +e.target.value)} step="0.05" min="0.4" max="0.9" /></div>
          <div className="form-group"><label className="form-label">Val Ratio</label><input type="number" className="form-control" value={form.valRatio} onChange={e => setF('valRatio', +e.target.value)} step="0.05" min="0.05" max="0.3" /></div>
          <div className="form-group"><label className="form-label">Horizon (periods)</label><input type="number" className="form-control" value={form.horizon} onChange={e => setF('horizon', +e.target.value)} min="1" /></div>
          <div className="form-group"><label className="form-label">Holdout Periods</label><input type="number" className="form-control" value={form.holdout} onChange={e => setF('holdout', +e.target.value)} min="0" /></div>
          <div className="form-group"><label className="form-label">Transform</label>
            <select className="form-control" value={form.transform} onChange={e => setF('transform', e.target.value)}>
              {['auto','none','log','sqrt','boxcox'].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div className="form-group"><label className="form-label">Scale Method</label>
            <select className="form-control" value={form.scale} onChange={e => setF('scale', e.target.value)}>
              {['minmax','standard','robust','none'].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Interval &amp; Frequency</div>
        <div className="grid-2">
          <div className="form-group"><label className="form-label">Interval Mode</label>
            <select className="form-control" value={form.intervalMode} onChange={e => setF('intervalMode', e.target.value)}>
              <option value="session">Use Session Interval {state.detectedFreq ? `(${state.detectedFreq})` : ''}</option>
              <option value="manual">Manual</option>
            </select>
          </div>
          <div className="form-group"><label className="form-label">Target Frequency</label>
            <select className="form-control" value={form.targetFreq} onChange={e => setF('targetFreq', e.target.value)} disabled={form.intervalMode !== 'manual'}>
              {['D','W','M','Q','Y','H','T'].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div className="form-group"><label className="form-label">Quantity Type</label>
            <select className="form-control" value={form.qtyType} onChange={e => setF('qtyType', e.target.value)}>
              <option value="flow">Flow (sales, demand)</option>
              <option value="stock">Stock (inventory, rate)</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Missing &amp; Outliers</div>
        <div className="grid-2">
          <div className="form-group"><label className="form-label">Missing Method</label>
            <select className="form-control" value={form.missingMethod} onChange={e => setF('missingMethod', e.target.value)}>
              {['auto','linear','spline','seasonal','forward_fill','backward_fill','knn','mean','median','zero'].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div className="form-group"><label className="form-label">Missing Seasonal Period</label>
            <input type="number" className="form-control" value={form.missingPeriod} onChange={e => setF('missingPeriod', +e.target.value)} min="2" />
          </div>
        </div>
        <div className="grid-3" style={{ marginTop: 6 }}>
          <Toggle id="zero-missing" checked={form.zeroMissing} onChange={v => setF('zeroMissing', v)} label="Treat zero as missing" />
          <Toggle id="apply-outlier" checked={form.applyOutlier} onChange={v => setF('applyOutlier', v)} label="Apply Outlier Treatment" />
          <div className="form-group"><label className="form-label">Outlier Treatment</label>
            <select className="form-control" value={form.outlierTreatment} onChange={e => setF('outlierTreatment', e.target.value)}>
              <option value="cap">Cap (IQR fence)</option>
              <option value="remove">Remove + Impute</option>
              <option value="keep">Keep (detect only)</option>
            </select>
          </div>
        </div>
        <div className="form-group" style={{ marginTop: 6 }}>
          <label className="form-label">Outlier Methods</label>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {[['outlierIqr','IQR'],['outlierZscore','Z-Score'],['outlierIsof','IsolationForest'],['outlierLof','LOF']].map(([key, label]) => (
              <label key={key} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 13, color: 'var(--tx2)' }}>
                <input type="checkbox" checked={form[key]} onChange={e => setF(key, e.target.checked)} /> {label}
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Features &amp; Export</div>
        <div className="grid-2">
          <div className="form-group"><label className="form-label">Rolling Windows (comma-separated)</label>
            <input type="text" className="form-control" value={form.windows} onChange={e => setF('windows', e.target.value)} />
          </div>
        </div>
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginTop: 6 }}>
          <Toggle id="calendar"      checked={form.calendar}     onChange={v => setF('calendar', v)}     label="Calendar features" />
          <Toggle id="neg-allowed"   checked={form.negAllowed}   onChange={v => setF('negAllowed', v)}   label="Negative Forecast Allowed" />
          <Toggle id="combo-models"  checked={form.comboModels}  onChange={v => setF('comboModels', v)}  label="Combination Models" />
        </div>
        <div className="form-group" style={{ marginTop: 10 }}>
          <label className="form-label">Export Format</label>
          <div className="radio-pills">
            <div className="radio-pill"><input type="radio" name="prep-out-fmt" id="out-excel" value="excel" checked={form.outputFormat === 'excel'} onChange={() => setF('outputFormat', 'excel')} /><label htmlFor="out-excel">Excel (.xlsx)</label></div>
            <div className="radio-pill"><input type="radio" name="prep-out-fmt" id="out-csv"   value="csv"   checked={form.outputFormat === 'csv'}   onChange={() => setF('outputFormat', 'csv')}   /><label htmlFor="out-csv">CSV Bundle (.zip)</label></div>
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 10 }}>
          <button className="btn btn-primary btn-lg" onClick={run} disabled={loading}>Run Forecast Preparation</button>
        </div>
        {loading && <Loader message="Preparing series…" />}
      </div>

      {/* Results */}
      {result && (
        <>
          <div className="grid-4" style={{ marginBottom: 16 }}>
            {(() => {
              const sp    = result.series_profile || {};
              const split = result.split_counts   || {};
              const splitsTxt = Object.entries(split).map(([k, v]) => `${k}:${v}`).join(' | ');
              return <>
                <StatCard value={sp.n_total    || 0}   label="Observations" />
                <StatCard value={sp.n_features || 0}   label="Features" />
                <StatCard value={sp.freq       || '—'} label="Frequency" color="var(--a6)" />
                <StatCard value={splitsTxt     || 'n/a'} label="Splits" />
              </>;
            })()}
          </div>

          {/* Timeline */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 12 }}>
              <div className="card-title" style={{ margin: 0 }}>📈 Timeline</div>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 12px', borderRadius: 20, border: '1px solid var(--border)', cursor: 'pointer', fontSize: 12, color: 'var(--tx2)' }}>
                <input type="checkbox" style={{ display: 'none' }} checked={forecastOnly} onChange={e => setForecastOnly(e.target.checked)} />
                🔭 {forecastOnly ? 'Forecast Only' : 'Full Timeline'}
              </label>
            </div>
            {/* Model selector */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
              {(result.available_models || []).filter(Boolean).map(name => {
                const isSel  = name === selectedModel;
                const isCmp  = compareModels.includes(name);
                const color  = getModelColor(name, true);
                return (
                  <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderRadius: 8, border: `1px solid ${isSel ? color : 'var(--border)'}`, background: isSel ? color + '22' : 'var(--bg2)', cursor: 'pointer' }}
                    onClick={() => setSelectedModel(name)}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                    <span style={{ fontSize: 12, color: isSel ? 'var(--tx)' : 'var(--tx2)', fontFamily: 'var(--font-mono)' }}>{name}</span>
                    {!isSel && <button className="btn btn-ghost btn-sm" style={{ padding: '2px 8px', fontSize: 10 }} onClick={e => { e.stopPropagation(); toggleCompare(name); }}>{isCmp ? 'Compared' : 'Compare'}</button>}
                  </div>
                );
              })}
            </div>
            <canvas id="prep-timeline-chart" />
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Model Comparison (MAPE)</div>
            <div className="tbl-wrap" style={{ maxHeight: 220, overflow: 'auto' }}>
              <DataTable rows={(result.model_comparison || []).map(x => ({ model: x.model || '—', mape: x.mape == null ? '—' : x.mape, rmse: x.rmse == null ? '—' : x.rmse, mae: x.mae == null ? '—' : x.mae, status: x.status || '—' }))} />
            </div>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Feature Matrix Preview</div>
            <div className="tbl-wrap" style={{ maxHeight: 280, overflow: 'auto' }}><DataTable rows={result.feature_preview || []} /></div>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Future Frame Preview</div>
            <div className="tbl-wrap" style={{ maxHeight: 220, overflow: 'auto' }}><DataTable rows={result.future_preview || []} /></div>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Agent Report</div>
            <pre style={{ whiteSpace: 'pre-wrap', color: 'var(--tx2)', fontSize: 12, lineHeight: 1.45, margin: 0 }}>{result.report || 'No report generated.'}</pre>
          </div>

          <div className="card">
            <div className="card-title">Export</div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <button className="btn btn-download btn-lg" onClick={download}>⬇ Download Forecast Prep Output</button>
              <span style={{ fontSize: 12, color: 'var(--tx2)' }}>Ready: {(state.forecastDownloadExt || '').toUpperCase()} export</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
