import React, { useState, useEffect, useRef } from 'react';
import { useAppState } from '../../context/AppContext';
import { apiCall } from '../../services/api';
import { fmtNum, formatShort, stdDev, STAB_PALETTE } from '../../utils/helpers';
import { Alert, Loader, DataTable } from '../shared/UI';
import styles from './StabilityTab.module.css';

const INTERVALS = [
  { key: 'D', label: 'Day' }, { key: 'W', label: 'Week' }, { key: 'M', label: 'Month' },
  { key: 'Q', label: 'Quarter' }, { key: '6M', label: 'Semi-year' }, { key: 'Y', label: 'Year' },
  { key: 'native', label: 'Native' },
];

export default function StabilityTab({ toast }) {
  const { state } = useAppState();
  const [interval, setInterval]   = useState('W');
  const [levelCol, setLevelCol]   = useState('');
  const [parentPath, setParentPath] = useState({});
  const [valCol, setValCol]       = useState('');
  const [agg, setAgg]             = useState('sum');
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [visible, setVisible]     = useState(new Set());
  const [showEnvelope, setShowEnvelope] = useState(false);
  const chartsRef = useRef({});

  const hierCols = state.hierCols || [];
  const valCols  = (state.depCols?.length ? state.depCols : state.valueCols) || [];

  useEffect(() => {
    if (hierCols.length) setLevelCol(hierCols[0]);
    if (valCols.length)  setValCol(valCols[0]);
  }, [hierCols.length, valCols.length]);

  const levelIdx    = hierCols.indexOf(levelCol);
  const parentLevels = hierCols.slice(0, levelIdx);

  // Build and render all charts when result arrives
  useEffect(() => {
    if (!result) return;
    let Chart;
    import('chart.js/auto').then(m => {
      Chart = m.default;
      renderMainChart(Chart);
      renderEnvelopeCharts(Chart);
      renderSeasonalCharts(Chart);
    });
    return () => Object.values(chartsRef.current).forEach(c => c?.destroy());
  }, [result, visible, showEnvelope]);

  const renderMainChart = (Chart) => {
    const ctx = document.getElementById('stab-main-chart');
    if (!ctx || !result) return;
    if (chartsRef.current['main']) chartsRef.current['main'].destroy();
    const { labels, series_data, series_names } = result;
    const datasets = (series_names || [])
      .filter(n => visible.has(n))
      .map((name, i) => ({
        label: name, data: series_data?.[name] || [],
        borderColor: STAB_PALETTE[i % STAB_PALETTE.length],
        backgroundColor: 'transparent',
        pointRadius: 0, borderWidth: 1.5, tension: .2, fill: false,
      }));
    chartsRef.current['main'] = new Chart(ctx, {
      type: 'line',
      data: { labels: labels || [], datasets },
      options: { responsive: true, animation: { duration: 300 }, plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } }, scales: { x: { ticks: { color: '#3d5a7a', maxTicksLimit: 8, font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } }, y: { ticks: { color: '#3d5a7a', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } } } },
    });
  };

  const renderEnvelopeCharts = (Chart) => {
    const env = result?.envelope || {};
    if (!env.labels) return;
    ['stab-envelope-chart', 'stab-cv-chart'].forEach(id => {
      if (chartsRef.current[id]) chartsRef.current[id].destroy();
    });
    const ctx1 = document.getElementById('stab-envelope-chart');
    if (ctx1) chartsRef.current['stab-envelope-chart'] = new Chart(ctx1, {
      type: 'line',
      data: { labels: env.labels, datasets: [
        { label: 'Mean+Std', data: env.upper_band, borderColor: 'rgba(59,130,246,.4)', backgroundColor: 'rgba(59,130,246,.1)', pointRadius: 0, fill: '+1', tension: .2 },
        { label: 'Mean',     data: env.mean,       borderColor: '#3b82f6', pointRadius: 0, borderWidth: 2, tension: .2, fill: false },
        { label: 'Mean-Std', data: env.lower_band, borderColor: 'rgba(59,130,246,.4)', backgroundColor: 'rgba(59,130,246,.1)', pointRadius: 0, fill: false, tension: .2 },
      ]},
      options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#3d5a7a', maxTicksLimit: 8, font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } }, y: { ticks: { color: '#3d5a7a', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } } } },
    });
    const ctx2 = document.getElementById('stab-cv-chart');
    if (ctx2) chartsRef.current['stab-cv-chart'] = new Chart(ctx2, {
      type: 'line',
      data: { labels: env.labels, datasets: [{ label: 'CV', data: env.cv, borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,.12)', pointRadius: 0, borderWidth: 1.5, fill: true, tension: .2 }] },
      options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#3d5a7a', maxTicksLimit: 8, font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } }, y: { ticks: { color: '#3d5a7a', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } }, suggestedMax: 1 } },
    });
  };

  const renderSeasonalCharts = (Chart) => {
    ['stab-season-chart', 'stab-season-cv-chart'].forEach(id => {
      if (chartsRef.current[id]) chartsRef.current[id].destroy();
    });
    const sp = result?.seasonal_profile || {};
    if (!sp.labels) return;
    const ctx3 = document.getElementById('stab-season-chart');
    if (ctx3) chartsRef.current['stab-season-chart'] = new Chart(ctx3, {
      type: 'bar',
      data: { labels: sp.labels, datasets: [{ label: 'Avg', data: sp.avg, backgroundColor: 'rgba(59,130,246,.6)', borderColor: 'var(--a1)', borderWidth: 1 }] },
      options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#3d5a7a', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } }, y: { ticks: { color: '#3d5a7a', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } } } },
    });
    const ctx4 = document.getElementById('stab-season-cv-chart');
    if (ctx4) chartsRef.current['stab-season-cv-chart'] = new Chart(ctx4, {
      type: 'bar',
      data: { labels: sp.labels, datasets: [{ label: 'CV', data: sp.cv, backgroundColor: 'rgba(249,115,22,.6)', borderColor: 'var(--a2)', borderWidth: 1 }] },
      options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { ticks: { color: '#3d5a7a', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } }, y: { ticks: { color: '#3d5a7a', font: { family: 'IBM Plex Mono', size: 10 } }, grid: { color: 'rgba(31,45,66,.5)' } } } },
    });
  };

  const run = async () => {
    if (!state.token || !levelCol) { toast('Select a hierarchy level first', 'err'); return; }
    setLoading(true); setResult(null);
    try {
      const r = await apiCall('/api/level-stability', {
        token: state.token, level_col: levelCol, parent_path: parentPath,
        value_col: valCol, agg_method: agg, interval, max_series: 30,
      });
      setResult(r);
      setVisible(new Set(r.series_names || []));
      toast(`Stability analysis complete — ${r.n_series} series ✓`, 'ok');
    } catch (e) { toast(`Stability failed: ${e.message}`, 'err'); }
    finally { setLoading(false); }
  };

  if (!state.token) return (
    <div className="tab-panel-enter">
      <div className="section-h">Level <em>Stability Analysis</em></div>
      <Alert type="info">⬆ Upload and confirm a hierarchical dataset in Tab 1 first.</Alert>
    </div>
  );

  if (!hierCols.length) return (
    <div className="tab-panel-enter">
      <div className="section-h">Level <em>Stability Analysis</em></div>
      <Alert type="warn">⚠ No hierarchy columns detected. Assign hierarchy roles in Tab 1.</Alert>
    </div>
  );

  return (
    <div className="tab-panel-enter">
      <div className="section-h">Level <em>Stability Analysis</em></div>
      <p className="section-sub">Superimpose all series at a chosen hierarchy level. Analyse stability across time intervals — Day · Week · Month · Quarter · Semi-year · Year.</p>

      {/* Controls */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, alignItems: 'flex-end' }}>
          <div className="form-group" style={{ flex: '0 0 200px', margin: 0 }}>
            <label className="form-label">Hierarchy Level</label>
            <select className="form-control" value={levelCol} onChange={e => setLevelCol(e.target.value)} style={{ padding: '7px 10px' }}>
              {hierCols.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          {/* Parent filters */}
          {parentLevels.map((lvl, i) => (
            <div key={lvl} className="form-group" style={{ flex: '0 0 150px', margin: 0 }}>
              <label className="form-label">{lvl} (filter)</label>
              <select className="form-control" value={parentPath[lvl] || ''} onChange={e => setParentPath(p => { const np = { ...p }; if (e.target.value) np[lvl] = e.target.value; else delete np[lvl]; return np; })} style={{ padding: '7px 10px', fontSize: 12 }}>
                <option value="">All</option>
                {(state.hierLevelValues?.[lvl] || []).map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          ))}
          <div className="form-group" style={{ flex: '0 0 160px', margin: 0 }}>
            <label className="form-label">Value Column</label>
            <select className="form-control" value={valCol} onChange={e => setValCol(e.target.value)} style={{ padding: '7px 10px' }}>
              {valCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="form-group" style={{ flex: '0 0 130px', margin: 0 }}>
            <label className="form-label">Aggregate</label>
            <select className="form-control" value={agg} onChange={e => setAgg(e.target.value)} style={{ padding: '7px 10px' }}>
              <option value="sum">Sum</option>
              <option value="mean">Mean</option>
            </select>
          </div>
          <button className="btn btn-primary" onClick={run} disabled={loading}>▶ Analyse</button>
        </div>

        {/* Interval pills */}
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 11, color: 'var(--tx3)', letterSpacing: '.5px', textTransform: 'uppercase', marginBottom: 8 }}>Time Interval for Analysis</div>
          <div className={styles.intervalPills}>
            {INTERVALS.map(iv => (
              <div key={iv.key} className={`${styles.ipill}${interval === iv.key ? ' ' + styles.active : ''}`} onClick={() => setInterval(iv.key)}>{iv.label}</div>
            ))}
          </div>
        </div>
      </div>

      {loading && <Loader message={`Computing ${interval} stability for level "${levelCol}"…`} />}

      {result && (
        <>
          {/* Main superimposed chart */}
          <div className="card" style={{ marginBottom: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 12, flexWrap: 'wrap' }}>
              <div className="card-title" style={{ margin: 0 }}>📈 Superimposed Series</div>
              <div style={{ display: 'flex', gap: 8, marginLeft: 'auto', flexShrink: 0 }}>
                <button className="btn btn-sm btn-ghost" onClick={() => setShowEnvelope(v => !v)}>{showEnvelope ? 'Hide Envelope' : 'Show Envelope'}</button>
                <button className="btn btn-sm btn-ghost" onClick={() => setVisible(new Set(result.series_names || []))}>Show All</button>
                <button className="btn btn-sm btn-ghost" onClick={() => setVisible(new Set())}>Hide All</button>
              </div>
            </div>
            {/* Interactive legend */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 14 }}>
              {(result.series_names || []).map((name, i) => {
                const color = STAB_PALETTE[i % STAB_PALETTE.length];
                const isVis = visible.has(name);
                return (
                  <div key={name} onClick={() => setVisible(v => { const ns = new Set(v); ns.has(name) ? ns.delete(name) : ns.add(name); return ns; })}
                    style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '3px 10px', borderRadius: 4, border: `1px solid ${isVis ? color : 'var(--border)'}`, background: isVis ? color + '22' : 'transparent', cursor: 'pointer', fontSize: 11, fontFamily: 'var(--font-mono)', color: isVis ? 'var(--tx)' : 'var(--tx3)', transition: 'all .15s' }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: isVis ? color : 'var(--tx3)' }} />
                    {name}
                  </div>
                );
              })}
            </div>
            <canvas id="stab-main-chart" height="240" />
          </div>

          {/* Envelope */}
          {showEnvelope && (
            <div className="card" style={{ marginBottom: 18 }}>
              <div className="card-title">📊 Cross-Series Envelope (Mean ± Std · CV)</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <div><div className="chart-title">Mean ± Std Band</div><canvas id="stab-envelope-chart" height="160" /></div>
                <div><div className="chart-title">Coefficient of Variation (CV)</div><canvas id="stab-cv-chart" height="160" /></div>
              </div>
            </div>
          )}

          {/* Seasonal profiles */}
          <div className="grid-2" style={{ gap: 18, marginBottom: 18 }}>
            <div className="card">
              <div className="card-title">🗓 Seasonal Profile (avg per sub-period)</div>
              <div style={{ fontSize: 12, color: 'var(--tx2)', marginBottom: 12 }}>Average value at each calendar sub-period. Taller bars = stronger seasonal peak.</div>
              <canvas id="stab-season-chart" height="180" />
            </div>
            <div className="card">
              <div className="card-title">📐 Seasonal CV (stability per sub-period)</div>
              <div style={{ fontSize: 12, color: 'var(--tx2)', marginBottom: 12 }}>How much series <em>disagree</em> within each season. Lower CV = more consistent.</div>
              <canvas id="stab-season-cv-chart" height="180" />
            </div>
          </div>

          {/* YoY heatmap */}
          <div className="card" style={{ marginBottom: 18 }}>
            <div className="card-title">📅 Year-over-Year Heatmap (series × year)</div>
            <div style={{ fontSize: 12, color: 'var(--tx2)', marginBottom: 12 }}>Each cell = total/average for that series in that year. Colour intensity shows relative level.</div>
            <div style={{ overflowX: 'auto' }}>
              <YoYHeatmap result={result} />
            </div>
          </div>

          {/* Stability stats table */}
          <div className="card">
            <div className="card-title">📋 Per-Series Stability Statistics</div>
            <div className="tbl-wrap">
              <StabilityTable result={result} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─── YoY Heatmap ─────────────────────────────────────────────
function YoYHeatmap({ result }) {
  const yoy   = result.yoy_matrix  || {};
  const years = result.yoy_years   || [];
  const names = result.series_names || [];
  if (!names.length || !years.length) return <div style={{ color: 'var(--tx3)', fontSize: 13, padding: 8 }}>Insufficient data for year-over-year analysis.</div>;

  return (
    <>
      <table style={{ borderCollapse: 'separate', borderSpacing: 3 }}>
        <thead>
          <tr>
            <th style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--tx3)', padding: '4px 10px', textAlign: 'left', minWidth: 120 }}>Series</th>
            {years.map(y => <th key={y} style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--tx2)', padding: '4px 8px', textAlign: 'center' }}>{y}</th>)}
            <th style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--tx3)', padding: '4px 8px', textAlign: 'center' }}>Stability</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name, si) => {
            const rowVals = years.map(y => yoy[name]?.[y] ?? null);
            const valid   = rowVals.filter(v => v != null);
            const rowMax  = valid.length ? Math.max(...valid) : 1;
            const rowMin  = valid.length ? Math.min(...valid) : 0;
            const rowRange = rowMax - rowMin || 1;
            const color   = STAB_PALETTE[si % STAB_PALETTE.length];

            let stabLabel = '—';
            if (valid.length > 1) {
              const rowCV = stdDev(valid) / (Math.abs(valid.reduce((a, b) => a + b, 0) / valid.length) + 1e-9);
              const stCls = rowCV < 0.1 ? 'stab-stable' : rowCV < 0.3 ? 'stab-moderate' : 'stab-volatile';
              const stTxt = rowCV < 0.1 ? 'stable' : rowCV < 0.3 ? 'moderate' : 'volatile';
              stabLabel = <span className={`stab-badge ${stCls}`}>{stTxt}</span>;
            }

            return (
              <tr key={name}>
                <td style={{ padding: '3px 10px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--tx)' }}>{name}</span>
                  </div>
                </td>
                {years.map(y => {
                  const v = yoy[name]?.[y];
                  if (v == null) return <td key={y}><div className="heatmap-cell" style={{ background: 'var(--bg2)', color: 'var(--tx3)' }}>—</div></td>;
                  const frac  = (v - rowMin) / rowRange;
                  const r255  = Math.round(59  + frac * (239 - 59));
                  const g255  = Math.round(130 + frac * (68  - 130));
                  const b255  = Math.round(246 + frac * (68  - 246));
                  const bg    = `rgba(${r255},${g255},${b255},0.6)`;
                  const tCol  = frac > 0.6 ? '#fff' : 'var(--tx)';
                  return <td key={y} title={`${name} · ${y}: ${v.toLocaleString()}`}><div className="heatmap-cell" style={{ background: bg, color: tCol }}>{formatShort(v)}</div></td>;
                })}
                <td style={{ textAlign: 'center' }}>{stabLabel}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 12, fontSize: 11, color: 'var(--tx3)' }}>
        <span>Row-relative colour scale:</span>
        <div style={{ display: 'flex', borderRadius: 4, overflow: 'hidden', height: 12, width: 140 }}>
          <div style={{ flex: 1, background: 'rgba(59,130,246,0.6)' }} />
          <div style={{ flex: 1, background: 'rgba(149,99,157,0.6)' }} />
          <div style={{ flex: 1, background: 'rgba(239,68,68,0.6)' }} />
        </div>
        <span>Low → High (within each series)</span>
      </div>
    </>
  );
}

// ─── Stability Stats Table ────────────────────────────────────
function StabilityTable({ result }) {
  const rows = (result.series_names || []).map(name => {
    const st      = result.stability_stats?.[name] || {};
    const stLabel = st.stability || '—';
    const badgeCls = stLabel === 'stable' ? 'stab-stable' : stLabel === 'moderate' ? 'stab-moderate' : 'stab-volatile';
    return {
      Series: name,
      Mean: fmtNum(st.mean, 2), Std: fmtNum(st.std, 2), CV: fmtNum(st.cv, 4),
      Min: fmtNum(st.min, 2), Max: fmtNum(st.max, 2), Range: fmtNum(st.range, 2),
      'N obs': st.n_obs || '—',
      'Avg YoY%': st.avg_yoy_pct_change != null ? fmtNum(st.avg_yoy_pct_change, 2) + '%' : '—',
      _stability: stLabel, _cls: badgeCls,
    };
  });

  if (!rows.length) return <div style={{ padding: 10, color: 'var(--tx3)' }}>No data</div>;
  const headers = ['Series','Mean','Std','CV','Min','Max','Range','N obs','Avg YoY%','Stability'];

  return (
    <table>
      <thead><tr>{headers.map(h => <th key={h}>{h}</th>)}</tr></thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {headers.map(h => {
              if (h === 'Stability') return <td key={h}><span className={`stab-badge ${row._cls}`}>{row._stability}</span></td>;
              return <td key={h}>{row[h] ?? '—'}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
