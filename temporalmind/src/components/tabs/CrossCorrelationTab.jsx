import React, { useState, useEffect } from 'react';
import { useAppState } from '../../context/AppContext';
import { apiCall } from '../../services/api';
import { fmtNum } from '../../utils/helpers';
import { Alert, Loader, StatCard, StrengthBar, DataTable } from '../shared/UI';

export default function CrossCorrelationTab({ toast }) {
  const { state } = useAppState();
  const [depCol, setDepCol]       = useState('');
  const [maxLags, setMaxLags]     = useState(20);
  const [eventWindow, setEventWindow] = useState(5);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);

  useEffect(() => {
    if (state.depCols?.length) setDepCol(state.depCols[0]);
    else if (state.valueCols?.length) setDepCol(state.valueCols[0]);
  }, [state.depCols, state.valueCols]);

  const run = async () => {
    if (!state.token) return;
    setLoading(true); setResult(null);
    try {
      const r = await apiCall('/api/cross-correlation', { token: state.token, dependent_col: depCol, max_lags: maxLags, event_window: eventWindow });
      setResult(r);
      toast('Cross-correlation complete ✓', 'ok');
    } catch (e) { toast(`CC failed: ${e.message}`, 'err'); }
    finally { setLoading(false); }
  };

  if (!state.token) return (
    <div className="tab-panel-enter">
      <div className="section-h">Cross-Variable <em>Analysis</em></div>
      <Alert type="info">⬆ Upload and confirm a multi-variable dataset in Tab 1 first.</Alert>
    </div>
  );

  const allCols = state.valueCols || [];
  const depCols  = state.depCols  || [];
  const indepCols = state.indepCols || [];
  const eventCols = state.eventCols || [];

  return (
    <div className="tab-panel-enter">
      <div className="section-h">Cross-Variable <em>Analysis</em></div>
      <p className="section-sub">Understand relationships between your dependent, independent, and event variables. Detect lead/lag effects, event lifts, and Granger-causal links.</p>

      {/* Variable roles */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-title">🗂 Variable Roles</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {depCols.map(c   => <VarRoleCard key={c} col={c} role="dependent"   tag="tag-dep"   icon="🎯" />)}
          {indepCols.map(c => <VarRoleCard key={c} col={c} role="independent" tag="tag-indep" icon="📈" />)}
          {eventCols.map(c => <VarRoleCard key={c} col={c} role="event"       tag="tag-event" icon="⚡" />)}
          {!depCols.length && !indepCols.length && !eventCols.length && (
            <span style={{ color: 'var(--tx3)', fontSize: 13 }}>Confirm schema in Tab 1 to set variable roles.</span>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'flex-end' }}>
          <div className="form-group" style={{ flex: '0 0 200px', margin: 0 }}>
            <label className="form-label">Dependent Variable</label>
            <select className="form-control" value={depCol} onChange={e => setDepCol(e.target.value)} style={{ padding: '7px 10px' }}>
              {allCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="form-group" style={{ flex: '0 0 120px', margin: 0 }}>
            <label className="form-label">Max CCF Lags</label>
            <input type="number" className="form-control" value={maxLags} onChange={e => setMaxLags(+e.target.value)} min="5" max="60" />
          </div>
          <div className="form-group" style={{ flex: '0 0 140px', margin: 0 }}>
            <label className="form-label">Event Window (post)</label>
            <input type="number" className="form-control" value={eventWindow} onChange={e => setEventWindow(+e.target.value)} min="1" max="30" />
          </div>
          <button className="btn btn-primary" onClick={run} disabled={loading}>▶ Run Analysis</button>
        </div>
      </div>

      {loading && <Loader message="Computing cross-correlations…" />}

      {result && (
        <>
          {/* Correlation heatmap */}
          <div className="card" style={{ marginBottom: 18 }}>
            <div className="card-title">🌡 Pearson Correlation Matrix</div>
            <CorrelationHeatmap matrix={result.corr_matrix} cols={result.corr_cols} />
          </div>

          {/* CCF charts */}
          {Object.keys(result.ccf_data || {}).length > 0 && (
            <div style={{ marginBottom: 18 }}>
              <div className="card-title" style={{ marginBottom: 12 }}>📉 Cross-Correlation Functions (vs {result.dependent_col})</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
                {Object.entries(result.ccf_data || {}).map(([col, cd]) => (
                  <CCFCard key={col} col={col} cd={cd} depCol={result.dependent_col} eventCols={result.event_cols || []} />
                ))}
              </div>
            </div>
          )}

          {/* Event impact */}
          {Object.keys(result.event_impacts || {}).length > 0 && (
            <div style={{ marginBottom: 18 }}>
              <div className="card">
                <div className="card-title">⚡ Event Impact Analysis</div>
                <p style={{ fontSize: 13, color: 'var(--tx2)', marginBottom: 16 }}>How event periods affect the dependent variable — average lift, Cohen's d effect size, and post-event window response.</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 14 }}>
                  {Object.entries(result.event_impacts || {}).map(([col, imp]) => (
                    <EventImpactCard key={col} col={col} imp={imp} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Granger */}
          {Object.keys(result.granger_results || {}).length > 0 && (
            <div className="card" style={{ marginBottom: 18 }}>
              <div className="card-title">🔗 Granger Causality Proxy</div>
              <p style={{ fontSize: 13, color: 'var(--tx2)', marginBottom: 14 }}>Incremental R² gained by adding each independent variable (lagged) to an AR(1) baseline model.</p>
              <div className="tbl-wrap">
                <DataTable
                  rows={Object.entries(result.granger_results || {}).map(([col, g]) => ({
                    Variable: col, 'Best Lag': g.best_lag, 'R² Gain': fmtNum(g.r2_gain, 4),
                    'AR(1) R²': fmtNum(g.r2_base, 4), Useful: g.useful ? '✓ Yes' : '✗ No',
                  }))}
                />
              </div>
            </div>
          )}

          {/* Feature recommendations */}
          <div className="card">
            <div className="card-title">✅ Feature Recommendations for Forecasting</div>
            <p style={{ fontSize: 13, color: 'var(--tx2)', marginBottom: 14 }}>Variables ranked by their statistical evidence of relationship with the dependent variable.</p>
            {(result.feature_recommendations || []).length === 0
              ? <div style={{ color: 'var(--tx3)', fontSize: 13 }}>No recommendations available.</div>
              : (result.feature_recommendations || []).map(rec => <FeatureRecRow key={rec.variable} rec={rec} />)
            }
          </div>
        </>
      )}
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────

function VarRoleCard({ col, role, tag, icon }) {
  const cls = role === 'dependent' ? 'role-dep' : role === 'event' ? 'role-event' : 'role-indep';
  return (
    <div className={`var-role-card ${cls}`}>
      <span className={`td-tag ${tag}`} style={{ fontSize: 11 }}>{icon} {role}</span>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--tx)', marginTop: 6 }}>{col}</div>
    </div>
  );
}

function CorrelationHeatmap({ matrix, cols }) {
  if (!matrix || !cols || !cols.length) return <div style={{ color: 'var(--tx3)', fontSize: 13, padding: 8 }}>Insufficient data for correlation matrix.</div>;
  return (
    <>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'separate', borderSpacing: 3 }}>
          <thead>
            <tr>
              <td style={{ width: 120 }} />
              {cols.map(c => <th key={c} style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--tx2)', padding: '2px 4px', textAlign: 'center', maxWidth: 52, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={c}>{c.substring(0, 8)}</th>)}
            </tr>
          </thead>
          <tbody>
            {cols.map(row => (
              <tr key={row}>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--tx2)', padding: '2px 6px', whiteSpace: 'nowrap', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }} title={row}>{row.substring(0, 14)}</td>
                {cols.map(col => {
                  const v = (matrix[row] || {})[col];
                  const numV = v == null ? 0 : v;
                  const abs  = Math.min(1, Math.abs(numV));
                  const bg   = v == null ? 'rgba(31,45,66,0.4)' : numV > 0 ? `rgba(59,130,246,${0.12 + abs * 0.7})` : `rgba(239,68,68,${0.12 + abs * 0.7})`;
                  const textColor = abs > 0.5 ? '#fff' : numV > 0 ? 'var(--a1)' : 'var(--a5)';
                  return (
                    <td key={col} title={`${row} × ${col}: ${v != null ? v.toFixed(3) : '—'}`}>
                      <div className="corr-cell" style={{ background: bg, color: textColor }}>{v != null ? v.toFixed(2) : '—'}</div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 16, fontSize: 11, color: 'var(--tx3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 80, height: 10, borderRadius: 3, background: 'linear-gradient(90deg,rgba(239,68,68,.7),rgba(31,45,66,.4),rgba(59,130,246,.9))' }} />
          <span>−1 → 0 → +1</span>
        </div>
        <span>Red = negative · Blue = positive · Intensity = magnitude</span>
      </div>
    </>
  );
}

function CCFCard({ col, cd, depCol, eventCols }) {
  const lags    = cd.lags || [];
  const vals    = cd.ccf  || [];
  const maxAbs  = Math.max(...vals.map(Math.abs), 0.01);
  const barH    = 80;
  const isEvent = eventCols.includes(col);
  const tagCls  = isEvent ? 'tag-event' : 'tag-indep';
  const icon    = isEvent ? '⚡' : '📈';

  return (
    <div className="card" style={cd.significant ? { borderColor: 'rgba(59,130,246,.35)' } : {}}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span className={`td-tag ${tagCls}`}>{icon}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--tx)' }}>{col}</span>
        {cd.significant && <span className="chip chip-blue" style={{ marginLeft: 'auto' }}>Significant</span>}
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 0, paddingBottom: 4, borderBottom: '1px solid var(--border)', height: barH }}>
        {lags.map((lag, i) => {
          const v     = vals[i] || 0;
          const frac  = Math.abs(v) / maxAbs;
          const h     = Math.round(frac * barH);
          const isBest = lag === cd.best_lag;
          const isSig  = Math.abs(v) > (cd.sig_threshold || 0.2);
          const bg     = isBest ? 'var(--a2)' : isSig ? (v >= 0 ? 'var(--a1)' : 'var(--a5)') : 'rgba(59,130,246,0.2)';
          return (
            <div key={lag} title={`lag=${lag} ccf=${v.toFixed(3)}`} style={{ display: 'inline-flex', flexDirection: 'column-reverse', alignItems: 'center', width: lags.length <= 25 ? 12 : 8, margin: '0 1px', height: barH, justifyContent: 'flex-start' }}>
              <div style={{ width: '100%', height: h, background: bg, borderRadius: '2px 2px 0 0' }} />
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--tx3)' }}>
        <span>lag −{Math.max(...lags.map(Math.abs))}</span><span>0</span><span>lag +{Math.max(...lags.map(Math.abs))}</span>
      </div>
      <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 11, color: 'var(--tx2)' }}>Best lag: <strong style={{ color: 'var(--a2)' }}>{cd.best_lag >= 0 ? '+' : ''}{cd.best_lag}</strong></div>
        <div style={{ fontSize: 11, color: 'var(--tx2)' }}>Max |CCF|: <strong style={{ color: 'var(--tx)' }}>{Math.abs(cd.max_ccf || 0).toFixed(3)}</strong></div>
        <div style={{ fontSize: 11, color: 'var(--tx2)' }}>Sig: ±{fmtNum(cd.sig_threshold, 3)}</div>
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: 'var(--tx2)', lineHeight: 1.5 }}>{cd.interpretation || ''}</div>
    </div>
  );
}

function EventImpactCard({ col, imp }) {
  const lift      = imp.lift_pct || 0;
  const d         = imp.cohens_d || 0;
  const eff       = imp.effect_size || 'small';
  const liftColor = lift > 0 ? 'var(--a3)' : lift < 0 ? 'var(--a5)' : 'var(--tx2)';
  const effColor  = eff === 'large' ? 'var(--a3)' : eff === 'medium' ? 'var(--a2)' : 'var(--tx3)';
  return (
    <div className="card" style={{ borderColor: 'rgba(245,158,11,.3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <span className="td-tag tag-event">⚡ event</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, color: 'var(--tx)' }}>{col}</span>
      </div>
      <div className="grid-2" style={{ gap: 8 }}>
        <StatCard value={`${fmtNum(lift, 1)}%`} label="Lift vs No-Event" color={liftColor} />
        <StatCard value={fmtNum(d, 3)} label="Cohen d" />
        <StatCard value={imp.n_event_periods || 0} label="Event Periods" />
        <StatCard value={eff} label="Effect Size" color={effColor} />
      </div>
      <div style={{ marginTop: 12 }}>
        <StrengthBar label="Effect magnitude" value={Math.min(1, Math.abs(d) / 1.5)} colorClass="fill-amber" note={eff} />
      </div>
      <div style={{ fontSize: 12, color: 'var(--tx2)', marginTop: 8 }}>
        Mean on event: <strong style={{ color: 'var(--tx)' }}>{fmtNum(imp.mean_on_event, 2)}</strong> &nbsp;vs&nbsp;
        off event: <strong style={{ color: 'var(--tx)' }}>{fmtNum(imp.mean_off_event, 2)}</strong>
        &nbsp;·&nbsp; Post-event avg: <strong style={{ color: 'var(--a6)' }}>{fmtNum(imp.post_event_avg, 2)}</strong>
      </div>
    </div>
  );
}

function FeatureRecRow({ rec }) {
  const inc     = rec.include_in_model;
  const tagCls  = rec.role === 'event' ? 'tag-event' : 'tag-indep';
  const icon    = rec.role === 'event' ? '⚡' : '📈';
  const scoreStr = rec.role === 'event'
    ? `lift=${fmtNum(rec.lift_pct, 1)}% d=${fmtNum(rec.cohens_d, 3)}`
    : `CCF=${fmtNum(rec.max_ccf, 3)} lag=${rec.recommended_lag}`;
  return (
    <div className={`feat-rec-row ${inc ? 'included' : 'excluded'}`}>
      <span className={`td-tag ${tagCls}`}>{icon} {rec.role}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--tx)', flex: 1 }}>{rec.variable}</span>
      <span style={{ fontSize: 11, color: 'var(--tx2)', flex: 1 }}>{(rec.reason || '').substring(0, 55)}</span>
      <span style={{ fontSize: 11, color: 'var(--tx3)', fontFamily: 'var(--font-mono)', minWidth: 120, textAlign: 'right' }}>{scoreStr}</span>
      <span className={`td-tag ${inc ? 'tag-value' : 'tag-other'}`} style={{ minWidth: 80, textAlign: 'center' }}>{inc ? '✓ Include' : '✗ Skip'}</span>
    </div>
  );
}
