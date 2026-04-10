import React from 'react';
import { fmtNum } from '../../utils/helpers';

// ─── Spinner / Loader ────────────────────────────────────────
export function Loader({ message = 'Loading…' }) {
  return (
    <div className="loader-wrap">
      <div className="spinner" />
      <span className="loader-text">{message}</span>
    </div>
  );
}

// ─── Alert ───────────────────────────────────────────────────
export function Alert({ type = 'info', children }) {
  return <div className={`alert alert-${type}`}>{children}</div>;
}

// ─── Stat Card ───────────────────────────────────────────────
export function StatCard({ value, label, sub, color = 'var(--tx)' }) {
  return (
    <div className="stat-card">
      <div className="stat-val" style={{ color }}>{value}</div>
      <div className="stat-label">{label}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

// ─── Strength Bar ────────────────────────────────────────────
export function StrengthBar({ label, value, colorClass, note }) {
  const pct = Math.round((value || 0) * 100);
  const n = note || (pct > 60 ? 'Strong' : pct > 30 ? 'Moderate' : 'Weak');
  return (
    <div className="strength-bar-wrap">
      <div className="strength-label">
        <span>{label}</span>
        <span>{fmtNum(value, 3)} — {n}</span>
      </div>
      <div className="strength-track">
        <div className={`strength-fill ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ─── Info Row ────────────────────────────────────────────────
export function InfoRow({ label, value }) {
  return (
    <div className="info-row">
      <span className="info-key">{label}</span>
      <span className="info-val">{value ?? '—'}</span>
    </div>
  );
}

// ─── Divider ─────────────────────────────────────────────────
export function Divider() {
  return <div className="divider" />;
}

// ─── Data Table ──────────────────────────────────────────────
const TAG_CLASSES = {
  split: (v) => `td-tag tag-${v}`,
  model_type_recommendation: () => 'td-tag tag-model',
};

export function DataTable({ rows, columns }) {
  if (!rows || !rows.length) {
    return <div style={{ padding: '10px', color: 'var(--tx3)', fontSize: '13px' }}>No data</div>;
  }
  const headers = columns || Object.keys(rows[0]);
  return (
    <table>
      <thead>
        <tr>{headers.map(h => <th key={h}>{h}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, ri) => (
          <tr key={ri}>
            {headers.map(col => {
              const v = row[col];
              const tagFn = TAG_CLASSES[col];
              if (tagFn) {
                return <td key={col}><span className={tagFn(v)}>{v ?? '—'}</span></td>;
              }
              return <td key={col}>{v == null ? '—' : String(v).substring(0, 40)}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ─── Toggle ──────────────────────────────────────────────────
export function Toggle({ id, checked, onChange, label }) {
  return (
    <div className="toggle-group">
      <label className="toggle">
        <input type="checkbox" id={id} checked={checked} onChange={e => onChange(e.target.checked)} />
        <span className="toggle-slider" />
      </label>
      {label && <span>{label}</span>}
    </div>
  );
}

// ─── Chip ────────────────────────────────────────────────────
export function Chip({ variant = 'blue', children, style }) {
  return <span className={`chip chip-${variant}`} style={style}>{children}</span>;
}

// ─── Section Header ──────────────────────────────────────────
export function SectionHeader({ title, em, sub }) {
  return (
    <>
      <div className="section-h">
        {title} {em && <em>{em}</em>}
      </div>
      {sub && <p className="section-sub">{sub}</p>}
    </>
  );
}

// ─── Mode Toggle Button Group ────────────────────────────────
export function ModeToggle({ modes, active, onChange }) {
  return (
    <div style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden', flexShrink: 0 }}>
      {modes.map((m, i) => (
        <button
          key={m.value}
          className="btn btn-sm"
          style={{
            borderRadius: 0,
            border: 'none',
            borderLeft: i > 0 ? '1px solid var(--border)' : 'none',
            background: active === m.value ? 'var(--a1)' : 'transparent',
            color: active === m.value ? '#fff' : 'var(--tx2)',
          }}
          onClick={() => onChange(m.value)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
