import React, { useState, useRef } from 'react';
import { useAppState } from '../../context/AppContext';
import { uploadFile, confirmSchema } from '../../services/api';
import { Alert, StatCard, DataTable, Loader } from '../shared/UI';
import styles from './UploadTab.module.css';

const ROLE_OPTIONS = [
  { value: 'timestamp',   label: '⏱ Timestamp' },
  { value: 'dependent',   label: '🎯 Dependent (target)' },
  { value: 'independent', label: '📈 Independent (exog.)' },
  { value: 'event',       label: '⚡ Event (binary 0/1)' },
  { value: 'hierarchy',   label: '🌲 Hierarchy' },
  { value: 'ignore',      label: '— Ignore' },
];

const ROLE_TAG = {
  timestamp: 'tag-ts', dependent: 'tag-dep', independent: 'tag-indep',
  event: 'tag-event', hierarchy: 'tag-hier', ignore: 'tag-other',
};

export default function UploadTab({ toast, onSwitchTab }) {
  const { state, update } = useAppState();
  const [dragging, setDragging]     = useState(false);
  const [uploading, setUploading]   = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [colRoles, setColRoles]     = useState({});
  const fileInputRef = useRef();

  const processFile = async (file) => {
    console.log('Processing file:', file);
    if (!file) return;
    console.log('Starting upload for file:', file.name);
    setUploading(true);
    toast(`Uploading ${file.name}…`, 'info');
    try {
      const meta = await uploadFile(file);
      update({ token: meta.token, filename: meta.filename, uploadMeta: meta, schemaResult: null });
      const roles = {};
      (meta.columns || []).forEach(col => {
        let r = (meta.suggested_roles || {})[col] || 'ignore';
        if (r === 'value') r = 'dependent';
        roles[col] = r;
      });
      setColRoles(roles);
      toast(`${file.name} uploaded ✓`, 'ok');
    } catch (e) {
      toast(`Upload error: ${e.message}`, 'err');
    } finally { setUploading(false); }
  };

  const handleConfirm = async () => {
    const cols = state.uploadMeta?.columns || [];
    let tsCol = null;
    const valCols = [], hierCols = [], depCols = [], indepCols = [], eventCols = [];
    const varRoles = {};
    cols.forEach(col => {
      const r = colRoles[col] || 'ignore';
      varRoles[col] = r;
      if (r === 'timestamp') tsCol = col;
      else if (r === 'hierarchy') hierCols.push(col);
      else if (r === 'dependent')   { valCols.push(col); depCols.push(col); }
      else if (r === 'independent') { valCols.push(col); indepCols.push(col); }
      else if (r === 'event')       { valCols.push(col); eventCols.push(col); }
    });
    if (!tsCol) { toast('Assign a Timestamp column', 'err'); return; }
    if (!depCols.length && !valCols.length) { toast('Assign at least one Dependent column', 'err'); return; }

    setConfirming(true);
    toast('Ingesting…', 'info');
    try {
      const r = await confirmSchema({ token: state.token, timestamp_col: tsCol, value_cols: valCols, hierarchy_cols: hierCols, variable_roles: varRoles });
      const dep  = r.dependent_cols   || depCols;
      const indep = r.independent_cols || indepCols;
      const evts  = r.event_cols       || eventCols;
      update({
        varRoles, depCols: dep, indepCols: indep, eventCols: evts,
        valueCols: dep.concat(indep).concat(evts),
        hierCols, isHierarchy: hierCols.length > 0,
        detectedFreq: r.schema?.detected_freq || null,
        schemaResult: { r, tsCol, valCols, hierCols, dep, indep, evts },
      });
      toast('Dataset ingested ✓', 'ok');
    } catch (e) {
      toast(`Ingestion error: ${e.message}`, 'err');
    } finally { setConfirming(false); }
  };

  const handleReset = () => {
    update({ token: null, filename: null, uploadMeta: null, schemaResult: null });
    setColRoles({});
    if (fileInputRef.current) fileInputRef.current.value = '';
    toast('Reset', 'info');
  };

  const { uploadMeta, schemaResult } = state;

  return (
    <div className="tab-panel-enter">
      <div className="section-h">Upload &amp; <em>Schema</em></div>
      <p className="section-sub">Upload a CSV or Excel file. Assign column roles, then confirm to ingest.</p>

      {/* Sample hint */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-title">📂 Sample Dataset Shape</div>
        <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap' }}>
          <div><div className={styles.hintLabel}>Columns</div><code className={styles.hintCode}>date, qty, promo_flag, region, store, sku</code></div>
          <div><div className={styles.hintLabel}>Hierarchy</div><code className={styles.hintCode}>date, region, store, sku, qty</code></div>
        </div>
      </div>

      {/* Upload zone */}
      {!uploadMeta && (
        <div className="card" style={{ marginBottom: 18 }}>
          <div
            className={`upload-zone${dragging ? ' dragover' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]); }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input ref={fileInputRef} type="file" accept=".csv,.xlsx,.xls" onChange={e => e.target.files[0] && processFile(e.target.files[0])} />
            <div style={{ fontSize: 36, marginBottom: 10 }}>📁</div>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>Drop a CSV or Excel file here</div>
            <div style={{ fontSize: 13, color: 'var(--tx2)' }}>or click to browse — .csv, .xlsx supported</div>
            {uploading && <Loader message="Uploading…" />}
          </div>
        </div>
      )}

      {/* Preview + schema confirm */}
      {uploadMeta && !schemaResult && (
        <>
          <div className="card" style={{ marginBottom: 18 }}>
            <div className="card-title">👁 Data Preview</div>
            <div className="grid-4" style={{ marginBottom: 14 }}>
              <StatCard value={uploadMeta.n_rows?.toLocaleString()} label="Rows" />
              <StatCard value={uploadMeta.columns?.length} label="Columns" />
              <StatCard value={uploadMeta.ts_candidates?.length} label="Date Cols" color="var(--a6)" />
              <StatCard value={uploadMeta.is_hierarchy ? 'YES' : 'NO'} label="Hierarchy" color={uploadMeta.is_hierarchy ? 'var(--a4)' : 'var(--tx3)'} />
            </div>
            <div className="tbl-wrap" style={{ maxHeight: 260, overflowY: 'auto' }}>
              <DataTable rows={uploadMeta.preview || []} />
            </div>
          </div>

          <div className="card" style={{ marginBottom: 18 }}>
            <div className="card-title">🗂 Assign Column Roles</div>
            {uploadMeta.needs_confirm && <Alert type="warn">⚠ Auto-detection uncertain. Please confirm column roles.</Alert>}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 16 }}>
              {(uploadMeta.columns || []).map(col => (
                <div key={col} className="col-row">
                  <div className="col-name" style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <span className={`td-tag ${ROLE_TAG[colRoles[col]] || 'tag-other'}`}>{colRoles[col] || '—'}</span>
                    <span style={{ color: 'var(--tx)' }}>{col}</span>
                    <span style={{ color: 'var(--tx3)', fontSize: 10 }}>{(uploadMeta.dtypes || {})[col] || ''}</span>
                  </div>
                  <span style={{ fontSize: 10, color: 'var(--tx3)', fontFamily: 'var(--font-mono)' }}>{(uploadMeta.nunique || {})[col] || ''} uniq</span>
                  <select className="form-control" value={colRoles[col] || 'ignore'} style={{ padding: '5px 9px', fontSize: 12 }} onChange={e => setColRoles(p => ({ ...p, [col]: e.target.value }))}>
                    {ROLE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </select>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 10 }}>
              <button className="btn btn-primary" onClick={handleConfirm} disabled={confirming}>{confirming ? 'Ingesting…' : '✓ Confirm Schema'}</button>
              <button className="btn btn-secondary" onClick={handleReset}>↺ Reset</button>
            </div>
            {confirming && <Loader message="Ingesting dataset…" />}
          </div>
        </>
      )}

      {/* Schema confirmed result */}
      {schemaResult && (
        <div className="card">
          <div className="card-title">✓ Schema Confirmed</div>
          <SchemaResult result={schemaResult} />
          <div style={{ marginTop: 14, display: 'flex', gap: 10 }}>
            <button className="btn btn-primary" onClick={() => onSwitchTab(1)}>→ Analyze</button>
            <button className="btn btn-secondary" onClick={() => onSwitchTab(3)}>⚙ Prepare Data</button>
            <button className="btn btn-ghost btn-sm" onClick={handleReset}>↺ Upload New File</button>
          </div>
        </div>
      )}
    </div>
  );
}

function SchemaResult({ result }) {
  const { r, tsCol, hierCols, dep, indep, evts } = result;
  const schema = r.schema || {};
  return (
    <>
      <div className="grid-4" style={{ gap: 8, marginBottom: 12 }}>
        <StatCard value={schema.n_rows} label="Rows" />
        <StatCard value={schema.detected_freq || '—'} label="Freq" color="var(--a6)" />
        <StatCard value={hierCols.length} label="Hier Levels" color={hierCols.length ? 'var(--a4)' : 'var(--tx3)'} />
        <StatCard value={dep.length + indep.length + evts.length} label="Value Cols" />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--tx2)' }}>⏱ {tsCol}</div>
        {dep.length   > 0 && <TagRow tag="tag-dep"   icon="🎯 Dependent"   cols={dep} />}
        {indep.length > 0 && <TagRow tag="tag-indep" icon="📈 Independent" cols={indep} />}
        {evts.length  > 0 && <TagRow tag="tag-event" icon="⚡ Events"       cols={evts} />}
        {hierCols.length > 0 && (
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            <span className="td-tag tag-hier">🌲 Hierarchy</span>
            <span style={{ fontSize: 11, color: 'var(--a4)', fontFamily: 'var(--font-mono)' }}>{hierCols.join(' → ')}</span>
          </div>
        )}
        {schema.date_range && <div style={{ fontSize: 11, color: 'var(--tx3)', fontFamily: 'var(--font-mono)' }}>📅 {schema.date_range[0]?.substring(0, 10)} → {schema.date_range[1]?.substring(0, 10)}</div>}
      </div>
      {(r.warnings || []).length > 0 && <Alert type="warn" style={{ marginTop: 10 }}>{r.warnings.join('; ')}</Alert>}
    </>
  );
}

function TagRow({ tag, icon, cols }) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
      <span className={`td-tag ${tag}`}>{icon}</span>
      {cols.map(c => <span key={c} style={{ fontSize: 11, color: 'var(--tx)', fontFamily: 'var(--font-mono)' }}>{c}</span>)}
    </div>
  );
}
