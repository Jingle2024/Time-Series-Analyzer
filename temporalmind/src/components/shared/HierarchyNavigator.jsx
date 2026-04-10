import React, { useState, useEffect } from 'react';
import { useAppState } from '../../context/AppContext';
import { useHierarchy } from '../../hooks/useHierarchy';
import { Chip } from '../shared/UI';

/**
 * Reusable hierarchy navigator with:
 * - Collapsible visual tree
 * - Cascade dropdowns
 * - Breadcrumb of selected path
 * - Node meta chips
 *
 * Props:
 *   idPrefix   – string prefix for dropdown IDs (prevents collisions between tabs)
 *   onPathChange(path) – called when selection changes
 */
export default function HierarchyNavigator({ idPrefix = 'hn', onPathChange }) {
  const { state } = useAppState();
  const { hierLevels, hierLevelValues, hierTree } = state;
  const { fetchChildValues } = useHierarchy();

  const [path, setPath]         = useState({});
  const [treeOpen, setTreeOpen] = useState(false);
  const [dropVals, setDropVals] = useState({});   // { levelIdx: [values] }

  // Populate level 0 on mount
  useEffect(() => {
    if (!hierLevels.length) return;
    const vals = hierLevelValues[hierLevels[0]] || [];
    setDropVals({ 0: vals });
  }, [hierLevels, hierLevelValues]);

  const handleChange = async (levelIdx, value) => {
    // Build new path
    const newPath = {};
    for (let i = 0; i < levelIdx; i++) {
      const k = hierLevels[i];
      if (path[k]) newPath[k] = path[k];
    }
    if (value) newPath[hierLevels[levelIdx]] = value;

    // Populate downstream level
    const newDropVals = { ...dropVals };
    for (let i = levelIdx + 1; i < hierLevels.length; i++) {
      newDropVals[i] = [];
    }
    if (value && levelIdx + 1 < hierLevels.length) {
      try {
        const vals = await fetchChildValues(newPath, levelIdx + 1);
        newDropVals[levelIdx + 1] = vals;
      } catch (_) {}
    }

    setPath(newPath);
    setDropVals(newDropVals);
    onPathChange?.(newPath);
  };

  const depth  = Object.keys(path).length;
  const total  = hierLevels.length;
  const isLeaf = depth === total;

  const totalLeaves = (() => {
    if (!hierTree) return 0;
    const count = (node) => {
      if (!node || typeof node !== 'object' || Array.isArray(node)) return 1;
      return Object.values(node).reduce((s, v) => s + count(v), 0);
    };
    return count(hierTree);
  })();

  return (
    <div className="hier-nav-panel">
      <div className="hier-nav-header">
        🌲 Hierarchy Navigator
        <span className="chip chip-violet" style={{ marginLeft: 'auto' }}>{totalLeaves} leaves</span>
      </div>

      {/* Visual tree toggle */}
      {hierTree && (
        <>
          <div className="vis-tree-toggle" onClick={() => setTreeOpen(o => !o)}>
            <span>{treeOpen ? '▼' : '▶'}</span>
            <span>Visual Tree</span>
            <span style={{ marginLeft: 'auto', fontSize: 10 }}>{treeOpen ? 'click to collapse' : 'click to expand'}</span>
          </div>
          {treeOpen && (
            <div className="vis-tree-wrap">
              <VisualTreeNode
                label="Total (all series)"
                levelIdx={-1}
                path={{}}
                subtree={hierTree}
                hierLevels={hierLevels}
                onSelect={p => { setPath(p); onPathChange?.(p); }}
              />
            </div>
          )}
        </>
      )}

      {/* Cascading dropdowns */}
      <div style={{ padding: '14px 16px', borderTop: '1px solid var(--border)' }}>
        <div style={{ fontSize: 11, color: 'var(--tx3)', letterSpacing: '.5px', textTransform: 'uppercase', marginBottom: 10 }}>
          Cascading Selection
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {hierLevels.map((lvl, idx) => (
            <div key={lvl} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--a4)',
                background: 'rgba(139,92,246,.1)', border: '1px solid rgba(139,92,246,.25)',
                borderRadius: 4, padding: '2px 7px', whiteSpace: 'nowrap', minWidth: 80, textAlign: 'center',
              }}>{lvl}</span>
              <select
                id={`${idPrefix}-cs-${idx}`}
                className="form-control"
                style={{ fontSize: 12 }}
                value={path[lvl] || ''}
                disabled={idx > 0 && !path[hierLevels[idx - 1]]}
                onChange={e => handleChange(idx, e.target.value)}
              >
                <option value="">— All —</option>
                {(dropVals[idx] || []).map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          ))}
        </div>
      </div>

      {/* Breadcrumb */}
      <div className="node-breadcrumb">
        <div className="breadcrumb-label">Selected Node</div>
        <div className="breadcrumb-path">
          {Object.entries(path).length === 0
            ? <span style={{ color: 'var(--tx3)', fontSize: 12 }}>No node selected yet</span>
            : Object.entries(path).map(([k, v], i, arr) => (
                <React.Fragment key={k}>
                  <span className="crumb">{k}: {v}</span>
                  {i < arr.length - 1 && <span className="crumb-sep">›</span>}
                </React.Fragment>
              ))
          }
        </div>
      </div>

      {/* Node meta */}
      {depth > 0 && (
        <div className="node-meta-row">
          <Chip variant="violet">Depth {depth}/{total}</Chip>
          <Chip variant={isLeaf ? 'green' : 'amber'}>{isLeaf ? 'Leaf Node' : 'Aggregate Node'}</Chip>
          <Chip variant="blue">{depth} level{depth > 1 ? 's' : ''} selected</Chip>
        </div>
      )}
    </div>
  );
}

// ─── Visual Tree Node (recursive, lazy) ─────────────────────
function VisualTreeNode({ label, levelIdx, path, subtree, hierLevels, onSelect }) {
  const [open, setOpen]     = useState(false);
  const [built, setBuilt]   = useState(false);
  const [selected, setSelected] = useState(false);

  const hasChildren = subtree && typeof subtree === 'object' && !Array.isArray(subtree) && Object.keys(subtree).length > 0;

  const handleToggle = (e) => {
    e.stopPropagation();
    if (hasChildren) {
      if (!built) setBuilt(true);
      setOpen(o => !o);
    }
  };

  const handleSelect = () => {
    setSelected(true);
    onSelect(levelIdx >= 0 ? { ...path } : {});
  };

  return (
    <div>
      <div
        className={`vtree-node${selected ? ' selected' : ''}`}
        onClick={handleSelect}
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 0, padding: '3px 0', fontSize: 12, fontFamily: 'var(--font-mono)' }}
      >
        {/* Indent */}
        {Array.from({ length: Math.max(0, levelIdx + 1) }).map((_, i) => (
          <div key={i} style={{ display: 'inline-block', width: 20, position: 'relative', flexShrink: 0 }}>
            {i < levelIdx && <div style={{ position: 'absolute', left: 9, top: 0, bottom: 0, width: 1, background: 'var(--border2)' }} />}
            {i === levelIdx && <div style={{ position: 'absolute', left: 9, top: '50%', width: 11, height: 1, background: 'var(--border2)' }} />}
          </div>
        ))}
        {/* Toggle button */}
        <div
          onClick={handleToggle}
          style={{
            width: 16, height: 16, borderRadius: 3, flexShrink: 0, marginRight: 5,
            border: '1px solid var(--border2)', background: 'var(--bg3)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 9, cursor: hasChildren ? 'pointer' : 'default',
            color: 'var(--tx2)', opacity: hasChildren ? 1 : 0.3,
          }}
        >
          {hasChildren ? (open ? '▼' : '▶') : '·'}
        </div>
        <span style={{ color: selected ? 'var(--a1)' : 'var(--tx2)' }}>
          {label}
          {hasChildren && <span style={{ color: 'var(--tx3)', marginLeft: 4, fontSize: 10 }}>({Object.keys(subtree).length})</span>}
        </span>
      </div>
      {/* Children */}
      {open && built && hasChildren && (
        <div>
          {Object.keys(subtree).map(val => {
            const childLvl  = levelIdx + 1;
            const childPath = { ...path };
            if (childLvl < hierLevels.length) childPath[hierLevels[childLvl]] = val;
            const childSubtree = subtree[val];
            return (
              <VisualTreeNode
                key={val}
                label={val}
                levelIdx={childLvl}
                path={childPath}
                subtree={childSubtree}
                hierLevels={hierLevels}
                onSelect={onSelect}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
