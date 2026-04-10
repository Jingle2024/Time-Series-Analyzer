import { API_BASE } from '../context/AppContext';

/**
 * Generic JSON API call (POST by default).
 */
export async function apiCall(path, body = null, method = 'POST') {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(API_BASE + path, opts);
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      msg = j.detail || JSON.stringify(j);
    } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

/**
 * Upload a file (multipart/form-data POST).
 */
export async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch(API_BASE + '/api/upload', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
  return r.json();
}

/**
 * Confirm schema after upload.
 */
export async function confirmSchema(payload) {
  return apiCall('/api/confirm-schema', payload);
}

/**
 * Run time-series analysis.
 */
export async function runAnalysis(payload) {
  return apiCall('/api/analyze', payload);
}

/**
 * Run cross-correlation analysis.
 */
export async function runCrossCorrelation(payload) {
  return apiCall('/api/cross-correlation', payload);
}

/**
 * Run forecast preparation.
 */
export async function runForecastPrep(payload) {
  return apiCall('/api/forecast-prep', payload);
}

/**
 * Get hierarchy children for cascade dropdown.
 */
export async function getHierarchyChildren(payload) {
  return apiCall('/api/hierarchy-children', payload);
}

/**
 * Run level stability analysis.
 */
export async function runLevelStability(payload) {
  return apiCall('/api/level-stability', payload);
}

/**
 * Check server health.
 */
export async function checkHealth() {
  const r = await fetch(API_BASE + '/health');
  return r.ok;
}

/**
 * Build a download URL for forecast-prep output.
 */
export function getDownloadUrl(token, ext) {
  return `${API_BASE}/api/download/${token}.${ext}`;
}
