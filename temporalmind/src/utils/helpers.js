/**
 * Format a number to fixed decimal places, returns '—' for null/NaN.
 */
export function fmtNum(v, d = 2) {
  if (v == null || v !== v) return '—';
  return typeof v === 'number' ? v.toFixed(d) : v;
}

/**
 * Format a large number in shorthand (1K, 1M).
 */
export function formatShort(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  return v.toFixed(0);
}

/**
 * Encode a column name so it's safe as a DOM id.
 */
export function encodeColId(col) {
  return encodeURIComponent(col).replace(/%/g, '_P_');
}

/**
 * Escape HTML characters.
 */
export function escapeHtml(v) {
  return String(v ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch])
  );
}

/**
 * Return a human-readable stability label from CV value.
 */
export function cv2label(v) {
  return v < 0.2 ? 'stable' : v < 0.5 ? 'moderate' : 'volatile';
}

/**
 * Compute standard deviation of an array.
 */
export function stdDev(arr) {
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
}

/**
 * Shared Chart.js default options for the dark theme.
 */
export const darkChartDefaults = {
  tooltipBg: 'rgba(12,16,25,.9)',
  titleColor: '#e8f0fe',
  bodyColor: '#8faec8',
  borderColor: 'rgba(59,130,246,.3)',
  gridColor: 'rgba(31,45,66,.5)',
  tickColor: '#3d5a7a',
};

/**
 * Build default line chart options for a given axis label.
 */
export function buildLineChartOptions(overrides = {}) {
  return {
    responsive: true,
    animation: { duration: 350 },
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: darkChartDefaults.tooltipBg,
        titleColor: darkChartDefaults.titleColor,
        bodyColor: darkChartDefaults.bodyColor,
        borderColor: darkChartDefaults.borderColor,
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        ticks: { color: darkChartDefaults.tickColor, maxTicksLimit: 8, font: { family: 'IBM Plex Mono', size: 10 } },
        grid: { color: darkChartDefaults.gridColor },
      },
      y: {
        ticks: { color: darkChartDefaults.tickColor, font: { family: 'IBM Plex Mono', size: 10 } },
        grid: { color: darkChartDefaults.gridColor },
      },
    },
    ...overrides,
  };
}

/**
 * Stability colour palette (for level-stability charts).
 */
export const STAB_PALETTE = [
  '#3b82f6','#10b981','#f97316','#8b5cf6','#06b6d4',
  '#ef4444','#84cc16','#f59e0b','#ec4899','#14b8a6',
];
