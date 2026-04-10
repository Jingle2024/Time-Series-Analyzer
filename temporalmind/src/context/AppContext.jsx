import React, { createContext, useContext, useState, useCallback } from 'react';

// ─── API base ────────────────────────────────────────────────
export const API_BASE = "https://goldfish-app-avb6e.ondigitalocean.app";
// export const API_BASE = 'http://127.0.0.1:8080';

// ─── Initial state ───────────────────────────────────────────
const initialState = {
  token: null,
  filename: null,
  uploadMeta: null,

  valueCols: [],
  hierCols: [],
  isHierarchy: false,
  varRoles: {},
  depCols: [],
  indepCols: [],
  eventCols: [],
  detectedFreq: null,
  targetFreq: null,

  downloadToken: null,
  downloadExt: 'csv',
  forecastDownloadToken: null,
  forecastDownloadExt: 'xlsx',

  prepResult: null,
  prepSelectedModel: null,
  prepCompareModels: [],
  prepForecastOnly: false,
  prepMode: 'single',

  // Hierarchy tree state
  hierTree: null,
  hierLevels: [],
  hierLevelValues: {},
  hierTreeDeferred: false,
  hierChildCache: {},
  currentNodePath: {},
  analyzeMode: 'single',

  // API status
  apiStatus: 'unknown', // 'connected' | 'offline' | 'unknown'
};

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [state, setState] = useState(initialState);

  const update = useCallback((patch) => {
    setState(prev => ({ ...prev, ...patch }));
  }, []);

  const reset = useCallback(() => {
    setState(initialState);
  }, []);

  return (
    <AppContext.Provider value={{ state, update, reset }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppState() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppState must be used within AppProvider');
  return ctx;
}
