import React, { useState, useEffect } from 'react';
import { AppProvider } from './context/AppContext';
import { useAppState } from './hooks/useAppState';
import { checkHealth } from './services/api';
import { useToast } from './hooks/useToast';
import Header from './components/layout/Header';
import TabNav from './components/layout/TabNav';
import ToastContainer from './components/shared/ToastContainer';
import UploadTab from './components/tabs/UploadTab';
import AnalyzerTab from './components/tabs/AnalyzerTab';
import CrossCorrelationTab from './components/tabs/CrossCorrelationTab';
import PrepareDataTab from './components/tabs/PrepareDataTab';
import StabilityTab from './components/tabs/StabilityTab';
import './styles/global.css';

function AppInner() {
  const { state, update } = useAppState();
  const { toasts, toast } = useToast();
  const [activeTab, setActiveTab] = useState(0);

  // Check API health on mount
  useEffect(() => {
    checkHealth()
      .then(ok => update({ apiStatus: ok ? 'connected' : 'offline' }))
      .catch(() => update({ apiStatus: 'offline' }));
  }, []);

  const switchTab = (idx) => {
    setActiveTab(idx);
  };

  const TAB_COMPONENTS = [
    <UploadTab          key="upload" toast={toast} onSwitchTab={switchTab} />,
    <AnalyzerTab        key="analyze" toast={toast} />,
    <CrossCorrelationTab key="cc"    toast={toast} />,
    <PrepareDataTab     key="prep"   toast={toast} />,
    <StabilityTab       key="stab"   toast={toast} />,
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1 }}>
      <Header apiStatus={state.apiStatus} />
      <TabNav activeTab={activeTab} onSwitch={switchTab} />
      <main style={{ flex: 1, maxWidth: 1480, margin: '0 auto', padding: 28, width: '100%' }}>
        {TAB_COMPONENTS[activeTab]}
      </main>
      <ToastContainer toasts={toasts} />
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppInner />
    </AppProvider>
  );
}
