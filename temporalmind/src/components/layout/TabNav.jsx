import React from 'react';
import styles from './TabNav.module.css';

const TABS = [
  { label: 'Upload & Schema' },
  { label: 'Analyzer' },
  { label: 'Cross-Correlation' },
  { label: 'Prepare Data' },
  { label: 'Level Stability' },
];

export default function TabNav({ activeTab, onSwitch }) {
  return (
    <nav className={styles.tabsWrapper}>
      {TABS.map((tab, i) => (
        <button
          key={i}
          id={`tab-${i}`}
          className={`${styles.tabBtn} ${activeTab === i ? styles.active : ''}`}
          onClick={() => onSwitch(i)}
        >
          <span className={styles.tabNum}>{i + 1}</span>
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
