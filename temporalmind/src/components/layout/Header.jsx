import React from 'react';
import styles from './Header.module.css';

export default function Header({ apiStatus }) {
  const statusClass = apiStatus === 'connected' ? 'chip chip-green'
    : apiStatus === 'offline'    ? 'chip chip-amber'
    : 'chip chip-blue';

  const statusText = apiStatus === 'connected' ? '● CONNECTED'
    : apiStatus === 'offline'    ? '● OFFLINE (demo)'
    : '● CHECKING…';

  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <div className={styles.logoMark}>⏱</div>
        Temporal<em>Mind</em>
      </div>

      <div className={styles.headerRight}>
        <span className={`chip chip-blue ${styles.tagline}`}>Time Series Intelligence</span>
        <span id="api-status" className={statusClass}>{statusText}</span>
      </div>
    </header>
  );
}
