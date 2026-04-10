import React from 'react';

export default function ToastContainer({ toasts }) {
  return (
    <div id="toast-root">
      {toasts.map(t => (
        <div key={t.id} className={`toast ${t.type}`}>
          {t.msg}
        </div>
      ))}
    </div>
  );
}
