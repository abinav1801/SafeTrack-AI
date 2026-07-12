import React from 'react';

export default function MetricsPanel({ isTargetRegistered, logs = [] }) {
  const defaultLogs = [
    { time: "16:20:10", msg: "[INFO] Running YOLOv8 on device: CPU" },
    { time: "16:20:12", msg: "[INFO] Loading YOLOv8 nano model..." },
    { time: "16:20:15", msg: "[INFO] Initializing InsightFace model..." },
    { time: "16:20:17", msg: "[INFO] Scaling face detector resolution to 1280x1280..." },
    { time: "16:20:19", msg: "[SUCCESS] Pipeline ready. Surveillance active." },
  ];

  const activeLogs = logs.length > 0 ? logs : defaultLogs;

  return (
    <div className="card glass-card metrics-card">
      <h2 className="card-title text-gradient">System Diagnostics</h2>
      
      <div className="metrics-grid">
        <div className="metric-box">
          <span className="metric-value font-mono">0.45</span>
          <span className="metric-label font-mono">Face Threshold</span>
        </div>
        <div className="metric-box">
          <span className="metric-value font-mono">0.50</span>
          <span className="metric-label font-mono">Color Threshold</span>
        </div>
        <div className="metric-box">
          <span className="metric-value font-mono text-success">Active</span>
          <span className="metric-label font-mono">Biometrics Override</span>
        </div>
      </div>

      <div className="log-panel">
        <h3 className="log-title font-mono">Console Logs</h3>
        <div className="log-output font-mono">
          {activeLogs.map((log, index) => (
            <div key={index} className="log-line">
              <span className="log-time text-muted">[{log.time}]</span>
              <span className="log-msg">{log.msg}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
