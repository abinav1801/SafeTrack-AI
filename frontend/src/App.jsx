import React, { useState } from 'react';
import TargetForm from './components/TargetForm';
import VideoStream from './components/VideoStream';
import MetricsPanel from './components/MetricsPanel';

export default function App() {
  const [targetRegistered, setTargetRegistered] = useState(false);
  const [targetData, setTargetData] = useState(null);
  const [logs, setLogs] = useState([]);

  const handleTargetRegister = (data) => {
    setTargetRegistered(true);
    setTargetData(data);
    
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [
      ...prev,
      { time: timestamp, msg: `[SUCCESS] Registered target profile. Override: ${data.colorName}` },
      { time: timestamp, msg: `[INFO] Initializing surveillance webcam capture...` }
    ]);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title text-gradient">SafeTrack AI</h1>
        <p className="app-tagline font-mono">INTELLIGENT SURVEILLANCE & TARGET SEARCH PIPELINE</p>
      </header>

      <main className="dashboard-grid">
        <div className="sidebar-col">
          <TargetForm onTargetRegister={handleTargetRegister} />
          <MetricsPanel isTargetRegistered={targetRegistered} logs={logs} />
        </div>
        <div className="main-col">
          <VideoStream isTargetRegistered={targetRegistered} targetData={targetData} />
        </div>
      </main>
    </div>
  );
}
