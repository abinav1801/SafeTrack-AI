import React, { useState } from 'react';

export default function VideoStream({ isTargetRegistered, targetData }) {
  const [active, setActive] = useState(false);

  const [loading, setLoading] = useState(false);

  const toggleStream = async () => {
    setLoading(true);
    const apiEndpoint = active ? "stop" : "start";
    try {
      const response = await fetch(`http://localhost:8000/api/${apiEndpoint}`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Failed to ${apiEndpoint} surveillance feed.`);
      }

      setActive(!active);
    } catch (err) {
      alert(`Error toggling camera feed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card glass-card stream-panel">
      <div className="card-header-row">
        <h2 className="card-title text-gradient">Live Feed Surveillance</h2>
        {active && <span className="badge badge-danger blink">REC</span>}
      </div>
      
      <div className="video-viewport">
        {active ? (
          <img 
            src="http://localhost:8000/api/stream" 
            alt="Live Surveillance Feed" 
            className="preview-image"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <div className="video-placeholder idle-feed">
            <span className="video-placeholder-icon">📺</span>
            <p className="font-mono text-muted">Webcam Stream Offline</p>
          </div>
        )}
      </div>

      <div className="controls-row">
        <button 
          onClick={toggleStream} 
          className={`btn ${active ? 'btn-danger' : 'btn-success'}`}
          disabled={(!isTargetRegistered && !active) || loading}
        >
          {loading ? "Processing..." : (active ? "Stop Surveillance Feed" : "Start Live Surveillance")}
        </button>
        {!isTargetRegistered && !active && (
          <p className="status-hint font-mono text-warning">⚠️ Please register a target profile first</p>
        )}
      </div>
    </div>
  );
}
