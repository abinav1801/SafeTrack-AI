import React, { useState } from 'react';

export default function TargetForm({ onTargetRegister }) {
  const [colorName, setColorName] = useState('none');
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const fileInput = e.target.querySelector('input[type="file"]');
    const file = fileInput?.files[0];
    if (!file) {
      alert("Please select a target photo first!");
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("color_name", colorName);

    try {
      const response = await fetch("http://localhost:8000/api/register", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to register target on the server.");
      }

      const result = await response.json();
      onTargetRegister({
        colorName,
        imagePreview: preview,
        faceDetected: result.face_detected,
        dominantColorHsv: result.dominant_color_hsv
      });
    } catch (err) {
      alert("Error registering target profile: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card glass-card">
      <h2 className="card-title text-gradient">Biometric Registration</h2>
      <p className="card-subtitle">Upload target template photo or specify manual overrides</p>
      
      <form onSubmit={handleSubmit} className="form-layout">
        <div className="upload-container">
          <label className="upload-box">
            {preview ? (
              <img src={preview} alt="Target Preview" className="preview-image" />
            ) : (
              <div className="upload-placeholder">
                <span className="upload-icon">📷</span>
                <span>Select Target Photo</span>
                <span className="upload-hint">Supports JPG, PNG</span>
              </div>
            )}
            <input type="file" accept="image/*" onChange={handleFileChange} className="hidden-input" />
          </label>
        </div>

        <div className="input-group">
          <label htmlFor="color-select" className="input-label">Manual Color Override</label>
          <select 
            id="color-select" 
            value={colorName} 
            onChange={(e) => setColorName(e.target.value)}
            className="styled-select"
          >
            <option value="none">None (Extract from Photo)</option>
            <option value="red">Red</option>
            <option value="blue">Blue</option>
            <option value="green">Green</option>
            <option value="yellow">Yellow</option>
            <option value="orange">Orange</option>
            <option value="purple">Purple</option>
            <option value="black">Black</option>
            <option value="white">White</option>
            <option value="gray">Gray</option>
          </select>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? (
            <span className="spinner-loader">⏳</span>
          ) : (
            "Register Target Profile"
          )}
        </button>
      </form>
    </div>
  );
}
