import React, { useState, useEffect } from 'react';
import { 
  Target, 
  Play, 
  Sparkles, 
  Code2, 
  CheckCircle, 
  Copy
} from 'lucide-react';
import { api } from '../api';

export default function PlaygroundView({
  activeDataset,
  activeSession: _activeSession,
  selectedModelId,
  selectedModelName,
  sessions: _sessions,
  onSelectModel: _onSelectModel,
}) {
  const [modelId, setModelId] = useState(selectedModelId || '');
  const [modelName, setModelName] = useState(selectedModelName || '');
  const [featureInputs, setFeatureInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [copied, setCopied] = useState(false);

  // Sync selectedModelId if passed
  useEffect(() => {
    if (selectedModelId) {
      setModelId(selectedModelId);
      setModelName(selectedModelName || 'Selected Model');
    }
  }, [selectedModelId, selectedModelName]);

  // If no modelId is set, find the best experiment from activeSession or first session
  useEffect(() => {
    if (!modelId && _activeSession) {
      api.getSession(_activeSession).then((data) => {
        const bestId = data.session?.best_experiment_id || data.experiments?.[0]?.id;
        const bestModel = data.experiments?.find(e => e.id === bestId) || data.experiments?.[0];
        if (bestId) {
          setModelId(bestId);
          setModelName(bestModel?.model_name || 'Recommended Champion');
        }
      }).catch(console.error);
    }
  }, [_activeSession, modelId]);

  // Load preview sample row to pre-fill inputs
  useEffect(() => {
    if (activeDataset?.id) {
      api.getDatasetPreview(activeDataset.id, 1).then((res) => {
        if (res.rows && res.rows.length > 0) {
          const row = { ...res.rows[0] };
          if (activeDataset.target_column) {
            delete row[activeDataset.target_column];
          }
          setFeatureInputs(row);
        }
      }).catch(console.error);
    }
  }, [activeDataset]);

  const handleInputChange = (field, value) => {
    setFeatureInputs((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handlePredict = async (e) => {
    e?.preventDefault();
    if (!modelId) {
      alert('Please select a trained model experiment.');
      return;
    }

    setPredicting(true);
    setPredictionResult(null);

    // Format types: cast numeric strings to numbers if numeric
    const cleanRow = {};
    for (const [key, val] of Object.entries(featureInputs)) {
      if (val === '' || val === null || val === undefined) {
        cleanRow[key] = null;
      } else if (!isNaN(Number(val)) && typeof val !== 'boolean') {
        cleanRow[key] = Number(val);
      } else {
        cleanRow[key] = val;
      }
    }

    const t0 = performance.now();
    try {
      const res = await api.predict(modelId, [cleanRow]);
      const latencyMs = Math.round(performance.now() - t0);
      setPredictionResult({
        ...res,
        latencyMs,
      });
    } catch (err) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setPredicting(false);
    }
  };

  const copyCode = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const pythonCode = `# Production inference snippet
import joblib
import pandas as pd

# 1. Load the trained pipeline artifact
artifact = joblib.load("model_${modelId || 'champion'}.joblib")
pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact

# 2. Prepare feature payload
features_df = pd.DataFrame([
${JSON.stringify(featureInputs, null, 4)}
])

# 3. Generate predictions
predictions = pipeline.predict(features_df)
print("Predicted target:", predictions)
`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Top Banner */}
      <div className="glass-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="card-icon"><Target size={20} /></div>
            <div>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '0.2rem' }}>Live Inference Playground</h2>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Test your trained model pipeline with live input parameters and inspect real-time predictions.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span className="badge badge-cyan">
              Model: {modelName || 'No model selected'}
            </span>
          </div>
        </div>
      </div>

      <div className="grid-cols-1-2">
        {/* Input Parameters Form */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon"><Sparkles size={18} /></div>
              <div>
                <h3 className="card-title">Feature Inputs</h3>
                <p className="card-subtitle">Adjust values and click Predict</p>
              </div>
            </div>
          </div>

          <form onSubmit={handlePredict}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', maxHeight: '420px', overflowY: 'auto', paddingRight: '0.5rem' }}>
              {Object.keys(featureInputs).length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '1rem 0' }}>
                  No feature columns detected yet. Select an active dataset in Data Studio first.
                </div>
              ) : (
                Object.entries(featureInputs).map(([col, val]) => (
                  <div key={col} className="form-group" style={{ marginBottom: 0 }}>
                    <label className="form-label" style={{ fontSize: '0.78rem' }}>
                      {col}
                    </label>
                    <input 
                      type="text" 
                      className="form-input" 
                      style={{ padding: '0.5rem 0.75rem', fontSize: '0.85rem' }}
                      value={val !== null && val !== undefined ? String(val) : ''}
                      onChange={(e) => handleInputChange(col, e.target.value)}
                    />
                  </div>
                ))
              )}
            </div>

            <button 
              type="submit" 
              className="btn btn-primary btn-lg" 
              style={{ width: '100%', marginTop: '1.25rem' }}
              disabled={predicting || !modelId}
            >
              <Play size={18} />
              {predicting ? 'Computing Inference...' : 'Execute Live Prediction'}
            </button>
          </form>
        </div>

        {/* Prediction Results & Code Integration */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Prediction Result Card */}
          <div className="glass-card" style={{ 
            background: predictionResult ? 'linear-gradient(135deg, rgba(0, 242, 254, 0.08) 0%, rgba(16, 24, 40, 0.8) 100%)' : 'var(--bg-card)',
            border: predictionResult ? '1px solid rgba(0, 242, 254, 0.3)' : '1px solid var(--border-subtle)',
          }}>
            <h3 style={{ fontSize: '1.05rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Target size={16} color="var(--accent-cyan)" /> Live Prediction Verdict
            </h3>

            {predictionResult ? (
              <div>
                <div style={{ 
                  background: 'rgba(10, 16, 28, 0.8)', 
                  border: '1px solid rgba(0, 242, 254, 0.25)', 
                  borderRadius: 'var(--radius-md)',
                  padding: '1.5rem',
                  textAlign: 'center',
                  marginBottom: '1rem'
                }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Predicted Value
                  </div>
                  <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent-cyan)', margin: '0.25rem 0' }}>
                    {typeof predictionResult.predictions[0] === 'number' 
                      ? (Number.isInteger(predictionResult.predictions[0]) 
                          ? predictionResult.predictions[0] 
                          : predictionResult.predictions[0].toFixed(4))
                      : String(predictionResult.predictions[0])}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                    Model: <strong>{predictionResult.model_name}</strong> • Latency: <strong style={{ color: 'var(--accent-emerald)' }}>{predictionResult.latencyMs}ms</strong>
                  </div>
                </div>

                {/* Class Probabilities (if classification) */}
                {predictionResult.probabilities && predictionResult.probabilities[0] && (
                  <div>
                    <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                      Class Confidence Distribution:
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                      {predictionResult.probabilities[0].map((prob, idx) => (
                        <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <span style={{ fontSize: '0.8rem', minWidth: '70px', fontFamily: 'var(--font-mono)' }}>Class {idx}</span>
                          <div style={{ flex: 1 }}>
                            <div className="progress-track" style={{ height: '8px' }}>
                              <div className="progress-fill" style={{ width: `${Math.round(prob * 100)}%` }} />
                            </div>
                          </div>
                          <span style={{ fontSize: '0.8rem', minWidth: '45px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            {Math.round(prob * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
                Click "Execute Live Prediction" to generate real-time inferences with the model pipeline.
              </div>
            )}
          </div>

          {/* Production Export Code Snippet */}
          <div className="glass-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Code2 size={16} color="var(--accent-purple)" />
                <h4 style={{ fontSize: '0.95rem' }}>Production Deployment Code</h4>
              </div>
              <button 
                className="btn btn-secondary btn-sm"
                onClick={() => copyCode(pythonCode)}
              >
                {copied ? <CheckCircle size={14} color="var(--accent-emerald)" /> : <Copy size={14} />}
                {copied ? 'Copied!' : 'Copy Python'}
              </button>
            </div>

            <pre style={{ 
              background: 'rgba(10, 14, 22, 0.95)', 
              padding: '1rem', 
              borderRadius: 'var(--radius-md)',
              fontSize: '0.78rem',
              color: 'var(--text-secondary)',
              maxHeight: '180px',
              overflowY: 'auto',
              border: '1px solid var(--border-subtle)',
            }}>
              {pythonCode}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
