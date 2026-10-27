import React, { useState } from 'react';
import { 
  UploadCloud, 
  Database, 
  Sparkles, 
  Play, 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  BarChart2, 
  Layers, 
  Table as TableIcon,
  RefreshCw,
  Trash2
} from 'lucide-react';
import { api } from '../api';

export default function DataStudioView({
  datasets,
  activeDataset,
  onSelectDataset,
  onDatasetUploaded,
  onLaunchExperiment,
  onDatasetDeleted,
}) {
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState(activeDataset?.target_column || '');
  const [selectedProblemType, setSelectedProblemType] = useState(activeDataset?.problem_type || 'regression');
  const [uploadError, setUploadError] = useState(null);

  // Update target when active dataset changes
  React.useEffect(() => {
    if (activeDataset) {
      setSelectedTarget(activeDataset.target_column || (activeDataset.columns?.[0] || ''));
      setSelectedProblemType(activeDataset.problem_type || 'regression');
    }
  }, [activeDataset]);

  const processFile = async (file) => {
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const res = await api.uploadDataset(file);
      onDatasetUploaded(res.dataset);
    } catch (err) {
      setUploadError(err.message || 'Dataset upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    processFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    processFile(file);
  };

  const handleRunAnalysis = async () => {
    if (!activeDataset || !selectedTarget) return;

    setAnalyzing(true);
    try {
      const res = await api.analyzeDataset(activeDataset.id, selectedTarget, selectedProblemType);
      onDatasetUploaded(res.dataset);
    } catch (err) {
      alert(`Analysis failed: ${err.message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleDeleteDataset = async () => {
    if (!activeDataset) return;
    if (!window.confirm(`Are you sure you want to remove dataset "${activeDataset.name}"?`)) return;

    try {
      await api.deleteDataset(activeDataset.id);
      if (onDatasetDeleted) onDatasetDeleted(activeDataset.id);
    } catch (err) {
      alert(`Failed to delete dataset: ${err.message}`);
    }
  };

  const analysis = activeDataset?.analysis || {};
  const linearity = analysis.linearity;
  const relationships = analysis.feature_target_relationships || [];
  const insights = analysis.insights || [];
  const missing = analysis.missing_values || {};
  const duplicates = analysis.duplicates || {};
  const formattedFileSize = activeDataset?.file_size_bytes 
    ? `${(activeDataset.file_size_bytes / 1024).toFixed(1)} KB` 
    : '—';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Top Action Bar */}
      <div className="glass-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="card-icon"><Database size={20} /></div>
            <div>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '0.2rem' }}>Dataset Studio & EDA Engine</h2>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Upload datasets, configure target variables, probe linearity, and inspect automatic data profiles.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {/* Dataset Selector */}
            {datasets.length > 0 && (
              <select 
                className="form-select" 
                style={{ width: 'auto', minWidth: '220px' }}
                value={activeDataset?.id || ''}
                onChange={(e) => onSelectDataset(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.n_rows.toLocaleString()} rows)
                  </option>
                ))}
              </select>
            )}

            {/* Quick Upload Button */}
            <label className="btn btn-secondary btn-sm" style={{ cursor: 'pointer' }}>
              <UploadCloud size={16} />
              <span>{uploading ? 'Uploading...' : 'Quick Upload'}</span>
              <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileInput} style={{ display: 'none' }} />
            </label>

            {activeDataset && (
              <button 
                className="btn btn-secondary btn-sm" 
                title="Remove current dataset"
                onClick={handleDeleteDataset}
              >
                <Trash2 size={15} color="var(--accent-rose)" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Modern Drag & Drop Zone */}
      <div 
        className={`glass-card dropzone ${isDragging ? 'dropzone-active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          padding: '2rem 1.5rem',
          textAlign: 'center',
          border: isDragging ? '2px dashed var(--accent-cyan)' : '2px dashed var(--border-subtle)',
          background: isDragging ? 'rgba(0, 242, 254, 0.08)' : 'rgba(16, 24, 40, 0.5)',
          cursor: 'pointer',
          position: 'relative'
        }}
      >
        <input 
          type="file" 
          accept=".csv,.xlsx,.xls" 
          onChange={handleFileInput} 
          style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, opacity: 0, cursor: 'pointer' }} 
        />
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem' }}>
          <div style={{ 
            width: '52px', 
            height: '52px', 
            borderRadius: '50%', 
            background: 'rgba(0, 242, 254, 0.1)', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            color: 'var(--accent-cyan)'
          }}>
            <UploadCloud size={26} />
          </div>
          <h3 style={{ fontSize: '1.1rem' }}>
            {uploading ? 'Processing Dataset...' : 'Drag & Drop Dataset Files Here'}
          </h3>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', maxWidth: '480px' }}>
            Supports CSV, Excel (.xlsx, .xls) up to 100 MB. Files are validated for schema integrity, column types, and resource limits.
          </p>
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.2rem' }}>
            <span className="badge badge-gray">.CSV</span>
            <span className="badge badge-gray">.XLSX</span>
            <span className="badge badge-gray">Auto Type Inference</span>
          </div>
          {uploadError && (
            <div style={{ color: 'var(--accent-rose)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
              ⚠️ {uploadError}
            </div>
          )}
        </div>
      </div>

      {activeDataset ? (
        <>
          {/* Configuration & Quick Specs Card */}
          <div className="grid-cols-1-2">
            {/* Target & Task Configuration */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1.05rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles size={16} color="var(--accent-cyan)" /> Experiment Target Setup
              </h3>

              <div className="form-group">
                <label className="form-label">
                  Target Variable
                  <span style={{ fontSize: '0.72rem', color: 'var(--accent-cyan)' }}>Column to Predict</span>
                </label>
                <select 
                  className="form-select"
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                >
                  {(activeDataset.columns || []).map((col) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Task Type</label>
                <select 
                  className="form-select"
                  value={selectedProblemType}
                  onChange={(e) => setSelectedProblemType(e.target.value)}
                >
                  <option value="regression">Regression (Continuous Numerical Value)</option>
                  <option value="classification">Classification (Categories / Discrete Classes)</option>
                </select>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem' }}>
                <button 
                  className="btn btn-secondary btn-sm" 
                  style={{ flex: 1 }}
                  onClick={handleRunAnalysis}
                  disabled={analyzing}
                >
                  <RefreshCw size={14} className={analyzing ? 'spin' : ''} />
                  {analyzing ? 'Analyzing...' : 'Re-Run EDA Analysis'}
                </button>
                <button 
                  className="btn btn-primary btn-sm"
                  style={{ flex: 1 }}
                  onClick={() => onLaunchExperiment({
                    dataset_id: activeDataset.id,
                    target_column: selectedTarget,
                    problem_type: selectedProblemType,
                  })}
                >
                  <Play size={14} /> Train Models
                </button>
              </div>
            </div>

            {/* Dataset Metadata Highlights */}
            <div className="glass-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Layers size={16} color="var(--accent-purple)" /> Dataset Specifications
                </h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  File Size: <strong style={{ color: 'var(--text-primary)' }}>{formattedFileSize}</strong>
                </span>
              </div>

              <div className="grid-cols-4" style={{ gap: '1rem', marginBottom: '1.25rem' }}>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.75rem', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Row Count</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700 }}>{activeDataset.n_rows.toLocaleString()}</div>
                </div>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.75rem', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Feature Columns</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700 }}>{activeDataset.n_columns}</div>
                </div>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.75rem', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Missing Cells</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700, color: (missing.total_missing || 0) > 0 ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
                    {missing.total_missing || 0}
                  </div>
                </div>
                <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.75rem', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Duplicate Rows</div>
                  <div style={{ fontSize: '1.35rem', fontWeight: 700, color: (duplicates.duplicate_rows || 0) > 0 ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>
                    {duplicates.duplicate_rows || 0}
                  </div>
                </div>
              </div>

              {/* Linearity / Complexity Probe */}
              {linearity && (
                <div style={{ 
                  background: 'rgba(0, 242, 254, 0.05)', 
                  border: '1px solid rgba(0, 242, 254, 0.2)', 
                  padding: '1rem', 
                  borderRadius: 'var(--radius-md)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: '1rem'
                }}>
                  <div>
                    <span className="badge badge-cyan" style={{ marginBottom: '0.4rem' }}>Linearity & Complexity Probe</span>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                      Verdict: {linearity.verdict}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      Linear CV R²: <strong>{linearity.linear_cv_r2}</strong> • Depth-6 Tree CV R²: <strong>{linearity.tree_cv_r2}</strong>
                    </div>
                  </div>
                  <span className={`badge ${linearity.is_approximately_linear ? 'badge-emerald' : 'badge-purple'}`}>
                    {linearity.is_approximately_linear ? 'Linear Baseline Suitable' : 'Ensemble / Non-linear Recommended'}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Automated Insights & Top Feature Relationships */}
          <div className="grid-cols-2">
            {/* Feature Relationships */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1.05rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart2 size={16} color="var(--accent-blue)" /> Strongest Feature Relationships
              </h3>

              {relationships.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {relationships.slice(0, 6).map((rel, idx) => (
                    <div 
                      key={idx} 
                      style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'space-between',
                        padding: '0.65rem 0.85rem',
                        background: 'rgba(255, 255, 255, 0.02)',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-subtle)',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                          {rel.feature}
                        </span>
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>({rel.feature_kind})</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <span className={`badge ${rel.strength > 0.4 ? 'badge-emerald' : rel.strength > 0.2 ? 'badge-cyan' : 'badge-gray'}`}>
                          {rel.relationship} ({rel.measure}: {rel.value})
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No correlations computed yet.</p>
              )}
            </div>

            {/* Automated Agent Insights */}
            <div className="glass-card">
              <h3 style={{ fontSize: '1.05rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles size={16} color="var(--accent-amber)" /> Automated Exploratory Insights
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '340px', overflowY: 'auto' }}>
                {insights.map((item, idx) => {
                  const isWarning = item.level === 'warning';
                  const isSuccess = item.level === 'success';
                  return (
                    <div 
                      key={idx} 
                      style={{ 
                        padding: '0.75rem 0.9rem',
                        borderRadius: 'var(--radius-sm)',
                        background: isWarning ? 'rgba(245, 158, 11, 0.08)' : isSuccess ? 'rgba(16, 185, 129, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                        border: `1px solid ${isWarning ? 'rgba(245, 158, 11, 0.25)' : isSuccess ? 'rgba(16, 185, 129, 0.25)' : 'var(--border-subtle)'}`,
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.2rem' }}>
                        {isWarning ? <AlertTriangle size={14} color="var(--accent-amber)" /> : isSuccess ? <CheckCircle size={14} color="var(--accent-emerald)" /> : <Info size={14} color="var(--accent-cyan)" />}
                        <strong style={{ fontSize: '0.85rem', color: isWarning ? 'var(--accent-amber)' : isSuccess ? 'var(--accent-emerald)' : 'var(--text-primary)' }}>
                          {item.title}
                        </strong>
                      </div>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        {item.detail}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Interactive Data Preview Table */}
          <div className="glass-card">
            <div className="card-header">
              <div className="card-title-group">
                <div className="card-icon"><TableIcon size={18} /></div>
                <div>
                  <h3 className="card-title">Dataset Preview</h3>
                  <p className="card-subtitle">Sample records from {activeDataset.name}</p>
                </div>
              </div>
            </div>

            <PreviewTable datasetId={activeDataset.id} columns={activeDataset.columns || []} />
          </div>
        </>
      ) : (
        <div className="glass-card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
          <Database size={48} color="var(--text-muted)" style={{ margin: '0 auto 1rem' }} />
          <h3>No Dataset Selected</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
            Upload a CSV/Excel file or choose a sample dataset to get started.
          </p>
        </div>
      )}
    </div>
  );
}

function PreviewTable({ datasetId, columns }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);

  React.useEffect(() => {
    let isMounted = true;
    setLoading(true);
    api.getDatasetPreview(datasetId, 15)
      .then((res) => {
        if (isMounted) {
          setRows(res.rows || []);
          setLoading(false);
        }
      })
      .catch(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [datasetId]);

  if (loading) {
    return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Loading preview...</div>;
  }

  return (
    <div className="table-container">
      <table className="data-table">
        <thead>
          <tr>
            {columns.slice(0, 10).map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {columns.slice(0, 10).map((col) => (
                <td key={col} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                  {row[col] !== undefined && row[col] !== null ? String(row[col]) : '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
