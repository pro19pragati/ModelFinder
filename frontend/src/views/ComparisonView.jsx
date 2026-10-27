import React, { useState, useEffect } from 'react';
import { 
  Award, 
  Download, 
  BarChart3, 
  CheckCircle, 
  Target, 
  Sliders, 
  TrendingUp, 
  FileSpreadsheet 
} from 'lucide-react';
import { api } from '../api';

export default function ComparisonView({
  activeSession,
  sessions,
  onSelectSession,
  onSelectModelForPredict,
  setActiveTab
}) {
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedExpForDetails, setSelectedExpForDetails] = useState(null);
  const [activeChartTab, setActiveChartTab] = useState('performance');

  useEffect(() => {
    if (!activeSession) return;
    setLoading(true);
    api.getSessionComparison(activeSession)
      .then((data) => {
        setComparison(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load comparison:', err);
        setLoading(false);
      });
  }, [activeSession]);

  if (!activeSession) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
        <Award size={48} color="var(--accent-amber)" style={{ margin: '0 auto 1rem' }} />
        <h3>No Session Selected</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
          Select an experimentation session from the list below to view its leaderboard, model comparison charts, and artifacts.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          {sessions.map((s) => (
            <button 
              key={s.id} 
              className="btn btn-secondary btn-sm"
              onClick={() => onSelectSession(s.id)}
            >
              {s.name || s.dataset_name} ({s.problem_type})
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '4rem 2rem', color: 'var(--text-muted)' }}>
        Generating comparative leaderboard and performance charts...
      </div>
    );
  }

  if (!comparison) {
    return null;
  }

  const recommendation = comparison.recommendation || {};
  const isAvailable = recommendation.available;
  const rows = comparison.rows || [];
  const charts = comparison.charts || {};
  const primaryMetric = comparison.primary_metric || 'metric';

  // Sort rows: completed rows by rank, then others
  const sortedRows = [...rows].sort((a, b) => {
    if (a.rank && b.rank) return a.rank - b.rank;
    if (a.rank) return -1;
    if (b.rank) return 1;
    return (a.experiment_number || 0) - (b.experiment_number || 0);
  });

  // Calculate max score and max duration for relative bar scaling
  const maxScore = Math.max(1, ...(charts.performance || []).map(p => p.metric || 0));
  const maxDuration = Math.max(0.1, ...(charts.durations || []).map(d => d.training_seconds || 0));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Session Switcher Bar */}
      <div className="glass-card" style={{ padding: '1rem 1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Current Session:</span>
          <select 
            className="form-select" 
            style={{ width: 'auto', minWidth: '240px' }}
            value={activeSession}
            onChange={(e) => onSelectSession(e.target.value)}
          >
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name || s.dataset_name} ({s.problem_type})
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <span className="badge badge-cyan">{comparison.problem_type}</span>
            <span className="badge badge-purple">Primary: {primaryMetric.toUpperCase()}</span>
            <span className="badge badge-gray">Priority: {comparison.priority}</span>
          </div>

          <a 
            className="btn btn-primary btn-sm"
            href={api.getExportCsvUrl(activeSession)}
            download
            title="Download full experiment results as CSV"
          >
            <FileSpreadsheet size={14} /> Export Results (.csv)
          </a>
        </div>
      </div>

      {/* Champion Spotlight Banner */}
      {isAvailable && (
        <div 
          className="glass-card" 
          style={{ 
            background: 'linear-gradient(135deg, rgba(0, 242, 254, 0.12) 0%, rgba(168, 85, 247, 0.08) 100%)',
            border: '1px solid rgba(0, 242, 254, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 242, 254, 0.12)'
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.5rem' }}>
            <div style={{ maxWidth: '750px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
                <span className="badge badge-amber" style={{ fontSize: '0.8rem', padding: '0.3rem 0.7rem' }}>
                  👑 Recommended Model Champion
                </span>
                <span className="badge badge-cyan">{recommendation.model_family}</span>
                <span className="badge badge-gray">{recommendation.kind}</span>
              </div>

              <h2 style={{ fontSize: '1.8rem', marginBottom: '0.5rem' }}>
                {recommendation.model_name}
              </h2>

              <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1.25rem' }}>
                {recommendation.reason}
              </p>

              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                <button 
                  className="btn btn-primary"
                  onClick={() => {
                    onSelectModelForPredict(recommendation.experiment_id, recommendation.model_name);
                    setActiveTab('playground');
                  }}
                >
                  <Target size={16} /> Test Live in Playground
                </button>

                {recommendation.experiment_id && (
                  <a 
                    className="btn btn-secondary"
                    href={api.getArtifactUrl(recommendation.experiment_id)}
                    download
                  >
                    <Download size={16} /> Download Pipeline (.joblib)
                  </a>
                )}

                <a 
                  className="btn btn-secondary"
                  href={api.getExportCsvUrl(activeSession)}
                  download
                >
                  <FileSpreadsheet size={16} /> Export CSV
                </a>
              </div>
            </div>

            {/* Score Showcase Box */}
            <div style={{ 
              background: 'rgba(16, 24, 40, 0.85)', 
              border: '1px solid rgba(0, 242, 254, 0.25)', 
              borderRadius: 'var(--radius-md)',
              padding: '1.25rem 1.5rem',
              textAlign: 'center',
              minWidth: '220px'
            }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Held-Out Test Score ({primaryMetric.toUpperCase()})
              </div>
              <div style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--accent-cyan)', margin: '0.2rem 0', fontFamily: 'var(--font-mono)' }}>
                {recommendation.test_metrics?.[primaryMetric] != null 
                  ? Number(recommendation.test_metrics[primaryMetric]).toFixed(4)
                  : (recommendation.primary_metric != null ? Number(recommendation.primary_metric).toFixed(4) : '—')}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.3rem' }}>
                <CheckCircle size={12} /> Evaluated on untouched test split
              </div>
              {recommendation.training_duration_seconds != null && (
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
                  Fit Duration: <strong style={{ color: 'var(--text-primary)' }}>{Number(recommendation.training_duration_seconds).toFixed(2)}s</strong>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Model Leaderboard Table */}
      <div className="glass-card">
        <div className="card-header">
          <div className="card-title-group">
            <div className="card-icon"><BarChart3 size={18} /></div>
            <div>
              <h3 className="card-title">Comparative Model Leaderboard</h3>
              <p className="card-subtitle">Ranked by held-out generalization, duration, and optimization priority</p>
            </div>
          </div>

          <a 
            className="btn btn-secondary btn-sm"
            href={api.getExportCsvUrl(activeSession)}
            download
          >
            <Download size={14} /> Download Table (.csv)
          </a>
        </div>

        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model Name</th>
                <th>Kind</th>
                <th>{primaryMetric.toUpperCase()}</th>
                <th>CV Mean</th>
                <th>Fit Duration</th>
                <th>Peak RAM</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row) => {
                const isWinner = row.is_recommended || row.rank === 1;
                return (
                  <tr key={row.experiment_id} className={isWinner ? 'winner-row' : ''}>
                    <td>
                      {row.rank ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                          {isWinner ? <span style={{ color: 'var(--accent-amber)', fontSize: '1rem' }}>👑</span> : null}
                          <strong style={{ color: isWinner ? 'var(--accent-cyan)' : 'inherit' }}>
                            #{row.rank}
                          </strong>
                        </div>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>—</span>
                      )}
                    </td>
                    <td>
                      <strong>{row.model_name}</strong>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Exp #{row.experiment_number}</div>
                    </td>
                    <td>
                      <span className={`badge ${
                        row.kind === 'optimization' ? 'badge-purple' :
                        row.kind === 'initial' ? 'badge-cyan' : 'badge-gray'
                      }`}>
                        {row.kind}
                      </span>
                    </td>
                    <td>
                      <strong style={{ color: 'var(--accent-cyan)', fontSize: '0.95rem', fontFamily: 'var(--font-mono)' }}>
                        {row.primary_metric != null ? Number(row.primary_metric).toFixed(4) : '—'}
                      </strong>
                    </td>
                    <td>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        {row.cv_metrics?.[primaryMetric] != null ? Number(row.cv_metrics[primaryMetric]).toFixed(4) : '—'}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                        {row.training_seconds != null ? `${Number(row.training_seconds).toFixed(2)}s` : '—'}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        {row.resource_usage?.peak_memory_mb != null ? `${row.resource_usage.peak_memory_mb} MB` : '—'}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.4rem' }}>
                        <button 
                          className="btn btn-secondary btn-sm"
                          onClick={() => setSelectedExpForDetails(row)}
                        >
                          Details
                        </button>
                        <button 
                          className="btn btn-primary btn-sm"
                          onClick={() => {
                            onSelectModelForPredict(row.experiment_id, row.model_name);
                            setActiveTab('playground');
                          }}
                        >
                          Predict
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Performance Comparison Charts */}
      <div className="glass-card">
        <div className="card-header" style={{ flexWrap: 'wrap', gap: '1rem' }}>
          <div className="card-title-group">
            <div className="card-icon"><TrendingUp size={18} /></div>
            <div>
              <h3 className="card-title">Comparative Visual Analytics</h3>
              <p className="card-subtitle">Visual benchmark of model scores, fit speeds, and feature importance</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.4rem' }}>
            <button 
              className={`btn btn-sm ${activeChartTab === 'performance' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveChartTab('performance')}
            >
              Performance ({primaryMetric.toUpperCase()})
            </button>
            <button 
              className={`btn btn-sm ${activeChartTab === 'durations' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveChartTab('durations')}
            >
              Training Durations
            </button>
            <button 
              className={`btn btn-sm ${activeChartTab === 'progression' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveChartTab('progression')}
            >
              Progression Curve
            </button>
          </div>
        </div>

        {/* Tab 1: Primary Metric Performance Bars */}
        {activeChartTab === 'performance' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', padding: '0.5rem 0' }}>
            {(charts.performance || []).map((item, idx) => {
              const val = item.metric != null ? item.metric : 0;
              const widthPct = Math.max(6, Math.min(100, (val / maxScore) * 100));
              const isBest = item.label?.includes(recommendation.model_name) || idx === 0;

              return (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ minWidth: '220px', fontSize: '0.85rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {item.label}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div className="progress-track" style={{ height: '14px', background: 'rgba(255, 255, 255, 0.04)' }}>
                      <div 
                        className="progress-fill" 
                        style={{ 
                          width: `${widthPct}%`,
                          background: isBest ? 'var(--grad-cyan-blue)' : 'var(--grad-purple-pink)',
                          transition: 'width 0.4s ease'
                        }} 
                      />
                    </div>
                  </div>
                  <div style={{ minWidth: '70px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 700, color: isBest ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
                    {val.toFixed(4)}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Tab 2: Training Durations Bars */}
        {activeChartTab === 'durations' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', padding: '0.5rem 0' }}>
            {(charts.durations || []).map((item, idx) => {
              const dur = item.training_seconds || 0;
              const widthPct = Math.max(5, Math.min(100, (dur / maxDuration) * 100));

              return (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ minWidth: '220px', fontSize: '0.85rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {item.label}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div className="progress-track" style={{ height: '14px', background: 'rgba(255, 255, 255, 0.04)' }}>
                      <div 
                        className="progress-fill" 
                        style={{ 
                          width: `${widthPct}%`,
                          background: 'var(--grad-emerald-teal)',
                          transition: 'width 0.4s ease'
                        }} 
                      />
                    </div>
                  </div>
                  <div style={{ minWidth: '70px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    {dur.toFixed(2)}s
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Tab 3: Progression Curve */}
        {activeChartTab === 'progression' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', padding: '0.5rem 0' }}>
            {(charts.progression || []).map((item, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ minWidth: '220px', fontSize: '0.85rem', fontWeight: 600 }}>
                  Exp #{item.experiment_number}: {item.label}
                </div>
                <div style={{ flex: 1 }}>
                  <div className="progress-track" style={{ height: '14px', background: 'rgba(255, 255, 255, 0.04)' }}>
                    <div 
                      className="progress-fill" 
                      style={{ 
                        width: `${Math.min(100, Math.max(5, (item.best_so_far / maxScore) * 100))}%`,
                        background: 'var(--grad-cyan-blue)',
                        transition: 'width 0.4s ease'
                      }} 
                    />
                  </div>
                </div>
                <div style={{ minWidth: '90px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>
                  Best: {item.best_so_far?.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Feature Importance & Explainability */}
      {charts.feature_importance?.length > 0 && (
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon"><Sliders size={18} /></div>
              <div>
                <h3 className="card-title">Feature Importance Analysis</h3>
                <p className="card-subtitle">Most influential predictors learned by the recommended model</p>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {charts.feature_importance.map((item, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <span style={{ minWidth: '180px', fontSize: '0.85rem', fontWeight: 600 }}>
                  {item.feature}
                </span>
                <div style={{ flex: 1 }}>
                  <div className="progress-track" style={{ height: '10px' }}>
                    <div 
                      className="progress-fill" 
                      style={{ 
                        width: `${Math.min(100, Math.max(5, item.share_pct || (item.importance * 100)))}%`,
                        background: idx === 0 ? 'var(--grad-cyan-blue)' : idx === 1 ? 'var(--grad-purple-pink)' : 'var(--grad-emerald-teal)'
                      }} 
                    />
                  </div>
                </div>
                <span style={{ minWidth: '60px', textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {item.share_pct != null ? `${item.share_pct}%` : item.importance?.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Experiment Details Modal / Drawer */}
      {selectedExpForDetails && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          padding: '1rem',
        }}>
          <div className="glass-card" style={{ maxWidth: '650px', width: '100%', maxHeight: '90vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div>
                <h3 style={{ fontSize: '1.2rem' }}>{selectedExpForDetails.model_name}</h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Experiment #{selectedExpForDetails.experiment_number} ({selectedExpForDetails.kind})
                </span>
              </div>
              <button className="btn btn-secondary btn-sm" onClick={() => setSelectedExpForDetails(null)}>
                ✕ Close
              </button>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--accent-cyan)' }}>Held-Out Test & Validation Metrics</h4>
              <pre style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem' }}>
                {JSON.stringify(selectedExpForDetails.metrics, null, 2)}
              </pre>
            </div>

            <div style={{ marginBottom: '1.25rem' }}>
              <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: 'var(--accent-purple)' }}>Hyperparameters</h4>
              <pre style={{ background: 'rgba(0, 0, 0, 0.4)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem' }}>
                {JSON.stringify(selectedExpForDetails.hyperparameters, null, 2)}
              </pre>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <a 
                className="btn btn-primary btn-sm"
                href={api.getArtifactUrl(selectedExpForDetails.experiment_id)}
                download
              >
                <Download size={14} /> Download Model (.joblib)
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
