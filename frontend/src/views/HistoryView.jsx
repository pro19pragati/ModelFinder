import React, { useState } from 'react';
import { 
  History, 
  Layers, 
  BarChart3, 
  Play, 
  Download, 
  Trash2, 
  Brain, 
  ArrowRight,
  Database,
  Award
} from 'lucide-react';
import { api } from '../api';

export default function HistoryView({
  sessions,
  datasets,
  onSelectSession,
  setActiveTab,
  onRefresh,
}) {
  const [filterType, setFilterType] = useState('all');
  const [actionLoading, setActionLoading] = useState(false);
  const [selectedSessionForDecisions, setSelectedSessionForDecisions] = useState(null);
  const [sessionDetailLoading, setSessionDetailLoading] = useState(false);
  const [sessionDetailData, setSessionDetailData] = useState(null);

  // Filter sessions
  const filteredSessions = sessions.filter((s) => {
    if (filterType === 'all') return true;
    if (filterType === 'classification') return s.problem_type === 'classification';
    if (filterType === 'regression') return s.problem_type === 'regression';
    if (filterType === 'completed') return s.status === 'completed';
    return true;
  });

  const handleOpenDecisions = async (sessionId) => {
    setSelectedSessionForDecisions(sessionId);
    setSessionDetailLoading(true);
    try {
      const data = await api.getSession(sessionId);
      setSessionDetailData(data);
    } catch (err) {
      console.error('Failed to load session details:', err);
    } finally {
      setSessionDetailLoading(false);
    }
  };

  const handleResume = async (sessionId) => {
    setActionLoading(true);
    try {
      await api.resumeSession(sessionId);
      if (onRefresh) await onRefresh();
      onSelectSession(sessionId);
      setActiveTab('arena');
    } catch (err) {
      alert(`Could not resume session: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = async (sessionId) => {
    if (!window.confirm('Are you sure you want to delete this session and its trained model artifacts?')) {
      return;
    }
    setActionLoading(true);
    try {
      await api.deleteSession(sessionId);
      if (onRefresh) await onRefresh();
    } catch (err) {
      alert(`Delete failed: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  // Compute aggregate stats
  const totalRuns = sessions.length;
  const completedRuns = sessions.filter(s => s.status === 'completed').length;
  const totalModelsTrained = sessions.reduce((acc, s) => acc + (s.completed_experiments || 0), 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Top Banner */}
      <div className="glass-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="card-icon"><History size={20} /></div>
            <div>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '0.2rem' }}>Experimentation History & Archives</h2>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Durable record of all autonomous sessions, decisions, hyperparameter logs, and saved model pipelines.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button 
              className={`btn btn-sm ${filterType === 'all' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilterType('all')}
            >
              All Runs ({sessions.length})
            </button>
            <button 
              className={`btn btn-sm ${filterType === 'regression' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilterType('regression')}
            >
              Regression
            </button>
            <button 
              className={`btn btn-sm ${filterType === 'classification' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setFilterType('classification')}
            >
              Classification
            </button>
          </div>
        </div>
      </div>

      {/* Aggregate Metrics Bar */}
      <div className="grid-cols-3">
        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '0.4rem' }}>
            <span className="card-subtitle">Total Autonomous Sessions</span>
            <Layers size={18} color="var(--accent-purple)" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800 }}>{totalRuns}</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {completedRuns} completed successfully
          </span>
        </div>

        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '0.4rem' }}>
            <span className="card-subtitle">Total Models Evaluated</span>
            <Brain size={18} color="var(--accent-cyan)" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>
            {totalModelsTrained}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Trained with full CV & test sets
          </span>
        </div>

        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '0.4rem' }}>
            <span className="card-subtitle">Active Datasets Connected</span>
            <Database size={18} color="var(--accent-emerald)" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
            {datasets.length}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Available for new experimentation
          </span>
        </div>
      </div>

      {/* Sessions History Table */}
      <div className="glass-card">
        <div className="card-header">
          <div className="card-title-group">
            <div className="card-icon"><BarChart3 size={18} /></div>
            <div>
              <h3 className="card-title">Previous Experiment Runs</h3>
              <p className="card-subtitle">Click any session to open its comparison matrix, inspect reasoning, or export results</p>
            </div>
          </div>
        </div>

        {filteredSessions.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '3.5rem 1rem', color: 'var(--text-muted)' }}>
            <Layers size={36} style={{ margin: '0 auto 0.75rem', opacity: 0.5 }} />
            <p>No experimentation sessions found matching this filter.</p>
            <button 
              className="btn btn-primary btn-sm" 
              style={{ marginTop: '1rem' }}
              onClick={() => setActiveTab('datastudio')}
            >
              Start New Experiment
            </button>
          </div>
        ) : (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session & Target</th>
                  <th>Task Type</th>
                  <th>Priority</th>
                  <th>Status</th>
                  <th>Budget Used</th>
                  <th>Champion Model</th>
                  <th>Best Score</th>
                  <th>Date</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredSessions.map((session) => {
                  const isRunning = session.status === 'running' || session.status === 'waiting';
                  const createdDate = session.created_at ? new Date(session.created_at).toLocaleDateString(undefined, {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                  }) : '—';

                  return (
                    <tr key={session.id}>
                      <td>
                        <strong style={{ fontSize: '0.9rem' }}>{session.name || session.dataset_name || 'Session'}</strong>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          Dataset: <strong style={{ color: session.dataset_name ? 'var(--text-primary)' : 'var(--text-muted)', fontStyle: session.dataset_name ? 'normal' : 'italic' }}>{session.dataset_name || 'Dataset deleted'}</strong>
                        </div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          Target: <strong style={{ color: 'var(--text-primary)' }}>{session.target_column}</strong>
                        </div>
                      </td>

                      <td>
                        <span className={`badge ${session.problem_type === 'classification' ? 'badge-emerald' : 'badge-cyan'}`}>
                          {session.problem_type}
                        </span>
                      </td>

                      <td>
                        <span className="badge badge-gray">{session.priority}</span>
                      </td>

                      <td>
                        <span className={`badge ${
                          session.status === 'completed' ? 'badge-emerald' : 
                          session.status === 'running' ? 'badge-cyan' : 
                          session.status === 'waiting' ? 'badge-amber' : 'badge-rose'
                        }`}>
                          {session.status}
                        </span>
                      </td>

                      <td>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                          {session.completed_experiments} / {session.max_experiments}
                        </span>
                      </td>

                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                          {session.best_model_name ? (
                            <>
                              <Award size={14} color="var(--accent-amber)" />
                              <strong style={{ fontSize: '0.85rem' }}>{session.best_model_name}</strong>
                            </>
                          ) : (
                            <span style={{ color: 'var(--text-muted)' }}>—</span>
                          )}
                        </div>
                      </td>

                      <td>
                        <strong style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                          {session.best_primary_metric != null ? Number(session.best_primary_metric).toFixed(4) : '—'}
                        </strong>
                      </td>

                      <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {createdDate}
                      </td>

                      <td>
                        <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                          <button 
                            className="btn btn-secondary btn-sm"
                            title="Inspect Agent Reasoning"
                            onClick={() => handleOpenDecisions(session.id)}
                          >
                            <Brain size={13} /> Reasoning
                          </button>

                          <button 
                            className="btn btn-primary btn-sm"
                            title="Open Comparison & Leaderboard"
                            onClick={() => {
                              onSelectSession(session.id);
                              setActiveTab('comparison');
                            }}
                          >
                            <BarChart3 size={13} /> Leaderboard
                          </button>

                          <a 
                            className="btn btn-secondary btn-sm"
                            title="Export Results CSV"
                            href={api.getExportCsvUrl(session.id)}
                            download
                          >
                            <Download size={13} />
                          </a>

                          {(session.status === 'stopped' || session.status === 'waiting') && (
                            <button 
                              className="btn btn-secondary btn-sm"
                              title="Resume Session"
                              onClick={() => handleResume(session.id)}
                            >
                              <Play size={13} color="var(--accent-emerald)" /> Resume
                            </button>
                          )}

                          {!isRunning && (
                            <button 
                              className="btn btn-secondary btn-sm"
                              title="Delete Session"
                              onClick={() => handleDelete(session.id)}
                            >
                              <Trash2 size={13} color="var(--accent-rose)" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Decisions Drawer / Modal */}
      {selectedSessionForDecisions && (
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
          <div className="glass-card" style={{ maxWidth: '720px', width: '100%', maxHeight: '90vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Brain size={20} color="var(--accent-cyan)" />
                <div>
                  <h3 style={{ fontSize: '1.2rem' }}>Agent Strategic Decisions Timeline</h3>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Autonomous decisions recorded during this session
                  </p>
                </div>
              </div>
              <button 
                className="btn btn-secondary btn-sm" 
                onClick={() => { setSelectedSessionForDecisions(null); setSessionDetailData(null); }}
              >
                ✕ Close
              </button>
            </div>

            {sessionDetailLoading ? (
              <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
                Loading agent decisions...
              </div>
            ) : sessionDetailData?.decisions?.length > 0 ? (
              <div className="decision-timeline" style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                {sessionDetailData.decisions.map((dec, idx) => (
                  <div key={dec.id || idx} className="decision-item">
                    <div className="decision-headline">
                      <span>{dec.headline || dec.decision}</span>
                      <span className={`badge ${
                        dec.decision === 'stop' ? 'badge-rose' :
                        dec.decision === 'optimize' ? 'badge-purple' :
                        dec.decision === 'initial_model' ? 'badge-cyan' : 'badge-emerald'
                      }`}>
                        {dec.decision}
                      </span>
                    </div>
                    <p className="decision-reasoning">{dec.reasoning}</p>
                    {dec.metrics_snapshot && (
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.4rem', fontFamily: 'var(--font-mono)' }}>
                        Snapshot: {JSON.stringify(dec.metrics_snapshot)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', padding: '2rem 1rem', textAlign: 'center' }}>
                No decision steps were logged for this session.
              </p>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.25rem' }}>
              <a 
                className="btn btn-secondary btn-sm"
                href={api.getExportCsvUrl(selectedSessionForDecisions)}
                download
              >
                <Download size={14} /> Export Results (.csv)
              </a>
              <button 
                className="btn btn-primary btn-sm"
                onClick={() => {
                  onSelectSession(selectedSessionForDecisions);
                  setSelectedSessionForDecisions(null);
                  setActiveTab('comparison');
                }}
              >
                Open Leaderboard <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
