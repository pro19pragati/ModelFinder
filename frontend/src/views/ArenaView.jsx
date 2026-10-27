import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  XCircle, 
  Sliders, 
  Brain, 
  Terminal,
  Layers,
  Award,
  Zap
} from 'lucide-react';
import { api } from '../api';

export default function ArenaView({
  datasets,
  activeDataset,
  activeSession,
  onSessionStarted,
  onSelectSession,
  setActiveTab,
}) {
  // Launch configuration state
  const [selectedDatasetId, setSelectedDatasetId] = useState(activeDataset?.id || datasets[0]?.id || '');
  const [priority, setPriority] = useState('balanced');
  const [maxExperiments, setMaxExperiments] = useState(8);
  const [interactive, setInteractive] = useState(false);
  const [sessionName, setSessionName] = useState('');
  const [isStarting, setIsStarting] = useState(false);

  // Live session state
  const [sessionData, setSessionData] = useState(null);
  const [liveProgress, setLiveProgress] = useState(null);
  const terminalEndRef = useRef(null);

  // Sync dataset id when activeDataset changes
  useEffect(() => {
    if (activeDataset?.id) {
      setSelectedDatasetId(activeDataset.id);
    }
  }, [activeDataset]);

  // Load session data when activeSession changes
  useEffect(() => {
    let timer = null;
    let isMounted = true;

    async function fetchSession() {
      if (!activeSession) {
        setSessionData(null);
        setLiveProgress(null);
        return;
      }

      try {
        const data = await api.getSession(activeSession);
        if (isMounted) {
          setSessionData(data);
          setLiveProgress(data.progress);

          const status = data.progress?.status || data.session?.status;
          if (status === 'running' || status === 'waiting') {
            timer = setTimeout(fetchSession, 1200);
          }
        }
      } catch (err) {
        console.error('Failed to poll session:', err);
      }
    }

    fetchSession();

    return () => {
      isMounted = false;
      if (timer) clearTimeout(timer);
    };
  }, [activeSession]);

  // Auto-scroll terminal log
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [liveProgress?.log]);

  const handleStartSession = async (e) => {
    e.preventDefault();
    const dataset = datasets.find(d => d.id === selectedDatasetId);
    if (!dataset) {
      alert('Please select a dataset');
      return;
    }
    if (!dataset.is_analyzed) {
      alert('Dataset must be analyzed before launching an experiment session. Please analyze it in Data Studio.');
      return;
    }

    setIsStarting(true);
    try {
      const newSession = await api.startSession({
        dataset_id: dataset.id,
        target_column: dataset.target_column,
        problem_type: dataset.problem_type,
        priority: priority,
        max_experiments: parseInt(maxExperiments, 10),
        interactive: interactive,
        auto_start: true,
        name: sessionName.trim() || `${dataset.name} - Autonomous Run`,
      });
      onSessionStarted(newSession.id);
    } catch (err) {
      alert(`Failed to launch session: ${err.message}`);
    } finally {
      setIsStarting(false);
    }
  };

  const handleContinue = async () => {
    if (!activeSession) return;
    try {
      await api.continueSession(activeSession);
    } catch (err) {
      alert(`Continue failed: ${err.message}`);
    }
  };

  const handleStop = async (mode = 'compare') => {
    if (!activeSession) return;
    try {
      await api.stopSession(activeSession, mode);
    } catch (err) {
      alert(`Stop failed: ${err.message}`);
    }
  };

  const handleCancelCurrent = async () => {
    if (!activeSession) return;
    try {
      await api.cancelCurrentExperiment(activeSession);
    } catch (err) {
      alert(`Cancel current failed: ${err.message}`);
    }
  };

  const currentDataset = datasets.find(d => d.id === selectedDatasetId) || activeDataset;
  const isRunning = liveProgress?.status === 'running';
  const isWaiting = liveProgress?.status === 'waiting';
  const isFinished = liveProgress?.status === 'completed' || liveProgress?.status === 'stopped';
  const progressPct = Math.round((liveProgress?.progress || 0) * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Top Banner / New Session Trigger */}
      <div className="glass-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="card-icon"><Brain size={20} /></div>
            <div>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '0.2rem' }}>Autonomous Experimentation Arena</h2>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                Watch the AI agent explore models, execute Optuna HPO, and transparently reason through every step.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {sessionData && (
              <button 
                className="btn btn-secondary btn-sm"
                onClick={() => onSelectSession(null)}
              >
                + Configure New Session
              </button>
            )}
            {isFinished && (
              <button 
                className="btn btn-primary btn-sm"
                onClick={() => setActiveTab('comparison')}
              >
                <Award size={14} /> View Leaderboard & Compare
              </button>
            )}
          </div>
        </div>
      </div>

      {!activeSession || !sessionData ? (
        /* Configuration Form to Launch a New Session */
        <div className="grid-cols-1-2">
          <div className="glass-card">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Sliders size={18} color="var(--accent-cyan)" /> Experiment Strategy Config
            </h3>

            <form onSubmit={handleStartSession}>
              <div className="form-group">
                <label className="form-label">Dataset</label>
                <select 
                  className="form-select"
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                >
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.problem_type || 'unassigned'})
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Session Name (Optional)</label>
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="e.g. Churn Optimization Run #1"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Optimization Priority</label>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
                  {[
                    { id: 'balanced', label: 'Balanced', desc: '70% score, 30% speed' },
                    { id: 'performance', label: 'Performance', desc: 'Highest metric score' },
                    { id: 'speed', label: 'Speed', desc: 'Fastest within 3% of best' },
                  ].map((p) => (
                    <button
                      type="button"
                      key={p.id}
                      className={`btn btn-sm ${priority === p.id ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ flexDirection: 'column', padding: '0.6rem 0.4rem', height: 'auto' }}
                      onClick={() => setPriority(p.id)}
                    >
                      <span style={{ fontSize: '0.85rem' }}>{p.label}</span>
                      <span style={{ fontSize: '0.65rem', opacity: 0.8 }}>{p.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="form-group" style={{ marginTop: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                  <label className="form-label">Experiment Budget</label>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                    {maxExperiments} models
                  </span>
                </div>
                <input 
                  type="range" 
                  min="1" 
                  max="20" 
                  value={maxExperiments}
                  onChange={(e) => setMaxExperiments(e.target.value)}
                  style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
                />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  The agent will stop early if a target metric plateau is reached.
                </span>
              </div>

              <div style={{ 
                margin: '1.25rem 0', 
                padding: '0.85rem', 
                background: 'rgba(255, 255, 255, 0.02)', 
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}>
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>Interactive Review Mode</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    Pause after each experiment to review decisions before proceeding.
                  </div>
                </div>
                <input 
                  type="checkbox" 
                  checked={interactive} 
                  onChange={(e) => setInteractive(e.target.checked)}
                  style={{ width: 18, height: 18, accentColor: 'var(--accent-cyan)', cursor: 'pointer' }}
                />
              </div>

              <button 
                type="submit" 
                className="btn btn-primary btn-lg" 
                style={{ width: '100%' }}
                disabled={isStarting}
              >
                <Zap size={18} />
                {isStarting ? 'Initiating Agent Loop...' : 'Launch Autonomous Session'}
              </button>
            </form>
          </div>

          {/* Quick Context Card */}
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Brain size={18} color="var(--accent-purple)" /> Autonomous Strategy Engine
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ padding: '0.85rem', background: 'rgba(0, 242, 254, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(0, 242, 254, 0.2)' }}>
                  <strong style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>1. Linear & Complexity Probing</strong>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    Runs lightweight pilot probes on subsamples to evaluate relationship linearity and predict wall-clock training times before touching the full dataset.
                  </p>
                </div>

                <div style={{ padding: '0.85rem', background: 'rgba(168, 85, 247, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(168, 85, 247, 0.2)' }}>
                  <strong style={{ fontSize: '0.85rem', color: 'var(--accent-purple)' }}>2. Adaptive Rule-Based Model Selection</strong>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    Selects from Linear/Ridge, Tree models, Random Forests, Gradient Boosting, or SVM based on data size, sparsity, and problem type.
                  </p>
                </div>

                <div style={{ padding: '0.85rem', background: 'rgba(16, 185, 129, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  <strong style={{ fontSize: '0.85rem', color: 'var(--accent-emerald)' }}>3. Bayesian Optuna HPO & Plateau Stopping</strong>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    Performs targeted hyperparameter optimization when a model shows strong signal. Automatically halts when targets are satisfied or plateau is detected.
                  </p>
                </div>
              </div>
            </div>

            {currentDataset && (
              <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Selected: <strong style={{ color: 'var(--text-primary)' }}>{currentDataset.name}</strong> • Target: <strong style={{ color: 'var(--accent-cyan)' }}>{currentDataset.target_column}</strong>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Active Experimentation Arena View */
        <>
          {/* Real-time Status Card */}
          <div className="glass-card" style={{ border: isRunning ? '1px solid rgba(0, 242, 254, 0.4)' : '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.25rem' }}>
                  <h3 style={{ fontSize: '1.3rem' }}>{sessionData.session.name}</h3>
                  <span className={`badge ${
                    isRunning ? 'badge-cyan' : 
                    isWaiting ? 'badge-amber' : 
                    isFinished ? 'badge-emerald' : 'badge-rose'
                  }`}>
                    {isRunning ? 'RUNNING' : isWaiting ? 'WAITING FOR USER' : isFinished ? 'COMPLETED' : 'STOPPED'}
                  </span>
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Target: <strong style={{ color: 'var(--text-primary)' }}>{sessionData.session.target_column}</strong> • Priority: <strong>{sessionData.session.priority}</strong>
                </div>
              </div>

              {/* Interactive Control Buttons */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                {isWaiting && (
                  <button className="btn btn-success btn-sm" onClick={handleContinue}>
                    <Play size={14} /> Continue to Next Experiment
                  </button>
                )}

                {isRunning && (
                  <button className="btn btn-secondary btn-sm" onClick={handleCancelCurrent}>
                    <XCircle size={14} /> Cancel Current
                  </button>
                )}

                {(isRunning || isWaiting) && (
                  <>
                    <button className="btn btn-secondary btn-sm" onClick={() => handleStop('save')} title="Stop and preserve progress to resume later">
                      <Pause size={14} /> Stop & Save
                    </button>
                    <button className="btn btn-danger btn-sm" onClick={() => handleStop('compare')} title="Stop and finalize comparison leaderboard">
                      <Square size={14} /> Stop & Compare
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Progress Bar & Status Text */}
            <div style={{ marginBottom: '0.75rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.4rem' }}>
                <span style={{ color: 'var(--text-secondary)' }}>
                  Stage: <strong style={{ color: 'var(--accent-cyan)' }}>{liveProgress?.stage || 'idle'}</strong> — {liveProgress?.message || 'Processing...'}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{progressPct}%</span>
              </div>
              <div className="progress-track">
                <div 
                  className={`progress-fill ${isRunning ? 'progress-fill-animated' : ''}`}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>

            {/* Time & Counter Stats */}
            <div style={{ display: 'flex', gap: '2rem', fontSize: '0.8rem', color: 'var(--text-muted)', flexWrap: 'wrap' }}>
              <div>Elapsed: <strong style={{ color: 'var(--text-primary)' }}>{liveProgress?.elapsed_seconds ? `${liveProgress.elapsed_seconds}s` : '0s'}</strong></div>
              {liveProgress?.eta_seconds != null && (
                <div>Est. Remaining: <strong style={{ color: 'var(--accent-amber)' }}>~{Math.round(liveProgress.eta_seconds)}s (estimate)</strong></div>
              )}
              <div>Completed: <strong style={{ color: 'var(--text-primary)' }}>{sessionData.experiments.length} / {sessionData.session.max_experiments}</strong></div>
              {sessionData.recommendation?.model_name && (
                <div>Recommended Winner: <strong style={{ color: 'var(--accent-cyan)' }}>{sessionData.recommendation.model_name}</strong></div>
              )}
            </div>
          </div>

          {/* Dual Panel: Agent Brain Stream & Live Experiments */}
          <div className="grid-cols-2-1">
            {/* The Agent Brain / Decisions Timeline */}
            <div className="glass-card">
              <div className="card-header">
                <div className="card-title-group">
                  <div className="card-icon"><Brain size={18} /></div>
                  <div>
                    <h3 className="card-title">Agent Brain & Decision Stream</h3>
                    <p className="card-subtitle">Transparent real-time reasoning behind every action</p>
                  </div>
                </div>
              </div>

              <div className="decision-timeline">
                {sessionData.decisions.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    Agent is initializing and compiling planning rules...
                  </div>
                ) : (
                  sessionData.decisions.map((dec, idx) => (
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
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Right Side: Completed Experiments List & Live Log */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* Completed Experiments */}
              <div className="glass-card">
                <h4 style={{ fontSize: '0.95rem', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Layers size={16} color="var(--accent-cyan)" /> Completed Models ({sessionData.experiments.length})
                </h4>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', maxHeight: '240px', overflowY: 'auto' }}>
                  {sessionData.experiments.length === 0 ? (
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>First model is currently in training...</div>
                  ) : (
                    sessionData.experiments.map((exp) => {
                      const primary = exp.metrics?.r2 ?? exp.metrics?.f1_weighted ?? exp.metrics?.accuracy;
                      return (
                        <div 
                          key={exp.id}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '0.6rem 0.8rem',
                            background: 'rgba(255, 255, 255, 0.02)',
                            borderRadius: 'var(--radius-sm)',
                            border: '1px solid var(--border-subtle)',
                          }}
                        >
                          <div>
                            <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                              #{exp.experiment_number} {exp.model_name}
                            </div>
                            <span className="badge badge-gray" style={{ fontSize: '0.68rem', padding: '0.1rem 0.4rem' }}>
                              {exp.kind}
                            </span>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                              {primary != null ? Number(primary).toFixed(4) : '—'}
                            </div>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                              {exp.training_duration_seconds ? `${exp.training_duration_seconds.toFixed(2)}s` : ''}
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              {/* Terminal Log Output */}
              <div className="glass-card" style={{ background: 'rgba(10, 14, 22, 0.95)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>
                  <Terminal size={14} />
                  <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>LIVE AGENT CONSOLE</span>
                </div>
                <div style={{ 
                  fontFamily: 'var(--font-mono)', 
                  fontSize: '0.75rem', 
                  color: 'var(--text-secondary)',
                  height: '180px',
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                }}>
                  {(liveProgress?.log || []).map((entry, idx) => (
                    <div key={idx} style={{ opacity: 0.9 }}>
                      <span style={{ color: 'var(--accent-cyan)', marginRight: '0.5rem' }}>&gt;</span>
                      <span>{typeof entry === 'string' ? entry : entry.text}</span>
                    </div>
                  ))}
                  <div ref={terminalEndRef} />
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
