import React from 'react';
import { 
  Zap, 
  Database, 
  Play, 
  Award, 
  ArrowRight, 
  Activity, 
  Layers,
  Sparkles,
  Sliders,
  Search,
  Brain
} from 'lucide-react';
import FloatingDashboardCards from '../components/FloatingDashboardCards';

export default function OverviewView({ 
  datasets, 
  sessions, 
  systemResources: _systemResources, 
  onSelectDataset: _onSelectDataset, 
  onQuickStartDemo, 
  setActiveTab,
  onSelectSession
}) {
  const sampleDemos = [
    {
      id: 'linear_regression',
      title: 'House Price Valuation',
      type: 'Linear Regression',
      rows: '6,000 rows',
      target: 'price',
      badge: 'regression',
      badgeColor: 'badge-cyan',
      description: 'Continuous real-estate valuation. The agent probes linearity (Linear CV R² = 0.952) and selects a regularized linear model as candidate baseline.',
      accent: 'linear-gradient(135deg, rgba(0, 242, 254, 0.12) 0%, rgba(56, 189, 248, 0.04) 100%)',
      borderColor: 'rgba(0, 242, 254, 0.3)',
    },
    {
      id: 'nonlinear_regression',
      title: 'Grid Energy Demand Forecasting',
      type: 'Non-linear Regression',
      rows: '6,000 rows',
      target: 'energy_demand_mw',
      badge: 'regression',
      badgeColor: 'badge-purple',
      description: 'Complex periodic energy consumption curve. The agent flags non-linearity (Tree CV R² = 0.985 vs Linear R² = 0.016) and selects Random Forest.',
      accent: 'linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(236, 72, 153, 0.04) 100%)',
      borderColor: 'rgba(168, 85, 247, 0.3)',
    },
    {
      id: 'classification',
      title: 'Customer Churn Prediction',
      type: 'Imbalanced Classification',
      rows: '6,025 rows',
      target: 'churn',
      badge: 'classification',
      badgeColor: 'badge-emerald',
      description: 'Class-imbalanced churn task (6.16:1 ratio). The agent applies class weighting, stratified splits, and weighted F1 scoring.',
      accent: 'linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(20, 184, 166, 0.04) 100%)',
      borderColor: 'rgba(16, 185, 129, 0.3)',
    },
  ];

  const workflowSteps = [
    { step: 1, title: 'Upload Dataset', desc: 'CSV or Excel with drag-and-drop' },
    { step: 2, title: 'Analyze', desc: 'Target, types, missing values & linearity' },
    { step: 3, title: 'Select Model', desc: 'Intelligent candidate hypothesis' },
    { step: 4, title: 'Train', desc: 'Non-blocking daemon worker' },
    { step: 5, title: 'Evaluate', desc: 'Hold-out test & CV validation' },
    { step: 6, title: 'Optimize', desc: 'Bayesian Optuna hyperparameter tuning' },
    { step: 7, title: 'Recommend', desc: 'Explainable champion selection & export' },
  ];

  const featureCards = [
    {
      icon: Search,
      color: 'var(--accent-cyan)',
      title: 'Dataset Analysis',
      desc: 'Automatically analyzes the uploaded dataset, identifies the target, detects the problem type, and examines feature relationships and linearity.',
    },
    {
      icon: Brain,
      color: 'var(--accent-purple)',
      title: 'Intelligent Model Selection',
      desc: 'Selects an appropriate initial model based on dataset size, sparsity, complexity, and feature-target relationship properties.',
    },
    {
      icon: Activity,
      color: 'var(--accent-emerald)',
      title: 'Autonomous Experimentation',
      desc: 'Evaluates models, measures improvements, runs Optuna HPO, and decides whether to optimize, try another model family, or halt.',
    },
    {
      icon: Sliders,
      color: 'var(--accent-amber)',
      title: 'Resource-Aware Optimization',
      desc: 'Estimates training time using pilot fits, monitors real CPU/RAM consumption, and allocates experiment budgets according to hardware capacity.',
    },
  ];

  // Derived stats
  const totalDatasets = datasets.length;
  const totalSessions = sessions.length;
  const completedExperiments = sessions.reduce((acc, s) => acc + (s.completed_experiments || 0), 0);
  const bestSession = sessions.find(s => s.best_primary_metric != null);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
      {/* =========================================================================
          HERO SECTION — Two-column modern AI startup layout with 3D tilted cards
          ========================================================================= */}
      <section className="hero-section-wrapper">
        <div className="hero-grid">
          {/* Left Column: Hero Content */}
          <div className="hero-left-content">
            <div className="hero-badge-row">
              <span className="badge badge-purple" style={{ padding: '0.3rem 0.75rem', fontSize: '0.78rem' }}>
                <Sparkles size={13} /> Autonomous Machine Learning Agent
              </span>
              <span className="badge badge-cyan" style={{ padding: '0.3rem 0.75rem', fontSize: '0.78rem' }}>
                Zero Hallucinations
              </span>
            </div>

            <h1 className="hero-headline">
              Autonomous ML <br />
              <span className="hero-headline-gradient">
                Experimentation
              </span>
            </h1>

            <p className="hero-subtitle">
              An intelligent ML agent that analyzes your data, selects suitable models, and autonomously experiments to find the right solution.
            </p>

            {/* CTAs */}
            <div className="hero-cta-group">
              <button 
                className="btn btn-primary btn-lg hero-btn-start" 
                onClick={() => setActiveTab('datastudio')}
              >
                <Zap size={18} />
                <span>Start Experiment</span>
              </button>
              <button 
                className="btn btn-secondary btn-lg hero-btn-explore"
                onClick={() => setActiveTab('arena')}
              >
                <Layers size={18} />
                <span>Explore Dashboard</span>
              </button>
            </div>

            {/* Supporting line */}
            <div className="hero-supporting-line">
              <span>Automated model selection</span>
              <span className="dot-sep">•</span>
              <span>Intelligent experimentation</span>
              <span className="dot-sep">•</span>
              <span>Resource-aware optimization</span>
            </div>
          </div>

          {/* Right Column: Layered, tilted 3D floating cards */}
          <div className="hero-right-visual">
            <FloatingDashboardCards />
          </div>
        </div>

        {/* Hero Bottom Capabilities Bar */}
        <div className="hero-capabilities-bar">
          <div className="capability-item">
            <Search size={16} color="var(--accent-cyan)" />
            <span>Automatic Dataset Analysis</span>
          </div>
          <div className="capability-item">
            <Brain size={16} color="var(--accent-purple)" />
            <span>Intelligent Model Selection</span>
          </div>
          <div className="capability-item">
            <Activity size={16} color="var(--accent-emerald)" />
            <span>Autonomous Experimentation</span>
          </div>
          <div className="capability-item">
            <Sliders size={16} color="var(--accent-amber)" />
            <span>Resource-Aware Optimization</span>
          </div>
        </div>
      </section>

      {/* =========================================================================
          PLATFORM METRICS ROW
          ========================================================================= */}
      <div className="grid-cols-4">
        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '0.4rem' }}>
            <span className="card-subtitle">Active Datasets</span>
            <Database size={18} color="var(--accent-cyan)" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800 }}>{totalDatasets}</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Ready for automated analysis</span>
        </div>

        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '0.4rem' }}>
            <span className="card-subtitle">Autonomous Sessions</span>
            <Layers size={18} color="var(--accent-purple)" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800 }}>{totalSessions}</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Pipelines launched & saved</span>
        </div>

        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '0.4rem' }}>
            <span className="card-subtitle">Total Experiments Run</span>
            <Activity size={18} color="var(--accent-emerald)" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800 }}>{completedExperiments}</div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Models trained & cross-validated</span>
        </div>

        <div className="glass-card">
          <div className="card-header" style={{ marginBottom: '0.4rem' }}>
            <span className="card-subtitle">Best Champion Score</span>
            <Award size={18} color="var(--accent-amber)" />
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
            {bestSession ? (bestSession.best_primary_metric?.toFixed(4) || '—') : '—'}
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {bestSession ? `${bestSession.best_model_name} (${bestSession.problem_type})` : 'Awaiting first run'}
          </span>
        </div>
      </div>

      {/* =========================================================================
          SECTION 5: PLATFORM WORKFLOW STEPPER
          ========================================================================= */}
      <div className="glass-card" style={{ padding: '2rem 1.75rem' }}>
        <div style={{ textAlign: 'center', maxWidth: '700px', margin: '0 auto 1.75rem' }}>
          <span className="badge badge-purple" style={{ marginBottom: '0.5rem' }}>Full ML Lifecycle</span>
          <h2 style={{ fontSize: '1.6rem', marginBottom: '0.4rem' }}>Autonomous Experimentation Workflow</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            From raw spreadsheet to calibrated production pipeline in a fully automated decision loop
          </p>
        </div>

        <div className="workflow-stepper">
          {workflowSteps.map((ws, idx) => (
            <React.Fragment key={ws.step}>
              <div className="workflow-step-card">
                <div className="step-circle">{ws.step}</div>
                <div className="step-title">{ws.title}</div>
                <div className="step-desc">{ws.desc}</div>
              </div>
              {idx < workflowSteps.length - 1 && (
                <div className="workflow-arrow">→</div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* =========================================================================
          SECTION 5: FOUR CORE CAPABILITY CARDS
          ========================================================================= */}
      <div>
        <div style={{ marginBottom: '1.25rem' }}>
          <h2 style={{ fontSize: '1.4rem', marginBottom: '0.25rem' }}>Core Autonomous Capabilities</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Four pillars powering the agent's zero-human-intervention experimentation loop
          </p>
        </div>

        <div className="grid-cols-4">
          {featureCards.map((feat, idx) => {
            const Icon = feat.icon;
            return (
              <div key={idx} className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                <div style={{ 
                  width: '42px', 
                  height: '42px', 
                  borderRadius: 'var(--radius-md)', 
                  background: 'rgba(255, 255, 255, 0.04)', 
                  border: '1px solid var(--border-subtle)',
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center' 
                }}>
                  <Icon size={20} color={feat.color} />
                </div>
                <h3 style={{ fontSize: '1.05rem' }}>{feat.title}</h3>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                  {feat.desc}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* =========================================================================
          1-CLICK INTERACTIVE DEMONSTRATION LAB
          ========================================================================= */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h2 style={{ fontSize: '1.4rem' }}>⚡ 1-Click Interactive Demonstration Lab</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Instantly preload realistic datasets, explore EDA insights, and launch the agent.
            </p>
          </div>
        </div>

        <div className="grid-cols-3">
          {sampleDemos.map((demo) => (
            <div 
              key={demo.id} 
              className="glass-card glass-card-interactive" 
              style={{ 
                background: demo.accent, 
                borderColor: demo.borderColor,
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                  <span className={`badge ${demo.badgeColor}`}>{demo.type}</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {demo.rows}
                  </span>
                </div>
                <h3 style={{ fontSize: '1.15rem', marginBottom: '0.5rem' }}>{demo.title}</h3>
                <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '1rem' }}>
                  {demo.description}
                </p>
              </div>

              <div style={{ paddingTop: '1rem', borderTop: '1px solid rgba(255, 255, 255, 0.06)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Target: <strong style={{ color: 'var(--text-primary)' }}>{demo.target}</strong>
                  </span>
                  <button 
                    className="btn btn-primary btn-sm"
                    onClick={() => onQuickStartDemo(demo.id)}
                  >
                    <Play size={14} /> Run Demo
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* =========================================================================
          RECENT SESSIONS TABLE
          ========================================================================= */}
      {sessions.length > 0 && (
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon"><Layers size={18} /></div>
              <div>
                <h3 className="card-title">Recent Autonomous Sessions</h3>
                <p className="card-subtitle">Experimentation runs recorded by the agent</p>
              </div>
            </div>
            <button className="btn btn-secondary btn-sm" onClick={() => setActiveTab('history')}>
              View Full History <ArrowRight size={14} />
            </button>
          </div>

          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session Name</th>
                  <th>Problem Type</th>
                  <th>Priority</th>
                  <th>Status</th>
                  <th>Experiments</th>
                  <th>Best Model</th>
                  <th>Best Score</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {sessions.slice(0, 5).map((s) => (
                  <tr key={s.id}>
                    <td>
                      <strong>{s.name || s.dataset_name || 'Session'}</strong>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Target: {s.target_column}</div>
                    </td>
                    <td>
                      <span className={`badge ${s.problem_type === 'classification' ? 'badge-emerald' : 'badge-cyan'}`}>
                        {s.problem_type}
                      </span>
                    </td>
                    <td>
                      <span className="badge badge-gray">{s.priority}</span>
                    </td>
                    <td>
                      <span className={`badge ${
                        s.status === 'completed' ? 'badge-emerald' : 
                        s.status === 'running' ? 'badge-cyan' : 
                        s.status === 'waiting' ? 'badge-amber' : 'badge-rose'
                      }`}>
                        {s.status}
                      </span>
                    </td>
                    <td>{s.completed_experiments} / {s.max_experiments}</td>
                    <td>{s.best_model_name || '—'}</td>
                    <td>
                      <strong style={{ color: 'var(--accent-cyan)' }}>
                        {s.best_primary_metric != null ? s.best_primary_metric.toFixed(4) : '—'}
                      </strong>
                    </td>
                    <td>
                      <button 
                        className="btn btn-secondary btn-sm" 
                        onClick={() => onSelectSession(s.id)}
                      >
                        View Results
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
