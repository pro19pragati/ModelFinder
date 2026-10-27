import React from 'react';
import { 
  Database, 
  Brain, 
  Activity, 
  Award, 
  CheckCircle2, 
  Sparkles
} from 'lucide-react';

export default function FloatingDashboardCards() {
  return (
    <div className="floating-cards-scene">
      {/* Ambient background glow orbs */}
      <div className="scene-glow-orb scene-glow-orb-1" />
      <div className="scene-glow-orb scene-glow-orb-2" />
      <div className="scene-glow-orb scene-glow-orb-3" />

      {/* Card 1: Dataset Analysis (top left tilted) */}
      <div className="hero-floating-card card-dataset-preview">
        <div className="card-mini-header">
          <div className="card-mini-title">
            <Database size={15} color="var(--accent-cyan)" />
            <span>Dataset Analysis</span>
          </div>
          <span className="badge badge-emerald" style={{ fontSize: '0.65rem', padding: '0.15rem 0.5rem' }}>
            <CheckCircle2 size={10} style={{ marginRight: '3px' }} /> Analyzed
          </span>
        </div>

        <div className="dataset-meta-grid">
          <div>
            <div className="meta-lbl">Dataset</div>
            <div className="meta-val">House Prices</div>
          </div>
          <div>
            <div className="meta-lbl">Problem</div>
            <div className="meta-val" style={{ color: 'var(--accent-cyan)' }}>Regression</div>
          </div>
          <div>
            <div className="meta-lbl">Rows</div>
            <div className="meta-val">10,000</div>
          </div>
          <div>
            <div className="meta-lbl">Features</div>
            <div className="meta-val">8</div>
          </div>
        </div>

        {/* Mini distribution histogram sparkline */}
        <div className="mini-chart-box">
          <div className="mini-chart-label">Feature Distribution (Target: price)</div>
          <div className="mini-bars-row">
            <span style={{ height: '35%', background: 'var(--accent-cyan)' }} />
            <span style={{ height: '60%', background: 'var(--accent-blue)' }} />
            <span style={{ height: '85%', background: 'var(--accent-cyan)' }} />
            <span style={{ height: '100%', background: 'var(--grad-cyan-blue)' }} />
            <span style={{ height: '70%', background: 'var(--accent-blue)' }} />
            <span style={{ height: '45%', background: 'var(--accent-cyan)' }} />
            <span style={{ height: '25%', background: 'var(--accent-purple)' }} />
          </div>
        </div>
      </div>

      {/* Card 2: Model Selection (middle left tilted) */}
      <div className="hero-floating-card card-model-selection">
        <div className="card-mini-header">
          <div className="card-mini-title">
            <Brain size={15} color="var(--accent-purple)" />
            <span>Autonomous Selection</span>
          </div>
          <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>
            Candidate #1
          </span>
        </div>

        <div style={{ margin: '0.5rem 0 0.4rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Selected Model</div>
          <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            Linear Regression
            <span className="badge badge-cyan" style={{ fontSize: '0.65rem', padding: '0.1rem 0.4rem' }}>Ready to train</span>
          </div>
        </div>

        <div className="reason-callout">
          <Sparkles size={13} color="var(--accent-purple)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>Strong linear relationship detected (Pearson r = 0.952, low residual curvature).</span>
        </div>
      </div>

      {/* Card 3: Experiment Progress (center overlapping) */}
      <div className="hero-floating-card card-experiment-progress">
        <div className="card-mini-header">
          <div className="card-mini-title">
            <Activity size={15} color="var(--accent-cyan)" />
            <span>Experiment Loop</span>
          </div>
          <span className="status-pill-pulsing">
            <span className="pulsing-dot" /> Training
          </span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', margin: '0.4rem 0' }}>
          <div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Current Model</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700 }}>Random Forest Regressor</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontWeight: 600 }}>
              Exp 3 of 10
            </span>
          </div>
        </div>

        <div style={{ marginTop: '0.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
            <span>Stage: Hyperparameter Tuning (Optuna)</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-primary)' }}>65%</span>
          </div>
          <div className="mini-progress-track">
            <div className="mini-progress-fill" style={{ width: '65%' }} />
          </div>
        </div>
      </div>

      {/* Card 4: Model Performance (bottom right tilted) */}
      <div className="hero-floating-card card-model-performance">
        <div className="card-mini-header">
          <div className="card-mini-title">
            <Award size={15} color="var(--accent-amber)" />
            <span>Top Performance</span>
          </div>
          <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>
            👑 Champion
          </span>
        </div>

        <div style={{ fontSize: '0.92rem', fontWeight: 700, margin: '0.3rem 0 0.6rem' }}>
          Gradient Boosting Regressor
        </div>

        <div className="metrics-triad">
          <div className="metric-box">
            <span className="m-lbl">R² SCORE</span>
            <span className="m-val" style={{ color: 'var(--accent-cyan)' }}>0.942</span>
          </div>
          <div className="metric-box">
            <span className="m-lbl">MAE</span>
            <span className="m-val">$14.2k</span>
          </div>
          <div className="metric-box">
            <span className="m-lbl">FIT TIME</span>
            <span className="m-val" style={{ color: 'var(--accent-emerald)' }}>1.84s</span>
          </div>
        </div>

        {/* Mini comparison bars */}
        <div style={{ marginTop: '0.6rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(255, 255, 255, 0.06)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)' }}>
            <span>Accuracy Improvement vs Baseline</span>
            <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>+12.8%</span>
          </div>
        </div>
      </div>

      {/* Decorative Floating Sparkle Badge */}
      <div className="floating-sparkle-badge">
        <Sparkles size={14} color="var(--accent-cyan)" />
        <span>Self-Driving Optimization Active</span>
      </div>
    </div>
  );
}
