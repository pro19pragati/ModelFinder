import React from 'react';
import { Sparkles, Database, PlayCircle, BarChart3, Target, Cpu, History as HistoryIcon } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, systemResources, isSessionRunning }) {
  const tabs = [
    { id: 'overview', label: 'Overview', icon: Sparkles },
    { id: 'datastudio', label: 'New Experiment', icon: Database },
    { id: 'arena', label: 'Experiments', icon: PlayCircle },
    { id: 'comparison', label: 'Model Comparison', icon: BarChart3 },
    { id: 'history', label: 'History', icon: HistoryIcon },
    { id: 'playground', label: 'Live Predict', icon: Target },
  ];

  return (
    <nav className="navbar">
      <div className="brand-section" onClick={() => setActiveTab('overview')} style={{ cursor: 'pointer' }}>
        <div className="brand-logo">⚡</div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span className="brand-title">ModelFinder</span>
            <span className="brand-badge">Autonomous Agent</span>
          </div>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Autonomous ML Experimentation</p>
        </div>
      </div>

      <div className="nav-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              className={`nav-tab-btn ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon size={16} />
              <span>{tab.label}</span>
              {tab.id === 'arena' && isSessionRunning && (
                <span className="status-dot" style={{ width: 6, height: 6 }} />
              )}
            </button>
          );
        })}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        {systemResources && (
          <div className="nav-status-pill">
            <Cpu size={14} color="var(--accent-cyan)" />
            <span style={{ fontSize: '0.78rem' }}>
              {systemResources.cpu_count} vCPU • {Math.round((systemResources.memory_available_mb || 0) / 1024 * 10) / 10}GB free
            </span>
          </div>
        )}
        <div className="nav-status-pill">
          <div className={`status-dot ${isSessionRunning ? 'active' : ''}`} />
          <span style={{ fontSize: '0.78rem', color: isSessionRunning ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
            {isSessionRunning ? 'Agent Active' : 'System Ready'}
          </span>
        </div>
      </div>
    </nav>
  );
}
