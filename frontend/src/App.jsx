import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import OverviewView from './views/OverviewView';
import DataStudioView from './views/DataStudioView';
import ArenaView from './views/ArenaView';
import ComparisonView from './views/ComparisonView';
import HistoryView from './views/HistoryView';
import PlaygroundView from './views/PlaygroundView';
import { api } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [datasets, setDatasets] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [systemResources, setSystemResources] = useState(null);
  const [activeDatasetId, setActiveDatasetId] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [selectedModelForPredict, setSelectedModelForPredict] = useState({ id: null, name: '' });
  const [loadingInitial, setLoadingInitial] = useState(true);

  // Load initial datasets, sessions, and system resources
  const refreshData = async () => {
    try {
      const [ds, sess, res] = await Promise.all([
        api.getDatasets().catch(() => []),
        api.getSessions().catch(() => []),
        api.getResources().catch(() => null),
      ]);
      setDatasets(ds);
      setSessions(sess);
      setSystemResources(res);

      if (ds.length > 0 && !activeDatasetId) {
        setActiveDatasetId(ds[0].id);
      }
      if (sess.length > 0 && !activeSessionId) {
        setActiveSessionId(sess[0].id);
      }
    } catch (err) {
      console.error('Error refreshing data:', err);
    } finally {
      setLoadingInitial(false);
    }
  };

  useEffect(() => {
    refreshData();
    const interval = setInterval(() => {
      api.getResources().then(setSystemResources).catch(() => {});
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const activeDataset = datasets.find(d => d.id === activeDatasetId) || datasets[0] || null;
  const isSessionRunning = sessions.some(s => s.status === 'running' || s.status === 'waiting');

  // Quick 1-click demo launcher from Overview
  const handleQuickStartDemo = async (sampleName) => {
    try {
      const loaded = await api.loadSampleDataset(sampleName);
      await refreshData();
      setActiveDatasetId(loaded.id);
      setActiveTab('datastudio');
    } catch (err) {
      alert(`Could not load sample dataset: ${err.message}`);
    }
  };

  const handleLaunchExperiment = async ({ dataset_id, target_column, problem_type }) => {
    try {
      const newSession = await api.startSession({
        dataset_id,
        target_column,
        problem_type,
        priority: 'balanced',
        max_experiments: 8,
        interactive: false,
        auto_start: true,
      });
      setActiveSessionId(newSession.id);
      await refreshData();
      setActiveTab('arena');
    } catch (err) {
      alert(`Failed to launch experiment: ${err.message}`);
    }
  };

  const handleSessionStarted = async (sessionId) => {
    setActiveSessionId(sessionId);
    await refreshData();
    setActiveTab('arena');
  };

  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
    if (sessionId) {
      setActiveTab('comparison');
    }
  };

  const handleSelectModelForPredict = (id, name) => {
    setSelectedModelForPredict({ id, name });
    setActiveTab('playground');
  };

  const handleDatasetUploaded = async (newDataset) => {
    await refreshData();
    setActiveDatasetId(newDataset.id);
    setActiveTab('datastudio');
  };

  const handleDatasetDeleted = async (deletedId) => {
    await refreshData();
    if (activeDatasetId === deletedId) {
      const remaining = datasets.filter(d => d.id !== deletedId);
      setActiveDatasetId(remaining[0]?.id || null);
    }
  };

  return (
    <div className="app-container">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        systemResources={systemResources}
        isSessionRunning={isSessionRunning}
      />

      <main className="main-content">
        {activeTab === 'overview' && (
          <OverviewView 
            datasets={datasets}
            sessions={sessions}
            systemResources={systemResources}
            onSelectDataset={(id) => { setActiveDatasetId(id); setActiveTab('datastudio'); }}
            onQuickStartDemo={handleQuickStartDemo}
            setActiveTab={setActiveTab}
            onSelectSession={handleSelectSession}
          />
        )}

        {activeTab === 'datastudio' && (
          <DataStudioView 
            datasets={datasets}
            activeDataset={activeDataset}
            onSelectDataset={setActiveDatasetId}
            onDatasetUploaded={handleDatasetUploaded}
            onDatasetDeleted={handleDatasetDeleted}
            onLaunchExperiment={handleLaunchExperiment}
          />
        )}

        {activeTab === 'arena' && (
          <ArenaView 
            datasets={datasets}
            activeDataset={activeDataset}
            activeSession={activeSessionId}
            onSessionStarted={handleSessionStarted}
            onSelectSession={setActiveSessionId}
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'comparison' && (
          <ComparisonView 
            activeSession={activeSessionId}
            sessions={sessions}
            onSelectSession={setActiveSessionId}
            onSelectModelForPredict={handleSelectModelForPredict}
            setActiveTab={setActiveTab}
          />
        )}

        {activeTab === 'history' && (
          <HistoryView 
            sessions={sessions}
            datasets={datasets}
            onSelectSession={setActiveSessionId}
            setActiveTab={setActiveTab}
            onRefresh={refreshData}
          />
        )}

        {activeTab === 'playground' && (
          <PlaygroundView 
            activeDataset={activeDataset}
            activeSession={activeSessionId}
            selectedModelId={selectedModelForPredict.id}
            selectedModelName={selectedModelForPredict.name}
            sessions={sessions}
            onSelectModel={(id, name) => setSelectedModelForPredict({ id, name })}
          />
        )}
      </main>
    </div>
  );
}
