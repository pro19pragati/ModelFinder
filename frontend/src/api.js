/**
 * ModelFinder API Client
 * Wraps backend FastAPI REST endpoints
 */

const API_BASE = '/api';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, options);
  
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = { message: `Request failed with status ${response.status}` };
    }
    const err = new Error(errorData.message || errorData.detail || 'An API error occurred');
    err.status = response.status;
    err.details = errorData;
    throw err;
  }
  
  return response.json();
}

export const api = {
  // System
  getHealth: () => request('/health'),
  getSettings: () => request('/settings'),
  getResources: () => request('/system/resources'),

  // Datasets
  getDatasets: () => request('/datasets'),
  getDataset: (id) => request(`/datasets/${id}`),
  getDatasetPreview: (id, limit = 20) => request(`/datasets/${id}/preview?limit=${limit}`),
  analyzeDataset: (id, targetColumn, problemType = null) =>
    request(`/datasets/${id}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_column: targetColumn, problem_type: problemType }),
    }),
  loadSampleDataset: (sampleName) =>
    request(`/datasets/sample/${sampleName}/load`, { method: 'POST' }),
  uploadDataset: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Upload failed');
    }
    return res.json();
  },

  deleteDataset: (id) =>
    request(`/datasets/${id}`, { method: 'DELETE' }),

  // Experiments & Sessions
  getSessions: () => request('/experiments/sessions'),
  getSession: (id) => request(`/experiments/sessions/${id}`),
  getSessionProgress: (id) => request(`/experiments/sessions/${id}/progress`),
  startSession: (payload) =>
    request('/experiments/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  continueSession: (id) =>
    request(`/experiments/sessions/${id}/continue`, { method: 'POST' }),
  resumeSession: (id) =>
    request(`/experiments/sessions/${id}/resume`, { method: 'POST' }),
  stopSession: (id, mode = 'compare') =>
    request(`/experiments/sessions/${id}/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    }),
  cancelCurrentExperiment: (id) =>
    request(`/experiments/sessions/${id}/cancel`, { method: 'POST' }),
  deleteSession: (id) =>
    request(`/experiments/sessions/${id}`, { method: 'DELETE' }),
  getSessionComparison: (id) => request(`/experiments/sessions/${id}/comparison`),
  getExportCsvUrl: (sessionId) => `${API_BASE}/experiments/sessions/${sessionId}/export/csv`,

  // Single Experiment
  getExperiment: (id) => request(`/experiments/${id}`),
  getArtifactUrl: (id) => `${API_BASE}/experiments/${id}/download`,
  predict: (experimentId, inputs) =>
    request(`/experiments/${experimentId}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs }),
    }),
};
