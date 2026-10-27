# ModelFinder — Autonomous ML Experimentation Agent

An intelligent, self-driving machine learning experimentation platform. Upload any tabular dataset and let the autonomous agent analyze feature distributions and relationships, probe linearity and complexity, allocate an optimal experimentation budget based on local system resources, train calibrated models via scikit-learn, tune hyperparameters using Optuna, and explain every strategic decision without hallucinations.

---

## ⚡ Key Features

1. **Design-Led AI Product Interface**
   - **Hero Section**: Two-column layout with bold typography, dark gradients, prominent rounded CTA buttons, and **floating, tilted 3D dashboard cards** with live/sample previews, mini visualizations, and micro-interactions.
   - **Capabilities & Workflow Stepper**: Visually explains the 7-step autonomous experimentation pipeline:
     $$\text{Upload Dataset} \to \text{Analyze} \to \text{Select Model} \to \text{Train} \to \text{Evaluate} \to \text{Optimize} \to \text{Recommend}$$

2. **Automated Exploratory Data Analysis (EDA)**
   - **Drag-and-Drop Ingestion**: Supports `.csv`, `.xlsx`, `.xls` datasets up to 100 MB with live schema validation.
   - **Automatic Problem Detection**: Accurately infers regression vs. binary/multiclass classification based on cardinality and data types.
   - **Linearity & Complexity Probing**: Fits pilot models on sub-samples to compare Linear CV $R^2$ vs. Non-linear Tree CV $R^2$, determining whether linear baselines or non-linear ensembles are appropriate.
   - **Dataset Profiles**: Computes missing values, duplicates, outliers, class imbalance ratios, and top Pearson/Mutual Information feature-target correlations.

3. **Intelligent Model Selection & Preprocessing**
   - **Regression Suite**: Linear Regression, Ridge, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regression (SVR).
   - **Classification Suite**: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN).
   - **Leakage-Free Pipelines**: Splits data before preprocessing. Automatically scales numeric features and One-Hot Encodes categorical features inside scikit-learn `Pipeline` and `ColumnTransformer`.

4. **Resource-Aware Budgeting & Training Calibration**
   - Measures available CPU cores and RAM before fitting.
   - Runs a lightweight pilot fit on sub-samples to calibrate wall-clock training time estimates and dynamically bounds the experiment budget (e.g. 8–10 experiments max).
   - Clearly labels all estimated durations as estimates ($low \sim high$ ranges).

5. **Autonomous Decision Engine & Optuna HPO**
   - **Transparent Agent Reasoning**: Evaluates held-out validation metrics after each run and chooses between:
     1. *Try another model family* (e.g., if linear model underfits non-linear patterns).
     2. *Optimize current model* (Bayesian hyperparameter optimization via Optuna).
     3. *Stop early* (if performance target is satisfied or metrics reach an improvement plateau).
   - **Interactive User Controls**:
     - **Continue**: Proceed to the next planned experiment in interactive review mode.
     - **Stop & Compare**: Gracefully halt future experimentation and generate the comparison leaderboard.
     - **Stop & Save**: Preserve results and pause the session to resume later.
     - **Cancel Current**: Cooperatively cancel the experiment currently training while retaining all prior results.

6. **Leaderboard, Visual Comparisons & Artifact Export**
   - Side-by-side comparison tables ranked by performance, speed, or balanced priorities.
   - Interactive comparative charts: Metric score comparison, training durations, progression curve, and feature importance.
   - **Export Results as CSV**: Download all session experiment metrics, hyperparameters, and rankings in one click.
   - **Download Trained Pipeline (.joblib)**: Ready for downstream production deployment.

7. **Live Interactive Prediction Playground**
   - Test any trained pipeline directly in the browser with real-time JSON/form inference and probability outputs.

8. **Session Archives & History**
   - Browse previous experiment runs, review historic agent decision streams, and resume paused sessions.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, Vite, Lucide React, Modern Vanilla CSS Design System |
| **Backend** | Python 3.10+, FastAPI, Pydantic v2, Uvicorn |
| **ML & Optimization** | Scikit-learn, Pandas, NumPy, Optuna, Joblib, OpenPyXL |
| **Database** | SQLite via SQLAlchemy ORM |

---

## 📂 Project Structure

```
AI_PRAGATI_ANNANT/
├── context.md                       # Master application specifications
├── README.md                        # Documentation and setup instructions
├── backend/
│   ├── config.py                    # System settings, thresholds & resource constraints
│   ├── database.py                  # SQLite database engine & session scopes
│   ├── models.py                    # SQLAlchemy ORM schemas (Dataset, Session, Experiment, Decision)
│   ├── schemas.py                   # Pydantic v2 request/response models
│   ├── main.py                      # FastAPI REST API endpoints
│   ├── logging_config.py            # Structured logging
│   ├── errors.py                    # Custom API exceptions & HTTP error mappings
│   ├── requirements.txt             # Python dependencies
│   ├── services/
│   │   ├── dataset_analyzer.py      # EDA, missing values, linearity & complexity probing
│   │   ├── preprocessing.py         # Sklearn ColumnTransformer & Pipeline builders
│   │   ├── model_selector.py        # Candidate selection rules & hyperparameters
│   │   ├── resource_monitor.py      # CPU, RAM, and disk utilization tracking
│   │   ├── time_estimator.py        # Pilot fits & duration estimation
│   │   ├── trainer.py               # Cross-validation, test fit & metric computation
│   │   ├── evaluator.py             # Classification/regression metric scorers & ranking
│   │   ├── optimizer.py             # Optuna Bayesian hyperparameter search
│   │   ├── decision_engine.py       # Autonomous rule engine (stop, optimize, explore)
│   │   ├── experiment_planner.py    # Initial strategy & budget planner
│   │   └── experiment_manager.py    # Threaded session worker & cooperative cancellation
│   ├── sample_data/
│   │   ├── sample_linear_regression.csv       # 6,000 rows house pricing dataset
│   │   ├── sample_nonlinear_regression.csv    # 6,000 rows energy consumption dataset
│   │   ├── sample_classification.csv          # 6,025 rows customer churn dataset
│   │   ├── sample_manifest.json               # Manifest metadata
│   │   └── generate_samples.py                # Synthetic generator for reproducible datasets
│   └── tests/
│       └── test_api.py              # End-to-end FastAPI endpoint and ML lifecycle tests
└── frontend/
    ├── package.json                 # Frontend scripts and dependencies
    ├── vite.config.js               # Vite bundler configuration
    ├── index.html                   # HTML5 shell
    └── src/
        ├── App.jsx                  # Main application container & router
        ├── api.js                   # API client mapping all FastAPI endpoints
        ├── index.css                # Glassmorphic dark design system & typography
        ├── components/
        │   ├── Navbar.jsx           # Sticky navigation with live system resource pill
        │   └── FloatingDashboardCards.jsx # 3D tilted preview cards
        └── views/
            ├── OverviewView.jsx     # Hero section, workflow stepper, sample demos
            ├── DataStudioView.jsx   # Drag-and-drop upload, target select, EDA charts
            ├── ArenaView.jsx        # Live training monitor, agent brain stream, controls
            ├── ComparisonView.jsx   # Leaderboard, comparison charts, model download & CSV export
            ├── HistoryView.jsx      # Historical sessions, decisions timeline & resume
            └── PlaygroundView.jsx   # Real-time model inference testing
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Node.js 18+** and **npm**

---

### Backend Setup

1. Open a terminal and navigate to `backend/`:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI backend server:
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.
   Interactive Swagger documentation is at `http://127.0.0.1:8000/docs`.

---

### Frontend Setup

1. Open a new terminal and navigate to `frontend/`:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Launch the Vite development server:
   ```bash
   npm run dev
   ```
   Open your browser at `http://localhost:5173`.

---

## 🧪 Running Tests

### Backend Test Suite
Run pytest from the `backend/` directory:
```bash
cd backend
python -m pytest
```
All 7 comprehensive end-to-end tests validate:
- System health and resource monitoring endpoints.
- Pre-bundled sample dataset loaders.
- CSV dataset upload, validation, and auto-EDA analysis.
- End-to-end autonomous session lifecycle: model selection, background training, Optuna tuning, and recommendation generation.
- Model artifact serialization and live prediction.
- Results CSV export, session deletion, and dataset deletion.

### Frontend Production Build
Validate that the client environment compiles cleanly without warnings:
```bash
cd frontend
npm run build
```

---

## 📊 End-to-End Walkthrough Scenarios

The platform includes three pre-bundled sample datasets for testing:

1. **Linear Regression (`sample_linear_regression.csv`)**
   - *Target*: `price`
   - *Behavior*: The agent tests linearity ($Linear\ R^2 = 0.952$), recognizes approximately linear feature-target relationships, and immediately selects Linear/Ridge Regression as candidate #1.

2. **Nonlinear Regression (`sample_nonlinear_regression.csv`)**
   - *Target*: `energy_demand_mw`
   - *Behavior*: The agent flags strong non-linear periodicity ($Tree\ R^2 = 0.985$ vs. $Linear\ R^2 = 0.016$), rejects linear regression, and autonomously selects Random Forest and Gradient Boosting.

3. **Imbalanced Classification (`sample_classification.csv`)**
   - *Target*: `churn`
   - *Behavior*: The agent detects class imbalance (6.16:1 ratio), automatically applies class-weighting and stratified train/test splitting, and optimizes weighted F1-score.

---

## 🔒 Known Limitations & Roadmap

- **GPU Acceleration**: Current estimators utilize scikit-learn CPU algorithms. Future releases can incorporate LightGBM, XGBoost, and CatBoost with CUDA support.
- **Deep Learning**: Designed primarily for structured tabular datasets. Future modules can add PyTorch tabular transformer architectures.
- **Automated Feature Engineering**: Currently focuses on automated imputation, scaling, and one-hot encoding; future updates will support polynomial feature generation and automated feature selection.
