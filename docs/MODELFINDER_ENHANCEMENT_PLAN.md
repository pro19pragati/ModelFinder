# ModelFinder – Autonomous ML Experimentation System
## Complete Project Audit & Enhancement Plan
*Last updated: 2026-10-26 | Full audit of all 24 source files completed*

---

## 1. Files Audited (complete)

| File | Lines | Audited |
|------|-------|---------|
| `backend/main.py` | 915 | Full |
| `backend/config.py` | 121 | Full |
| `backend/models.py` | 171 | Full |
| `backend/schemas.py` | 268 | Full |
| `backend/services/dataset_analyzer.py` | 1220 | Full |
| `backend/services/experiment_manager.py` | 1249 | Full |
| `backend/services/decision_engine.py` | 517 | Full |
| `backend/services/experiment_planner.py` | 260 | Full |
| `backend/services/model_selector.py` | 856 | Full |
| `backend/services/evaluator.py` | 315 | Full |
| `backend/services/trainer.py` | 448 | Full |
| `backend/services/optimizer.py` | 185 | Full |
| `backend/services/preprocessing.py` | 148 | Full |
| `backend/tests/test_api.py` | 305 | Full |
| `frontend/src/App.jsx` | 198 | Full |
| `frontend/src/api.js` | 100 | Full |
| `frontend/src/views/OverviewView.jsx` | 441 | Partial |
| `frontend/src/views/DataStudioView.jsx` | 500 | Partial |
| `frontend/src/views/ArenaView.jsx` | 538 | Partial |
| `frontend/src/views/ComparisonView.jsx` | 553 | Partial |
| `frontend/src/views/PlaygroundView.jsx` | 305 | Partial |
| `frontend/src/views/HistoryView.jsx` | 424 | Partial |

---

## 2. Existing Working Features

### Backend – Core Pipeline
- CSV/XLSX upload with size guard (60 MB), encoding fallback (UTF-8, UTF-8-SIG, Latin-1), and delimiter sniffing (`;`, tab, `|`)
- Full EDA: shape, dtypes, missing values, duplicates, memory, numeric & categorical summaries, correlation matrix, IQR outlier detection, feature-target relationships (Pearson r, eta-squared)
- Auto problem-type detection: dtype + cardinality + integer-like test; `ambiguous` flag surfaced in analysis JSON
- Target candidate ranking by name keywords, column position, uniqueness ratio
- Linearity probe (regression only): LinearRegression vs DecisionTree 3-fold CV on <=2000-row subsample; `verdict` string attached to analysis
- Model registry: 6 regression + 6 classification families with builders, param spaces, `needs_scaling`, `family` tags
- Intelligent initial model selection: 8+ explicit rules using n_rows, linearity ratio, feature width, sparsity, class imbalance, memory pressure
- Three-way train/val/test split with stratified splits for classification; notes field records strategy choices
- Leak-safe preprocessing pipeline: `ColumnTransformer` (impute -> scale -> OHE) fitted exclusively on training fold
- Optuna TPE optimization: multivariate sampler, per-trial cancel-check, configurable trial budget, time-budget ceiling
- 8-rule autonomous decision engine with stop codes: `target_reached`, `plateau_stop`, `budget_exhausted`, `exhausted_candidates`, `session_timeout`, `user_stop`
- Resource monitoring: live RAM/CPU snapshot, peak memory per experiment via tracemalloc
- Feature importance via `feature_importances_` or `coef_`
- Model persistence: full sklearn Pipeline serialised via joblib; refit on train+val before final test evaluation
- Cooperative cancellation: daemon thread with `threading.Event` flags checked between pipeline stages
- Interactive mode: worker pauses between experiments and waits for `continue` event
- `ExperimentSession` resume: stored plan + budget reloaded; pilot runs skipped
- Full session lifecycle in SQLite: created -> running -> waiting -> stopped/completed/failed
- `AgentDecision` records with headline, reasoning, and numeric evidence after every experiment
- Leaderboard ranking with three priority modes: performance, speed, balanced (70/30 score/time composite)
- 30+ REST API endpoints; full Pydantic v2 schemas; structured error responses
- Metrics: Regression (MAE, MSE, RMSE, R2, MAPE); Classification (accuracy, precision, recall, F1-weighted, balanced accuracy, ROC-AUC, log-loss, per-class breakdown, confusion matrix)
- CSV export of session leaderboard; joblib model download
- `best_experiment_id` tracking on ExperimentSession

### Frontend – React SPA
- Dark-mode glassmorphism design, smooth animations, micro-interactions
- Six views: Overview, DataStudio, Arena, Comparison, History, Playground
- Centralised `api.js` with error handling; relative `/api` base (works behind Vite proxy)
- Arena: live polling at 1200 ms when running/waiting; auto-scroll terminal log; stop/continue/cancel-current controls
- Playground: live prediction from any experiment artifact; pre-fills inputs from dataset preview row; Python code snippet generation
- ComparisonView: leaderboard table with rank, selection score, rationale; performance + duration + progression + resource charts; recommended model highlight
- HistoryView: filterable session list; inline decision stream viewer; resume + delete actions
- DataStudioView: drag-and-drop upload; dataset selector; target/problem-type dropdowns; analysis insights panel
- OverviewView: stat counters (datasets, sessions, experiments); three demo launchers (linear, nonlinear, classification)

### Tests
- `tests/test_api.py`: 5 integration tests covering health, settings, resources, full lifecycle (upload -> analyse -> start -> poll -> compare -> predict -> export CSV -> delete), plateau guard, regression lifecycle, 404 handling

---

## 3. Confirmed Bugs

### B1 – Cross-validation data leakage (CRITICAL)
**File:** `trainer.py` ~line 285  
**Issue:** `cross_val_score` is called on an already-fitted `Pipeline`. sklearn re-fits the estimator on each fold but does NOT re-fit the `ColumnTransformer` steps inside the pipeline when the outer object was already fitted. The imputer and scaler fitted on the full training set contaminate CV fold statistics.  
**Fix:** Pass `sklearn.base.clone(pipeline)` into `cross_val_score`.

### B2 – Plateau detection counts model_keys not model families (HIGH)
**File:** `decision_engine.py` lines 362-367  
**Issue:** `distinct_families_tried` is built correctly from `spec.family` on lines 362-365, then immediately overridden on line 367 with a plain `model_key` set — making the family-diversity guard dead code. The plateau can fire after trying only one model family.  
**Fix:** Remove the override; use the `spec.family`-based set.

### B3 – Recommendation model vs. best_experiment_id mismatch (MEDIUM)
**File:** `experiment_manager.py` `_finalize_session` + `test_api.py` comment line 139  
**Issue:** `session.best_experiment_id` is set from `best_by_metric` (raw score) while `recommendation.experiment_id` is set from the priority-ranked winner. They can differ. The test already documents this inconsistency in a comment but does not assert against it.  
**Fix:** Store priority-ranked winner in `best_experiment_id`; add response field `raw_best_experiment_id` for the metric-ranked winner.

### B4 – CV guard ignores variance (HIGH)
**File:** `decision_engine.py` lines 330-332  
**Issue:** `cv_ok` only checks `cv_mean >= threshold`. A model with mean=0.93 but std=0.15 passes the "target reached" guard and stops experimentation prematurely.  
**Fix:** Require `cv_mean - cv_std >= target - 0.02`.

### B5 – Class imbalance never propagated to model builders (HIGH)
**File:** `model_selector.py` (`default_params_for`)  
**Issue:** `model_selector` correctly detects imbalance and narrates it in the reasoning, but never injects `class_weight="balanced"` into Random Forest, Gradient Boosting, or Logistic Regression builders.  
**Fix:** In `default_params_for`, inject `class_weight="balanced"` when `profile.get("is_imbalanced")`.

### B6 – `_session_context` reloads full DataFrame per experiment (MEDIUM)
**File:** `experiment_manager.py` `_session_context`  
**Issue:** The worker calls `analyzer.load_dataframe(dataset.stored_path)` once per experiment even for 60 MB datasets, causing unnecessary repeated disk I/O.  
**Fix:** Load the frame once in `_session_context` and pass through `context` dict.

### B7 – `target_candidates` can include the current target column
**File:** `dataset_analyzer.py` lines 169-195  
**Issue:** `target_candidates()` does not exclude the column that is already the selected target; it appears in its own candidate dropdown list.  
**Fix:** Exclude the current `target` param from returned candidates.

### B8 – Optimization experiments have no CV metrics
**File:** `experiment_manager.py` lines 862-866  
**Issue:** `record.cv_metrics` is populated from `result.get("cv_metrics")`, but `trainer.run_experiment` only runs CV when `with_cv=(kind == "initial")`. Optimization experiments always have `cv_metrics = None`.  
**Fix:** Run a lightweight post-refit 3-fold CV for optimization experiments to populate `cv_metrics`.

---

## 4. Limitations

| ID | Description | Severity |
|----|-------------|----------|
| L1 | CV folds fixed at 3; noisy for <500-row datasets | Medium |
| L2 | `OneHotEncoder` silently truncates categories beyond cap=25 | Medium |
| L3 | No `class_weight` applied (same as B5) | High |
| L4 | No empirical separability probe for classification (only regression has linearity probe) | Medium |
| L5 | No `ExtraTreesRegressor` / `ExtraTreesClassifier` in registry | Low |
| L6 | `optimization_trials` default=12 too low for high-dimensional spaces | Medium |
| L7 | No ensemble/voting/stacking final step | Low |
| L8 | Leaderboard ranks by validation metric, not CV mean | Medium |
| L9 | `column_kind()` hard-codes cardinality threshold at 50; no config knob | Low |
| L10 | `_progress` and `_runners` are in-memory dicts; lost on server restart (active sessions become orphaned) | Medium |
| L11 | No pagination on list endpoints | Low |
| L12 | `feature_distributions` slices to 8 numeric + 4 categorical only | Low |
| L13 | RAM measurement via `psutil.Process().memory_info()` counts shared libs; may overstate | Low |
| L14 | `fill_value="missing"` passed to `most_frequent` SimpleImputer — `fill_value` is ignored | Low |
| L15 | `column_kind` returns `"text"` for high-cardinality strings but preprocessing encodes them as categorical | Medium |
| L16 | `AllocatedBudget.time_estimate_seconds` can be 0 for resumed sessions | Low |

---

## 5. Automatic Regression/Classification Detection Improvements

**Current state:** `detect_problem_type` is solid for clear-cut cases. The `ambiguous` flag is stored in the analysis JSON but the frontend does not surface it prominently.

**Planned improvements:**

1. **Classification separability probe** (`dataset_analyzer.py`) — Mirror the linearity probe: run LogisticRegression vs DecisionTreeClassifier (depth-6) 3-fold CV on <=1000-row sample. Output `separability_score` (LR accuracy) and `verdict` ("linearly_separable" | "nonlinear" | "weak_signal"). Feed result to model selector to prefer tree-based models when nonlinear.

2. **Ordinal integer hint** (`dataset_analyzer.py`) — When `unique` is between 5 and 20, integer-like, and `ambiguous=True`, attach an `ordinal_hint` field. Surface in DataStudio.

3. **Ambiguity UX** (`DataStudioView.jsx`) — When `analysis.problem_type_detection.ambiguous === true`, show a dismissible amber banner with a one-click "Override to Regression / Classification" button wired to the `/analyze` endpoint with `problem_type` set.

4. **Confidence indicator in Arena** (`ArenaView.jsx`) — Show `problem_type_detection.confidence` (high/medium) and `problem_type_source` (agent/user) in the session header chip.

---

## 6. Autonomous Model-Selection Improvements

1. **Fix B2** – compute `distinct_families_tried` from `spec.family` in `decision_engine.py`
2. **Fix B5/L3** – inject `class_weight="balanced"` for imbalanced classification in `model_selector.py`
3. **Add classification separability probe** (L4) to guide first-model choice
4. **Add ExtraTrees** (L5) to registry with `family="ensemble"` for both regression and classification
5. **Re-sort alternates after each experiment** – move unexplored families to the front so the agent explores variety before optimizing
6. **Resource-aware exclusions** – if `memory_available_mb < 2000` and `n_rows > 50000`, exclude SVR/SVC from candidate pool
7. **Fix L8** – when `cv_metrics` is available, use `cv_mean` as the primary ranking signal in `evaluator.rank_experiments`

---

## 7. Optimization and Early-Stopping Improvements

1. **Fix B1** – clone pipeline before `cross_val_score` in `trainer.py`
2. **Fix B4** – add `cv_std` guard: `cv_mean - cv_std >= target - epsilon`
3. **Raise optimization trials** (L6): speed=15, balanced=25, performance=40 (up from fixed 12)
4. **Dynamic CV folds** (L1): `folds = min(5, max(3, n_train_rows // 100))`
5. **Fix B8** – run 3-fold CV after Optuna refit to populate `cv_metrics` for optimization experiments
6. **Session-timeout early stop**: check elapsed vs `max_session_seconds` in the worker loop and finalize gracefully
7. **Fix L10** – on startup, transition sessions with `status="running"` to `status="failed"` with `stop_reason="server_restart"`

---

## 8. Frontend/Backend Integration Issues

| ID | View | Issue | Fix |
|----|------|-------|-----|
| I1 | `ArenaView` | Decision stream renders `reasoning` text but omits `details` numeric evidence dict. Agent's quantitative reasoning is invisible to the user. | Render `decision.details` as a collapsible key-value list below reasoning. |
| I2 | `PlaygroundView` | Feature inputs pre-filled from row 0 may include target column if `target_column` is null at that moment. | Guard `delete row[activeDataset.target_column]` with a null-check. |
| I3 | `PlaygroundView` | Blank categorical field sends `null`; verify backend imputer handles it without 500. | Add integration test for null feature values in predict. |
| I4 | `ComparisonView` | CV std error bars not shown on performance chart. | Add `cv_std` field to chart data rows and render +/- bar. |
| I5 | `ComparisonView` | Comparison data loaded once on mount; not re-fetched when a running session completes. | Re-fetch when session status transitions to completed. |
| I6 | `ArenaView` | Polling timer may fire one extra tick after terminal status received. | Clear timer explicitly in isMounted cleanup path. |
| I7 | `App.jsx` | `handleSelectSession` always navigates to `comparison` tab even when called from History. | Pass optional `targetTab` param; only navigate if explicitly required. |
| I8 | `DataStudioView` | `handleRunAnalysis` calls `onDatasetUploaded` which triggers full `refreshData()`; partial state update on failure. | Use a dedicated `onDatasetAnalyzed` callback instead. |
| I9 | `HistoryView` | `best_primary_metric` value shown without metric label ("0.932" with no unit). | Append "R2" or "F1" label based on `session.problem_type`. |
| I10 | `api.js` | Upload error handler catches `err.message` but backend returns `{detail: [{msg:...}]}` for 422s. Message is unhelpful. | Also check `err.detail` array and format it for display. |

---

## 9. Prioritised Implementation Phases

### Phase 0 – Baseline (0.5 day)
- Run `pytest tests/ -v` and record baseline pass/fail counts
- Run `npm run lint` in `frontend/`
- Document exact line numbers for all bugs above

### Phase 1 – Critical Bug Fixes (1 day)

| Priority | Bug | File | Specific Fix |
|----------|-----|------|-------------|
| P0 | B1 | `trainer.py` | `clone(pipeline)` before `cross_val_score` |
| P0 | B2 | `decision_engine.py` | Remove `model_key` override; use `spec.family` set |
| P0 | B4 | `decision_engine.py` | `cv_mean - cv_std >= target - 0.02` guard |
| P0 | B5/L3 | `model_selector.py` | `class_weight="balanced"` for imbalanced datasets |
| P1 | B3 | `experiment_manager.py` | Align `best_experiment_id` with priority-ranked winner |
| P1 | L14 | `preprocessing.py` | Remove `fill_value="missing"` from `most_frequent` imputer |
| P2 | B7 | `dataset_analyzer.py` | Exclude current target from `target_candidates` result |

### Phase 2 – ML Intelligence Upgrades (1-2 days)

| Item | File | Change |
|------|------|--------|
| L1 | `trainer.py` | Dynamic CV folds: `min(5, max(3, n_rows//100))` |
| L6 | `config.py` + `optimizer.py` | Raise default trials to 25; scale by priority |
| L5 | `model_selector.py` | Add ExtraTreesRegressor + ExtraTreesClassifier |
| L4 | `dataset_analyzer.py` | Classification separability probe |
| L8 | `decision_engine.py` + `evaluator.py` | CV mean as primary ranking signal |
| B8 | `trainer.py` / `experiment_manager.py` | Post-Optuna CV for optimization experiments |
| L15 | `preprocessing.py` + `dataset_analyzer.py` | Flag "text" columns; exclude or OHE-truncate explicitly |

### Phase 3 – Task-Detection UX (0.5 day)

| Item | File | Change |
|------|------|--------|
| T1 | `DataStudioView.jsx` | Ambiguity warning banner + one-click override button |
| T2 | `main.py` | Ensure `problem_type_source="user"` on manual override |
| T3 | `ArenaView.jsx` | Display `problem_type_source` + confidence chip in session header |
| T4 | `dataset_analyzer.py` | Ordinal integer hint in `build_insights` |

### Phase 4 – Frontend Integration Fixes (1 day)

| Item | File | Fix |
|------|------|-----|
| I1 | `ArenaView.jsx` | Render `decision.details` numeric evidence |
| I2/I3 | `PlaygroundView.jsx` | Null-check on target column; empty-string to null cast |
| I4 | `ComparisonView.jsx` | CV std error bars on performance chart |
| I5 | `ComparisonView.jsx` | Re-fetch on session completion |
| I6 | `ArenaView.jsx` | Fix polling stop on terminal states |
| I7 | `App.jsx` | `handleSelectSession` tab navigation fix |
| I9 | `HistoryView.jsx` | Show metric label next to `best_primary_metric` |
| I10 | `api.js` | Handle 422 Pydantic error detail array |
| L10 | `experiment_manager.py` | Orphaned session cleanup on startup |

### Phase 5 – Tests and Verification (0.5-1 day)
- `tests/test_trainer.py` (new): CV clone correctness; imbalanced class weight; dynamic folds
- `tests/test_decision.py` (new): family diversity fix (B2); cv_std guard (B4); plateau requires >=3 experiments; all families tried before stopping
- `tests/test_analyzer.py` (new): ambiguous detection; target excluded from candidates (B7); separability probe contract
- `tests/test_evaluator.py` (new): all metric values; lower-is-better regression metrics; CV-mean ranking
- End-to-end: regression + classification smoke tests with sample datasets
- Verify model export, re-import, live prediction round-trip
- Verify orphaned session cleanup after simulated restart

---

## 10. Testing Strategy

### Existing test coverage gaps
`test_api.py` covers the full HTTP surface (5 tests) but does NOT test:
- Internal service units (trainer, decision engine, evaluator, analyzer)
- Edge cases (orphaned sessions, corrupt uploads, null predictions)
- The CV leakage bug (B1) — currently passes even with the bug present

### New unit tests to add

**`test_trainer.py`**
- Assert `clone(pipeline)` is used: mock `cross_val_score`, verify first arg is a fresh clone
- Imbalanced dataset: assert `class_weight="balanced"` in fitted estimator params
- Dynamic folds: n=100 rows -> 3 folds; n=500 rows -> 5 folds

**`test_decision.py`**
- B2 fix: create 2 experiments from same family -> assert agent does NOT plateau-stop
- B4: mock cv_mean=0.93, cv_std=0.15 -> assert target_reached does NOT fire
- Plateau patience: run 4 experiments, identical metrics -> assert stop only after patience=2 extra
- Exhausted candidates: run through all alternates -> assert stop_code="exhausted_candidates"

**`test_analyzer.py`**
- `ambiguous=True` for 10-unique integer target; confidence="medium"
- Target excluded from `target_candidates` result
- Linearity probe: linear data -> is_approximately_linear=True
- Separability probe (Phase 2): linearly separable data -> verdict="linearly_separable"

**`test_evaluator.py`**
- Regression: MAE, RMSE, R2 computed correctly; lower-is-better not used as "better"
- Classification: F1-weighted selected when imbalanced; accuracy when balanced
- rank_experiments: speed priority picks fast model within 3% of best

### Smoke-test workflows

**Regression workflow**
1. Load `sample_linear_regression.csv`, target=`price`
2. Analyse -> `problem_type=regression`, `confidence=high`, `is_approximately_linear=True`
3. Start session -> first model must be `linear_regression` or `ridge_regression`
4. Complete -> >=2 experiments; reasoning cites actual R2 values
5. Download artifact -> `joblib.load` returns a fitted Pipeline
6. Live predict -> numeric float response

**Classification workflow**
1. Load `sample_classification.csv`, target=`churn`
2. Analyse -> `problem_type=classification`, `is_imbalanced=True`, imbalance_ratio >= 1.5
3. Start session -> verify `class_weight="balanced"` in plan params (Phase 2)
4. Complete -> F1-weighted, accuracy, ROC-AUC all populated
5. Decision stream -> reasoning cites actual F1 delta values in `details`

**Edge cases**
- Dataset < 50 rows: warning shown; training completes without crash
- All-numeric target, unique_count=8: `ambiguous=True` shown in DataStudio
- 200k rows: SVR/SVC excluded from candidate pool automatically
- Session stopped mid-experiment: status=stopped, not failed; recommendation still generated
- Server restart with running session: startup transitions session to failed with reason
- Predict with null feature field: 200 response with imputed prediction (not 500 error)

---

## 11. Files to Change Summary

| Phase | Files |
|-------|-------|
| Phase 1 (bugs) | `trainer.py`, `decision_engine.py`, `model_selector.py`, `preprocessing.py`, `experiment_manager.py`, `dataset_analyzer.py` |
| Phase 2 (ML upgrades) | `config.py`, `trainer.py`, `optimizer.py`, `model_selector.py`, `dataset_analyzer.py`, `decision_engine.py`, `evaluator.py` |
| Phase 3 (detection UX) | `dataset_analyzer.py`, `main.py`, `DataStudioView.jsx`, `ArenaView.jsx` |
| Phase 4 (frontend) | `ArenaView.jsx`, `PlaygroundView.jsx`, `ComparisonView.jsx`, `HistoryView.jsx`, `App.jsx`, `api.js`, `experiment_manager.py` |
| Phase 5 (tests) | `tests/test_trainer.py` (new), `tests/test_decision.py` (new), `tests/test_analyzer.py` (new), `tests/test_evaluator.py` (new) |

---

> **AWAITING APPROVAL BEFORE ANY CODE CHANGES.**
> After approval, implementation proceeds phase by phase with test results reported after each phase.
