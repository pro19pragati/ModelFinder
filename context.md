# Project: Autonomous ML Experimentation Agent

Act as a senior full-stack developer, UI/UX designer, and machine learning engineer. Build a complete, functional web application called **Autonomous ML Experimentation Agent**.

The application should allow users to upload datasets and let an autonomous agent analyze the data, select suitable ML models, train and evaluate them, and decide whether to conduct further experiments.

The application should support supervised machine learning:

* Classification
* Regression

The website must have a modern, visually appealing interface inspired by the attached design reference. The Overview page should closely follow the reference's visual style, while the other pages should have a consistent design suited to an AI/ML platform.

Build a working application, not just a static frontend.

---

# 1. Technology Stack

### Frontend

* React.js
* Vite
* Tailwind CSS
* Recharts
* Lucide React
* React Router

### Backend

* Python
* FastAPI
* Pandas
* NumPy
* Scikit-learn
* Optuna

### Database

* SQLite
* SQLAlchemy

Use REST APIs to connect the frontend and backend.

Keep the code modular and easy to maintain.

---

# 2. UI/UX Design — IMPORTANT

Use the attached screenshot as the primary visual reference for the **Overview page**.

The reference has:

* A dark background with a subtle gradient.
* Large, bold typography on the left.
* A short description below the heading.
* A prominent rounded CTA button.
* Floating, overlapping cards on the right.
* Tilted interface elements that create a layered 3D effect.
* Bright accent colors against a dark background.
* A minimal layout with generous spacing.

Recreate this visual style for the Autonomous ML Experimentation Agent.

Do not copy the social media content or branding from the reference. Replace it with relevant ML experimentation content.

## Overall visual theme

Use a modern AI startup aesthetic.

**Colors:**

* Deep navy and dark purple for the main background.
* Subtle purple and blue gradients.
* White and light grey for text.
* Lavender or violet for primary buttons.
* Light green for successful experiments.
* Orange or yellow for warnings and resource limits.

**Typography:**

* Large, bold headings.
* Modern sans-serif fonts.
* Clear text hierarchy.
* Readable body text.

**UI elements:**

* Rounded buttons and cards.
* Subtle shadows.
* Smooth hover effects.
* Soft gradients.
* Carefully spaced layouts.
* Minimal borders.

Avoid a generic admin dashboard appearance on the Overview page.

The Overview page should feel like the landing page of a modern AI product, with a distinctive hero section and floating dashboard previews.

For the other pages, use a practical dashboard layout with a consistent dark theme, readable charts, and clear navigation.

---

# 3. Overview Page — Follow the Reference Image

The Overview page is the most important part of the visual design.

Create a full-screen hero section inspired by the reference image.

Use a two-column layout.

## Left side: Hero content

Place the main content on the left side.

**Main heading:**

"Autonomous ML Experimentation"

Use very large, bold typography, with a subtle gradient or accent color on selected words.

**Subtitle:**

"An intelligent ML agent that analyzes your data, selects suitable models, and autonomously experiments to find the right solution."

Keep the description concise and visually balanced.

**Primary CTA button:**

"Start Experiment"

Use a large, rounded lavender or purple button.

When clicked, it should navigate to the New Experiment page.

**Secondary CTA:**

"Explore Dashboard"

When clicked, it should navigate to the experiment dashboard or show a preview of the platform.

Below the buttons, optionally display a small line of supporting text:

"Automated model selection · Intelligent experimentation · Resource-aware optimization"

## Right side: Floating dashboard illustration

Recreate the layered, tilted-card appearance from the reference image.

Instead of the profile cards shown in the reference, create floating cards that represent the actual functionality of the ML agent.

Use CSS transforms, positioning, shadows, and gradients to create the layered appearance.

The floating cards should include:

### Card 1: Dataset Analysis

Display a small preview of dataset information.

Example:

* Dataset: House Prices
* Rows: 10,000
* Features: 8
* Problem: Regression
* Status: Analyzed

Include a small chart or data visualization.

### Card 2: Model Selection

Display a model-selection preview.

Example:

* Selected model: Linear Regression
* Reason: Approximately linear feature-target relationships
* Status: Ready to train

Use a highlighted model badge.

### Card 3: Experiment Progress

Display a small experiment progress card.

Example:

* Experiment 3 of 10
* Current model: Random Forest
* Status: Training
* Progress bar: 65%

Use a progress indicator and a small status badge.

### Card 4: Model Performance

Display a small performance card with:

* Model name
* R² score
* MAE
* Training time

Use a small chart to visualize model performance.

All values in these cards should be illustrative placeholders when no experiment has been run. Clearly distinguish sample previews from actual experiment results.

## Floating-card appearance

Arrange the cards diagonally, with some cards overlapping.

Use:

* Slight rotations.
* Different card sizes.
* Different z-index values.
* Subtle shadows.
* Rounded corners.
* Soft purple and blue gradients.
* A few small decorative shapes.

The effect should resemble the floating UI cards in the supplied reference image.

Keep the cards visually connected to the hero section without making the page overcrowded.

On smaller screens, remove the rotations and stack the cards neatly below the hero content.

## Bottom section

At the bottom of the hero section, add a subtle row of product capabilities instead of company logos.

Display four items:

* Automatic Dataset Analysis
* Intelligent Model Selection
* Autonomous Experimentation
* Resource-Aware Optimization

Use simple icons and muted text.

## Hero page behavior

The hero section should occupy most of the initial viewport.

The user should immediately understand what the product does and how to start.

Add subtle hover animations to the CTA buttons and floating cards.

Do not use excessive animations that distract from the content.

---

# 4. Main Navigation

Add a navigation bar that fits the hero design.

Include:

* Product logo and name
* Overview
* New Experiment
* Experiments
* Model Comparison
* History

On the Overview page, use a transparent or dark navigation bar that blends into the hero background.

On internal pages, use a fixed sidebar or top navigation with the same dark color scheme.

Ensure navigation works correctly using React Router.

---

# 5. Overview Dashboard Content

Below the main hero section, add a section introducing the platform's capabilities.

Include four feature cards:

### Dataset Analysis

Automatically analyzes the uploaded dataset, identifies the target, detects the problem type, and examines feature relationships.

### Intelligent Model Selection

Selects an appropriate initial model based on the dataset's characteristics.

### Autonomous Experimentation

Evaluates models and decides whether to optimize, try another model, or stop.

### Resource-Aware Optimization

Estimates training time and allocates experiments according to available resources.

Use simple icons and concise descriptions.

Add another section explaining the workflow:

Upload Dataset → Analyze → Select Model → Train → Evaluate → Optimize → Recommend

Keep the layout visually consistent with the hero.

---

# 6. Dataset Upload Page

Create a dedicated page for uploading datasets.

Support:

* CSV files
* Excel files
* Drag-and-drop uploads
* File size display
* File replacement and removal
* Target column selection

After uploading, send the dataset to the FastAPI backend.

Validate the file and show meaningful error messages for:

* Unsupported formats
* Empty datasets
* Invalid files
* Missing target columns
* Datasets that exceed configured resource limits

Once the user selects the target column, provide an "Analyze Dataset" button.

---

# 7. Automatic Dataset Analysis

The agent must inspect the uploaded dataset itself.

Implement a dataset analyzer using Python, Pandas, and NumPy.

Analyze:

* Number of rows and columns
* Numerical and categorical features
* Missing values
* Duplicate rows
* Unique values
* Memory usage
* Feature distributions
* Target distribution
* Feature-target relationships
* Correlations
* Outliers
* Class imbalance

Automatically detect whether the problem is classification or regression.

Use the target data type and number of unique values as part of the decision. If the problem type is ambiguous, allow the user to correct it.

For regression, examine whether the relationships between features and target appear approximately linear.

For classification, analyze the class distribution and identify potential class imbalance.

Display the results on the Dataset Analysis page using:

* Summary cards
* Charts
* Correlation heatmaps
* Missing-value summaries
* Dataset preview tables

Include a section called **Agent's Dataset Insights** that explains the findings in simple language.

Generate these insights from the actual dataset, not hardcoded examples.

---

# 8. Automatic Model Selection

The agent should choose an initial model based on its analysis.

### Regression models

* Linear Regression
* Ridge Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regression

### Classification models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Classifier
* K-Nearest Neighbors

The agent should not train every model automatically.

For example, if a regression dataset exhibits approximately linear relationships, Linear Regression can be selected as the initial candidate.

If the data shows nonlinear patterns, consider tree-based models.

For classification, select a suitable classifier.

Initially, use transparent, explainable selection rules. Keep the model selector modular so that more sophisticated approaches can be added later.

The first selected model must be treated as a candidate, not automatically as the final best model.

---

# 9. Resource Analysis and Budget Allocation

The agent should automatically estimate how many experiments it can reasonably perform.

The user should not have to specify the training duration or the number of experiments.

Consider:

* Dataset size
* Number of rows and features
* Available RAM
* Available CPU resources
* Model complexity
* Estimated preprocessing cost

Set a configurable default maximum evaluation limit, such as 10 experiments.

The agent can stop earlier if it reaches a satisfactory result or another stopping condition.

Allow an optional user-defined maximum if needed.

Distinguish between:

* Hard resource limits
* Experimentation budget
* Optimization priorities

Estimate training time using dataset characteristics, model complexity, and available hardware.

Where practical, use a small pilot experiment to improve the estimate.

Show an estimated duration as a range and update it using actual training times.

Clearly label estimated resource usage rather than presenting estimates as measured values.

---

# 10. Experiment Planning

Implement an experiment planner that determines which experiment to run next.

The planner should:

1. Select the initial model.
2. Choose reasonable default hyperparameters.
3. Estimate the experimentation budget.
4. Run the initial experiment.
5. Evaluate its results.
6. Decide whether to stop, optimize the current model, or try another model.

After every experiment, consider:

* Model performance
* Training time
* Resource consumption
* Previous experiment results
* Remaining budget
* Optimization priority

Use Optuna for hyperparameter optimization where appropriate.

Avoid repeating identical experiments unnecessarily.

Start with a transparent, rule-based decision engine. Keep it modular for future improvements.

---

# 11. Data Preprocessing

Implement automatic preprocessing using Scikit-learn Pipelines and ColumnTransformer.

For numerical features:

* Handle missing values.
* Apply scaling when required.

For categorical features:

* Handle missing values.
* Apply One-Hot Encoding or another suitable encoding method.

Prevent data leakage:

* Split the data before fitting preprocessing transformations.
* Fit transformations only on the training data.
* Use validation data or cross-validation for model selection.
* Keep the test set separate for final evaluation.
* Use stratified splitting for classification when appropriate.

Adapt preprocessing to the selected model.

---

# 12. Model Training and Evaluation

Implement actual model training using Scikit-learn.

For regression, calculate:

* MAE
* MSE
* RMSE
* R² score

For classification, calculate:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Support binary and multiclass classification.

For every experiment, store:

* Experiment ID
* Model name
* Hyperparameters
* Evaluation metrics
* Training duration
* Dataset identifier
* Experiment status
* Resource usage, where available
* Timestamp

Use background tasks or a suitable job execution mechanism so that training does not block the web application.

---

# 13. Autonomous Decision Engine

This is the central component of the project.

After each completed experiment, the agent must decide whether to:

1. Stop experimentation.
2. Optimize the current model.
3. Try another model.

The decision should consider:

* Current performance
* Previous experiment results
* Improvement between experiments
* Remaining budget
* Resource consumption
* Optimization priority
* Stopping conditions

For example, if Linear Regression performs well on an approximately linear dataset, the agent may decide that further experimentation is unnecessary.

If it performs poorly, the agent may try another suitable model.

If a model performs well but has promising hyperparameters to explore, the agent may allocate more experiments to optimization.

Display the decisions in a section called **Agent's Reasoning**.

Explain decisions using actual evaluation results and resource information.

Do not fabricate reasoning or performance results.

---

# 14. Experiment Dashboard

Create an interactive experiment dashboard using the same visual theme.

Display:

* Current experiment number
* Current model
* Experiment status
* Progress bar
* Current training stage
* Estimated remaining time
* Completed experiments
* Remaining budget

Include an experiment history table that updates as experiments finish.

Provide three controls:

**Continue**

* Allow the agent to proceed with the next experiment.

**Stop & Compare**

* Stop future experimentation.
* Show a comparison of all completed experiments.
* Preserve all completed results.

**Stop & Save**

* Stop experimentation and save the current progress.
* Allow the user to view the results later.

If the user stops during an experiment, safely cancel it if possible.

Preserve all completed results and experiment history.

The comparison should remain accessible while experimentation is running.

---

# 15. Model Comparison Page

Create a dedicated comparison page.

Display all completed experiments in a comparison table.

Include:

* Model name
* Evaluation metrics
* Training time
* Hyperparameters
* Resource consumption, where available
* Experiment number

Add charts comparing:

* Model performance
* Training duration
* Performance improvements across experiments

Use the appropriate metrics for classification and regression.

Highlight the recommended model without hiding other results.

Allow the user to open the details of each experiment.

---

# 16. Final Model Recommendation

After experimentation ends, display the agent's recommendation.

Include:

* Recommended model
* Evaluation metrics
* Training time
* Hyperparameters
* Reason for recommendation
* Number of experiments completed
* Resource consumption, where available

Support three optimization priorities:

* Performance
* Speed
* Balanced

The recommendation should take the selected priority into account.

If the user stops early, clearly state that the recommendation is based on the experiments completed so far.

Allow users to download:

* Experiment results as CSV
* Trained model, where practical

---

# 17. Experiment History

Create a page for previous experiments.

Users should be able to:

* View previous experiments
* View dataset information
* View completed models and metrics
* Open previous comparisons
* Review agent decisions
* Resume an experiment if its state has been saved and resuming is supported

Store experiment metadata and results in SQLite.

Avoid unnecessarily duplicating large datasets.

---

# 18. Backend Architecture

Use a modular FastAPI backend.

Suggested structure:

backend/
main.py
database.py
models.py
schemas.py
routes/
datasets.py
experiments.py
results.py
services/
dataset_analyzer.py
preprocessing.py
model_selector.py
resource_monitor.py
time_estimator.py
experiment_planner.py
trainer.py
evaluator.py
optimizer.py
decision_engine.py
experiment_manager.py

Provide APIs for:

* Dataset upload
* Dataset analysis
* Starting experiments
* Retrieving experiment status
* Stopping experiments
* Continuing experiments
* Retrieving experiment history
* Comparing models
* Downloading results
* Downloading trained models

Use Pydantic schemas for validation.

Add error handling and logging.

---

# 19. Frontend Architecture

Suggested structure:

frontend/
src/
components/
Navbar.jsx
Sidebar.jsx
MetricCard.jsx
DatasetUploader.jsx
DatasetPreview.jsx
ProgressBar.jsx
ExperimentTable.jsx
ModelComparisonChart.jsx
AgentReasoning.jsx
FloatingDashboardCards.jsx
pages/
Overview.jsx
NewExperiment.jsx
DatasetAnalysis.jsx
ExperimentDashboard.jsx
ModelComparison.jsx
ExperimentHistory.jsx
Settings.jsx
services/
api.js
App.jsx
main.jsx

Use React Router.

Create reusable components and consistent styling.

The Overview page should use the custom hero design described above. Internal pages should use a functional dashboard layout.

Connect every page to the backend.

Show loading states, error messages, and empty states.

---

# 20. Example End-to-End Scenario

Test the website using three datasets:

1. A linear regression dataset.
2. A nonlinear regression dataset.
3. A classification dataset.

For a linear regression dataset:

1. The user uploads the dataset.
2. The user selects the target column.
3. The agent analyzes the data.
4. It detects a regression problem.
5. It identifies approximately linear relationships.
6. It selects Linear Regression as the initial candidate.
7. It estimates the training duration and experimentation budget.
8. It trains and evaluates Linear Regression.
9. It stores the results.
10. It decides whether further experimentation is worthwhile.
11. If necessary, it tries another model or optimizes the current one.
12. It repeats the evaluation and decision process within the available budget.
13. It compares all completed experiments.
14. It recommends a model based on the results and optimization priority.

Also verify that the agent does not always select Linear Regression for nonlinear datasets.

---

# 21. Development Instructions

Build the application in phases.

**Phase 1:** Create the frontend, backend, and API connection. Implement dataset upload.

**Phase 2:** Implement dataset analysis and problem-type detection.

**Phase 3:** Implement automatic model selection and preprocessing.

**Phase 4:** Implement model training and evaluation.

**Phase 5:** Implement resource estimation, experimentation budgeting, and experiment tracking.

**Phase 6:** Implement the autonomous decision engine.

**Phase 7:** Implement the experiment dashboard, comparison page, and final recommendation.

**Phase 8:** Test, debug, and document the application.

Before making changes, inspect the existing project directory. If it is empty, initialize the project. If it already contains code, understand it before modifying anything.

After each phase, run the relevant tests and fix errors before proceeding.

Do not stop after creating the frontend. Implement and integrate the backend and ML functionality.

Include:

* README with setup instructions
* requirements.txt
* package.json
* Sample datasets
* Basic backend tests

At the end, provide the project structure, installation instructions, startup commands, implemented functionality, and known limitations.

---

# Final Objective

Build a complete, functional web application with a distinctive, modern AI-product interface.

The Overview page should visually resemble the supplied reference, with a dark gradient hero, large typography, prominent CTA buttons, and floating dashboard cards.

The rest of the application should provide a practical interface for uploading datasets, viewing analysis, monitoring experiments, comparing models, and reviewing recommendations.

The autonomous agent must analyze the uploaded data, select an appropriate initial model, train and evaluate it, and decide whether to continue experimenting, optimize, or stop while respecting available resources.

Prioritize a working end-to-end application, reliable ML functionality, and a polished user experience.