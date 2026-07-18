# Intelligent Customer Behavior Analysis and Dynamic Marketing Strategy

This project is a production-grade customer relationship management (CRM) machine learning and reinforcement learning (RL) application. It cleans transaction-level retail sales logs, engineers customer-level Recency, Frequency, Monetary (RFM) and category preference features, projects customer vectors using dimensionality reduction (PCA/LDA), builds predictive models to classify high-value clients and forecast future spend, and trains reinforcement learning agents (Tabular Q-learning and PyTorch DQN) to recommend dynamic optimal marketing actions.

---

## Folder Structure

```text
project-root/
│
├── data/
│   ├── processed_customers.csv       # Customer RFM and projection features
│   ├── regression_customers.csv      # Months 1-9 feature to Months 10-12 target spend
│   └── plots/                        # Diagnostic and performance visualisations
│       ├── class_confusion_matrix.png
│       ├── class_roc_curve.png
│       ├── correlation_heatmap.png
│       ├── country_distribution.png
│       ├── feature_importance.png
│       ├── missing_values.png
│       ├── pca_analysis.png
│       ├── reg_predicted_vs_actual.png
│       ├── reg_residuals.png
│       ├── rfm_distributions.png
│       ├── rl_policy_comparison.png
│       └── rl_training_rewards.png
│
├── models/
│   ├── best_classifier.pkl           # Saved champion RandomForest/MLP classifier
│   ├── best_regressor.pkl            # Saved champion MLP/Linear regressor
│   ├── best_dqn.pt                   # Saved PyTorch DQN weights
│   ├── category_scaler.pkl           # Scaler for category spend percentages
│   ├── pca_model.pkl                 # Fitted PCA transformer
│   ├── lda_model.pkl                 # Fitted LDA transformer
│   ├── tabular_kmeans.pkl            # State clustering KMeans discretizer
│   └── tabular_q_table.pkl           # Trained Tabular Q-Table
│
├── src/                              # Reusable package code
│   ├── __init__.py
│   ├── preprocessing.py              # Cleaning and feature engineering pipeline
│   ├── dimensionality.py             # PCA, LDA, and metric scaling
│   ├── modeling.py                   # Classification, regression grid search, and evaluation
│   ├── rl_agent.py                   # Custom customer simulation environment and RL policies
│   └── utils.py                      # Diagnostic plotting and visualization helpers
│
├── notebooks/                        # Step-by-step educational notebooks
│   ├── 01_data_preprocessing.ipynb   # Cleaning, category mapping, and RFM distributions
│   ├── 02_pca_lda.ipynb              # Variance scree plots, LDA classification checks
│   ├── 03_classification.ipynb       # Hyperparameter grid search for RandomForest and MLP
│   ├── 04_regression.ipynb           # Forecasting spend (Months 1-9 to Months 10-12)
│   └── 05_qlearning_dqn.ipynb        # Q-learning and DQN model training loops
│
├── app.py                            # Streamlit dashboard application
├── requirements.txt                  # Python package dependencies
├── category_map.json                 # Product category regex mapping dictionary
└── run_all.py                        # Compilation and notebook execution pipeline script
```

---

## Installation & Setup

1. **Clone or navigate** to the project directory:
   ```bash
   cd final
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows (Command Prompt)
   venv\Scripts\activate.bat
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pipeline Execution & Training

To run the complete data processing, modeling, and training pipeline, execute the python runner script. This compiles all 5 notebooks in order, writes the preprocessed datasets, trains all ML models and RL policies, and saves the output plots and model artifacts:

```bash
python run_all.py
```

*Note: Notebooks can also be opened and run interactively using Jupyter notebook or JupyterLab:*
```bash
jupyter notebook
```

---

## Running the Streamlit Dashboard

To launch the interactive customer behavior analysis application:

```bash
streamlit run app.py
```

The Streamlit dashboard will automatically open in your default browser at `http://localhost:8501`.

### Dashboard Features:
- **Interactive Selectors**: Pick any customer from the database to view their profile dynamically.
- **Top Row Cards**: View Recency, Frequency, Monetary spend, and Segment class immediately.
- **Dimensionality Visualization**: View PCA/LDA projections highlighting the selected customer.
- **Spending Forecast & Classifications**: Real-time evaluation of purchase probabilities and future customer lifetime value.
- **RL Action Recommendations**: Compares DQN and Tabular Q-learning outputs to select the optimal discount or engagement campaigns.
- **Global Diagnostics Panels**: Toggle between training curves, confusion matrices, and feature importance bar plots.
