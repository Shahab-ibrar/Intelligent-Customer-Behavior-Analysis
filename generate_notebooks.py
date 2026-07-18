import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook: {filename}")

def build_cells():
    # -------------------------------------------------------------
    # NOTEBOOK 1: DATA PREPROCESSING
    # -------------------------------------------------------------
    nb1_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Notebook 1: Data Preprocessing & Customer Feature Engineering\n",
                "\n",
                "**Project Title**: Intelligent Customer Behavior Analysis and Dynamic Marketing Strategy  \n",
                "**Objective**: This notebook focuses on cleaning transaction-level data, mapping items to categories, engineering customer-level RFM features, splitting customers into Train/Validation/Test sets, and generating baseline distributions.\n",
                "\n",
                "### Why Data Preprocessing is Critical:\n",
                "Raw transactional datasets often contain anomalies like cancelled invoices, missing Customer IDs, and negative quantities. To construct a reliable predictive model, we must clean these anomalies and aggregate transactions into customer-level representations.\n",
                "\n",
                "### What We Will Accomplish:\n",
                "1. Load the raw transactional dataset.\n",
                "2. Clean anomalies: remove missing `CustomerID`, cancelled orders, non-positive quantities and prices.\n",
                "3. Engineer RFM (Recency, Frequency, Monetary) metrics, product diversity, average spend, and category spend percentage features.\n",
                "4. Segment customers using a 80-10-10 split based strictly on Customer IDs (reusable across all subsequent notebooks).\n",
                "5. Label customers as 'High Value' (top 20% by training Monetary value).\n",
                "6. Save clean datasets and plot diagnostic charts."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Import Libraries and Configure Environment\n",
                "We import pandas, numpy, and our custom preprocessing/plotting modules from the `src` directory."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add src directory to system path for importing local packages\n",
                "sys.path.append(os.path.abspath('../'))\n",
                "from src.preprocessing import load_and_clean_data, get_category_map, compute_customer_rfm, split_customers\n",
                "from src.utils import plot_missing_values, plot_country_distribution, plot_rfm_distributions, plot_correlation_heatmap\n",
                "\n",
                "# Configure plotting aesthetics\n",
                "%matplotlib inline\n",
                "print(\"Libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Load and Clean Dataset\n",
                "We load `online_retail_II.csv` and filter out missing values, cancellations (starts with 'C'), negative quantities, and prices."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset_path = '../online_retail_II.csv'\n",
                "print(f\"Cleaning transactional dataset from {dataset_path}...\")\n",
                "df_clean = load_and_clean_data(dataset_path)\n",
                "print(f\"Cleaned dataset size: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.\")\n",
                "df_clean.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Map Product Categories and Aggregate to Customer RFM Profiles\n",
                "Using the category mapping defined in `category_map.json`, we label descriptions and aggregate features to the customer level. We also compute category spend percentages."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "category_map = get_category_map('../category_map.json')\n",
                "print(\"Category Mapping definition:\")\n",
                "for cat, keywords in category_map.items():\n",
                "    print(f\"  - {cat}: {', '.join(keywords[:5])}...\")\n",
                "    \n",
                "print(\"\\nAggregating transactions to customer-level profiles...\")\n",
                "rfm_df = compute_customer_rfm(df_clean, category_map)\n",
                "print(f\"Generated profiles for {rfm_df.shape[0]} unique customers.\")\n",
                "rfm_df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Split Customers strictly at Customer Level\n",
                "We split customers using a fixed random state `42` to ensure reproducibility: 80% Train, 10% Validation, 10% Test."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_cust, val_cust, test_cust = split_customers(rfm_df['CustomerID'], random_state=42)\n",
                "print(f\"Dataset Splitting Results (Seed 42):\")\n",
                "print(f\"  - Train Customers: {len(train_cust)} ({len(train_cust)/len(rfm_df)*100:.1f}%)\")\n",
                "print(f\"  - Val Customers: {len(val_cust)} ({len(val_cust)/len(rfm_df)*100:.1f}%)\")\n",
                "print(f\"  - Test Customers: {len(test_cust)} ({len(test_cust)/len(rfm_df)*100:.1f}%)\")\n",
                "\n",
                "# Add split indicator column to rfm_df\n",
                "rfm_df['Split'] = 'Train'\n",
                "rfm_df.loc[rfm_df['CustomerID'].isin(val_cust), 'Split'] = 'Val'\n",
                "rfm_df.loc[rfm_df['CustomerID'].isin(test_cust), 'Split'] = 'Test'"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: High Value Label Definition\n",
                "The threshold for labeling customers as \"High Value\" (1) vs \"Low Value\" (0) is defined as the 80th percentile of **training set Monetary values** only. This static threshold is then applied to validation and test sets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "train_monetary = rfm_df[rfm_df['Split'] == 'Train']['Monetary']\n",
                "monetary_threshold = train_monetary.quantile(0.80)\n",
                "print(f\"80th Percentile of Training Monetary Spend: ${monetary_threshold:.2f}\")\n",
                "\n",
                "rfm_df['High_Value'] = (rfm_df['Monetary'] >= monetary_threshold).astype(int)\n",
                "print(\"High Value breakdown:\")\n",
                "print(rfm_df.groupby('Split')['High_Value'].value_counts(normalize=True).unstack())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Visualizations and Diagnostics\n",
                "We plot and save the required diagnostic figures: missing values, country distributions, transaction volumes, RFM distributions, and the correlation heatmap."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "os.makedirs('../data/plots', exist_ok=True)\n",
                "\n",
                "print(\"Generating and saving diagnostic plots...\")\n",
                "fig_miss = plot_missing_values(df_clean, save_path='../data/plots/missing_values.png')\n",
                "fig_cnt = plot_country_distribution(df_clean, save_path='../data/plots/country_distribution.png')\n",
                "fig_rfm = plot_rfm_distributions(rfm_df, save_path='../data/plots/rfm_distributions.png')\n",
                "fig_corr = plot_correlation_heatmap(rfm_df, title='Customer Metrics Correlation Heatmap', save_path='../data/plots/correlation_heatmap.png')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Save Processed Outputs\n",
                "We save the processed customer profile DataFrame, splits, and labels for downstream modeling notebooks."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "processed_path = '../data/processed_customers.csv'\n",
                "rfm_df.to_csv(processed_path, index=False)\n",
                "print(f\"Saved processed customer profiles to {processed_path} successfully!\")"
            ]
        }
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 2: PCA & LDA
    # -------------------------------------------------------------
    nb2_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Notebook 2: Dimensionality Reduction - PCA and LDA\n",
                "\n",
                "**Project Title**: Intelligent Customer Behavior Analysis and Dynamic Marketing Strategy  \n",
                "**Objective**: This notebook implements Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) to reduce dimensions, capture purchase preferences, and benchmark classifier projections.\n",
                "\n",
                "### Why PCA and LDA:\n",
                "- **PCA (Unsupervised)**: Standardizes and projects category spend percentages to orthogonal directions of maximal variance. This avoids multicollinearity and isolates key shopping styles.\n",
                "- **LDA (Supervised)**: Finds linear boundaries that maximize separation between High Value and Low Value customer labels.\n",
                "\n",
                "### What We Will Accomplish:\n",
                "1. Standardize *only* the category spend percentage features (using standard scaler fit on train data).\n",
                "2. Apply PCA to find components capturing at least 90% of the total variance.\n",
                "3. Visualize the PC1 vs PC2 space, colored by customer value label.\n",
                "4. Apply LDA using the High Value target label.\n",
                "5. Benchmark performance of a baseline classifier (Logistic Regression) on Raw, PCA, and LDA features in a comparison table."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Import Libraries and Load Data\n",
                "We import pandas, numpy, sklearn modules, and local packages from `src/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sys.path.append(os.path.abspath('../'))\n",
                "from src.dimensionality import standardize_features, run_pca, run_lda, compare_features_baseline\n",
                "\n",
                "%matplotlib inline\n",
                "print(\"Libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Extract Category Spend Columns and Standardize\n",
                "We load the customer profiles and standardize the category spend percentages. Scaling parameters are fit *only* on the training set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('../data/processed_customers.csv')\n",
                "print(f\"Loaded {df.shape[0]} customer records.\")\n",
                "\n",
                "# Identify category spend columns\n",
                "cat_cols = [col for col in df.columns if col.endswith('_Spend_Pct')]\n",
                "print(\"Category columns to project:\", cat_cols)\n",
                "\n",
                "# Filter splits\n",
                "train_df = df[df['Split'] == 'Train']\n",
                "val_df = df[df['Split'] == 'Val']\n",
                "test_df = df[df['Split'] == 'Test']\n",
                "\n",
                "# Standardize ONLY category spend % columns\n",
                "train_scaled, val_scaled, test_scaled, scaler = standardize_features(\n",
                "    train_df, val_df, test_df, cat_cols\n",
                ")\n",
                "print(\"Standardization complete.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Run Unsupervised PCA\n",
                "We fit PCA on the training set and project all subsets. We compute variance curves and automatically select the minimum number of components explaining at least 90% variance."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pca, train_pca, val_pca, test_pca, n_comp = run_pca(\n",
                "    train_scaled, val_scaled, test_scaled, cat_cols, target_variance=0.90\n",
                ")\n",
                "\n",
                "explained_var = pca.explained_variance_ratio_\n",
                "cum_var = np.cumsum(explained_var)\n",
                "print(f\"Optimal Components covering 90%+ variance: {n_comp}\")\n",
                "for i, ev in enumerate(explained_var):\n",
                "    print(f\"  - PC{i+1}: Explained Var = {ev*100:.2f}%, Cumulative = {cum_var[i]*100:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Plot PCA Variance Curves and PC1 vs PC2 Scatter Plot\n",
                "We draw an explained variance scree plot and a scatter plot of projected coordinates colored by the High Value label."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Subplot 1: Variance curves\n",
                "axes[0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.6, align='center', label='Individual Variance', color='darkblue')\n",
                "axes[0].step(range(1, len(cum_var)+1), cum_var, where='mid', label='Cumulative Variance', color='red', lw=2)\n",
                "axes[0].axhline(y=0.90, color='g', linestyle='--', label='90% Threshold')\n",
                "axes[0].set_xlabel('Principal Component Index')\n",
                "axes[0].set_ylabel('Explained Variance Ratio')\n",
                "axes[0].set_title('PCA Scree Plot & Cumulative Variance')\n",
                "axes[0].legend(loc='best')\n",
                "\n",
                "# Subplot 2: PC1 vs PC2 Scatter plot\n",
                "train_pca_labels = train_pca.copy()\n",
                "train_pca_labels['High_Value'] = train_scaled['High_Value'].values\n",
                "sns.scatterplot(\n",
                "    data=train_pca_labels, x='PC1', y='PC2', hue='High_Value', \n",
                "    palette={0: 'gray', 1: 'green'}, alpha=0.6, ax=axes[1]\n",
                ")\n",
                "axes[1].set_title('PC1 vs PC2 Space (Training Set)')\n",
                "axes[1].set_xlabel('Principal Component 1')\n",
                "axes[1].set_ylabel('Principal Component 2')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('../data/plots/pca_analysis.png', dpi=300)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Fit Supervised LDA\n",
                "We fit Linear Discriminant Analysis using the `High_Value` label as the supervised target to maximize separation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_train = train_scaled['High_Value']\n",
                "y_val = val_scaled['High_Value']\n",
                "y_test = test_scaled['High_Value']\n",
                "\n",
                "lda, train_lda, val_lda, test_lda = run_lda(\n",
                "    train_scaled, val_scaled, test_scaled, cat_cols, y_train\n",
                ")\n",
                "\n",
                "print(\"LDA coefficients for categories:\")\n",
                "for col, coef in zip(cat_cols, lda.coef_[0]):\n",
                "    print(f\"  - {col}: {coef:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Compare Features using Baseline Classifier\n",
                "We compare performance of a baseline Logistic Regression classifier trained on: Raw category features, PCA features, and LDA features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_cols = cat_cols\n",
                "comp_results = compare_features_baseline(\n",
                "    train_scaled[raw_cols], y_train,\n",
                "    val_scaled[raw_cols], y_val,\n",
                "    train_pca, val_pca,\n",
                "    train_lda, val_lda\n",
                ")\n",
                "print(\"Classification Performance on Validation Set across Feature Projections:\")\n",
                "comp_results"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Save Scaled Coordinates and Projection Models\n",
                "We merge components PC1 and PC2 back into our main Customer dataset and save projection tools so they can be loaded by the Streamlit application."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import joblib\n",
                "\n",
                "# Add PC1 & PC2 to dataframe\n",
                "df['PC1'] = 0.0\n",
                "df['PC2'] = 0.0\n",
                "\n",
                "# Fill projected components for each split\n",
                "df.loc[df['Split'] == 'Train', ['PC1', 'PC2']] = train_pca[['PC1', 'PC2']].values\n",
                "df.loc[df['Split'] == 'Val', ['PC1', 'PC2']] = val_pca[['PC1', 'PC2']].values\n",
                "df.loc[df['Split'] == 'Test', ['PC1', 'PC2']] = test_pca[['PC1', 'PC2']].values\n",
                "\n",
                "df.to_csv('../data/processed_customers.csv', index=False)\n",
                "\n",
                "# Save Scaler and PCA models\n",
                "os.makedirs('../models', exist_ok=True)\n",
                "joblib.dump(scaler, '../models/category_scaler.pkl')\n",
                "joblib.dump(pca, '../models/pca_model.pkl')\n",
                "joblib.dump(lda, '../models/lda_model.pkl')\n",
                "print(\"Scaler, PCA, and LDA models saved to models/ successfully!\")"
            ]
        }
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 3: CLASSIFICATION
    # -------------------------------------------------------------
    nb3_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Notebook 3: Customer Value Classification Models\n",
                "\n",
                "**Project Title**: Intelligent Customer Behavior Analysis and Dynamic Marketing Strategy  \n",
                "**Objective**: This notebook builds a classification framework to predict whether a customer will be \"High Value\" using RFM and shopping profile features.\n",
                "\n",
                "### Classification Workflow:\n",
                "We train and tune two classifier models on the training set, validate hyperparameters using cross-validation, and perform final model selection on the test set:\n",
                "- **Model 1**: MLPClassifier (scikit-learn) with a specified network structure (Input -> 32 -> 16 -> Output).\n",
                "- **Model 2**: RandomForestClassifier tuned using `GridSearchCV` with 5-fold cross-validation.\n",
                "\n",
                "### What We Will Accomplish:\n",
                "1. Prepare feature matrices using Recency, Frequency, Monetary, PC1, and PC2.\n",
                "2. Fit MLPClassifier and RandomForest (GridSearchCV).\n",
                "3. Print classification metrics: Accuracy, Precision, Recall, F1, and confusion matrix.\n",
                "4. Render ROC curves and save the best classifier model to disk."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Import Libraries and Load Preprocessed Data\n",
                "Import metrics, modeling classes, and plotting helpers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sys.path.append(os.path.abspath('../'))\n",
                "from src.modeling import train_classifier_mlp, train_classifier_rf_grid, evaluate_classifier, save_model\n",
                "from src.utils import plot_confusion_matrix, plot_roc_curves\n",
                "\n",
                "%matplotlib inline\n",
                "print(\"Libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Build Feature Matrices\n",
                "We use the continuous customer metrics (`Recency`, `Frequency`, `Monetary`, `PC1`, `PC2`) to define the feature space."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('../data/processed_customers.csv')\n",
                "feature_cols = ['Recency', 'Frequency', 'Monetary', 'PC1', 'PC2']\n",
                "print(\"Feature Columns:\", feature_cols)\n",
                "\n",
                "# Split features and target labels\n",
                "X_train = df[df['Split'] == 'Train'][feature_cols]\n",
                "y_train = df[df['Split'] == 'Train']['High_Value']\n",
                "\n",
                "X_val = df[df['Split'] == 'Val'][feature_cols]\n",
                "y_val = df[df['Split'] == 'Val']['High_Value']\n",
                "\n",
                "X_test = df[df['Split'] == 'Test'][feature_cols]\n",
                "y_test = df[df['Split'] == 'Test']['High_Value']\n",
                "\n",
                "print(f\"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Train MLP Classifier\n",
                "We fit our custom MLP network (Input -> 32 -> 16 -> Output)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training MLP Classifier (32, 16 layers)...\")\n",
                "mlp_model = train_classifier_mlp(X_train, y_train, random_state=42)\n",
                "mlp_val_metrics = evaluate_classifier(mlp_model, X_val, y_val, 'MLP')\n",
                "print(\"MLP Validation Accuracy:\", mlp_val_metrics['Accuracy'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Train Random Forest with GridSearchCV\n",
                "We execute 5-fold cross-validation on the training set to search for the best RF hyperparameters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Running GridSearchCV for Random Forest (5-fold CV)...\")\n",
                "rf_grid = train_classifier_rf_grid(X_train, y_train, random_state=42)\n",
                "rf_model = rf_grid.best_estimator_\n",
                "print(\"Best Parameters:\", rf_grid.best_params_)\n",
                "rf_val_metrics = evaluate_classifier(rf_model, X_val, y_val, 'RandomForest')\n",
                "print(\"Random Forest Validation F1-Score:\", rf_val_metrics['F1'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Final Evaluation on Test Set\n",
                "We select the best performing model based on Validation F1-score, and run it on the unseen Test set to print reports."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Choose best model\n",
                "best_model_name = 'RandomForest' if rf_val_metrics['F1'] >= mlp_val_metrics['F1'] else 'MLP'\n",
                "best_model = rf_model if best_model_name == 'RandomForest' else mlp_model\n",
                "print(f\"Best model selected based on Validation performance: {best_model_name}\\n\")\n",
                "\n",
                "# Evaluate on test\n",
                "test_metrics = evaluate_classifier(best_model, X_test, y_test, best_model_name)\n",
                "print(f\"=== TEST RESULTS FOR {best_model_name} ===\")\n",
                "print(f\"Accuracy:  {test_metrics['Accuracy']:.4f}\")\n",
                "print(f\"Precision: {test_metrics['Precision']:.4f}\")\n",
                "print(f\"Recall:    {test_metrics['Recall']:.4f}\")\n",
                "print(f\"F1-Score:  {test_metrics['F1']:.4f}\\n\")\n",
                "print(\"Classification Report:\\n\", test_metrics['Classification_Report'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Render Confusion Matrix and ROC Curve\n",
                "We plot and save diagnostic performance curves for validation and test checks."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot confusion matrix\n",
                "cm_fig = plot_confusion_matrix(y_test, best_model.predict(X_test), classes=['Low Value', 'High Value'], \n",
                "                               title=f'Test Confusion Matrix - {best_model_name}', save_path='../data/plots/class_confusion_matrix.png')\n",
                "\n",
                "# Prepare ROC curve variables\n",
                "curves_dict = {}\n",
                "if 'ROC_Curve' in mlp_val_metrics:\n",
                "    curves_dict['MLP'] = (*mlp_val_metrics['ROC_Curve'], mlp_val_metrics['ROC_AUC'])\n",
                "if 'ROC_Curve' in rf_val_metrics:\n",
                "    curves_dict['Random Forest'] = (*rf_val_metrics['ROC_Curve'], rf_val_metrics['ROC_AUC'])\n",
                "\n",
                "roc_fig = plot_roc_curves(curves_dict, save_path='../data/plots/class_roc_curve.png')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Feature Importance & Save Classifier\n",
                "We look at RF feature importances and save the champion classifier to the models folder."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if best_model_name == 'RandomForest':\n",
                "    importances = best_model.feature_importances_\n",
                "    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)\n",
                "    \n",
                "    plt.figure(figsize=(8, 4))\n",
                "    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')\n",
                "    plt.title('Random Forest Feature Importance')\n",
                "    plt.xlabel('Importance Value')\n",
                "    plt.tight_layout()\n",
                "    plt.savefig('../data/plots/feature_importance.png', dpi=300)\n",
                "    plt.show()\n",
                "    \n",
                "# Save best classifier\n",
                "model_save_path = '../models/best_classifier.pkl'\n",
                "save_model(best_model, model_save_path)\n",
                "print(f\"Champion classifier model saved to {model_save_path} successfully!\")"
            ]
        }
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 4: REGRESSION
    # -------------------------------------------------------------
    nb4_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Notebook 4: Customer Spend Regression Models\n",
                "\n",
                "**Project Title**: Intelligent Customer Behavior Analysis and Dynamic Marketing Strategy  \n",
                "**Objective**: This notebook implements regression modeling to forecast future spend on a customer level using the operational window definitions.\n",
                "\n",
                "### Regression Operational Window Setup:\n",
                "- **Features (Months 1-9)**: Customer Recency, Frequency, Monetary spend, PC1, and PC2 computed from transactions spanning `2009-12-01` to `2010-08-31`.\n",
                "- **Target (Future Spend, Months 10-12)**: Total sum of transactions for each customer spanning `2010-09-01` to `2010-11-30` (defaults to 0 for customers who did not buy in this window).\n",
                "\n",
                "### What We Will Accomplish:\n",
                "1. Construct the regression datasets at the customer level.\n",
                "2. Standardize features and perform Train/Val/Test customer splits (reusing the same seed 42).\n",
                "3. Fit **Model 1**: MLPRegressor and **Model 2**: LinearRegression.\n",
                "4. Compare performance metrics: MSE, RMSE, MAE, R² in a table.\n",
                "5. Render predicted vs actual scatter plots and residual plots.\n",
                "6. Save the champion regression model for use in Reinforcement Learning."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Import Modules and Load Raw Data\n",
                "Import preprocessing, regression models, and plotting helpers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import joblib\n",
                "\n",
                "sys.path.append(os.path.abspath('../'))\n",
                "from src.preprocessing import load_and_clean_data, get_category_map, create_regression_dataset, split_customers\n",
                "from src.modeling import train_regressor_mlp, train_regressor_linear, evaluate_regressor, save_model\n",
                "from src.utils import plot_predicted_vs_actual, plot_residual_plot\n",
                "\n",
                "%matplotlib inline\n",
                "print(\"Libraries successfully imported!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Build Regression Dataset at Customer Level\n",
                "We split clean transactions into the feature window (Months 1-9) and target window (Months 10-12)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_df = load_and_clean_data('../online_retail_II.csv')\n",
                "category_map = get_category_map('../category_map.json')\n",
                "\n",
                "print(\"Building Months 1-9 features and Months 10-12 target spend...\")\n",
                "reg_df = create_regression_dataset(\n",
                "    raw_df, category_map,\n",
                "    f_start='2009-12-01', f_end='2010-08-31',\n",
                "    t_start='2010-09-01', t_end='2010-11-30'\n",
                ")\n",
                "print(f\"Regression customer profiles dataset shape: {reg_df.shape}\")\n",
                "reg_df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Project PCA Features and Apply Splitting\n",
                "We load our saved category scaler and PCA projection matrices to append PC1 and PC2, then perform customer splitting."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load saved scaler and PCA models\n",
                "scaler = joblib.load('../models/category_scaler.pkl')\n",
                "pca = joblib.load('../models/pca_model.pkl')\n",
                "\n",
                "# Extract and standardize category features for regression dataset\n",
                "cat_cols = [c for c in reg_df.columns if c.endswith('_Spend_Pct')]\n",
                "scaled_cat = scaler.transform(reg_df[cat_cols])\n",
                "pc_proj = pca.transform(scaled_cat)\n",
                "\n",
                "reg_df['PC1'] = pc_proj[:, 0]\n",
                "reg_df['PC2'] = pc_proj[:, 1]\n",
                "\n",
                "# Perform customer level split using customer IDs\n",
                "train_cust, val_cust, test_cust = split_customers(reg_df['CustomerID'], random_state=42)\n",
                "\n",
                "reg_df['Split'] = 'Train'\n",
                "reg_df.loc[reg_df['CustomerID'].isin(val_cust), 'Split'] = 'Val'\n",
                "reg_df.loc[reg_df['CustomerID'].isin(test_cust), 'Split'] = 'Test'\n",
                "\n",
                "print(f\"Split counts - Train: {len(train_cust)}, Val: {len(val_cust)}, Test: {len(test_cust)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Extract Regressor Matrices\n",
                "We prepare features (`Recency`, `Frequency`, `Monetary`, `PC1`, `PC2`) and targets (`Future_Spend`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = ['Recency', 'Frequency', 'Monetary', 'PC1', 'PC2']\n",
                "\n",
                "X_train = reg_df[reg_df['Split'] == 'Train'][feature_cols]\n",
                "y_train = reg_df[reg_df['Split'] == 'Train']['Future_Spend']\n",
                "\n",
                "X_val = reg_df[reg_df['Split'] == 'Val'][feature_cols]\n",
                "y_val = reg_df[reg_df['Split'] == 'Val']['Future_Spend']\n",
                "\n",
                "X_test = reg_df[reg_df['Split'] == 'Test'][feature_cols]\n",
                "y_test = reg_df[reg_df['Split'] == 'Test']['Future_Spend']"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Fit Regression Models\n",
                "We fit MLPRegressor and LinearRegression models on the training set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training MLP Regressor...\")\n",
                "mlp_reg = train_regressor_mlp(X_train, y_train, random_state=42)\n",
                "\n",
                "print(\"Training Linear Regression...\")\n",
                "lr_reg = train_regressor_linear(X_train, y_train)\n",
                "\n",
                "print(\"Training complete.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Evaluate Models and Report Metrics\n",
                "We score both regressors on validation data and choose the best performer to run on the Test set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "mlp_val = evaluate_regressor(mlp_reg, X_val, y_val)\n",
                "lr_val = evaluate_regressor(lr_reg, X_val, y_val)\n",
                "\n",
                "val_summary = pd.DataFrame([mlp_val, lr_val], index=['MLP Regressor', 'Linear Regression'])\n",
                "print(\"Validation Set Comparison:\")\n",
                "print(val_summary)\n",
                "\n",
                "# Select best based on validation R^2\n",
                "best_name = 'MLP Regressor' if mlp_val['R2'] >= lr_val['R2'] else 'Linear Regression'\n",
                "best_model = mlp_reg if best_name == 'MLP Regressor' else lr_reg\n",
                "print(f\"\\nChampion Regressor model: {best_name}\")\n",
                "\n",
                "# Evaluate on test\n",
                "test_metrics = evaluate_regressor(best_model, X_test, y_test)\n",
                "print(f\"=== TEST SET EVALUATION ({best_name}) ===\")\n",
                "for m, val in test_metrics.items():\n",
                "    print(f\"{m}: {val:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Plot Predicted vs Actual Spend & Residual Plots\n",
                "We plot diagnostics using our helper utilities and save the figures."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_test_pred = best_model.predict(X_test)\n",
                "\n",
                "fig_pva = plot_predicted_vs_actual(y_test.values, y_test_pred, title=f'Test Predicted vs Actual Spend - {best_name}', save_path='../data/plots/reg_predicted_vs_actual.png')\n",
                "fig_res = plot_residual_plot(y_test.values, y_test_pred, title=f'Test Residuals - {best_name}', save_path='../data/plots/reg_residuals.png')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8: Save Regression Dataset and Model\n",
                "We save the regression target mappings and serialize the best regression model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "reg_df.to_csv('../data/regression_customers.csv', index=False)\n",
                "save_model(best_model, '../models/best_regressor.pkl')\n",
                "print(\"Regressor model saved to models/best_regressor.pkl successfully!\")"
            ]
        }
    ]

    # -------------------------------------------------------------
    # NOTEBOOK 5: REINFORCEMENT LEARNING
    # -------------------------------------------------------------
    nb5_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Notebook 5: Reinforcement Learning for Dynamic Marketing Action Selection\n",
                "\n",
                "**Project Title**: Intelligent Customer Behavior Analysis and Dynamic Marketing Strategy  \n",
                "**Objective**: This notebook implements Tabular Q-learning and Deep Q-Networks (DQN) to select optimal marketing actions for customers in order to maximize long-term marketing rewards.\n",
                "\n",
                "### Reinforcement Learning Formulation:\n",
                "- **State Space**: Continuous vectors `[Recency, Frequency, Monetary, PC1, PC2]`.\n",
                "- **Actions**:\n",
                "  - `0`: No Action (Adjust: 0.30, Cost: 0)\n",
                "  - `1`: 10% Coupon (Adjust: 0.55, Cost: 1)\n",
                "  - `2`: Premium Trial (Adjust: 0.75, Cost: 5)\n",
                "- **Reward Formula**: `predicted_spend * prob_adjustment[action] - cost[action]`, where spend is predicted dynamically by the saved regression model.\n",
                "\n",
                "### What We Will Accomplish:\n",
                "1. Set up the customer environment simulator using the best regressor.\n",
                "2. Implement Tabular Q-learning:\n",
                "   - Discretize states into 8 clusters using KMeans.\n",
                "   - Train for 500 episodes and record rewards.\n",
                "3. Implement Deep Q-Networks (DQN) in PyTorch:\n",
                "   - Input -> 64 -> 64 -> 3 model.\n",
                "   - Experience Replay buffer (capacity >= 2000).\n",
                "   - Target network updated every 20 episodes.\n",
                "   - Train for 300 episodes.\n",
                "4. Compare DQN, Tabular Q-learning, Random, and Always No Action policies.\n",
                "5. Save the trained DQN model weights to models/."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Import Libraries and Load Models\n",
                "Import PyTorch, scikit-learn, local modules, and load the regression simulator."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import torch\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import random\n",
                "from tqdm import tqdm\n",
                "\n",
                "sys.path.append(os.path.abspath('../'))\n",
                "from src.modeling import load_model\n",
                "from src.rl_agent import CustomerEnv, TabularQLearningAgent, DQNAgent\n",
                "from src.utils import plot_rl_reward_curves\n",
                "\n",
                "%matplotlib inline\n",
                "print(\"RL Libraries and environment successfully loaded!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Initialize Customer Simulation Environment\n",
                "We instantiate the environment using the pre-computed customer profiles and the saved champion regressor."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "regressor = load_model('../models/best_regressor.pkl')\n",
                "df_cust = pd.read_csv('../data/processed_customers.csv')\n",
                "\n",
                "# Setup environment\n",
                "env = CustomerEnv(df_cust, regressor, max_steps=10)\n",
                "state = env.reset()\n",
                "print(\"Example Customer Initial State:\", state)\n",
                "print(\"Environment initialized successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Train Tabular Q-learning Agent\n",
                "We fit KMeans (k=8) on the complete customer state matrix to discretize continuous states. We then execute epsilon-greedy tabular updates for 500 episodes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Fitting KMeans on state matrices...\")\n",
                "feature_cols = ['Recency', 'Frequency', 'Monetary', 'PC1', 'PC2']\n",
                "all_states = df_cust[feature_cols].values\n",
                "\n",
                "tabular_agent = TabularQLearningAgent(n_clusters=8, action_dim=3, epsilon=0.2)\n",
                "tabular_agent.fit_kmeans(all_states)\n",
                "\n",
                "# Tabular Training Loop\n",
                "tabular_rewards = []\n",
                "print(\"Training Tabular Q-Learning Agent (500 episodes)...\")\n",
                "for ep in tqdm(range(500)):\n",
                "    state = env.reset()\n",
                "    state_idx = tabular_agent.discretize(state)\n",
                "    total_reward = 0\n",
                "    done = False\n",
                "    \n",
                "    while not done:\n",
                "        action = tabular_agent.choose_action(state_idx)\n",
                "        next_state, reward, done, _ = env.step(action)\n",
                "        next_state_idx = tabular_agent.discretize(next_state)\n",
                "        \n",
                "        tabular_agent.update(state_idx, action, reward, next_state_idx, done)\n",
                "        \n",
                "        state_idx = next_state_idx\n",
                "        total_reward += reward\n",
                "        \n",
                "    tabular_rewards.append(total_reward)\n",
                "\n",
                "print(f\"Tabular Q-learning training completed. Final Average Reward: {np.mean(tabular_rewards[-50:]):.2f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Train Deep Q-Network (DQN) Agent\n",
                "We configure a PyTorch network and train for 300 episodes, updating target weights every 20 episodes. We store transitions in the replay buffer."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dqn_agent = DQNAgent(state_dim=5, action_dim=3, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, decay_steps=1500)\n",
                "\n",
                "dqn_rewards = []\n",
                "print(\"Training DQN Agent (300 episodes)...\")\n",
                "for ep in tqdm(range(300)):\n",
                "    state = env.reset()\n",
                "    total_reward = 0\n",
                "    done = False\n",
                "    \n",
                "    while not done:\n",
                "        action = dqn_agent.choose_action(state)\n",
                "        next_state, reward, done, _ = env.step(action)\n",
                "        \n",
                "        # Save transition\n",
                "        dqn_agent.memory.push(state, action, reward, next_state, done)\n",
                "        \n",
                "        # Gradient update step\n",
                "        dqn_agent.update()\n",
                "        \n",
                "        state = next_state\n",
                "        total_reward += reward\n",
                "        dqn_agent.decay_epsilon()\n",
                "        \n",
                "    dqn_rewards.append(total_reward)\n",
                "    \n",
                "    # Target network update interval\n",
                "    if ep % 20 == 0:\n",
                "        dqn_agent.update_target_network()\n",
                "\n",
                "print(f\"DQN training completed. Final Average Reward: {np.mean(dqn_rewards[-50:]):.2f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Comparative Evaluation\n",
                "We evaluate performance across 100 episodes for four policies: DQN, Tabular Q-learning, Random, and Always No Action."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_policy(policy_type, num_eps=100):\n",
                "    eval_rewards = []\n",
                "    for _ in range(num_eps):\n",
                "        state = env.reset()\n",
                "        total_reward = 0\n",
                "        done = False\n",
                "        \n",
                "        while not done:\n",
                "            if policy_type == 'dqn':\n",
                "                action = dqn_agent.choose_action(state, eval_mode=True)\n",
                "            elif policy_type == 'tabular':\n",
                "                s_idx = tabular_agent.discretize(state)\n",
                "                action = tabular_agent.choose_action(s_idx, eval_mode=True)\n",
                "            elif policy_type == 'random':\n",
                "                action = random.randint(0, 2)\n",
                "            elif policy_type == 'no_action':\n",
                "                action = 0\n",
                "                \n",
                "            state, reward, done, _ = env.step(action)\n",
                "            total_reward += reward\n",
                "            \n",
                "        eval_rewards.append(total_reward)\n",
                "    return np.mean(eval_rewards)\n",
                "\n",
                "dqn_eval = evaluate_policy('dqn')\n",
                "tabular_eval = evaluate_policy('tabular')\n",
                "random_eval = evaluate_policy('random')\n",
                "no_action_eval = evaluate_policy('no_action')\n",
                "\n",
                "comparison_df = pd.DataFrame({\n",
                "    'Policy': ['DQN', 'Tabular Q-Learning', 'Random', 'Always No Action'],\n",
                "    'Average Reward': [dqn_eval, tabular_eval, random_eval, no_action_eval]\n",
                "})\n",
                "print(\"Evaluation Comparison:\")\n",
                "print(comparison_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Render Charts and Save Q-table / DQN Weights\n",
                "We draw training progression curves, render reward summaries, and save model assets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot training curves\n",
                "fig_prog = plot_rl_reward_curves(tabular_rewards, dqn_rewards, save_path='../data/plots/rl_training_rewards.png')\n",
                "\n",
                "# Plot evaluation bar chart\n",
                "plt.figure(figsize=(8, 5))\n",
                "sns.barplot(data=comparison_df, x='Policy', y='Average Reward', palette='coolwarm')\n",
                "plt.title('Policy Average Reward Evaluation (100 Episodes)')\n",
                "plt.ylabel('Expected Reward')\n",
                "plt.tight_layout()\n",
                "plt.savefig('../data/plots/rl_policy_comparison.png', dpi=300)\n",
                "plt.show()\n",
                "\n",
                "# Save DQN Weights\n",
                "dqn_agent.save('../models/best_dqn.pt')\n",
                "\n",
                "# Save Tabular Q-Table and KMeans discretizer for Streamlit\n",
                "import joblib\n",
                "joblib.dump(tabular_agent.q_table, '../models/tabular_q_table.pkl')\n",
                "joblib.dump(tabular_agent.kmeans, '../models/tabular_kmeans.pkl')\n",
                "print(\"RL Models saved to models/ successfully!\")"
            ]
        }
    ]

    create_notebook("notebooks/01_data_preprocessing.ipynb", nb1_cells)
    create_notebook("notebooks/02_pca_lda.ipynb", nb2_cells)
    create_notebook("notebooks/03_classification.ipynb", nb3_cells)
    create_notebook("notebooks/04_regression.ipynb", nb4_cells)
    create_notebook("notebooks/05_qlearning_dqn.ipynb", nb5_cells)

if __name__ == '__main__':
    build_cells()
