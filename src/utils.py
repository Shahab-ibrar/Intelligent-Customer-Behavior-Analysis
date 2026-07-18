import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Dict, Any, Optional

# Set style for all plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

def plot_missing_values(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Creates a bar plot showing missing values by column.
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct}).reset_index()
    missing_df = missing_df.rename(columns={'index': 'Column'})
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(missing_df) == 0:
        ax.text(0.5, 0.5, 'No missing values found!', ha='center', va='center', fontsize=12)
        ax.set_axis_off()
    else:
        sns.barplot(data=missing_df, x='Percentage', y='Column', ax=ax, palette='Blues_r')
        ax.set_title('Percentage of Missing Values per Column')
        ax.set_xlabel('Percentage (%)')
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_country_distribution(df: pd.DataFrame, top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots the count distribution of transactions across countries.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    country_counts = df['Country'].value_counts().head(top_n).reset_index()
    country_counts.columns = ['Country', 'Count']
    
    sns.barplot(data=country_counts, x='Count', y='Country', ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Countries by Transaction Volume')
    ax.set_xscale('log')
    ax.set_xlabel('Count (Log Scale)')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_rfm_distributions(rfm_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots subplots of the distribution of Recency, Frequency, and Monetary values.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recency
    sns.histplot(rfm_df['Recency'], bins=30, kde=True, ax=axes[0], color='navy')
    axes[0].set_title('Recency Distribution (Days)')
    axes[0].set_xlabel('Recency')
    
    # Frequency
    sns.histplot(rfm_df['Frequency'], bins=30, kde=True, ax=axes[1], color='teal')
    axes[1].set_title('Frequency Distribution (Invoices)')
    axes[1].set_xlabel('Frequency')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Count (Log)')
    
    # Monetary
    sns.histplot(rfm_df['Monetary'], bins=30, kde=True, ax=axes[2], color='darkgreen')
    axes[2].set_title('Monetary Distribution ($)')
    axes[2].set_xlabel('Monetary')
    axes[2].set_yscale('log')
    axes[2].set_ylabel('Count (Log)')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots a correlation heatmap for numerical columns.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], title: str = "Confusion Matrix", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots a clean, beautiful confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_roc_curves(models_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots ROC curves for multiple models.
    models_curves format: {model_name: (fpr, tpr, roc_auc)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for name, (fpr, tpr, auc_score) in models_curves.items():
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predicted vs Actual", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots a scatter plot of predicted spend versus actual spend.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color='darkblue')
    
    # Perfect fit line
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Spend')
    ax.set_ylabel('Predicted Spend')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residual Plot", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots residuals (actual - predicted) vs predicted.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, color='teal')
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    
    ax.set_xlabel('Predicted Spend')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig

def plot_rl_reward_curves(rewards_tabular: List[float], rewards_dqn: List[float], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots reward curves for Tabular Q-learning and DQN agents.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Apply rolling average to smooth reward curves
    def rolling_avg(arr, window=10):
        return pd.Series(arr).rolling(window=window, min_periods=1).mean()
        
    ax.plot(rolling_avg(rewards_tabular), label='Tabular Q-Learning (Rolling 10)', color='darkorange', alpha=0.9)
    ax.plot(rolling_avg(rewards_dqn), label='DQN (Rolling 10)', color='blue', alpha=0.9)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reinforcement Learning Training Progress')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    return fig
