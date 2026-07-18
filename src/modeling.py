import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Tuple, Dict, Any, Optional, Union

def train_classifier_mlp(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> MLPClassifier:
    """
    Trains an MLPClassifier with input -> 32 ReLU -> 16 ReLU -> Sigmoid/Softmax output.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=random_state
    )
    mlp.fit(X_train, y_train)
    return mlp

def train_classifier_rf_grid(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> GridSearchCV:
    """
    Uses GridSearchCV (5-fold CV) to tune a RandomForestClassifier.
    """
    rf = RandomForestClassifier(random_state=random_state)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_classifier(model: Any, X: pd.DataFrame, y: pd.Series, name: str) -> Dict[str, Any]:
    """
    Evaluates a classifier and returns core metrics.
    """
    y_pred = model.predict(X)
    
    # Try to get prediction probability for ROC AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None
        
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'Confusion_Matrix': confusion_matrix(y, y_pred),
        'Classification_Report': classification_report(y, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['ROC_AUC'] = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        metrics['ROC_Curve'] = (fpr, tpr)
        
    return metrics

def train_regressor_mlp(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> MLPRegressor:
    """
    Trains an MLPRegressor.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=random_state
    )
    mlp.fit(X_train, y_train)
    return mlp

def train_regressor_linear(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Trains a LinearRegression model.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def evaluate_regressor(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluates a regressor and returns MSE, RMSE, MAE, and R^2.
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2)
    }

def save_model(model: Any, filepath: str) -> None:
    """
    Saves a trained model to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath: str) -> Any:
    """
    Loads a saved model from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)
