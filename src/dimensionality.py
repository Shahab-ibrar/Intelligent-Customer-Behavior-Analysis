import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from typing import Tuple, List, Dict, Any

def standardize_features(train_features: pd.DataFrame, val_features: pd.DataFrame, test_features: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardizes specified columns using a StandardScaler fit ONLY on the training set.
    """
    scaler = StandardScaler()
    
    # Fit on training and transform all splits
    train_scaled = train_features.copy()
    val_scaled = val_features.copy()
    test_scaled = test_features.copy()
    
    train_scaled[columns] = scaler.fit_transform(train_features[columns])
    val_scaled[columns] = scaler.transform(val_features[columns])
    test_scaled[columns] = scaler.transform(test_features[columns])
    
    return train_scaled, val_scaled, test_scaled, scaler

def run_pca(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, 
            columns: List[str], target_variance: float = 0.90) -> Tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Fits PCA on standardised training features to cover target_variance (90%).
    Returns:
    - pca: Fitted PCA model
    - train_pca: Train data projected to PCA components
    - val_pca: Val data projected to PCA components
    - test_pca: Test data projected to PCA components
    - n_components: Optimal number of components
    """
    # Fit PCA with enough components to capture up to target_variance
    # We can just fit full PCA first to determine components, then re-fit or slice
    full_pca = PCA(random_state=42)
    full_pca.fit(train_data[columns])
    
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    
    # Re-fit/initialize PCA with optimal n_components
    pca = PCA(n_components=n_components, random_state=42)
    
    train_proj = pca.fit_transform(train_data[columns])
    val_proj = pca.transform(val_data[columns])
    test_proj = pca.transform(test_data[columns])
    
    # Create column names PC1, PC2...
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    
    train_pca_df = pd.DataFrame(train_proj, columns=pc_cols, index=train_data.index)
    val_pca_df = pd.DataFrame(val_proj, columns=pc_cols, index=val_data.index)
    test_pca_df = pd.DataFrame(test_proj, columns=pc_cols, index=test_data.index)
    
    return pca, train_pca_df, val_pca_df, test_pca_df, int(n_components)

def run_lda(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, 
            columns: List[str], y_train: pd.Series) -> Tuple[LDA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fits LDA on standardised training features using high value target labels.
    Returns:
    - lda: Fitted LDA model
    - train_lda: Train data projected to LDA components
    - val_lda: Val data projected to LDA components
    - test_lda: Test data projected to LDA components
    """
    lda = LDA()
    train_proj = lda.fit_transform(train_data[columns], y_train)
    val_proj = lda.transform(val_data[columns])
    test_proj = lda.transform(test_data[columns])
    
    lda_cols = [f"LDA{i+1}" for i in range(train_proj.shape[1])]
    
    train_lda_df = pd.DataFrame(train_proj, columns=lda_cols, index=train_data.index)
    val_lda_df = pd.DataFrame(val_proj, columns=lda_cols, index=val_data.index)
    test_lda_df = pd.DataFrame(test_proj, columns=lda_cols, index=test_data.index)
    
    return lda, train_lda_df, val_lda_df, test_lda_df

def compare_features_baseline(
    X_train_raw: pd.DataFrame, y_train: pd.Series,
    X_val_raw: pd.DataFrame, y_val: pd.Series,
    X_train_pca: pd.DataFrame, X_val_pca: pd.DataFrame,
    X_train_lda: pd.DataFrame, X_val_lda: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compares Raw Features, PCA features, and LDA features using a baseline LogisticRegression.
    Returns a DataFrame containing Accuracy, Precision, Recall, and F1-Score for each space.
    """
    baseline = LogisticRegression(random_state=random_state, max_iter=1000)
    
    results = []
    
    configs = [
        ('Raw Features', X_train_raw, X_val_raw),
        ('PCA Features', X_train_pca, X_val_pca),
        ('LDA Features', X_train_lda, X_val_lda)
    ]
    
    for name, train_feats, val_feats in configs:
        baseline.fit(train_feats, y_train)
        y_pred = baseline.predict(val_feats)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            'Feature Set': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
    return pd.DataFrame(results)
