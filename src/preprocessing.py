import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Set, Optional

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw online retail CSV dataset, standardizes column names, and applies cleaning filters:
    - Renames 'Invoice' to 'InvoiceNo', 'Price' to 'UnitPrice', and 'Customer ID' to 'CustomerID' if needed.
    - Drops rows where CustomerID is null.
    - Removes cancelled invoices (where InvoiceNo starts with 'C').
    - Removes rows where Quantity <= 0.
    - Removes rows where UnitPrice <= 0.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Rename columns to standard names
    col_mapping = {
        'Invoice': 'InvoiceNo',
        'Price': 'UnitPrice',
        'Customer ID': 'CustomerID'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Check that required columns exist
    required_cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset. Found: {list(df.columns)}")
            
    # Clean data
    # 1. Drop CustomerID null values
    df = df.dropna(subset=['CustomerID'])
    
    # Cast CustomerID to int (or string if it has text, but retail CustomerID is numeric)
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # 2. Remove cancelled invoices (InvoiceNo starts with C)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C', na=False)]
    
    # 3. Remove Quantity <= 0
    df = df[df['Quantity'] > 0]
    
    # 4. Remove UnitPrice <= 0
    df = df[df['UnitPrice'] > 0]
    
    # Parse InvoiceDate
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)
    
    # Create TotalPrice
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    return df

def get_category_map(map_path: str = 'category_map.json') -> Dict[str, List[str]]:
    """
    Loads or creates category_map.json.
    """
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            return json.load(f)
    
    # Default category map from project requirements
    default_map = {
        "Homeware": ["HOME", "MUG", "CANDLE", "LANTERN", "CUSHION"],
        "Stationery": ["CARD", "NOTEBOOK", "PEN", "PAPER", "ENVELOPE"],
        "Gadgets": ["LIGHT", "CLOCK", "BATTERY", "ALARM"],
        "Decorations": ["CHRISTMAS", "DECORATION", "BUNTING", "GARLAND"],
        "Kitchenware": ["BAKING", "CAKE", "TIN", "JAR", "BOWL"],
        "Else": ["Other"]
    }
    with open(map_path, 'w') as f:
        json.dump(default_map, f, indent=2)
    return default_map

def assign_product_categories(df: pd.DataFrame, category_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Vectorized classification of product descriptions into categories.
    Adds a 'Category' column to the dataframe.
    """
    df = df.copy()
    df['Description'] = df['Description'].fillna('').astype(str)
    df['Category'] = 'Else'
    
    # Build regex patterns for categories (excluding 'Else')
    for cat, keywords in category_map.items():
        if cat == 'Else':
            continue
        pattern = '|'.join(keywords)
        # Vectorized check: if description contains pattern (case-insensitive) and currently marked 'Else'
        mask = (df['Category'] == 'Else') & df['Description'].str.contains(pattern, case=False, na=False)
        df.loc[mask, 'Category'] = cat
        
    return df

def compute_customer_rfm(df: pd.DataFrame, category_map: Dict[str, List[str]], snapshot_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Groups the cleaned transactional dataframe by CustomerID to compute customer-level RFM features:
    - Recency
    - Frequency
    - Monetary
    - Product Diversity
    - Average Spend Per Transaction
    - Category Spend % (for each of the categories)
    """
    # Standardize description categories
    df_cat = assign_product_categories(df, category_map)
    
    # Determine snapshot date
    if snapshot_date is None:
        snapshot_date = df_cat['InvoiceDate'].max() + pd.Timedelta(days=1)
        
    # Group by CustomerID
    cust_groups = df_cat.groupby('CustomerID')
    
    # Calculate RFM & General Features
    rfm = cust_groups.agg(
        Last_Date=('InvoiceDate', 'max'),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
        Product_Diversity=('StockCode', 'nunique')
    ).reset_index()
    
    rfm['Recency'] = (snapshot_date - rfm['Last_Date']).dt.days
    rfm = rfm.drop(columns=['Last_Date'])
    
    # Calculate Average Spend Per Transaction
    rfm['Average_Spend_Per_Transaction'] = rfm['Monetary'] / rfm['Frequency']
    
    # Pivot to get spend per category for each customer
    cat_spend = df_cat.groupby(['CustomerID', 'Category'])['TotalPrice'].sum().unstack(fill_value=0.0)
    
    # Ensure all mapping categories are present in columns
    all_categories = list(category_map.keys())
    for cat in all_categories:
        if cat not in cat_spend.columns:
            cat_spend[cat] = 0.0
            
    # Compute Category Spend Percentages
    for cat in all_categories:
        pct_col_name = f"{cat}_Spend_Pct"
        # Avoid division by zero
        rfm[pct_col_name] = rfm['CustomerID'].map(cat_spend[cat]) / rfm['Monetary']
        rfm[pct_col_name] = rfm[pct_col_name].fillna(0.0)
        
    return rfm

def split_customers(customer_ids: pd.Series, train_pct: float = 0.8, val_pct: float = 0.1, test_pct: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits unique customer IDs into Train (80%), Val (10%), and Test (10%) sets.
    Ensures splitting is done strictly at customer level to never mix transactions.
    """
    unique_custs = np.unique(customer_ids)
    
    # Set seed
    np.random.seed(random_state)
    shuffled_custs = np.random.permutation(unique_custs)
    
    total_len = len(shuffled_custs)
    train_end = int(total_len * train_pct)
    val_end = int(total_len * (train_pct + val_pct))
    
    train_custs = shuffled_custs[:train_end]
    val_custs = shuffled_custs[train_end:val_end]
    test_custs = shuffled_custs[val_end:]
    
    return train_custs, val_custs, test_custs

def create_regression_dataset(df: pd.DataFrame, category_map: Dict[str, List[str]], 
                               f_start: str = '2009-12-01', f_end: str = '2010-08-31', 
                               t_start: str = '2010-09-01', t_end: str = '2010-11-30') -> pd.DataFrame:
    """
    Builds the regression dataset based on operational definition:
    - Features are computed from transactions in Months 1-9 (f_start to f_end).
    - The target value (Future_Spend) is total spend in Months 10-12 (t_start to t_end).
    """
    f_start_dt = pd.to_datetime(f_start)
    f_end_dt = pd.to_datetime(f_end)
    t_start_dt = pd.to_datetime(t_start)
    t_end_dt = pd.to_datetime(t_end)
    
    # Filter transactions for features and targets
    feat_df = df[(df['InvoiceDate'] >= f_start_dt) & (df['InvoiceDate'] <= f_end_dt)]
    targ_df = df[(df['InvoiceDate'] >= t_start_dt) & (df['InvoiceDate'] <= t_end_dt)]
    
    # Compute features for active customers in Months 1-9
    # Snapshot date for feature window is the day after the feature window end
    snapshot_date = f_end_dt + pd.Timedelta(days=1)
    features = compute_customer_rfm(feat_df, category_map, snapshot_date=snapshot_date)
    
    # Compute target spend for each customer
    target_spend = targ_df.groupby('CustomerID')['TotalPrice'].sum().to_dict()
    
    # Map future spend to features, filling 0 for customers who did not buy in target window
    features['Future_Spend'] = features['CustomerID'].map(target_spend).fillna(0.0)
    
    return features
