import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, StratifiedKFold

def auto_split(df, target_column=None):
    """
    Automatically splits data into train, validation, and test sets based on best practices.
    """
    stratify = df[target_column] if target_column and df[target_column].nunique() < 20 else None

    train_val, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=stratify
    )

    stratify = train_val[target_column] if stratify is not None else None
    train, val = train_test_split(
        train_val, test_size=0.1765, random_state=42, stratify=stratify
    )

    return train, val, test

def manual_split(df, target_column=None, train_size=0.7, val_size=0.15, stratify=False):
    """
    Manually splits the data based on user-defined ratios.
    """
    test_size = 1 - train_size - val_size

    stratify_col = df[target_column] if stratify and target_column else None

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=stratify_col
    )

    stratify_col = train_val[target_column] if stratify and target_column else None
    val_ratio = val_size / (train_size + val_size)

    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=42, stratify=stratify_col
    )

    return train, val, test

def generate_kfold_splits(df, target_column=None, k=5, stratify=False):
    """
    Generate K-Fold or Stratified K-Fold splits.
    """
    splits = []
    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None

    if stratify and y is not None:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X, y):
            splits.append((train_idx, test_idx))
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            splits.append((train_idx, test_idx))
    return splits

def display_split_summary(train, val, test):
    st.subheader("📊 Data Split Summary")
    st.write(f"✅ Training Set: {train.shape[0]} rows")
    st.write(f"✅ Validation Set: {val.shape[0]} rows")
    st.write(f"✅ Test Set: {test.shape[0]} rows")
    st.dataframe(train.head())

def display_kfold_info(k_splits):
    st.subheader("📂 K-Fold Splits")
    st.write(f"Total folds: {len(k_splits)}")
    for i, (train_idx, test_idx) in enumerate(k_splits):
        st.write(f"Fold {i+1}: Train indices ({len(train_idx)}), Test indices ({len(test_idx)})")