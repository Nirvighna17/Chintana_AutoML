import streamlit as st
import pandas as pd

def detect_target_column(df):
    """
    Automatically detect a potential target column in the dataframe.
    Returns the name of the column most likely to be the target.
    """
    # Candidates with few unique values (often targets)
    potential_targets = [
        col for col in df.columns
        if df[col].nunique() <= 20 and df[col].dtype in ['object', 'bool', 'int64']
    ]

    # Prioritize binary columns
    binary_targets = [col for col in potential_targets if df[col].nunique() == 2]
    if binary_targets:
        return binary_targets[0]

    # Fallback: pick column with highest correlation to others (useful for regression)
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(~(pd.np.tril(pd.np.ones(corr_matrix.shape)).astype(bool)))
        strongest_target = upper.sum().sort_values(ascending=False).index[0]
        return strongest_target

    # If all else fails, return first potential target
    if potential_targets:
        return potential_targets[0]

    return None

def get_target_column(df):
    st.subheader("🎯 Target Column Selection")

    user_option = st.radio("Do you want to perform Supervised or Unsupervised Learning?",
                           ["Supervised Learning", "Unsupervised Learning"], key="task_choice")

    if user_option == "Unsupervised Learning":
        st.info("Since this is unsupervised learning, no target column is required.")
        st.markdown("""
        ### 📌 Suggested Tasks for Unsupervised Learning:
        - 🔗 **Clustering** (e.g., KMeans, DBSCAN)
        - 📊 **Dimensionality Reduction** (e.g., PCA, t-SNE)
        - 🛒 **Association Rule Mining** (e.g., Market Basket Analysis)
        """)

        task_type = st.radio(
            "🧠 Select a task you'd like to perform:",
            options=["Clustering", "Dimensionality Reduction", "Association Rule Mining"],
            key="unsupervised_task_choice"
        )

        st.success(f"🧪 Selected Unsupervised Task: **{task_type}**")
        return None, task_type

    # Try to suggest likely target columns based on name and uniqueness

    likely_targets = [
        col for col in df.columns
        if col.lower() in ["target", "label", "y"] or df[col].nunique() < len(df) * 0.5
    ]

    if likely_targets:
        default_index = df.columns.get_loc(likely_targets[0])
    else:
        default_index = 0

    target_column = st.selectbox("Select the target column:", df.columns, index=default_index, key="target_select")

    # Validation checks
    if df[target_column].isnull().any():
        st.warning(f"⚠ The selected target column '{target_column}' has missing values. Consider cleaning it.")

    if df[target_column].nunique() <= 1:
        st.error("❌ The selected target column has only one unique value. Please choose another column.")

    suggested_task = detect_task_type(df, target_column)
    st.info(f"🔍 Suggested Task Type: **{suggested_task}**")

    task_type = st.radio(
        "🧠 Select the type of ML task you want to perform:",
        options=["Classification", "Regression"],
        index=0 if suggested_task == "Classification" else 1,
        key="user_task_choice"
    )

    if task_type != suggested_task:
        st.warning(f"⚠ You selected **{task_type}**, but we suggested **{suggested_task}** based on the data. Make sure this is intentional.")

    st.success(f"🧪 Selected Task: **{task_type}**")

    return target_column, task_type


def detect_task_type(df, target_column):
    if target_column is None:
        return "Unsupervised"

    unique_vals = df[target_column].nunique()
    dtype = df[target_column].dtype

    if pd.api.types.is_numeric_dtype(dtype):
        if unique_vals <= 10:
            return "Classification"
        else:
            return "Regression"
    else:
        return "Classification"
