import streamlit as st


def suggest_models(task_type, df, target_column=None):
    suggestions = []
    all_models = []

    if task_type == "Classification":
        all_models = ["Logistic Regression","Decision Tree","Gradient Boosting","Naive Bayes","AdaBoost", "Random Forest", "SVM", "KNN"]

        if target_column and df[target_column].nunique() == 2:
            suggestions = ["Logistic Regression"]
        elif target_column and df[target_column].nunique() > 2:
            suggestions = ["Random Forest"]

    elif task_type == "Regression":
        all_models = ["Linear Regression", "Random Forest","Decision Tree","AdaBoost","Ridge","Lasso","ElasticNet","SVR"]
        suggestions = ["Linear Regression"]

    elif task_type == "Clustering":
        all_models = ["KMeans", "DBSCAN","Agglomerative Clustering","MeanShift","Birch"]
        suggestions = ["KMeans"]

    elif task_type == "Dimensionality Reduction":
        all_models = ["PCA", "t-SNE"]
        suggestions = ["PCA"]

    elif task_type == "Association Rule Mining":
        all_models = ["Apriori", "FPGrowth"]
        suggestions = ["Apriori"]

    st.subheader("Model Suggestion & Selection")

    if suggestions:
        st.info(f"✅ Recommended Model(s): {', '.join(suggestions)}")

    selected_models = st.multiselect(
        "Select model(s) to train:",
        options=all_models,
        default=suggestions if suggestions else []
    )

    return selected_models
