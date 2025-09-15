import streamlit as st
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import *
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import *
import numpy as np
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle
# UI to get model parameters



def get_model_parameters_ui(model_name, task_type):
    st.subheader("🔧 Model Hyperparameter Tuning")
    manual_tuning = st.checkbox("⚙️ Enable Manual Hyperparameter Tuning", value=False)

    custom_params = {}
    if not manual_tuning:
        return custom_params

    if task_type == "Classification":
        if model_name == "Logistic Regression":
            custom_params["C"] = st.number_input("C (Regularization strength)", 0.01, 10.0, 1.0)
            custom_params["solver"] = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
            custom_params["max_iter"] = st.slider("Max Iterations", 50, 1000, 100)

        elif model_name == "Random Forest":
            # custom_params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100)
            custom_params["max_depth"] = st.slider("Max Depth", 1, 100, 10)
            custom_params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)
            custom_params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 10, 1)
            custom_params["bootstrap"] = st.checkbox("Use Bootstrapping", value=True)

        elif model_name == "Gradient Boosting":
            custom_params["n_estimators"] = st.slider("Boosting Stages", 50, 500, 100)
            custom_params["learning_rate"] = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            custom_params["max_depth"] = st.slider("Max Depth", 1, 20, 3)

        elif model_name == "AdaBoost":
            custom_params["n_estimators"] = st.slider("Estimators", 50, 500, 100)
            custom_params["learning_rate"] = st.slider("Learning Rate", 0.01, 2.0, 1.0)

        elif model_name == "Decision Tree":
            custom_params["criterion"] = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
            custom_params["max_depth"] = st.slider("Max Depth", 1, 100, 10)

        elif model_name == "KNN":
            custom_params["n_neighbors"] = st.slider("Neighbors", 1, 30, 5)
            custom_params["weights"] = st.selectbox("Weights", ["uniform", "distance"])

        elif model_name == "SVM":
            custom_params["C"] = st.number_input("C (Penalty)", 0.01, 100.0, 1.0)
            custom_params["kernel"] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            custom_params["gamma"] = st.selectbox("Gamma", ["scale", "auto"])

    elif task_type == "Regression":
        if model_name == "Linear Regression":
            pass  # No major hyperparameters
        elif model_name == "Ridge":
            custom_params["alpha"] = st.number_input("Alpha", 0.01, 100.0, 1.0)
        elif model_name == "Lasso":
            custom_params["alpha"] = st.number_input("Alpha", 0.01, 100.0, 1.0)
        elif model_name == "ElasticNet":
            custom_params["alpha"] = st.number_input("Alpha", 0.01, 100.0, 0.1)
            custom_params["l1_ratio"] = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
        elif model_name == "Random Forest":
            custom_params["n_estimators"] = st.slider("Number of Trees", 10, 500, 100)
            custom_params["max_depth"] = st.slider("Max Depth", 1, 50, 10)
        elif model_name == "Gradient Boosting":
            custom_params["n_estimators"] = st.slider("Boosting Stages", 50, 500, 100)
            custom_params["learning_rate"] = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            custom_params["max_depth"] = st.slider("Max Depth", 1, 20, 3)
        elif model_name == "Decision Tree":
            custom_params["criterion"] = st.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])
            custom_params["max_depth"] = st.slider("Max Depth", 1, 100, 10)
        elif model_name == "KNN":
            custom_params["n_neighbors"] = st.slider("Neighbors", 1, 30, 5)
            custom_params["weights"] = st.selectbox("Weights", ["uniform", "distance"])
        elif model_name == "SVR":
            custom_params["C"] = st.number_input("C (Penalty)", 0.01, 100.0, 1.0)
            custom_params["kernel"] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])

    elif task_type == "Clustering":
        if model_name == "KMeans":
            custom_params["n_clusters"] = st.slider("Clusters", 2, 20, 3)
        elif model_name == "DBSCAN":
            custom_params["eps"] = st.slider("Epsilon", 0.1, 10.0, 0.5)
            custom_params["min_samples"] = st.slider("Min Samples", 1, 10, 5)
        elif model_name == "Agglomerative Clustering":
            custom_params["n_clusters"] = st.slider("Clusters", 2, 20, 3)
        elif model_name == "MeanShift":
            pass
        elif model_name == "Birch":
            custom_params["n_clusters"] = st.slider("Clusters", 2, 20, 3)

    elif task_type == "Dimensionality Reduction":
        if model_name == "PCA":
            custom_params["n_components"] = st.slider("Number of Components", 2, 50, 2)
        elif model_name == "TruncatedSVD":
            custom_params["n_components"] = st.slider("Number of Components", 2, 50, 2)
        elif model_name == "ICA":
            custom_params["n_components"] = st.slider("Number of Components", 2, 50, 2)
        elif model_name == "t-SNE":
            custom_params["n_components"] = st.slider("Components", 2, 3, 2)
            custom_params["perplexity"] = st.slider("Perplexity", 5, 50, 30)

    elif task_type == "Association Rule Mining":
        custom_params["min_support"] = st.slider("Min Support", 0.01, 1.0, 0.5)
        custom_params["metric"] = st.selectbox("Metric", ["lift", "confidence", "support"])
        custom_params["min_threshold"] = st.slider("Min Threshold", 0.01, 5.0, 1.0)

    return custom_params


# Main training function
def train_model(X_train, y_train, task_type, selected_model, custom_params=None, fit_status=None):
    st.subheader("📦 Model Training")
    model = None

    if custom_params is None:
        custom_params = {}

    # 🔁 Auto-adjust hyperparameters for overfitting or underfitting
    if fit_status == "Overfitting":
        st.info("🔁 Auto-adjusting model for Overfitting")
        if selected_model in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            custom_params["max_depth"] = custom_params.get("max_depth", 5)
            custom_params["min_samples_split"] = custom_params.get("min_samples_split", 4)
            custom_params["min_samples_leaf"] = custom_params.get("min_samples_leaf", 2)

    elif fit_status == "Underfitting":
        st.info("🔁 Auto-adjusting model for Underfitting")
        if selected_model in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            custom_params["max_depth"] = custom_params.get("max_depth", 30)
            custom_params["n_estimators"] = custom_params.get("n_estimators", 300)

    # ------------------------
    # CLASSIFICATION MODELS
    # ------------------------
    if task_type == "Classification":
        if selected_model == "Logistic Regression":
            model = LogisticRegression(**custom_params)
        elif selected_model == "Random Forest":
            model = RandomForestClassifier(**custom_params)
        elif selected_model == "Gradient Boosting":
            model = GradientBoostingClassifier(**custom_params)
        elif selected_model == "AdaBoost":
            model = AdaBoostClassifier(**custom_params)
        elif selected_model == "Decision Tree":
            model = DecisionTreeClassifier(**custom_params)
        elif selected_model == "KNN":
            model = KNeighborsClassifier(**custom_params)
        elif selected_model == "SVM":
            model = SVC(**custom_params)
        elif selected_model == "Naive Bayes":
            model = GaussianNB(**custom_params)

    # ------------------------
    # REGRESSION MODELS
    # ------------------------
    elif task_type == "Regression":
        if selected_model == "Linear Regression":
            model = LinearRegression(**custom_params)
        elif selected_model == "Random Forest":
            model = RandomForestRegressor(**custom_params)
        elif selected_model == "Gradient Boosting":
            model = GradientBoostingRegressor(**custom_params)
        elif selected_model == "AdaBoost":
            model = AdaBoostRegressor(**custom_params)
        elif selected_model == "Decision Tree":
            model = DecisionTreeRegressor(**custom_params)
        elif selected_model == "KNN":
            model = KNeighborsRegressor(**custom_params)
        elif selected_model == "Ridge":
            model = Ridge(**custom_params)
        elif selected_model == "Lasso":
            model = Lasso(**custom_params)
        elif selected_model == "ElasticNet":
            model = ElasticNet(**custom_params)
        elif selected_model == "SVR":
            model = SVR(**custom_params)

    # ------------------------
    # CLUSTERING
    # ------------------------
    elif task_type == "Clustering":
        if selected_model == "KMeans":
            model = KMeans(**custom_params)
        elif selected_model == "DBSCAN":
            model = DBSCAN(**custom_params)
        elif selected_model == "Agglomerative Clustering":
            model = AgglomerativeClustering(**custom_params)
        elif selected_model == "MeanShift":
            model = MeanShift(**custom_params)
        elif selected_model == "Birch":
            model = Birch(**custom_params)

    # ------------------------
    # DIMENSIONALITY REDUCTION
    # ------------------------
    elif task_type == "Dimensionality Reduction":
        if selected_model == "PCA":
            model = PCA(**custom_params)
        elif selected_model == "TruncatedSVD":
            model = TruncatedSVD(**custom_params)
        elif selected_model == "ICA":
            model = FastICA(**custom_params)
        elif selected_model == "t-SNE":
            model = TSNE(**custom_params)

    # ------------------------
    # ASSOCIATION RULE MINING
    # ------------------------
    elif task_type == "Association Rule Mining":
        if selected_model == "Apriori":
            st.info("👉 Applying Apriori algorithm to find frequent itemsets...")
            frequent_itemsets = apriori(X_train, min_support=custom_params.get("min_support", 0.5), use_colnames=True)
            rules = association_rules(
                frequent_itemsets,
                metric=custom_params.get("metric", "lift"),
                min_threshold=custom_params.get("min_threshold", 1.0)
            )
            st.success("✅ Association rules generated successfully!")
            st.dataframe(rules)
            return rules  # Return directly

    # ------------------------
    # MODEL TRAINING
    # ------------------------
    if model:
        if task_type not in ["Association Rule Mining", "Dimensionality Reduction"]:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)
        st.success("✅ Model trained successfully!")
        return model
    else:
        st.error("❌ Could not initialize the selected model. Please check your selection.")
        return None


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, log_loss, matthews_corrcoef,
    balanced_accuracy_score, mean_squared_error, mean_absolute_error,
    r2_score, silhouette_score, adjusted_rand_score,
    davies_bouldin_score, calinski_harabasz_score
)


def evaluate_model(model, X_train, y_train=None, X_val=None, y_val=None, task_type=None):
    task_type = task_type.lower().strip() if task_type else ""

    try:
        if task_type == "classification":
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            st.subheader("📊 Classification Report")
            train_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)).transpose()
            val_report = pd.DataFrame(classification_report(y_val, y_val_pred, output_dict=True)).transpose()
            st.write("### Training Report")
            st.dataframe(train_report.round(2))
            st.write("### Validation Report")
            st.dataframe(val_report.round(2))

            st.subheader("📈 Metrics Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Training")
                st.metric("Accuracy", f"{accuracy_score(y_train, y_train_pred):.4f}")
                st.metric("Precision", f"{precision_score(y_train, y_train_pred, average='weighted'):.4f}")
                st.metric("Recall", f"{recall_score(y_train, y_train_pred, average='weighted'):.4f}")
                st.metric("F1 Score", f"{f1_score(y_train, y_train_pred, average='weighted'):.4f}")
                st.metric("Balanced Accuracy", f"{balanced_accuracy_score(y_train, y_train_pred):.4f}")
                st.metric("MCC", f"{matthews_corrcoef(y_train, y_train_pred):.4f}")

            with col2:
                st.markdown("#### Validation")
                st.metric("Accuracy", f"{accuracy_score(y_val, y_val_pred):.4f}")
                st.metric("Precision", f"{precision_score(y_val, y_val_pred, average='weighted'):.4f}")
                st.metric("Recall", f"{recall_score(y_val, y_val_pred, average='weighted'):.4f}")
                st.metric("F1 Score", f"{f1_score(y_val, y_val_pred, average='weighted'):.4f}")
                st.metric("Balanced Accuracy", f"{balanced_accuracy_score(y_val, y_val_pred):.4f}")
                st.metric("MCC", f"{matthews_corrcoef(y_val, y_val_pred):.4f}")

                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_val)
                        if y_proba is not None:
                            st.metric("Log Loss", f"{log_loss(y_val, y_proba):.4f}")
                            if y_proba.ndim == 2:
                                roc_auc = roc_auc_score(y_val, y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba,
                                                        multi_class='ovr')
                                st.metric("ROC-AUC", f"{roc_auc:.4f}")
                except Exception as e:
                    st.warning(f"⚠️ Could not compute Log Loss / ROC-AUC: {e}")

            st.subheader("📉 Confusion Matrix (Validation)")
            cm = confusion_matrix(y_val, y_val_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            if accuracy_score(y_train, y_train_pred) - accuracy_score(y_val, y_val_pred) > 0.15:
                return "Overfitting"
            elif accuracy_score(y_val, y_val_pred) < 0.6:
                return "Underfitting"
            else:
                return "Good Fit"

        elif task_type == "regression":
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            st.subheader("📊 Regression Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Training")
                st.metric("R² Score", f"{r2_score(y_train, y_train_pred):.4f}")
                st.metric("MAE", f"{mean_absolute_error(y_train, y_train_pred):.4f}")
                st.metric("MSE", f"{mean_squared_error(y_train, y_train_pred):.4f}")
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
            with col2:
                st.markdown("#### Validation")
                st.metric("R² Score", f"{r2_score(y_val, y_val_pred):.4f}")
                st.metric("MAE", f"{mean_absolute_error(y_val, y_val_pred):.4f}")
                st.metric("MSE", f"{mean_squared_error(y_val, y_val_pred):.4f}")
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}")

            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            if train_r2 - val_r2 > 0.15:
                return "Overfitting"
            elif val_r2 < 0.5:
                return "Underfitting"
            else:
                return "Good Fit"

        elif task_type == "clustering":
            st.subheader("🔍 Clustering Evaluation")
            try:
                cluster_labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_train)
                st.metric("🧠 Silhouette Score", f"{silhouette_score(X_train, cluster_labels):.4f}")
                st.metric("📉 Davies-Bouldin Index", f"{davies_bouldin_score(X_train, cluster_labels):.4f}")
                st.metric("📈 Calinski-Harabasz Score", f"{calinski_harabasz_score(X_train, cluster_labels):.2f}")
                if y_train is not None:
                    st.metric("🔁 Adjusted Rand Index", f"{adjusted_rand_score(y_train, cluster_labels):.4f}")
                return "Good Fit"
            except Exception as e:
                st.error(f"Clustering Evaluation Failed: {e}")
                return "Error"

        elif task_type == "association rule mining":
            st.subheader("🔗 Association Rule Summary")
            try:
                if isinstance(model, pd.DataFrame) and 'support' in model.columns:
                    rules = association_rules(model, metric="lift", min_threshold=1.0)
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3))
                    st.success(f"✅ {len(rules)} strong rules discovered")
                    return "Completed"
                else:
                    st.error("❌ Invalid association rule model.")
                    return "Error"
            except Exception as e:
                st.error(f"Association evaluation failed: {e}")
                return "Error"

        else:
            st.warning("🚫 Task type not recognized for evaluation.")
            return "Unknown"

    except Exception as e:
        st.error(f"❌ Error during evaluation: {e}")
        return "Error"


def save_model(model, model_name, username):
    folder = f"models/{username}"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{model_name}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
