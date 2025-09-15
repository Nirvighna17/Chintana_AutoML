import streamlit as st
import pandas as pd
import sqlite3
import pickle
from src.data_loader import load_uploaded_file, get_predefined_dataset
from src.cleaner import clean_data
from src.transformer import full_transformation_pipeline
from src.visualizer import generate_visual_report
from src.ml_task_selector import get_target_column, detect_task_type
from src.model_selector import suggest_models
from src.data_splitter import auto_split, manual_split, generate_kfold_splits, display_split_summary, display_kfold_info
from src.train_model import train_model, evaluate_model, get_model_parameters_ui, save_model
import streamlit.components.v1 as components
import os
import shutil

if not shutil.which("google-chrome") and not shutil.which("chromium"):
    os.system("apt-get update && apt-get install -y chromium-browser")
    os.environ["PATH"] += os.pathsep + "/usr/bin"

components.html("""
    <script>
        history.pushState(null, '', location.href);
        window.onpopstate = function () {
            history.go(1);
        };
    </script>
""", height=0)

# --- PAGE CONFIG ---
st.set_page_config(page_title="CHINTANA AI", layout="wide")

# --- LOGIN CHECK ---
if not st.session_state.get("logged_in", False):
    st.error("🚫 Access Denied: Please login first.")
    st.page_link("pages/login_signup.py", label="Go to Login")
    st.stop()

username = st.session_state.get("username")
role = st.session_state.get("role", "user")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration Panel")
st.sidebar.markdown(f"👤 **{username}** ")


if st.session_state.get("logged_in", False):
    # Show only relevant pages
    st.sidebar.page_link("pages/dashboard.py", label="Dashboard")
    st.sidebar.page_link("pages/automl_app.py", label="AutoML App")

    # Add Logout Button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.success("Logged out successfully.")
        st.rerun()

if st.session_state["role"] == "admin":
    with st.expander("Admin Panel", expanded=False):
        st.success("Welcome, Admin! You have full access.")
        st.write("Here you can manage Chintana platform operations.")

        st.subheader("👥 Manage Users")
        DB_FILE = "users.db"
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # Get all users except current admin
        c.execute("SELECT * FROM users WHERE username != ?", (username,))
        users = c.fetchall()

        if users:
            for user in users:
                user_id, uname, _, _, user_role = user  # 5 columns: id, username, email, password, role

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**👤 {uname}** — *{user_role}*")
                with col2:
                    if st.button("❌ Delete", key=f"del_{user_id}"):
                        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
                        conn.commit()
                        st.success(f"User **{uname}** deleted successfully.")
                        st.rerun()
        else:
            st.info("No other users found.")


# --- ROLE BANNER ---
st.title("CHINTANA – Let Your Data Think.")

if role == "admin":
    st.subheader("Admin Panel")
    st.markdown("You have full access to user data and system control.")

    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT username, email, role FROM users")
        users = c.fetchall()
        conn.close()
        st.markdown("### 👥 Registered Users")
        st.dataframe(pd.DataFrame(users, columns=["Username", "Email", "Role"]))
    except Exception as e:
        st.error(f"Unable to fetch users: {e}")

    st.markdown("---")
    st.subheader("📊 Proceed with Chintana AutoML Tasks")
else:
    st.subheader(" Welcome to Chintana AutoML")

# --- SESSION KEYS SETUP ---
for key in ["df_transformed", "train_set", "val_set", "test_set", "split_done"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- FILE UPLOAD ---
available_datasets = [
    "Iris", "Wine", "Diabetes", "Breast Cancer", "California Housing",
    "Covertype", "Titanic", "Titanic (Full)", "Tips", "Penguins",
    "Flights", "Diamonds", "Planets", "Car Crashes", "Exercise",
    "FMRI", "Geyser", "Anscombe", "Brain Networks"
]

dataset_source = st.radio("Select Dataset Source", ["Upload your own", "Use a predefined dataset"])
df = None

if dataset_source == "Upload your own":
    uploaded_file = st.file_uploader("Upload a dataset file", type=["csv", "xlsx", "xls", "json", "parquet"])
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        if df is not None:
            st.success(f"\u2705 File '{uploaded_file.name}' uploaded successfully!")
else:
    selected_dataset = st.selectbox("Choose a predefined dataset", available_datasets)
    df = get_predefined_dataset(selected_dataset)
    if df is not None and not df.empty:
        st.success(f"\u2705 {selected_dataset} dataset loaded!")

if df is not None and not df.empty:
    st.write("Preview of Dataset")
    st.dataframe(df.head())
    st.markdown(f"**Raw Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")


# --- SIDEBAR CLEANING OPTIONS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Data Cleaning Options")
apply_cleaning = st.sidebar.checkbox("Apply Data Cleaning", value=True)
drop_duplicates = st.sidebar.checkbox("Drop Duplicate Rows", value=True)
drop_high_null_columns = st.sidebar.checkbox("Drop High Null Columns", value=True)
handle_missing_values = st.sidebar.checkbox("Handle Missing Values", value=True)
fix_column_types = st.sidebar.checkbox("Fix Column Types", value=True)
strip_whitespace = st.sidebar.checkbox("Strip Whitespace", value=True)
remove_constant_columns = st.sidebar.checkbox("Remove Constant Columns", value=True)
remove_outliers_iqr = st.sidebar.checkbox("Remove Outliers (IQR)", value=True)

# --- TRANSFORMATION OPTIONS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Data Transformation Options")
scaling_method = st.sidebar.selectbox("Feature Scaling Method", ["standard", "minmax", "robust", "maxabs"])
include_polynomial = st.sidebar.checkbox("Add Polynomial Features", value=False)
apply_log_transform = st.sidebar.checkbox("Apply Log Transformation to Skewed Features", value=True)
smart_encode = st.sidebar.checkbox("Smart Encode Categorical Features", value=True)
drop_highly_correlated = st.sidebar.checkbox("Drop Highly Correlated Features", value=True)
remove_low_variance = st.sidebar.checkbox("Remove Low Variance Features", value=True)
extract_datetime_features = st.sidebar.checkbox("Extract Features from Datetime Columns", value=True)

# --- MAIN PIPELINE ---
if df is not None:
    try:

        st.subheader("Column Type Distribution")
        col_types = df.dtypes.value_counts()
        st.bar_chart(col_types)

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing)
        else:
            st.info("No missing values detected.")

        # --- CLEANING ---
        df_cleaned = clean_data(
            df,
            drop_duplicates=drop_duplicates,
            drop_high_null_columns=drop_high_null_columns,
            handle_missing_values=handle_missing_values,
            fix_column_types=fix_column_types,
            strip_whitespace=strip_whitespace,
            remove_constant_columns=remove_constant_columns,
            remove_outliers_iqr=remove_outliers_iqr
        ) if apply_cleaning else df

        st.subheader(" Cleaned Data Preview")
        st.dataframe(df_cleaned.head())
        st.markdown(f"**Cleaned Data Shape:** {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
        st.download_button("📥 Download Cleaned CSV", df_cleaned.to_csv(index=False).encode(), file_name="cleaned_data.csv", mime="text/csv")

        # --- VISUAL REPORT ---
        if st.button("📊 Generate Smart Visual Report"):
            generate_visual_report(df_cleaned)

        # --- TASK DETECTION ---
        st.subheader("📌 ML Task Selection")
        target_column, task_type = get_target_column(df_cleaned)
        if task_type:
            st.success(f"Task Detected: **{task_type}**")

        selected_models = suggest_models(task_type, df_cleaned, target_column)
        if selected_models:
            st.success(f"✅ Selected Models: {', '.join(selected_models)}")

        y_target = df_cleaned[target_column]
        X = df_cleaned.drop(columns=[target_column])

        # --- TRANSFORMATION ---
        if st.button("⚙ Run Transformation"):
            x_transformed = full_transformation_pipeline(
                X,
                scale_method=scaling_method,
                include_polynomial=include_polynomial,
                apply_log=apply_log_transform,
                smart_encode=smart_encode,
                drop_corr=drop_highly_correlated,
                drop_low_var=remove_low_variance,
                extract_datetime=extract_datetime_features
            )
            df_transformed = pd.concat([x_transformed, y_target.reset_index(drop=True)], axis=1)
            st.session_state.df_transformed = df_transformed
            st.success("✅ Transformation Complete")

        df_transformed = st.session_state.df_transformed

        if df_transformed is not None:
            st.subheader("Transformed Data")
            st.dataframe(df_transformed.head())
            st.markdown(f"🔄 **Transformed Data Shape:** {df_transformed.shape[0]} rows × {df_transformed.shape[1]} columns")
            st.download_button("📥 Download Transformed CSV", df_transformed.to_csv(index=False).encode(), file_name="transformed_data.csv", mime="text/csv")

            # --- SPLIT DATA ---
            st.subheader("Data Split Configuration")
            split_mode = st.radio("Split Mode", ["Automatic", "Manual"])

            if split_mode == "Manual":
                train_ratio = st.slider("Train %", 10, 90, 70) / 100
                val_ratio = st.slider("Validation %", 5, 40, 15) / 100
                test_ratio = 1.0 - (train_ratio + val_ratio)

                stratify_split = st.checkbox("Stratify (for classification)", True)
                use_kfold = st.checkbox("Enable K-Fold", False)
                k = st.number_input("Number of Folds", 2, 20, 5) if use_kfold else None

                if st.button("Split Data"):
                    if use_kfold:
                        folds = generate_kfold_splits(df_transformed, target_column, k=k, stratify=stratify_split)
                        display_kfold_info(folds)
                    else:
                        train_set, val_set, test_set = manual_split(df_transformed, target_column, train_ratio, val_ratio, stratify_split)
                        display_split_summary(train_set, val_set, test_set)
                        st.session_state.train_set = train_set
                        st.session_state.val_set = val_set
                        st.session_state.test_set = test_set
                        st.session_state.split_done = True
            else:
                train_set, val_set, test_set = auto_split(df_transformed, target_column)
                display_split_summary(train_set, val_set, test_set)
                st.session_state.train_set = train_set
                st.session_state.val_set = val_set
                st.session_state.test_set = test_set
                st.session_state.split_done = True

        # --- MODEL TRAINING ---
        if all([
            st.session_state.split_done,
            st.session_state.train_set is not None,
            st.session_state.val_set is not None,
            target_column,
            task_type,
            selected_models
        ]):
            X_train = st.session_state.train_set.drop(columns=[target_column])
            y_train = st.session_state.train_set[target_column]
            X_val = st.session_state.val_set.drop(columns=[target_column])
            y_val = st.session_state.val_set[target_column]

            model_name = ", ".join(selected_models)
            custom_params = get_model_parameters_ui(model_name, task_type)

            trained_model = train_model(X_train, y_train, task_type, model_name, custom_params)
            fit_status = evaluate_model(trained_model, X_train, y_train, X_val, y_val, task_type)

            if fit_status in ["Overfitting", "Underfitting"]:
                st.warning(f"⚠️ Detected {fit_status} — Auto-adjusting...")
                trained_model = train_model(X_train, y_train, task_type, model_name, fit_status=fit_status)
                evaluate_model(trained_model, X_train, y_train, X_val, y_val, task_type)

            if trained_model:
                model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"

                # Save model
                with open(model_filename, "wb") as f:
                    pickle.dump(trained_model, f)
                    save_model(trained_model, "model_name", username)

                # Open file in binary mode for downloading
                with open(model_filename, "rb") as f:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=f,
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("📌 Upload your dataset to begin.")
