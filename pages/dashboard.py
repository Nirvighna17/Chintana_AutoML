import streamlit as st
import os

# --- LOGIN CHECK ---
# --- LOGIN CHECK ---
if not st.session_state.get("logged_in", False):
    st.error("🚫 Access Denied: Please login first.")
    st.page_link("pages/login_signup.py", label="Go to Login")
    st.stop()

st.set_page_config(page_title="User Dashboard", layout="wide")
st.title("📊 Your Model Dashboard")

username = st.session_state.get("username")
model_folder = f"models/{username}"

if not os.path.exists(model_folder):
    st.info("You haven't trained any models yet.")
else:
    model_files = os.listdir(model_folder)
    if model_files:
        for model_file in model_files:
            st.markdown(f"✅ **{model_file}**")
            with open(os.path.join(model_folder, model_file), "rb") as f:
                st.download_button("⬇️ Download", f, file_name=model_file)
    else:
        st.info("No models found in your dashboard.")
