import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from fpdf import FPDF
import tempfile
import uuid
import plotly.io as pio  # Required for saving plotly figures


def generate_visual_report(df_cleaned):
    st.subheader("📊 Smart Data Visualizations")

    numeric_cols = df_cleaned.select_dtypes(include='number').columns.tolist()
    cat_cols = df_cleaned.select_dtypes(include='object').columns.tolist()
    bool_cols = df_cleaned.select_dtypes(include='bool').columns.tolist()
    datetime_cols = df_cleaned.select_dtypes(include=['datetime64']).columns.tolist()

    view_mode = st.radio("🔀 Select View Mode:", ["Interactive (Plotly)", "Static (Seaborn/Matplotlib)"])

    image_paths = []

    def save_fig(fig, title):
        path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.png")
        fig.write_image(path, format="png")
        image_paths.append((title, path))

    if view_mode == "Interactive (Plotly)":
        if len(numeric_cols) >= 2:
            st.write("### 🔥 Correlation Heatmap")
            corr = df_cleaned[numeric_cols].corr().values
            labels = df_cleaned[numeric_cols].columns.tolist()
            fig = ff.create_annotated_heatmap(z=corr, x=labels, y=labels, colorscale='Viridis')
            st.plotly_chart(fig, use_container_width=True, key=str(hash("corr_heatmap")))
            save_fig(fig, "Correlation Heatmap")

        for col in numeric_cols:
            st.write(f"### 📉 Histogram & KDE of {col}")
            values = df_cleaned[col].dropna()
            if values.nunique() > 1:
                try:
                    fig = ff.create_distplot([values], group_labels=[col], show_hist=True, show_rug=False)
                    st.plotly_chart(fig, use_container_width=True, key=str(hash(f"dist_{col}")))
                    save_fig(fig, f"Histogram & KDE of {col}")
                except Exception as e:
                    st.warning(f"KDE failed for {col}: {e}")
            else:
                st.info(f"Skipping KDE for {col} — not enough variation.")

            st.write(f"### 📦 Box & 🎻 Violin Plot of {col}")
            fig_box = px.box(df_cleaned, y=col, title=f"Box Plot of {col}")
            fig_violin = px.violin(df_cleaned, y=col, box=True, title=f"Violin Plot of {col}")
            st.plotly_chart(fig_box, use_container_width=True, key=str(hash(f"box_{col}")))
            st.plotly_chart(fig_violin, use_container_width=True, key=str(hash(f"violin_{col}")))
            save_fig(fig_box, f"Box Plot of {col}")
            save_fig(fig_violin, f"Violin Plot of {col}")

            st.write(f"### 📈 Line Plot of {col}")
            fig_line = px.line(df_cleaned[col].dropna().reset_index(), y=col, title=f"Line Plot of {col}")
            st.plotly_chart(fig_line, use_container_width=True, key=str(hash(f"line_{col}")))
            save_fig(fig_line, f"Line Plot of {col}")

        for col in cat_cols:
            st.write(f"### 🧱 Bar Chart of {col}")
            bar_data = df_cleaned[col].value_counts().reset_index()
            bar_data.columns = [col, 'count']
            fig = px.bar(bar_data, x=col, y='count', title=f"Bar Chart of {col}")
            st.plotly_chart(fig, use_container_width=True, key=str(hash(f"bar_{col}")))
            save_fig(fig, f"Bar Chart of {col}")

            if df_cleaned[col].nunique() <= 10:
                st.write(f"### 🥧 Pie Chart of {col}")
                fig = px.pie(df_cleaned, names=col, title=f"Pie Chart of {col}")
                st.plotly_chart(fig, use_container_width=True, key=str(hash(f"pie_{col}")))
                save_fig(fig, f"Pie Chart of {col}")

        for col in bool_cols:
            st.write(f"### ✅ Countplot of {col}")
            fig = px.bar(df_cleaned[col].value_counts().reset_index(), x='index', y=col, title=f"Count Plot of {col}")
            st.plotly_chart(fig, use_container_width=True, key=str(hash(f"bool_{col}")))
            save_fig(fig, f"Count Plot of {col}")

        if len(numeric_cols) > 1 and df_cleaned.shape[0] <= 500:
            st.write("### 🔄 Scatter Matrix (Pairplot)")
            fig = px.scatter_matrix(df_cleaned[numeric_cols].dropna())
            st.plotly_chart(fig, use_container_width=True, key=str(hash("pairplot")))
            save_fig(fig, "Scatter Matrix")

        if datetime_cols:
            for col in datetime_cols:
                for num_col in numeric_cols:
                    st.write(f"### ⏳ Time Series: {num_col} over {col}")
                    fig = px.line(df_cleaned.sort_values(col), x=col, y=num_col, title=f"{num_col} over {col}")
                    st.plotly_chart(fig, use_container_width=True, key=str(hash(f"time_{col}_{num_col}")))
                    save_fig(fig, f"Time Series: {num_col} over {col}")

        st.write("### ❓ Missing Data Heatmap")
        fig = go.Figure(data=go.Heatmap(z=df_cleaned.isnull().astype(int).T.values,
                                        x=df_cleaned.index,
                                        y=df_cleaned.columns,
                                        colorscale='Viridis'))
        fig.update_layout(title="Missing Data Heatmap", xaxis_title="Index", yaxis_title="Columns")
        st.plotly_chart(fig, use_container_width=True, key=str(hash("missing_heatmap")))
        save_fig(fig, "Missing Data Heatmap")

        if 'target' in df_cleaned.columns:
            st.write("### 🎯 Correlation with Target")
            target_corr = df_cleaned.corr()['target'].drop('target').sort_values()
            fig = px.bar(target_corr, orientation='h', title="Correlation with Target")
            st.plotly_chart(fig, use_container_width=True, key=str(hash("target_corr")))
            save_fig(fig, "Correlation with Target")

    if st.button("📥 Export Visual Report to PDF"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Smart Data Visual Report", ln=True, align='C')
        pdf.ln(10)

        for title, img_path in image_paths:
            pdf.cell(0, 10, title, ln=True)
            pdf.image(img_path, w=180)
            pdf.ln(5)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, "rb") as file:
                st.download_button(
                    label="📥 Download Visual Report as PDF",
                    data=file.read(),
                    file_name="visual_report.pdf",
                    mime="application/pdf",
                    key="download_visual_pdf"
                )
            os.unlink(tmp.name)

        for _, path in image_paths:
            os.remove(path)
