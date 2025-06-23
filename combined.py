# Install libraries if needed:
# pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 🚀 Title
# ===============================
st.set_page_config(page_title="Test Case Prioritization Dashboard", page_icon="🧠", layout="wide")
st.title(" Test Case Prioritization Dashboard")

# Sidebar upload
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File uploaded successfully!")

    # Preserve original for comparison
    preserved_columns = df.copy()

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Dataset Information")
    st.sidebar.write(f"Rows: {df.shape[0]}")
    st.sidebar.write(f"Columns: {df.shape[1]}")

    # ===============================
    # 🧠 Dynamic Preprocessing
    # ===============================

    # Step 1: Try convert
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Step 2: Refresh column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Step 3: Fill missing
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    if len(categorical_cols) > 0:
        mode_values = df[categorical_cols].mode()
        if not mode_values.empty:
            df[categorical_cols] = df[categorical_cols].fillna(mode_values.iloc[0])

    # Step 4: Scaling
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Step 5: Encode categoricals
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # ===============================
    # ⚙️ Feature Engineering
    # ===============================
    if 'Time' in df.columns and 'Cost' in df.columns:
        df['Time_to_Cost'] = df['Time'] / (df['Cost'] + 1e-9)

    if 'Cost' in df.columns and 'Complexity' in df.columns:
        df['Cost_to_Complexity'] = df['Cost'] / (df['Complexity'] + 1e-9)
        df['Cost_to_Complexity'] = np.clip(df['Cost_to_Complexity'], 0, 1000)

    if 'Complexity' in df.columns and 'Time' in df.columns:
        df['Complexity_x_Time'] = df['Complexity'] * df['Time']

    if 'Complexity' in df.columns:
        df['Zero_Complexity'] = (df['Complexity'] == 0).astype(int)

    # ===============================
    # 🔥 Tabs for Workflow
    # ===============================
    tab1, tab2, tab3 = st.tabs(["🔍 Data Preview", "⚙️ Model Training", "📥 Download"])

    with tab1:
        st.subheader("🔍 Raw and Preprocessed Data")

        with st.expander("See Raw Uploaded Data"):
            st.dataframe(preserved_columns, use_container_width=True)

        with st.expander("See Preprocessed Data"):
            st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("⚙️ Model Training and Prediction")

        if 'Priority' not in df.columns:
            st.error("❗ 'Priority' column missing. Cannot train model.")
            st.stop()

        non_features = ['Priority']
        extra_cols = ['FP', 'R_Priority', 'Weights']
        non_features += [col for col in extra_cols if col in df.columns]

        X = df.drop(columns=[col for col in non_features if col in df.columns])
        y = df['Priority']
        X = X.select_dtypes(include=[np.number])

        # ✅ Safe Patch: Encode y if needed
        if len(X) > 100:
            if y.dtype == 'object' or y.dtype.name == 'category':
                y_encoder = LabelEncoder()
                y = y_encoder.fit_transform(y)

            smoteenn = SMOTEENN(random_state=42)
            X_balanced, y_balanced = smoteenn.fit_resample(X, y)
        else:
            X_balanced, y_balanced = X, y

        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_balanced)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_pca, y_balanced)

        X_full = pca.transform(X)
        predictions = model.predict(X_full)

        df['Predicted_Priority'] = predictions

        # Map numeric prediction to label if necessary
        df['Predicted_Priority_Label'] = df['Predicted_Priority'].map({0: 'Low', 1: 'Medium', 2: 'High'})

        # ===============================
        # 📈 Metrics & Visualization
        # ===============================
        col1, col2, col3 = st.columns(3)
        with col1:
            high_count = int((df['Predicted_Priority_Label'] == 'High').sum())
            st.metric(label="🔥 High Priority", value=high_count)
        with col2:
            medium_count = int((df['Predicted_Priority_Label'] == 'Medium').sum())
            st.metric(label="🟡 Medium Priority", value=medium_count)
        with col3:
            low_count = int((df['Predicted_Priority_Label'] == 'Low').sum())
            st.metric(label="🟢 Low Priority", value=low_count)

        st.subheader("📊 Prediction Distribution")
        fig = px.histogram(df, x="Predicted_Priority_Label", color="Predicted_Priority_Label",
                           color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Final Prioritized Data")
        st.dataframe(df, use_container_width=True)

    with tab3:
        st.subheader("📥 Download Prioritized Dataset")

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="Prioritized_TestCases.csv",
            mime="text/csv"
        )

else:
    st.warning("👈 Please upload a file from the sidebar to begin.")

# Footer
st.markdown("---")
st.caption("🚀 Built by Kavin.")
