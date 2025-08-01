import streamlit as st
import pandas as pd

from utils.loader import load_data
from utils.preprocess import preprocess_data
from utils.visual import show_plots
from utils.trainer import train_model

st.set_page_config(page_title="ML Fundamental Tools", layout="wide")

st.title("🤖 Machine Learning Fundamental Tools")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Statistik Data")
    st.write(df.describe())

    st.subheader("🎨 Visualisasi")
    try:
        show_plots(df)
    except Exception as e:
        st.error(f"Gagal menampilkan visualisasi: {e}")

    target_column = st.selectbox("🎯 Pilih kolom target (yang ingin diprediksi)", df.columns)

    model_name = st.selectbox("🧠 Pilih algoritma ML", ["Linear Regression", "KNN", "Decision Tree"])

    if st.button("🚀 Jalankan Model"):
        try:
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
            result = train_model(model_name, X_train, X_test, y_train, y_test)
            st.success("✅ Model berhasil dilatih!")
            st.json(result)
        except Exception as e:
            st.error(f"Gagal melatih model: {e}")
else:
    st.info("⬆️ Silakan upload file CSV untuk memulai.")
