import streamlit as st
import pandas as pd
import csv

from utils.loader import load_data
from utils.preprocess import preprocess_data
from utils.visual import show_plots
from utils.trainer import train_model

st.set_page_config(page_title="ML Fundamental Tools", layout="wide")

st.title("🔧🤖 Machine Learning Fundamental Tools")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    # 🧠 Auto-detect delimiter (, or ;) sebelum load
    try:
        content = uploaded_file.read().decode("utf-8")
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(content[:1024]).delimiter
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=delimiter)
        st.info(f"📌 Detected delimiter: `{delimiter}`")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

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
