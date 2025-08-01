import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def show_plots(df):
    st.write("📌 Korelasi Fitur")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("📌 Distribusi Target")
    if df.select_dtypes(include='number').shape[1] > 0:
        target = df.columns[-1]
        fig2, ax2 = plt.subplots()
        sns.histplot(df[target], kde=True, ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Tidak ditemukan kolom numerik untuk ditampilkan.")
