import pandas as pd

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)  # coba delimiter default (,)
        if df.shape[1] == 1:
            # jika hanya 1 kolom, kemungkinan pakai delimiter ;
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, delimiter=';')
        return df
    except Exception as e:
        raise ValueError(f"Gagal memuat file: {e}")
