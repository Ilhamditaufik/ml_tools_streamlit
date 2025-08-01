import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_column, mode='regression'):
    # Jika mode classification, ubah target numerik jadi kategori
    if mode == 'classification':
        df[target_column] = df[target_column].apply(
            lambda x: 'Low' if x <= 5 else ('Medium' if x == 6 else 'High')
        )

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encoding untuk fitur kategorikal
    X = pd.get_dummies(X)

    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
