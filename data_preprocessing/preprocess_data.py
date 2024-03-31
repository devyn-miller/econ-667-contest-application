import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    df = pd.read_csv('data/collected_data.csv')
    # Example preprocessing steps
    df.fillna(0, inplace=True)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_scaled.to_csv('data/preprocessed_data.csv', index=False)

if __name__ == "__main__":
    preprocess_data()