import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_data(file_path):
    data = pd.read_excel(file_path)
    # Data cleaning and preprocessing steps
    # ...
    # Normalize data using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return normalized_data
