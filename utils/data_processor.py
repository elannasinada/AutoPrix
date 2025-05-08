import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_input(data, encoders, model_type):
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Define categorical and numerical columns
    categorical_cols = ['etat', 'marque', 'modele', 'boite', 'carburant', 'premiere_main']
    numeric_cols = ['annee', 'kilometrage', 'puissance_fiscale', 'nombre_portres']
    
    # Ensure all columns are present and in the correct type
    for col in categorical_cols:
        if col not in df.columns or pd.isna(df[col].iloc[0]):
            df[col] = 'Unknown'
        df[col] = df[col].astype(str)
    
    for col in numeric_cols:
        if col not in df.columns or pd.isna(df[col].iloc[0]):
            df[col] = 0
        df[col] = df[col].astype(float)
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    
    # Combine with numerical columns
    df_processed = pd.concat([df[numeric_cols], df_encoded], axis=1)
    
    # Scale numerical columns
    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    # Align with expected columns from training data
    expected_cols = encoders.get(f'{model_type}_columns', [])
    for col in expected_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[expected_cols]
    
    return df_processed