import os
import pickle
import subprocess
import numpy as np
import pandas as pd
import joblib

def load_models():
    """
    Charger les modèles de régression linéaire, lasso et xgboost pré-entraînés
    ainsi que leurs encodeurs respectifs pour les caractéristiques catégoriques
    
    Retourne:
        tuple: (linear_model, lasso_model, xgboost_model, encoders)
    """
    try:
        # Vérifier les chemins d'accès
        print(f"Répertoire de travail actuel: {os.getcwd()}")
        
        # Vérifier si le fichier CSV existe
        csv_path = "data/avito_cars_clean.csv"
        if not os.path.exists(csv_path):
            csv_path = "avito_cars_clean.csv"  # Try alternate path
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Le fichier de données n'a pas été trouvé")
        
        # Charger le dataset
        print(f"Chargement du dataset depuis {csv_path}...")
        df = pd.read_csv(csv_path)

        # Charger les encodeurs à partir des données du fichier CSV
        print("Création des encodeurs à partir des données...")
        
        encoders = {
            'make': {make: idx for idx, make in enumerate(df['marque'].dropna().unique())},
            'model': {model: idx for idx, model in enumerate(df['modele'].dropna().unique())},
            'condition': {condition: idx for idx, condition in enumerate(df['etat'].dropna().unique())},
            'boite': {boite: idx for idx, boite in enumerate(df['boite'].dropna().unique())},
            'carburant': {carburant: idx for idx, carburant in enumerate(df['carburant'].dropna().unique())}
        }
        
        # Get the expected columns for one-hot encoding
        sample_df = df[['annee', 'kilometrage', 'puissance_fiscale', 'etat', 'marque', 'modele', 'boite', 'carburant']].head(1)
        sample_encoded = pd.get_dummies(sample_df, drop_first=True)
        encoders['expected_columns'] = sample_encoded.columns.tolist()
        
        features_df = df[['annee', 'kilometrage', 'puissance_fiscale', 'etat', 'marque', 'modele', 'boite', 'carburant']]
        encoded_df = pd.get_dummies(features_df, drop_first=True)
        encoders['column_order'] = encoded_df.columns.tolist()
        
        # Utiliser joblib pour charger les modèles
        pkl_dir = "pkl-files"
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir, exist_ok=True)
            raise FileNotFoundError(f"Le répertoire {pkl_dir} n'existait pas et a été créé. Placez-y vos modèles.")

        # Chargement pour Linear Regression
        linear_model_path = os.path.join(pkl_dir, 'linear_regression_model.pkl')
        print(f"Chargement du modèle linéaire depuis {linear_model_path}...")
        linear_model = joblib.load(linear_model_path)
        linear_columns_path = os.path.join(pkl_dir, 'linear_regression_columns.pkl')
        if os.path.exists(linear_columns_path):
            encoders['linear_columns'] = joblib.load(linear_columns_path)
        else:
            raise FileNotFoundError(f"Le fichier {linear_columns_path} est manquant.")

        # Chargement pour Lasso
        lasso_model_path = os.path.join(pkl_dir, 'lasso_model.pkl')
        print(f"Chargement du modèle Lasso depuis {lasso_model_path}...")
        lasso_model = joblib.load(lasso_model_path)
        lasso_columns_path = os.path.join(pkl_dir, 'lasso_columns.pkl')
        if os.path.exists(lasso_columns_path):
            encoders['lasso_columns'] = joblib.load(lasso_columns_path)
        else:
            raise FileNotFoundError(f"Le fichier {lasso_columns_path} est manquant.")

        # Chargement pour XGBoost
        xgboost_model_path = os.path.join(pkl_dir, 'xgboost_model.pkl')
        print(f"Chargement du modèle XGBoost depuis {xgboost_model_path}...")
        xgboost_model = joblib.load(xgboost_model_path)
        xgboost_columns_path = os.path.join(pkl_dir, 'xgboost_columns.pkl')
        if os.path.exists(xgboost_columns_path):
            encoders['xgboost_columns'] = joblib.load(xgboost_columns_path)
        else:
            raise FileNotFoundError(f"Le fichier {xgboost_columns_path} est manquant.")
        
        print("Modèles chargés avec succès")
        return linear_model, lasso_model, xgboost_model, encoders
    
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {str(e)}")
        raise
