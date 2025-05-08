import os
import joblib

def load_models():
    """
    Charger les modèles de régression linéaire, lasso et xgboost pré-entraînés
    ainsi que leurs encodeurs respectifs pour les caractéristiques catégoriques
    
    Retourne:
        tuple: (linear_regression_model, lasso_model, xgboost_model, encoders)
    """
    try:
        # Vérifier les chemins d'accès
        print(f"Répertoire de travail actuel: {os.getcwd()}")
        
        # Définir le chemin absolu du répertoire pkl-files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        pkl_dir = os.path.join(project_root, "pkl-files")
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir, exist_ok=True)
            raise FileNotFoundError(f"Le répertoire {pkl_dir} n'existait pas et a été créé. Placez-y vos modèles.")

        # Chargement pour Linear Regression
        linear_model_path = os.path.join(pkl_dir, 'linear_regression_model.pkl')
        print(f"Chargement du modèle linéaire depuis {linear_model_path}...")
        linear_model = joblib.load(linear_model_path)
        linear_columns_path = os.path.join(pkl_dir, 'linear_regression_columns.pkl')
        if os.path.exists(linear_columns_path):
            linear_columns = joblib.load(linear_columns_path)
        else:
            raise FileNotFoundError(f"Le fichier {linear_columns_path} est manquant.")

        # Chargement pour Lasso
        lasso_model_path = os.path.join(pkl_dir, 'lasso_model.pkl')
        print(f"Chargement du modèle Lasso depuis {lasso_model_path}...")
        lasso_model = joblib.load(lasso_model_path)
        lasso_columns_path = os.path.join(pkl_dir, 'lasso_columns.pkl')
        if os.path.exists(lasso_columns_path):
            lasso_columns = joblib.load(lasso_columns_path)
        else:
            raise FileNotFoundError(f"Le fichier {lasso_columns_path} est manquant.")

        # Chargement pour XGBoost
        xgboost_model_path = os.path.join(pkl_dir, 'xgboost_model.pkl')
        print(f"Chargement du modèle XGBoost depuis {xgboost_model_path}...")
        xgboost_model = joblib.load(xgboost_model_path)
        xgboost_columns_path = os.path.join(pkl_dir, 'xgboost_columns.pkl')
        if os.path.exists(xgboost_columns_path):
            xgboost_columns = joblib.load(xgboost_columns_path)
        else:
            raise FileNotFoundError(f"Le fichier {xgboost_columns_path} est manquant.")

        # Chargement des encodeurs pour les variables catégoriques
        # Note: This file is missing; temporarily set to empty dict until retrained
        label_encoders_path = os.path.join(pkl_dir, 'xgboost_label_encoders.pkl')
        print(f"Chargement des encodeurs depuis {label_encoders_path}...")
        if os.path.exists(label_encoders_path):
            label_encoders = joblib.load(label_encoders_path)
        else:
            print(f"Avertissement: Le fichier {label_encoders_path} est manquant. Utilisation d'un dictionnaire vide.")
            label_encoders = {}

        # Créer le dictionnaire des encodeurs
        encoders = {
            'linear_columns': linear_columns,
            'lasso_columns': lasso_columns,
            'xgboost_columns': xgboost_columns,
            'label_encoders': label_encoders  # Dictionary of LabelEncoder objects
        }
        
        print("Modèles et encodeurs chargés avec succès")
        return linear_model, lasso_model, xgboost_model, encoders
    
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {str(e)}")
        raise