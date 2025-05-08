from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
import csv
import subprocess
from models.model_loader import load_models
from utils.data_processor import preprocess_input
import joblib

app = Flask(__name__)
encoders = {
    'linear_columns': joblib.load('pkl-files/linear_regression_columns.pkl')
}

# Charger les correspondances entre marques et modèles au démarrage de l'application
MAKE_MODEL_MAPPING = {}
try:
    df = pd.read_csv('data/avito_cars_clean.csv')
    # Create mapping from makes to models
    for marque in df['marque'].unique():
        models = df[df['marque'] == marque]['modele'].unique().tolist()
        MAKE_MODEL_MAPPING[marque] = models
    print(f"Chargement réussi de {len(MAKE_MODEL_MAPPING)} marques de voitures depuis le dataset")
except Exception as e:
    print(f"Erreur lors du chargement des marques et modèles: {str(e)}")

# Charger les modèles de prédiction au démarrage de l'application
try:
    models = load_models()
    linear_pipeline = models[0]
    lasso_pipeline = models[1]
    xgboost_pipeline = models[2]
    encoders = models[3] if len(models) > 3 else None
    print("Tous les modèles ont été chargés avec succès")
except Exception as e:
    print(f"Erreur lors du chargement des modèles: {str(e)}")
    linear_pipeline, lasso_pipeline, xgboost_pipeline, encoders = None, None, None, None

@app.route('/')
def index():
    """Afficher la page principale avec le formulaire d'entrée"""
    if None in (linear_pipeline, lasso_pipeline, xgboost_pipeline, encoders):
        return render_template('error.html', message="Les modèles n'ont pas pu être chargés. Veuillez vérifier les logs du serveur.")
    
    # Obtenir toutes les marques pour le menu déroulant
    makes = list(MAKE_MODEL_MAPPING.keys())
    
    # Obtenir les options d'état
    try:
        if os.path.exists("avito_cars_clean.csv"):
            df = pd.read_csv('avito_cars_clean.csv')
            conditions = df['etat'].unique().tolist()
        else:
            conditions = ['Excellent', 'Très bon', 'Bon', 'Moyen']
    except:
        conditions = ['Excellent', 'Très bon', 'Bon', 'Moyen']
    
    return render_template('index.html', makes=makes, conditions=conditions)

@app.route('/car-models/<marque>')
def get_car_models(marque):
    """Return car models for a specific make"""
    if marque not in MAKE_MODEL_MAPPING:
        return jsonify(['Aucun modèle disponible pour cette marque'])
    return jsonify(MAKE_MODEL_MAPPING[marque])


# ... (previous imports and model loading remain unchanged)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate form data
        required_fields = ['year', 'mileage', 'engine_power', 'condition', 'make', 'model', 'gearbox', 'fuel', 'first_hand', 'num_doors']
        for field in required_fields:
            if field not in request.form or not request.form[field]:
                return jsonify({'success': False, 'error': f"Champ requis manquant: {field}"})
        
        # Convert form data
        try:
            form_data = {
                'annee': int(request.form['year']),
                'kilometrage': float(request.form['mileage']),
                'puissance_fiscale': int(request.form['engine_power']),
                'etat': request.form['condition'],
                'marque': request.form['make'],
                'modele': request.form['model'],
                'boite': request.form['gearbox'],
                'carburant': request.form['fuel'],
                'premiere_main': request.form['first_hand'],
                'nombre_portres': int(request.form['num_doors'])
            }
            # Debug: Log the form data
            print("[DEBUG] Form Data Received:", form_data)
        except ValueError as e:
            return jsonify({'success': False, 'error': f"Format de valeur invalide: {str(e)}"})
        
        # Basic validation
        current_year = 2025
        if not (1970 <= form_data['annee'] <= current_year):
            return jsonify({'success': False, 'error': f"L'année doit être comprise entre 1970 et {current_year}"})
        if form_data['kilometrage'] < 0:
            return jsonify({'success': False, 'error': "Le kilométrage ne peut pas être négatif"})
        if form_data['puissance_fiscale'] <= 0:
            return jsonify({'success': False, 'error': "La puissance fiscale doit être positive"})
        if form_data['nombre_portres'] not in [3, 4, 5]:
            return jsonify({'success': False, 'error': "Le nombre de portes doit être 3, 4 ou 5"})
        if form_data['premiere_main'] not in ['Oui', 'Non']:
            return jsonify({'success': False, 'error': "Première main doit être 'Oui' ou 'Non'"})
        
        # Preprocess data for each model
        processed_data_linear = preprocess_input(form_data, encoders, model_type='linear')
        processed_data_lasso = preprocess_input(form_data, encoders, model_type='lasso')
        processed_data_xgboost = preprocess_input(form_data, encoders, model_type='xgboost')

        # Debugging: print the processed data
        print(f"[DEBUG] Processed data for Linear Regression: {processed_data_linear}")
        print(f"[DEBUG] Processed data for Lasso: {processed_data_lasso}")
        print(f"[DEBUG] Processed data for XGBoost: {processed_data_xgboost}")

        # Get predictions
        linear_prediction = linear_pipeline.predict(processed_data_linear)[0]
        lasso_prediction = lasso_pipeline.predict(processed_data_lasso)[0]
        xgboost_prediction = xgboost_pipeline.predict(processed_data_xgboost)[0]

        # Debugging: print the predictions
        print(f"[DEBUG] Linear Prediction: {linear_prediction}")
        print(f"[DEBUG] Lasso Prediction: {lasso_prediction}")
        print(f"[DEBUG] XGBoost Prediction: {xgboost_prediction}")

        # Adjust predictions (assuming Linear and Lasso are in log scale)
        linear_prediction = np.exp(linear_prediction) if linear_prediction > 0 else 0
        lasso_prediction = np.exp(lasso_prediction) if lasso_prediction > 0 else 0
        xgboost_prediction = float(xgboost_prediction)

        # Ensure predictions are positive
        linear_prediction = max(0, linear_prediction)
        lasso_prediction = max(0, lasso_prediction)
        xgboost_prediction = max(0, xgboost_prediction)

        # Round predictions to the nearest thousand
        linear_prediction = round(linear_prediction / 1000) * 1000
        lasso_prediction = round(lasso_prediction / 1000) * 1000
        xgboost_prediction = round(xgboost_prediction / 1000) * 1000

        # Calculate average prediction
        average_prediction = (linear_prediction + lasso_prediction + xgboost_prediction) / 3
        average_prediction = round(average_prediction / 1000) * 1000

        # Format predictions as strings with thousands separator
        result = {
            'linear_prediction': f"{int(linear_prediction):,d} DH",
            'lasso_prediction': f"{int(lasso_prediction):,d} DH",
            'xgboost_prediction': f"{int(xgboost_prediction):,d} DH",
            'average_prediction': f"{int(average_prediction):,d} DH",
            'success': True
        }

        return jsonify(result)

    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({'success': False, 'error': f"Erreur de prédiction: {str(e)}"})

# ... (rest of the file remains unchanged)
if __name__ == '__main__':
    app.run(debug=True)  # Mettre debug=False en production
