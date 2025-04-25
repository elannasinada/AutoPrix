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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Valider que les données du formulaire sont présentes
        required_fields = ['year', 'mileage', 'engine_power', 'condition', 'make', 'model', 'gearbox', 'fuel']
        for field in required_fields:
            if field not in request.form or not request.form[field]:
                return jsonify({'success': False, 'error': f"Champ requis manquant: {field}"})
        
        # Obtenir et convertir les données du formulaire
        try:
            form_data = {
                'annee': int(request.form['year']),
                'kilometrage': float(request.form['mileage']),
                'puissance_fiscale': int(request.form['engine_power']),
                'etat': request.form['condition'],
                'marque': request.form['make'],
                'modele': request.form['model'],
                'boite': request.form['gearbox'],
                'carburant': request.form['fuel']
            }
        except ValueError as e:
            return jsonify({'success': False, 'error': f"Format de valeur invalide: {str(e)}"})
        
        # Validation de base
        current_year = 2025
        if not (1970 <= form_data['annee'] <= current_year):
            return jsonify({'success': False, 'error': f"L'année doit être comprise entre 1970 et {current_year}"})
        if form_data['kilometrage'] < 0:
            return jsonify({'success': False, 'error': "Le kilométrage ne peut pas être négatif"})
        if form_data['puissance_fiscale'] <= 0:
            return jsonify({'success': False, 'error': "La puissance fiscale doit être positive"})
        
        # Prétraiter les données d'entrée pour chaque modèle
        processed_data_linear = preprocess_input(form_data, encoders, model_type='linear')
        processed_data_lasso = preprocess_input(form_data, encoders, model_type='lasso')
        processed_data_xgboost = preprocess_input(form_data, encoders, model_type='xgboost')
        
        # Obtenir les prédictions de tous les modèles
        linear_prediction = linear_pipeline.predict(processed_data_linear)[0]
        lasso_prediction = lasso_pipeline.predict(processed_data_lasso)[0]
        xgboost_prediction = xgboost_pipeline.predict(processed_data_xgboost)[0]
        
        # S'assurer que les prédictions sont positives
        linear_prediction = max(0, linear_prediction)
        lasso_prediction = max(0, lasso_prediction)
        xgboost_prediction = max(0, xgboost_prediction)
        
        # Arrondir les prédictions au millier le plus proche
        linear_prediction = round(linear_prediction / 1000) * 1000
        lasso_prediction = round(lasso_prediction / 1000) * 1000
        xgboost_prediction = round(xgboost_prediction / 1000) * 1000
        
        # Calculer la prédiction moyenne
        average_prediction = (linear_prediction + lasso_prediction + xgboost_prediction) / 3
        average_prediction = round(average_prediction / 1000) * 1000  # Arrondir également la moyenne
        
        result = {
            'linear_prediction': f"{linear_prediction:,.0f} DH",
            'lasso_prediction': f"{lasso_prediction:,.0f} DH",
            'xgboost_prediction': f"{xgboost_prediction:,.0f} DH",
            'average_prediction': f"{average_prediction:,.0f} DH",
            'success': True
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({'success': False, 'error': f"Erreur de prédiction: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)  # Mettre debug=False en production
