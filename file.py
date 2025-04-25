from flask import Flask, request, render_template, jsonify
from models.model_loader import load_models
import pandas as pd
import os

app = Flask(__name__)

# Load car makes and models mapping
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

# Load pipelines
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
def home():
    return render_template('index.html')

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
        required_fields = ['year', 'mileage', 'engine_power', 'condition', 'make', 'model']
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
                'modele': request.form['model']
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
        
        # Prétraiter les données d'entrée
        processed_data = preprocess_input(form_data, encoders)
        
        # Obtenir les prédictions de tous les modèles
        linear_prediction = linear_pipeline.predict(processed_data)[0]
        lasso_prediction = lasso_pipeline.predict(processed_data)[0]
        xgboost_prediction = xgboost_pipeline.predict(processed_data)[0] 
        
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
