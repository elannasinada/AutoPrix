# Prédicteur de Prix de Voitures d'Occasion

Une application web Flask qui utilise des modèles d'apprentissage automatique (Régression Linéaire et Lasso) pour prédire le prix des voitures d'occasion en fonction de diverses caractéristiques.


## Fonctionnalités

- Interface utilisateur épurée
- Formulaire d'entrée en plusieurs étapes pour les détails de la voiture
- Support pour les modèles de régression linéaire et lasso
- Prédiction de prix en temps réel
- Design réactif pour tous les appareils

## Installation

1. Clonez ce dépôt :
```
git clone <repository-url>
cd used-car-price-predictor
```

2. Créez un environnement virtuel et activez-le :
```
python -m venv venv
source venv/Scripts/activate
```

3. Installez les dépendances:
```
pip install -r requirements.txt
```

4. Lancez l'application:
```
python app.py
```

5. Ouvrez votre navigateur et allez à `http://localhost:5000`

## Structure du Projet

- `app.py`: Application Flask principale
- `models/`: Contient le code de chargement et de gestion des modèles
- `utils/`: Fonctions d'assistance pour le traitement des données
- `static/`: Fichiers CSS, JavaScript et autres fichiers statiques
- `templates/`: Modèles HTML pour l'interface web
- `data/`: Jeu de données des voitures usagées
- `pkl-files/`:  Modèles entraînés et transformateurs serialisés

## Technologies Utilisées

- Flask: Framework web
- Scikit-learn: Modèles d'apprentissage automatique
- NumPy & Pandas: Traitement des données
- HTML/CSS/JavaScript: Frontend

## Informations sur le Modèle

Cette application utilise trois modèles d'apprentissage automatique :

1. Régression Linéaire : Un modèle linéaire de base pour la prédiction des prix
2. Régression Lasso : Un modèle linéaire régularisé qui aide à la sélection des caractéristiques
3. XGBoost : Un modèle basé sur les arbres de décision optimisé pour des performances élevées

Les modèles ont été entraînés sur un ensemble de données d'annonces de voitures d'occasion au Maroc, avec des caractéristiques incluant :

- Année
- Boîte
- Carburant
- Kilométrage
- Marque
- Modèle
- Nombre des portres
- Premiere main
- Puissance du moteur
- État

## License

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.