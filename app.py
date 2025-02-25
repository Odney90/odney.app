import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from scipy.stats import poisson

# Fonction pour générer les prédictions avec les modèles
def generate_predictions(team1_data, team2_data):
    # Fusionner les données des deux équipes
    team1_values = list(team1_data.values())
    team2_values = list(team2_data.values())
    
    # Créer un DataFrame avec ces valeurs pour les prédictions
    X = pd.DataFrame([team1_values + team2_values], columns=list(team1_data.keys()) + list(team2_data.keys()))
    
    # Exemple de cible dummy, il faut la remplacer par la vraie cible
    y = np.array([1])  # Cible fictive pour l'exemple, à ajuster en fonction de la logique de ton modèle
    
    # Modèle de régression logistique
    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)
    
    # Modèle Random Forest
    rf = RandomForestClassifier()
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    
    # Modèle XGBoost
    xgboost_model = xgb.XGBClassifier(use_label_encoder=False)
    xgboost_model.fit(X, y)
    xgboost_prediction = xgboost_model.predict(X)
    
    # Calcul des probabilités Poisson pour les buts marqués par chaque équipe
    poisson_team1_prob = poisson.pmf(2, team1_data['⚽ Attaque'])
    poisson_team2_prob = poisson.pmf(2, team2_data['⚽ Attaque'])
    
    return poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf.mean(), xgboost_prediction

# Données fictives pour les équipes 1 et 2
team1_data = {
    '🧑‍💼 Nom de l\'équipe': 'Équipe 1',
    '⚽ Attaque': 1.2,
    '🛡️ Défense': 0.8,
    '🔥 Forme récente': 0.7,
    '💔 Blessures': 0.3,
    '💪 Motivation': 0.9,
    '🔄 Tactique': 0.75,
    '📊 Historique face à face': 0.6,
    '⚽xG': 1.4,
    '🔝 Nombre de corners': 5
}

team2_data = {
    '🧑‍💼 Nom de l\'équipe': 'Équipe 2',
    '⚽ Attaque': 0.8,
    '🛡️ Défense': 1.0,
    '🔥 Forme récente': 0.6,
    '💔 Blessures': 0.4,
    '💪 Motivation': 0.7,
    '🔄 Tactique': 0.65,
    '📊 Historique face à face': 0.5,
    '⚽xG': 1.1,
    '🔝 Nombre de corners': 6
}

# Prédictions
poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf_mean, xgboost_prediction = generate_predictions(team1_data, team2_data)

# Affichage des résultats
print(f"⚽ Probabilité Poisson de l'Équipe 1 : {poisson_team1_prob}")
print(f"⚽ Probabilité Poisson de l'Équipe 2 : {poisson_team2_prob}")
print(f"📊 Prédiction de la régression logistique : {logreg_prediction}")
print(f"📊 Moyenne des scores de validation croisée (Random Forest) : {cv_scores_rf_mean:.2f}")
print(f"📊 Prédiction de XGBoost : {xgboost_prediction}")
