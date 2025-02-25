import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import poisson
import xgboost as xgb

# Fonction pour calculer la force d'attaque et la force de défense
def calculate_forces(data):
    attack_strength = data['buts_marques'] / data['matchs_joues']
    defense_strength = data['buts_subis'] / data['matchs_joues']
    return attack_strength, defense_strength

# Fonction pour générer les prédictions avec les modèles
def generate_predictions(team1_data, team2_data):
    # Variables pour l'équipe 1 et équipe 2
    X = pd.DataFrame([team1_data + team2_data], columns=team1_data.keys() + team2_data.keys())
    y = np.array([1])  # Dummy label, à ajuster selon la cible de Double Chance

    # Modèle de régression logistique
    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)

    # Modèle Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    cv_scores_rf = cross_val_score(rf, X, y, cv=10)
    rf.fit(X, y)
    rf_prediction = rf.predict(X)

    # Modèle XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=10)
    xgb_model.fit(X, y)
    xgb_prediction = xgb_model.predict(X)

    # Calcul des probabilités Poisson pour les buts
    poisson_team1_prob = poisson.pmf(2, team1_data['buts_marques'])  # Exemple avec 2 buts
    poisson_team2_prob = poisson.pmf(2, team2_data['buts_marques'])

    return poisson_team1_prob, poisson_team2_prob, logreg_prediction, rf_prediction, xgb_prediction, cv_scores_rf.mean(), cv_scores_xgb.mean()

# Définition des variables de l'équipe 1 et équipe 2 avec emojis
team1_data = {
    '🧑‍💼 Nom de l\'équipe': 'Équipe 1',
    '⚽ Attaque': 0.8,
    '🛡️ Défense': 0.6,
    '🔥 Forme récente': 0.7,
    '💔 Blessures': 0.3,
    '💪 Motivation': 0.9,
    '🔄 Tactique': 0.75,
    '📊 Historique face à face': 0.5,
    '⚽xG': 1.4,
    '🔝 Nombre de corners': 5,
    '⚽ Buts à domicile': 1.5,
    '⚽ Buts à l\'extérieur': 1.2
}

team2_data = {
    '🧑‍💼 Nom de l\'équipe': 'Équipe 2',
    '⚽ Attaque': 0.6,
    '🛡️ Défense': 0.8,
    '🔥 Forme récente': 0.7,
    '💔 Blessures': 0.4,
    '💪 Motivation': 0.8,
    '🔄 Tactique': 0.7,
    '📊 Historique face à face': 0.6,
    '⚽xG': 1.2,
    '🔝 Nombre de corners': 6,
    '⚽ Buts à domicile': 1.3,
    '⚽ Buts à l\'extérieur': 1.0
}

# Calcul des forces
attack_strength_1, defense_strength_1 = calculate_forces(team1_data)
attack_strength_2, defense_strength_2 = calculate_forces(team2_data)

# Affichage des résultats avec emojis
st.title("Analyse du Match ⚽")
st.write("### Team 1 vs Team 2")

# Affichage des données des équipes
st.write("### Variables pour Équipe 1:")
for key, value in team1_data.items():
    st.write(f"{key}: {value}")

st.write("### Variables pour Équipe 2:")
for key, value in team2_data.items():
    st.write(f"{key}: {value}")

# Prédictions
poisson_team1_prob, poisson_team2_prob, logreg_prediction, rf_prediction, xgb_prediction, cv_scores_rf_mean, cv_scores_xgb_mean = generate_predictions(team1_data, team2_data)

# Affichage des prédictions
st.write(f"⚽ Probabilité Poisson de l'Équipe 1 : {poisson_team1_prob:.2f}")
st.write(f"⚽ Probabilité Poisson de l'Équipe 2 : {poisson_team2_prob:.2f}")
st.write(f"📊 Prédiction de la régression logistique : {'Victoire Équipe 1' if logreg_prediction[0] == 1 else 'Match Nul ou Victoire Équipe 2'}")
st.write(f"📊 Prédiction de Random Forest : {'Victoire Équipe 1' if rf_prediction[0] == 1 else 'Match Nul ou Victoire Équipe 2'}")
st.write(f"📊 Prédiction de XGBoost : {'Victoire Équipe 1' if xgb_prediction[0] == 1 else 'Match Nul ou Victoire Équipe 2'}")
st.write(f"📊 Moyenne des scores de validation croisée (Random Forest) : {cv_scores_rf_mean:.2f}")
st.write(f"📊 Moyenne des scores de validation croisée (XGBoost) : {cv_scores_xgb_mean:.2f}")

# Option de téléchargement des résultats
st.download_button(
    label="Télécharger les prédictions en format DOC",
    data="Les données et prédictions ici",  # Remplacer par les données formatées en DOCX
    file_name="predictions.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
