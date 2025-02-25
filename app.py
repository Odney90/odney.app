import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import poisson

# Fonction pour calculer la force d'attaque et la force de défense
def calculate_forces(data):
    attack_strength = data['buts_marques'] / data['matchs_joues']
    defense_strength = data['buts_subis'] / data['matchs_joues']
    return attack_strength, defense_strength

# Fonction pour générer les prédictions avec les modèles
def generate_predictions(team1_data, team2_data):
    # Données de l'équipe 1 et équipe 2
    X = pd.DataFrame([team1_data + team2_data], columns=team1_data.keys() + team2_data.keys())
    y = np.array([1])  # Dummy label, à ajuster selon les critères de comparaison

    # Modèle Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_  # Poids attribués par Random Forest

    # Calcul des probabilités Poisson
    poisson_team1_prob = poisson.pmf(2, team1_data['buts_marques'])
    poisson_team2_prob = poisson.pmf(2, team2_data['buts_marques'])

    return poisson_team1_prob, poisson_team2_prob, feature_importances

# Données fictives des équipes 1 et 2 avec les nouvelles variables
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
    '🏠 Buts marqués à domicile': 15,
    '🏠 Buts encaissés à domicile': 10,
    '🌍 Buts marqués à l’extérieur': 18,
    '🌍 Buts encaissés à l’extérieur': 12,
    '🏠 Forme à domicile': 0.6,
    '🌍 Forme à l’extérieur': 0.7,
    '🏠 Taux de victoire à domicile': 0.55,
    '🌍 Taux de victoire à l’extérieur': 0.65,
    '🏠 Historique à domicile vs top 10': 0.4,
    '🌍 Historique à l’extérieur vs top 10': 0.6,
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
    '🏠 Buts marqués à domicile': 20,
    '🏠 Buts encaissés à domicile': 15,
    '🌍 Buts marqués à l’extérieur': 25,
    '🌍 Buts encaissés à l’extérieur': 18,
    '🏠 Forme à domicile': 0.7,
    '🌍 Forme à l’extérieur': 0.5,
    '🏠 Taux de victoire à domicile': 0.65,
    '🌍 Taux de victoire à l’extérieur': 0.60,
    '🏠 Historique à domicile vs top 10': 0.5,
    '🌍 Historique à l’extérieur vs top 10': 0.65,
}

# Calcul des forces
attack_strength_1, defense_strength_1 = calculate_forces(team1_data)
attack_strength_2, defense_strength_2 = calculate_forces(team2_data)

# Affichage des résultats avec emojis
st.title("Analyse du Match ⚽")
st.write("### Team 1 vs Team 2")

# Affichage des variables
st.write("### Variables pour Équipe 1:")
for key, value in team1_data.items():
    st.write(f"{key}: {value}")

st.write("### Variables pour Équipe 2:")
for key, value in team2_data.items():
    st.write(f"{key}: {value}")

# Prédictions
poisson_team1_prob, poisson_team2_prob, feature_importances = generate_predictions(team1_data, team2_data)

st.write(f"⚽ Probabilité Poisson de l'Équipe 1 : {poisson_team1_prob}")
st.write(f"⚽ Probabilité Poisson de l'Équipe 2 : {poisson_team2_prob}")

# Affichage de l'importance des variables par Random Forest
st.write("📊 Importance des variables par Random Forest :")
importance_df = pd.DataFrame({
    'Variable': list(team1_data.keys()) + list(team2_data.keys()),
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
st.write(importance_df)

# Option de téléchargement des résultats
st.download_button(
    label="Télécharger les prédictions en format DOC",
    data="Les données et prédictions ici",  # Remplace avec le format DOC généré
    file_name="predictions.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
