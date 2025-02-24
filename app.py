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
    # Exemple de calcul pour la force d'attaque et la défense
    attack_strength = data['buts_marques'] / data['matchs_joues']
    defense_strength = data['buts_subis'] / data['matchs_joues']
    return attack_strength, defense_strength

# Fonction pour générer les prédictions avec les modèles
def generate_predictions(team1_data, team2_data):
    # Données de l'équipe 1 et équipe 2
    X = pd.DataFrame([team1_data + team2_data], columns=team1_data.keys() + team2_data.keys())
    y = np.array([1])  # Dummy label, à ajuster selon les critères de comparaison

    # Modèle de régression logistique
    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)

    # Modèle Random Forest
    rf = RandomForestClassifier()
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)

    # Calcul des probabilités Poisson
    poisson_team1_prob = poisson.pmf(2, team1_data['buts_marques'])
    poisson_team2_prob = poisson.pmf(2, team2_data['buts_marques'])

    return poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf.mean()

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
    '🔝 Nombre de corners': 5
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
    '🔝 Nombre de corners': 6
}

# Calcul des forces
attack_strength_1, defense_strength_1 = calculate_forces(team1_data)
attack_strength_2, defense_strength_2 = calculate_forces(team2_data)

# Affichage des résultats avec emojis
st.title("Analyse du Match ⚽")
st.write("### Team 1 vs Team 2")

st.write("### Variables pour Équipe 1:")
st.write(f"🧑‍💼 Nom de l'équipe : {team1_data['🧑‍💼 Nom de l\'équipe']}")
st.write(f"⚽ Force d'attaque : {team1_data['⚽ Attaque']}")
st.write(f"🛡️ Force de défense : {team1_data['🛡️ Défense']}")
st.write(f"🔥 Forme récente : {team1_data['🔥 Forme récente']}")
st.write(f"💔 Blessures : {team1_data['💔 Blessures']}")
st.write(f"💪 Motivation : {team1_data['💪 Motivation']}")
st.write(f"🔄 Tactique : {team1_data['🔄 Tactique']}")
st.write(f"📊 Historique face à face : {team1_data['📊 Historique face à face']}")
st.write(f"⚽xG : {team1_data['⚽xG']}")
st.write(f"🔝 Nombre de corners : {team1_data['🔝 Nombre de corners']}")

st.write("### Variables pour Équipe 2:")
st.write(f"🧑‍💼 Nom de l'équipe : {team2_data['🧑‍💼 Nom de l\'équipe']}")
st.write(f"⚽ Force d'attaque : {team2_data['⚽ Attaque']}")
st.write(f"🛡️ Force de défense : {team2_data['🛡️ Défense']}")
st.write(f"🔥 Forme récente : {team2_data['🔥 Forme récente']}")
st.write(f"💔 Blessures : {team2_data['💔 Blessures']}")
st.write(f"💪 Motivation : {team2_data['💪 Motivation']}")
st.write(f"🔄 Tactique : {team2_data['🔄 Tactique']}")
st.write(f"📊 Historique face à face : {team2_data['📊 Historique face à face']}")
st.write(f"⚽xG : {team2_data['⚽xG']}")
st.write(f"🔝 Nombre de corners : {team2_data['🔝 Nombre de corners']}")

# Prédictions
poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf_mean = generate_predictions(team1_data, team2_data)

st.write(f"⚽ Probabilité Poisson de l'Équipe 1 : {poisson_team1_prob}")
st.write(f"⚽ Probabilité Poisson de l'Équipe 2 : {poisson_team2_prob}")
st.write(f"📊 Prédiction de la régression logistique : {logreg_prediction}")
st.write(f"📊 Moyenne des scores de validation croisée (Random Forest) : {cv_scores_rf_mean:.2f}")

# Option de télécharger les résultats
st.download_button(
    label="Télécharger les prédictions en format DOC",
    data="Les données et prédictions ici",  # Remplace avec le format DOC généré
    file_name="predictions.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
