import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy.stats import poisson

# Fonction pour générer les prédictions avec les modèles
def generate_predictions(team1_data, team2_data):
    # Données des équipes 1 et 2
    X = pd.DataFrame([team1_data + team2_data], columns=team1_data.keys() + team2_data.keys())
    y = np.array([1])  # Dummy label, à ajuster en fonction des critères de comparaison

    # Modèle de régression logistique
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)[0]

    # Modèle Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)
    rf_prediction = rf.predict(X)[0]
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    rf_cv_score = cv_scores_rf.mean()

    # Modèle XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X, y)
    xgb_prediction = xgb_model.predict(X)[0]

    # Calcul des probabilités Poisson
    poisson_team1_prob = poisson.pmf(2, team1_data['⚽ Attaque'])
    poisson_team2_prob = poisson.pmf(2, team2_data['⚽ Attaque'])

    return poisson_team1_prob, poisson_team2_prob, logreg_prediction, rf_prediction, rf_cv_score, xgb_prediction

# Interface utilisateur Streamlit pour saisir les données des équipes
st.title("Analyse des Paris Sportifs ⚽")

# Saisie des données pour l'équipe 1
st.write("### Données pour l'Équipe 1:")
team1_name = st.text_input("Nom de l'Équipe 1", "Équipe 1")
attack1 = st.slider("Force d'attaque Équipe 1", 0.0, 5.0, 1.2)
defense1 = st.slider("Force de défense Équipe 1", 0.0, 5.0, 1.0)
recent_form1 = st.slider("Forme récente Équipe 1 (0 à 5)", 0.0, 5.0, 3.5)
injuries1 = st.slider("Blessures Équipe 1 (0 à 5)", 0.0, 5.0, 1.5)
motivation1 = st.slider("Motivation Équipe 1 (0 à 5)", 0.0, 5.0, 4.0)
tactic1 = st.slider("Tactique Équipe 1 (0 à 5)", 0.0, 5.0, 3.5)
h2h1 = st.slider("Historique face-à-face Équipe 1 (0 à 5)", 0.0, 5.0, 3.0)
xg1 = st.slider("xG Équipe 1", 0.0, 5.0, 1.4)
corners1 = st.slider("Nombre de corners Équipe 1", 0, 20, 5)

# Saisie des données pour l'équipe 2
st.write("### Données pour l'Équipe 2:")
team2_name = st.text_input("Nom de l'Équipe 2", "Équipe 2")
attack2 = st.slider("Force d'attaque Équipe 2", 0.0, 5.0, 0.8)
defense2 = st.slider("Force de défense Équipe 2", 0.0, 5.0, 1.0)
recent_form2 = st.slider("Forme récente Équipe 2 (0 à 5)", 0.0, 5.0, 3.0)
injuries2 = st.slider("Blessures Équipe 2 (0 à 5)", 0.0, 5.0, 2.0)
motivation2 = st.slider("Motivation Équipe 2 (0 à 5)", 0.0, 5.0, 3.5)
tactic2 = st.slider("Tactique Équipe 2 (0 à 5)", 0.0, 5.0, 3.0)
h2h2 = st.slider("Historique face-à-face Équipe 2 (0 à 5)", 0.0, 5.0, 2.5)
xg2 = st.slider("xG Équipe 2", 0.0, 5.0, 1.1)
corners2 = st.slider("Nombre de corners Équipe 2", 0, 20, 6)

# Données d'entrée sous forme de dictionnaire
team1_data = {
    '🧑‍💼 Nom de l\'équipe': team1_name,
    '⚽ Attaque': attack1,
    '🛡️ Défense': defense1,
    '🔥 Forme récente': recent_form1,
    '💔 Blessures': injuries1,
    '💪 Motivation': motivation1,
    '🔄 Tactique': tactic1,
    '📊 Historique face à face': h2h1,
    '⚽xG': xg1,
    '🔝 Nombre de corners': corners1
}

team2_data = {
    '🧑‍💼 Nom de l\'équipe': team2_name,
    '⚽ Attaque': attack2,
    '🛡️ Défense': defense2,
    '🔥 Forme récente': recent_form2,
    '💔 Blessures': injuries2,
    '💪 Motivation': motivation2,
    '🔄 Tactique': tactic2,
    '📊 Historique face-à-face': h2h2,
    '⚽xG': xg2,
    '🔝 Nombre de corners': corners2
}

# Prédictions
poisson_team1_prob, poisson_team2_prob, logreg_prediction, rf_prediction, rf_cv_score, xgb_prediction = generate_predictions(team1_data, team2_data)

# Affichage des résultats
st.write("### Résultats des Prédictions:")

# Prédictions des modèles
st.write(f"⚽ **Probabilité Poisson pour Équipe 1** : {poisson_team1_prob:.4f}")
st.write(f"⚽ **Probabilité Poisson pour Équipe 2** : {poisson_team2_prob:.4f}")

st.write(f"📊 **Prédiction de la régression logistique (Équipe 1)** : {logreg_prediction}")
st.write(f"📊 **Prédiction de Random Forest (Équipe 1)** : {rf_prediction}")
st.write(f"📊 **Moyenne des scores de validation croisée (Random Forest)** : {rf_cv_score:.2f}")
st.write(f"📊 **Prédiction de XGBoost (Équipe 1)** : {xgb_prediction}")

# Option de téléchargement des résultats
st.download_button(
    label="Télécharger les prédictions en format DOC",
    data="Les données et prédictions ici",  # Remplace avec le format DOC généré
    file_name="predictions.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
