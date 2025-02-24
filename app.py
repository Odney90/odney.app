import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import poisson
from io import StringIO
from docx import Document

# Fonction pour calculer la probabilité implicite d'une cote
def odds_to_probability(odds):
    return 1 / odds

# Fonction pour détecter le value bet
def check_value_bet(predicted_prob, odds):
    implied_prob = odds_to_probability(odds)
    return predicted_prob > implied_prob

# Fonction pour prédire les résultats avec Poisson, Random Forest et Logistic Regression
def generate_predictions(team1_data, team2_data):
    # Variables d'exemple (peut être élargi à plus de variables pertinentes)
    X = pd.DataFrame({
        'attaque_1': [team1_data['attaque']],
        'defense_1': [team1_data['defense']],
        'forme_1': [team1_data['forme']],
        'blessures_1': [team1_data['blessures']],
        'attaque_2': [team2_data['attaque']],
        'defense_2': [team2_data['defense']],
        'forme_2': [team2_data['forme']],
        'blessures_2': [team2_data['blessures']],
    })
    
    y = [team1_data['resultat']]  # La colonne 'resultat' doit être définie avec les résultats historiques (1 pour victoire équipe 1, 2 pour victoire équipe 2, X pour match nul)

    # Poisson
    attack1, attack2 = team1_data['attaque'], team2_data['attaque']
    poisson_team1_prob = poisson.pmf(1, attack1)
    poisson_team2_prob = poisson.pmf(1, attack2)
    
    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X, y)
    rf_prediction = rf.predict(X)[0]
    
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)[0]

    # Validation croisée K=5 pour le Random Forest
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    
    return poisson_team1_prob, poisson_team2_prob, rf_prediction, logreg_prediction, cv_scores_rf

# Interface utilisateur Streamlit
st.title("Analyse de Match de Football ⚽")

# Persistance des données dans la session
if 'team1_name' not in st.session_state:
    st.session_state.team1_name = "Équipe A"
if 'team2_name' not in st.session_state:
    st.session_state.team2_name = "Équipe B"

# Renseigner les noms des équipes
team1_name = st.text_input("Nom de l'Équipe 1", st.session_state.team1_name)
team2_name = st.text_input("Nom de l'Équipe 2", st.session_state.team2_name)

# Sauvegarder les noms dans la session
st.session_state.team1_name = team1_name
st.session_state.team2_name = team2_name

# Variables pour l'Équipe 1
team1_attack = st.slider("Force d'attaque de l'équipe 1 (0-1)", 0.0, 1.0, 0.5)
team1_defense = st.slider("Force de défense de l'équipe 1 (0-1)", 0.0, 1.0, 0.5)
team1_form = st.slider("Forme récente de l'équipe 1 (0-1)", 0.0, 1.0, 0.5)
team1_injuries = st.slider("Blessures dans l'équipe 1 (0-1)", 0.0, 1.0, 0.0)

# Variables pour l'Équipe 2
team2_attack = st.slider("Force d'attaque de l'équipe 2 (0-1)", 0.0, 1.0, 0.5)
team2_defense = st.slider("Force de défense de l'équipe 2 (0-1)", 0.0, 1.0, 0.5)
team2_form = st.slider("Forme récente de l'équipe 2 (0-1)", 0.0, 1.0, 0.5)
team2_injuries = st.slider("Blessures dans l'équipe 2 (0-1)", 0.0, 1.0, 0.0)

# Collecte des données et prédiction
team1_data = {
    'attaque': team1_attack,
    'defense': team1_defense,
    'forme': team1_form,
    'blessures': team1_injuries,
    'resultat': 1  # Exemple de résultat historique
}

team2_data = {
    'attaque': team2_attack,
    'defense': team2_defense,
    'forme': team2_form,
    'blessures': team2_injuries,
    'resultat': 2  # Exemple de résultat historique
}

# Calcul des prédictions
poisson_team1_prob, poisson_team2_prob, rf_prediction, logreg_prediction, cv_scores_rf = generate_predictions(team1_data, team2_data)

# Affichage des résultats avec visuels et tableaux
st.subheader("Prédictions")

# Création d'un tableau pour afficher les prédictions
predictions_df = pd.DataFrame({
    'Modèle': ['Poisson - Équipe 1', 'Poisson - Équipe 2', 'Random Forest', 'Logistic Regression'],
    'Probabilité / Résultat': [
        poisson_team1_prob, poisson_team2_prob, rf_prediction, logreg_prediction
    ]
})

# Affichage du tableau avec pandas
st.dataframe(predictions_df)

# Graphique pour comparer les probabilités
fig, ax = plt.subplots(figsize=(8, 6))
bar_data = {
    'Modèle': ['Poisson - Équipe 1', 'Poisson - Équipe 2', 'Random Forest', 'Logistic Regression'],
    'Probabilité': [poisson_team1_prob, poisson_team2_prob, rf_prediction, logreg_prediction]
}

bar_df = pd.DataFrame(bar_data)
sns.barplot(x='Modèle', y='Probabilité', data=bar_df, ax=ax, palette="viridis")
ax.set_title("Comparaison des Probabilités des Modèles")
st.pyplot(fig)

# Graphique de la validation croisée (K=5) pour Random Forest
st.subheader("Validation Croisée (K=5) - Random Forest")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=cv_scores_rf, ax=ax)
ax.set_title("Distribution des Scores de Validation Croisée (K=5)")
st.pyplot(fig)

# Option de téléchargement
if st.button("Télécharger les données et prédictions en format DOC"):
    doc = generate_doc(team1_name, team2_name, team1_data, team2_data, poisson_team1_prob, poisson_team2_prob, rf_prediction, logreg_prediction, cv_scores_rf)
    doc.save("/mnt/data/match_analysis.docx")
    st.download_button("Télécharger le fichier DOC", "/mnt/data/match_analysis.docx", "Télécharger le fichier DOC")
