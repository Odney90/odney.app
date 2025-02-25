import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Fonction pour calculer les probabilités implicites à partir des cotes
def odds_to_probability(odds):
    return 1 / odds

# Génération de données factices pour les équipes (pour l'exemple)
def generate_fake_data():
    # Variables non chiffrées : Forme des équipes, Etat physique, etc.
    return {
        "team1_name": "Team A",
        "team2_name": "Team B",
        "team1_attack": 0.8,
        "team2_attack": 0.7,
        "team1_defense": 0.75,
        "team2_defense": 0.7,
        "team1_recent_form": 0.85,
        "team2_recent_form": 0.75,
        "team1_injury": 0.1,
        "team2_injury": 0.15,
        "team1_home_advantage": 0.65,
        "team2_home_advantage": 0.5,
    }

# Interface utilisateur pour saisir les données
st.title("Analyse de Match de Football 🏟️")

# Saisie des données de l'équipe 1
team1_name = st.text_input("Nom de l'équipe 1", "Team A")
team2_name = st.text_input("Nom de l'équipe 2", "Team B")

team1_attack = st.slider(f"Force d'attaque de {team1_name} (0-1)", 0.0, 1.0, 0.8)
team2_attack = st.slider(f"Force d'attaque de {team2_name} (0-1)", 0.0, 1.0, 0.7)

team1_defense = st.slider(f"Force de défense de {team1_name} (0-1)", 0.0, 1.0, 0.75)
team2_defense = st.slider(f"Force de défense de {team2_name} (0-1)", 0.0, 1.0, 0.7)

team1_recent_form = st.slider(f"Forme récente de {team1_name} (0-1)", 0.0, 1.0, 0.85)
team2_recent_form = st.slider(f"Forme récente de {team2_name} (0-1)", 0.0, 1.0, 0.75)

team1_injury = st.slider(f"Blessures dans {team1_name} (0-1)", 0.0, 1.0, 0.1)
team2_injury = st.slider(f"Blessures dans {team2_name} (0-1)", 0.0, 1.0, 0.15)

team1_home_advantage = st.slider(f"Avantage à domicile de {team1_name} (0-1)", 0.0, 1.0, 0.65)
team2_home_advantage = st.slider(f"Avantage à domicile de {team2_name} (0-1)", 0.0, 1.0, 0.5)

# Modèle Poisson
def poisson_prediction(team1_attack, team2_defense, team2_attack, team1_defense):
    team1_goals = np.random.poisson(lam=team1_attack * team2_defense * 2.5)  # La moyenne de buts
    team2_goals = np.random.poisson(lam=team2_attack * team1_defense * 2.5)
    return team1_goals, team2_goals

# Calcul des prédictions
if st.button("Générer les prédictions"):
    team1_goals, team2_goals = poisson_prediction(team1_attack, team2_defense, team2_attack, team1_defense)

    st.subheader(f"Prédiction des buts : {team1_name} vs {team2_name}")
    st.write(f"{team1_name} : {team1_goals} buts")
    st.write(f"{team2_name} : {team2_goals} buts")

    # Afficher les cotes et les probabilités
    st.subheader("Cotes et Probabilités Implicites")
    team1_odds = st.number_input(f"Cote pour {team1_name}", min_value=1.0, value=2.5)
    team2_odds = st.number_input(f"Cote pour {team2_name}", min_value=1.0, value=2.8)

    st.write(f"Probabilité implicite de {team1_name}: {odds_to_probability(team1_odds):.2%}")
    st.write(f"Probabilité implicite de {team2_name}: {odds_to_probability(team2_odds):.2%}")

    # Analyse avec le modèle Random Forest
    st.subheader("Importance des variables avec Random Forest")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X = pd.DataFrame({
        'attack': [team1_attack, team2_attack],
        'defense': [team1_defense, team2_defense],
        'recent_form': [team1_recent_form, team2_recent_form],
        'injury': [team1_injury, team2_injury],
        'home_advantage': [team1_home_advantage, team2_home_advantage]
    })

    y = [0, 1]  # Exemples de résultats fictifs (1 ou 0) pour l'entrainement du modèle

    rf.fit(X, y)
    feature_importances = np.round(rf.feature_importances_ * 100, 2)

    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance (%)': feature_importances
    })

    st.write("Importances des variables :")
    st.write(feature_df)

    # Visualisation des importances
    chart = alt.Chart(feature_df).mark_bar().encode(
        x='Importance (%)',
        y='Feature',
        color='Feature'
    ).properties(title="Importance des variables selon le modèle Random Forest")

    st.altair_chart(chart, use_container_width=True)

    # Mettre en couleur les variables utilisées par Poisson
    st.subheader("Variables utilisées par le modèle Poisson :")
    st.markdown("""
    - Force d'attaque des équipes
    - Force de défense des équipes
    - Forme récente des équipes
    - Avantage à domicile
    """, unsafe_allow_html=True)
