import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import scipy.stats as stats

# Fonction de Poisson pour la prédiction des buts
def poisson_prediction(attack, defense, form):
    avg_goals = 1.5  # Nombre moyen de buts en match (modifiable)
    team_goals = avg_goals * attack * (1 - defense) * form  # Estimation des buts
    poisson_dist = stats.poisson(team_goals)  # Distribution de Poisson
    return poisson_dist.mean()  # Prédiction du nombre de buts

# Fonction pour générer les prédictions
def generate_predictions(team1_data, team2_data):
    # Extraction des variables
    attack1, defense1, form1 = team1_data
    attack2, defense2, form2 = team2_data

    # Prédiction du nombre de buts avec Poisson
    poisson_team1_goals = poisson_prediction(attack1, defense2, form1)
    poisson_team2_goals = poisson_prediction(attack2, defense1, form2)

    # Définition des labels des issues possibles
    labels = ["1", "2", "X", "1X", "X2", "12"]
    label_map = {label: idx for idx, label in enumerate(labels)}

    # Déterminer l'issue selon Poisson
    if poisson_team1_goals > poisson_team2_goals:
        match_result = "1"
    elif poisson_team1_goals < poisson_team2_goals:
        match_result = "2"
    else:
        match_result = "X"

    y = [label_map[match_result]]  # Transformation en label numérique
    X = pd.DataFrame([team1_data + team2_data], columns=feature_names)

    # Modèles
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    logreg = LogisticRegression(max_iter=1000, random_state=42)

    # Entraîner les modèles
    rf.fit(X, y)
    logreg.fit(X, y)

    # Validation croisée (K=5)
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)

    # Prédictions
    rf_prediction = rf.predict(X)[0]
    logreg_prediction = logreg.predict(X)[0]

    return poisson_team1_goals, poisson_team2_goals, rf_prediction, logreg_prediction, cv_scores_rf

# Interface utilisateur Streamlit
st.title("Analyse de Match de Football ⚽")

team1_name = st.text_input("Nom de l'Équipe 1", "Équipe 1")
team2_name = st.text_input("Nom de l'Équipe 2", "Équipe 2")

st.subheader(f"Entrer les données pour {team1_name}")
team1_attack = st.slider("Force d'attaque (0-1)", 0.0, 1.0, 0.5)
team1_defense = st.slider("Force de défense (0-1)", 0.0, 1.0, 0.5)
team1_form = st.slider("Forme récente (0-1)", 0.0, 1.0, 0.5)

st.subheader(f"Entrer les données pour {team2_name}")
team2_attack = st.slider("Force d'attaque (0-1)", 0.0, 1.0, 0.5)
team2_defense = st.slider("Force de défense (0-1)", 0.0, 1.0, 0.5)
team2_form = st.slider("Forme récente (0-1)", 0.0, 1.0, 0.5)

# Lancer les prédictions
if st.button("Générer les prédictions"):
    team1_data = [team1_attack, team1_defense, team1_form]
    team2_data = [team2_attack, team2_defense, team2_form]

    # Générer les prédictions
    poisson_team1_goals, poisson_team2_goals, rf_prediction, logreg_prediction, cv_scores_rf = generate_predictions(team1_data, team2_data)

    # Affichage des résultats
    st.write("### Résultats des prédictions")
    st.write(f"Prédiction du nombre de buts (Poisson) pour {team1_name} : **{poisson_team1_goals:.2f}**")
    st.write(f"Prédiction du nombre de buts (Poisson) pour {team2_name} : **{poisson_team2_goals:.2f}**")
    st.write(f"Prédiction Random Forest : **{rf_prediction}**")
    st.write(f"Prédiction Logistic Regression : **{logreg_prediction}**")
    st.write(f"Validation croisée (K=5) - Score moyen Random Forest : **{cv_scores_rf.mean():.2f}**")

# Conversion des cotes en probabilités
def odds_to_probability(odds):
    return 1 / odds

st.subheader("Conversion des Cotes en Probabilités")
odds_home = st.number_input("Cote victoire équipe 1", min_value=1.0, value=2.5)
odds_draw = st.number_input("Cote match nul", min_value=1.0, value=3.0)
odds_away = st.number_input("Cote victoire équipe 2", min_value=1.0, value=3.0)

home_prob = odds_to_probability(odds_home)
draw_prob = odds_to_probability(odds_draw)
away_prob = odds_to_probability(odds_away)

st.write(f"Probabilité équipe 1 : **{home_prob:.2f}**")
st.write(f"Probabilité match nul : **{draw_prob:.2f}**")
st.write(f"Probabilité équipe 2 : **{away_prob:.2f}**")

# Détection des Value Bets
st.subheader("Détection de Value Bet")

predicted_home_prob = 0.55  # Exemple
predicted_draw_prob = 0.25  # Exemple
predicted_away_prob = 0.20  # Exemple

if home_prob > predicted_home_prob:
    st.write("💰 **Value Bet détecté sur la victoire de l'équipe 1**")
if draw_prob > predicted_draw_prob:
    st.write("💰 **Value Bet détecté sur le match nul**")
if away_prob > predicted_away_prob:
    st.write("💰 **Value Bet détecté sur la victoire de l'équipe 2**")
