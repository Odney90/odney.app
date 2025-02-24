import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import poisson
import altair as alt

# Fonction pour calculer la prédiction avec le modèle Poisson
def poisson_prediction(team1_attack, team2_attack):
    # Calcul des buts attendus avec le modèle Poisson
    team1_goals = poisson.pmf(np.arange(6), team1_attack)
    team2_goals = poisson.pmf(np.arange(6), team2_attack)
    
    team1_prob = np.round(team1_goals, 2)
    team2_prob = np.round(team2_goals, 2)
    
    return team1_prob, team2_prob

# Fonction pour générer les prédictions
def generate_predictions(team1_data, team2_data):
    # Données utilisées pour Poisson
    poisson_team1_attack = team1_data['attack']
    poisson_team2_attack = team2_data['attack']
    
    # Données utilisées pour RandomForest et Logistic Regression
    X = np.array([
        [team1_data['attack'], team1_data['defense'], team1_data['form'], team1_data['injuries'],
         team2_data['attack'], team2_data['defense'], team2_data['form'], team2_data['injuries']]
    ])
    
    # Labels (résultats des matchs) - pour simplifier, je mets un label fictif à ce stade
    y = [1]  # Résultat fictif (ex. victoire de l'équipe 1)
    
    # Modèle RandomForest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    
    # Validation croisée K=5
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    
    # Modèle Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X, y)
    
    # Prédiction Poisson
    poisson_team1_prob, poisson_team2_prob = poisson_prediction(poisson_team1_attack, poisson_team2_attack)
    
    # Résultats de prédiction
    rf_prediction = rf.predict(X)[0]
    logreg_prediction = logreg.predict(X)[0]
    
    return poisson_team1_prob, poisson_team2_prob, rf_prediction, logreg_prediction, cv_scores_rf

# Interface Streamlit
st.title("Analyse de Match de Football ⚽")

# Entrées des utilisateurs pour les équipes
team1_name = st.text_input("Nom de l'Équipe 1")
team2_name = st.text_input("Nom de l'Équipe 2")

# Données des équipes
team1_data = {
    'attack': st.slider("Force d'attaque de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'defense': st.slider("Force de défense de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'form': st.slider("Forme récente de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'injuries': st.slider("Blessures dans l'équipe 1 (0-1)", 0.0, 1.0, 0.0)
}

team2_data = {
    'attack': st.slider("Force d'attaque de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'defense': st.slider("Force de défense de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'form': st.slider("Forme récente de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'injuries': st.slider("Blessures dans l'équipe 2 (0-1)", 0.0, 1.0, 0.0)
}

# Générer les prédictions
if st.button("Générer les prédictions"):
    poisson_team1_prob, poisson_team2_prob, rf_prediction, logreg_prediction, cv_scores_rf = generate_predictions(team1_data, team2_data)
    
    # Affichage des résultats
    st.subheader(f"Prédiction Poisson pour {team1_name} vs {team2_name}")
    st.write(f"Probabilités des buts de {team1_name}: {poisson_team1_prob}")
    st.write(f"Probabilités des buts de {team2_name}: {poisson_team2_prob}")
    
    st.subheader(f"Prédictions avec Random Forest et Régression Logistique")
    st.write(f"Prédiction Random Forest: {rf_prediction}")
    st.write(f"Prédiction Régression Logistique: {logreg_prediction}")
    
    st.subheader(f"Validation Croisée K=5 pour Random Forest")
    st.write(f"Scores de validation croisée: {cv_scores_rf}")
    
    # Option de comparer les probabilités des bookmakers
    odds_home = st.number_input("Cote bookmaker pour l'Équipe 1", min_value=1.0)
    odds_away = st.number_input("Cote bookmaker pour l'Équipe 2", min_value=1.0)
    
    if odds_home and odds_away:
        st.subheader("Comparer les cotes avec les prédictions")
        home_prob = 1 / odds_home
        away_prob = 1 / odds_away
        st.write(f"Probabilité implicite de {team1_name}: {home_prob}")
        st.write(f"Probabilité implicite de {team2_name}: {away_prob}")
        
        # Valeur des paris (Value Bet)
        st.write(f"Value Bet pour {team1_name}: {'Oui' if home_prob < poisson_team1_prob[0] else 'Non'}")
        st.write(f"Value Bet pour {team2_name}: {'Oui' if away_prob < poisson_team2_prob[0] else 'Non'}")
