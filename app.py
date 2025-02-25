import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import poisson

# Titre de l'application
st.title("Analyse de Matchs de Football et Prédictions de Paris Sportifs")

# Saisie des données des équipes
st.header("Saisie des données des équipes")

# Variables pour l'équipe à domicile
st.subheader("Équipe à domicile")
home_team = st.text_input("Nom de l'équipe")
home_goals_scored = st.number_input("Buts marqués (total)", min_value=0)
home_goals_conceded = st.number_input("Buts encaissés (total)", min_value=0)
home_shots = st.number_input("Tirs (total)", min_value=0)
home_shots_on_target = st.number_input("Tirs cadrés", min_value=0)
home_possession = st.slider("Possession de balle (%)", 0, 100, 50)
home_pass_accuracy = st.slider("Précision des passes (%)", 0, 100, 75)
home_fouls_committed = st.number_input("Fautes commises", min_value=0)
home_corners = st.number_input("Corners obtenus", min_value=0)
home_yellow_cards = st.number_input("Cartons jaunes", min_value=0)
home_red_cards = st.number_input("Cartons rouges", min_value=0)
home_injuries = st.number_input("Joueurs blessés", min_value=0)
home_recent_form = st.number_input("Forme récente (points)", min_value=0, max_value=15)
home_defensive_strength = st.number_input("Force défensive (1-10)", 1, 10)
home_offensive_strength = st.number_input("Force offensive (1-10)", 1, 10)
home_head_to_head = st.number_input("Historique des confrontations (points)", min_value=0)
home_manager_experience = st.number_input("Expérience du manager (années)", min_value=0)

# Variables pour l'équipe à l'extérieur
st.subheader("Équipe à l'extérieur")
away_team = st.text_input("Nom de l'équipe")
away_goals_scored = st.number_input("Buts marqués (total)", min_value=0)
away_goals_conceded = st.number_input("Buts encaissés (total)", min_value=0)
away_shots = st.number_input("Tirs (total)", min_value=0)
away_shots_on_target = st.number_input("Tirs cadrés", min_value=0)
away_possession = st.slider("Possession de balle (%)", 0, 100, 50)
away_pass_accuracy = st.slider("Précision des passes (%)", 0, 100, 75)
away_fouls_committed = st.number_input("Fautes commises", min_value=0)
away_corners = st.number_input("Corners obtenus", min_value=0)
away_yellow_cards = st.number_input("Cartons jaunes", min_value=0)
away_red_cards = st.number_input("Cartons rouges", min_value=0)
away_injuries = st.number_input("Joueurs blessés", min_value=0)
away_recent_form = st.number_input("Forme récente (points)", min_value=0, max_value=15)
away_defensive_strength = st.number_input("Force défensive (1-10)", 1, 10)
away_offensive_strength = st.number_input("Force offensive (1-10)", 1, 10)
away_head_to_head = st.number_input("Historique des confrontations (points)", min_value=0)
away_manager_experience = st.number_input("Expérience du manager (années)", min_value=0)

# Modèles de prédiction
def predict_poisson(home_goals, away_goals):
    home_prob = poisson.pmf(range(6), home_goals)
    away_prob = poisson.pmf(range(6), away_goals)
    return home_prob, away_prob

# Ajouter les autres modèles selon les besoins
def logistic_regression_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def random_forest_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def xgboost_model(X, y):
    model = XGBClassifier()
    model.fit(X, y)
    return model

# Afficher un message pour prédire les résultats
if st.button("Prédire les résultats"):
    # Hypothétique, en fonction des variables d'entrée pour Poisson
    home_goals = home_goals_scored / max(1, home_goals_conceded)  # Logique simplifiée
    away_goals = away_goals_scored / max(1, away_goals_conceded)  # Logique simplifiée
    
    home_prob, away_prob = predict_poisson(home_goals, away_goals)
    st.subheader("Probabilités de buts")
    st.write("Équipe à domicile :", home_prob)
    st.write("Équipe à l'extérieur :", away_prob)

    # Autres prédictions peuvent être ajoutées ici

# Analyse avec validation croisée
# Intégrer une section ici pour la validation croisée K=10 selon le modèle

# Options d'exportation
st.write("Télécharger les résultats au format DOC")
# Ajouter des logiques pour télécharger les données au format approprié

# Affichage des résultats
st.header("Résultats de prédiction")
# Logique pour afficher les résultats ici
