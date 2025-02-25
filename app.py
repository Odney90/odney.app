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
home_team = st.text_input("Nom de l'équipe à domicile")
away_team = st.text_input("Nom de l'équipe à l'extérieur")

# Saisie des statistiques des équipes
home_goals = st.number_input("Buts marqués à domicile", min_value=0)
away_goals = st.number_input("Buts marqués à l'extérieur", min_value=0)
home_possession = st.slider("Possession de balle à domicile (%)", 0, 100, 50)
away_possession = st.slider("Possession de balle à l'extérieur (%)", 0, 100, 50)

# Ajouter d'autres variables selon les besoins

# Modèles de prédiction
def predict_poisson(home_goals, away_goals):
    # Modèle Poisson - Exemple simple
    home_prob = poisson.pmf(range(6), home_goals)  # Probabilités pour 0-5 buts
    away_prob = poisson.pmf(range(6), away_goals)
    return home_prob, away_prob

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

# Afficher un message pour prédier les résultats
if st.button("Prédire les résultats"):
    home_prob, away_prob = predict_poisson(home_goals, away_goals)
    st.subheader("Probabilités de buts")
    st.write("Équipe à domicile :", home_prob)
    st.write("Équipe à l'extérieur :", away_prob)

    # Autres prédictions peuvent être ajoutées ici
    # Par exemple : utilisation des modèles de régression logistique, Random Forest, et XGBoost

# Analyse avec validation croisée
# Vous pouvez ajouter une section ici pour la validation croisée K=10 selon le modèle

# Gestion de la bankroll
st.header("Gestion de la bankroll")
stake = st.number_input("Montant de mise ", min_value=0, step=1)
# Intégrer des calculs de mise de Kelly ici

# Options d'exportation
st.write("Télécharger les résultats au format DOC")
# Ajouter logiques pour télécharger les données au format approprié

# Affichage des résultats
st.header("Résultats de prédiction")
# Logique pour afficher les résultats ici
