import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Initialisation de session_state si non existant
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'model_scores' not in st.session_state:
    st.session_state.model_scores = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'poisson_results' not in st.session_state:
    st.session_state.poisson_results = None
if 'value_bets' not in st.session_state:
    st.session_state.value_bets = None

@st.cache_resource
def train_models(X_train, y_train):
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        st.error("Erreur: Les données ne contiennent pas au moins deux classes distinctes pour l'entraînement.")
        return None
    
    if st.session_state.trained_models is None:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
            "SVM": SVC(probability=True)
        }
        st.session_state.trained_models = {name: model.fit(X_train, y_train) for name, model in models.items()}
    return st.session_state.trained_models

def poisson_prediction(goals_pred):
    return np.array([poisson.pmf(i, goals_pred) for i in range(6)])

def evaluate_models(X, y):
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        st.error("Erreur: Les données ne contiennent pas au moins deux classes distinctes pour l'évaluation.")
        return None
    
    if st.session_state.model_scores is None:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
            "SVM": SVC(probability=True)
        }
        st.session_state.model_scores = {name: cross_val_score(model, X, y, cv=3).mean() for name, model in models.items()}
    return st.session_state.model_scores

def calculate_implied_prob(odds):
    return 1 / odds

def detect_value_bet(predicted_prob, implied_prob, threshold=0.05):
    return predicted_prob > (implied_prob + threshold)

st.set_page_config(page_title="Prédiction de Matchs", layout="wide")
st.title("🏆 Analyse et Prédictions Football")

st.header("📋 Saisie des données des équipes")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Équipe à Domicile")
    home_team = st.text_input("🏠 Nom de l'équipe à domicile", value="Équipe A")
with col2:
    st.subheader("Équipe à Extérieur")
    away_team = st.text_input("🏟️ Nom de l'équipe à l'extérieur", value="Équipe B")

if st.button("🔍 Obtenir les résultats"):
    with st.spinner("Calcul en cours..."):
        # Simuler les probabilités des modèles
        predicted_probs = np.random.uniform(0, 1, 3)
        predicted_probs /= predicted_probs.sum()
        
        # Calcul des cotes implicites
        odds_home = st.number_input("🏠 Cote Domicile", min_value=1.0, value=1.8)
        odds_away = st.number_input("🏟️ Cote Extérieur", min_value=1.0, value=2.2)
        implied_home_prob = calculate_implied_prob(odds_home)
        implied_away_prob = calculate_implied_prob(odds_away)
        
        # Détection des value bets
        value_bet_home = detect_value_bet(predicted_probs[0], implied_home_prob)
        value_bet_away = detect_value_bet(predicted_probs[2], implied_away_prob)
        
        st.subheader("📊 Résultats de la prédiction")
        st.write(f"Probabilités prédites: Domicile {predicted_probs[0]*100:.2f}%, Nul {predicted_probs[1]*100:.2f}%, Extérieur {predicted_probs[2]*100:.2f}%")
        
        st.subheader("💰 Conseils de Paris")
        if value_bet_home:
            st.success(f"Value Bet détecté sur {home_team}!")
        if value_bet_away:
            st.success(f"Value Bet détecté sur {away_team}!")
        if not value_bet_home and not value_bet_away:
            st.info("Aucun value bet détecté.")
