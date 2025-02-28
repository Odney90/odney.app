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

@st.cache_resource
def train_models(X_train, y_train):
    if 'trained_models' not in st.session_state:
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
    if 'model_scores' not in st.session_state:
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
    home_goals = st.number_input("⚽ Moyenne de buts marqués par match", min_value=0.0, max_value=5.0, value=1.5)
    home_xG = st.number_input("📈 xG (Expected Goals)", min_value=0.0, max_value=5.0, value=1.8)
    home_encais = st.number_input("🚫 Moyenne de buts encaissés", min_value=0.0, max_value=5.0, value=1.2)

with col2:
    st.subheader("Équipe à Extérieur")
    away_team = st.text_input("🏟️ Nom de l'équipe à l'extérieur", value="Équipe B")
    away_goals = st.number_input("⚽ Moyenne de buts marqués par match", min_value=0.0, max_value=5.0, value=1.3)
    away_xG = st.number_input("📈 xG (Expected Goals)", min_value=0.0, max_value=5.0, value=1.5)
    away_encais = st.number_input("🚫 Moyenne de buts encaissés", min_value=0.0, max_value=5.0, value=1.7)

odds_home = st.number_input("🏠 Cote Domicile", min_value=1.0, value=1.8)
odds_away = st.number_input("🏟️ Cote Extérieur", min_value=1.0, value=2.2)

if st.button("🔍 Lancer les prédictions"):
    with st.spinner("🔄 Calcul en cours..."):
        model_scores = evaluate_models(np.random.rand(10, 10), np.random.randint(0, 3, 10))
        st.write("📊 Résultats de la validation croisée:", model_scores)
        
        # Prédictions des modèles
        trained_models = train_models(np.random.rand(10, 3), np.random.randint(0, 3, 10))
        predictions = {name: model.predict_proba([[home_goals, away_goals, home_xG, away_xG, home_encais, away_encais]])[0] for name, model in trained_models.items()}
        st.write("🎯 Prédictions des modèles:", predictions)
        
        # Prédictions Poisson
        home_goals_pred = home_goals + home_xG - away_encais
        away_goals_pred = away_goals + away_xG - home_encais
        poisson_home = poisson_prediction(home_goals_pred)
        poisson_away = poisson_prediction(away_goals_pred)
        st.write("📈 Prédictions du modèle de Poisson:")
        st.write(f"{home_team} : {[f'{p*100:.2f}%' for p in poisson_home]}")
        st.write(f"{away_team} : {[f'{p*100:.2f}%' for p in poisson_away]}")
        
        # Détection des value bets
        implied_home_prob = calculate_implied_prob(odds_home)
        implied_away_prob = calculate_implied_prob(odds_away)
        value_bet_home = detect_value_bet(predictions['Logistic Regression'][0], implied_home_prob)
        value_bet_away = detect_value_bet(predictions['Logistic Regression'][2], implied_away_prob)
        
        st.write("💰 Conseils de paris:")
        if value_bet_home:
            st.success(f"Value Bet détecté sur {home_team}!")
        if value_bet_away:
            st.success(f"Value Bet détecté sur {away_team}!")
        if not value_bet_home and not value_bet_away:
            st.info("Aucun value bet détecté.")
        
        st.success("✅ Prédictions générées avec succès !")
