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
    home_possession = st.number_input("📊 Possession moyenne (%)", min_value=0.0, max_value=100.0, value=55.0)
    home_tirs_par_match = st.number_input("🔫 Nombre de tirs par match", min_value=0, max_value=30, value=15)
    home_passes_cles_par_match = st.number_input("📊 Nombre de passes clés par match", min_value=0, max_value=50, value=10)
    home_tirs_cadres = st.number_input("🎯 Tirs cadrés par match", min_value=0, max_value=15, value=5)
    home_touches_surface = st.number_input("⚽ Balles touchées dans la surface adverse", min_value=0, max_value=300, value=20)
    home_duels_defensifs = st.number_input("🤼 Duels défensifs gagnés", min_value=0, max_value=100, value=60)
    home_passes_reussies = st.number_input("✅ Passes réussies (%)", min_value=0.0, max_value=100.0, value=80.0)
    home_forme_recente = st.number_input("📈 Forme récente (points sur les 5 derniers matchs)", min_value=0, max_value=15, value=10)
    home_victories = st.number_input("🏆 Nombre de victoires à domicile", min_value=0, max_value=20, value=5)
    home_fautes_commises = st.number_input("⚠️ Fautes commises par match", min_value=0, max_value=30, value=12)
    home_interceptions = st.number_input("🛑 Interceptions par match", min_value=0, max_value=30, value=8)

with col2:
    st.subheader("Équipe à Extérieur")
    away_team = st.text_input("🏟️ Nom de l'équipe à l'extérieur", value="Équipe B")
    away_goals = st.number_input("⚽ Moyenne de buts marqués par match", min_value=0.0, max_value=5.0, value=1.3)
    away_xG = st.number_input("📈 xG (Expected Goals)", min_value=0.0, max_value=5.0, value=1.5)
    away_encais = st.number_input("🚫 Moyenne de buts encaissés", min_value=0.0, max_value=5.0, value=1.7)
    away_possession = st.number_input("📊 Possession moyenne (%)", min_value=0.0, max_value=100.0, value=50.0)
    away_tirs_par_match = st.number_input("🔫 Nombre de tirs par match", min_value=0, max_value=30, value=12)
    away_passes_cles_par_match = st.number_input("📊 Nombre de passes clés par match", min_value=0, max_value=50, value=8)
    away_tirs_cadres = st.number_input("🎯 Tirs cadrés par match", min_value=0, max_value=15, value=4)
    away_touches_surface = st.number_input("⚽ Balles touchées dans la surface adverse", min_value=0, max_value=300, value=15)
    away_duels_defensifs = st.number_input("🤼 Duels défensifs gagnés", min_value=0, max_value=100, value=55)
    away_passes_reussies = st.number_input("✅ Passes réussies (%)", min_value=0.0, max_value=100.0, value=75.0)
    away_forme_recente = st.number_input("📈 Forme récente (points sur les 5 derniers matchs)", min_value=0, max_value=15, value=8)
    away_victories = st.number_input("🏆 Nombre de victoires à l'extérieur", min_value=0, max_value=20, value=3)
    away_fautes_commises = st.number_input("⚠️ Fautes commises par match", min_value=0, max_value=30, value=14)
    away_interceptions = st.number_input("🛑 Interceptions par match", min_value=0, max_value=30, value=10)

odds_home = st.number_input("🏠 Cote Domicile", min_value=1.0, value=1.8)
odds_away = st.number_input("🏟️ Cote Extérieur", min_value=1.0, value=2.2)

if st.button("🔍 Lancer les prédictions"):
    st.write("⚡ Prédictions en cours...")
    model_scores = evaluate_models(np.random.rand(10, 10), np.random.randint(0, 3, 10))
    st.write("📊 Résultats de la validation croisée:", model_scores)
