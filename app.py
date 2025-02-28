import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import poisson
import matplotlib.pyplot as plt

def generate_synthetic_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'home_goals': np.random.randint(0, 3, size=500),
        'away_goals': np.random.randint(0, 3, size=500),
        'home_xG': np.random.uniform(0, 2, size=500),
        'away_xG': np.random.uniform(0, 2, size=500),
        'home_encais': np.random.uniform(0, 2, size=500),
        'away_encais': np.random.uniform(0, 2, size=500),
        'home_victories': np.random.randint(0, 20, size=500),
        'away_victories': np.random.randint(0, 20, size=500),
        'home_goals_scored': np.random.randint(0, 50, size=500),
        'away_goals_scored': np.random.randint(0, 50, size=500),
        'home_xGA': np.random.uniform(0, 2, size=500),
        'away_xGA': np.random.uniform(0, 2, size=500),
        'home_tirs_par_match': np.random.randint(0, 30, size=500),
        'away_tirs_par_match': np.random.randint(0, 30, size=500),
        'home_passes_cles_par_match': np.random.randint(0, 50, size=500),
        'away_passes_cles_par_match': np.random.randint(0, 50, size=500),
        'home_tirs_cadres': np.random.randint(0, 15, size=500),
        'away_tirs_cadres': np.random.randint(0, 15, size=500),
        'home_tirs_concedes': np.random.randint(0, 30, size=500),
        'away_tirs_concedes': np.random.randint(0, 30, size=500),
        'home_duels_defensifs': np.random.randint(0, 100, size=500),
        'away_duels_defensifs': np.random.randint(0, 100, size=500),
        'home_possession': np.random.uniform(0, 100, size=500),
        'away_possession': np.random.uniform(0, 100, size=500),
        'home_passes_reussies': np.random.uniform(0, 100, size=500),
        'away_passes_reussies': np.random.uniform(0, 100, size=500),
        'home_touches_surface': np.random.randint(0, 300, size=500),
        'away_touches_surface': np.random.randint(0, 300, size=500),
        'home_forme_recente': np.random.randint(0, 15, size=500),
        'away_forme_recente': np.random.randint(0, 15, size=500)
    })
    y = np.random.choice([0, 1, 2], size=500)
    return X, y

@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "SVM": SVC(probability=True)
    }
    trained_models = {name: model.fit(X_train, y_train) for name, model in models.items()}
    return trained_models

def poisson_prediction(goals_pred):
    return [poisson.pmf(i, goals_pred) for i in range(6)]

def evaluate_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "SVM": SVC(probability=True)
    }
    return {name: cross_val_score(model, X, y, cv=3).mean() for name, model in models.items()}

st.set_page_config(page_title="Prédiction de Matchs", layout="wide")
st.title("🏆 Analyse et Prédictions Football")

home_team = st.text_input("🏠 Nom de l'équipe à domicile", value="Équipe A")
away_team = st.text_input("🏟️ Nom de l'équipe à l'extérieur", value="Équipe B")

odds_home = st.number_input("🏠 Cote Domicile", min_value=1.0, value=1.8)
odds_away = st.number_input("🏟️ Cote Extérieur", min_value=1.0, value=2.2)

def calculate_implied_prob(odds):
    return 1 / odds

def predict_match(models, X_test):
    return {name: model.predict_proba(X_test) for name, model in models.items()}

if st.button("🔍 Prédire les résultats"):
    with st.spinner('Calcul en cours...'):
        X, y = generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = train_models(X_train, y_train)
        evaluation_results = evaluate_models(X, y)
        st.write("📊 Résultats Validation Croisée", evaluation_results)
        predictions = predict_match(models, X_test.iloc[:1])
        st.write("📊 Prédictions des modèles", predictions)
