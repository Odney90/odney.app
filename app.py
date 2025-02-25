
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import altair as alt

# Fonction pour prédire avec le modèle Poisson
def poisson_prediction(home_goals, away_goals):
    # Exemple simplifié de prédiction Poisson
    home_prob = np.exp(-home_goals) * (home_goals ** home_goals) / np.math.factorial(home_goals)
    away_prob = np.exp(-away_goals) * (away_goals ** away_goals) / np.math.factorial(away_goals)
    return home_prob, away_prob

# Fonction pour prédire avec la régression logistique
def logistic_regression_predict(X_train, y_train, X_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)

# Fonction pour prédire avec Random Forest
def random_forest_predict(X_train, y_train, X_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)

# Fonction pour prédire avec XGBoost
def xgboost_predict(X_train, y_train, X_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)

# Interface utilisateur
st.title("Analyse de Matchs de Football et Prédictions de Paris Sportifs")

# Saisie des données des équipes
st.header("Saisie des données des équipes")
home_team = st.text_input("Nom de l'équipe à domicile")
away_team = st.text_input("Nom de l'équipe à l'extérieur")

# Variables d'entrée
st.subheader("Statistiques de l'équipe à domicile")
home_stats = {
    "moyenne_buts_marques": st.number_input("Moyenne de buts marqués par match", min_value=0.0),
    "xG": st.number_input("xG (Expected Goals)", min_value=0.0),
    "moyenne_buts_encais": st.number_input("Moyenne de buts encaissés par match", min_value=0.0),
    "XGA": st.number_input("xGA (Expected Goals Against)", min_value=0.0),
    "tires_par_match": st.number_input("Nombres de tirs par match", min_value=0),
    "passes_menant_a_tir": st.number_input("Nombres de passes menant à un tir", min_value=0),
    "tirs_cadres": st.number_input("Tirs cadrés par match", min_value=0),
    "tires_concedes": st.number_input("Nombres de tirs concédés par match", min_value=0),
    "duels_defensifs": st.number_input("Duels défensifs gagnés (%)", min_value=0.0, max_value=100.0),
    "possession": st.number_input("Possession moyenne (%)", min_value=0.0, max_value=100.0),
    "passes_reussies": st.number_input("Passes réussies (%)", min_value=0.0, max_value=100.0),
    "touches_surface": st.number_input("Balles touchées dans la surface adverse", min_value=0),
    "forme_recente": st.number_input("Forme récente (points sur les 5 derniers matchs)", min_value=0),
}

st.subheader("Statistiques de l'équipe à l'extérieur")
away_stats = {
    "moyenne_buts_marques": st.number_input("Moyenne de buts marqués par match", min_value=0.0),
    "xG": st.number_input("xG (Expected Goals)", min_value=0.0),
    "moyenne_buts_encais": st.number_input("Moyenne de buts encaissés par match", min_value=0.0),
    "XGA": st.number_input("xGA (Expected Goals Against)", min_value=0.0),
    "tires_par_match": st.number_input("Nombres de tirs par match", min_value=0),
    "passes_menant_a_tir": st.number_input("Nombres de passes menant à un tir", min_value=0),
    "tirs_cadres": st.number_input("Tirs cadrés par match", min_value=0),
    "tires_concedes": st.number_input("Nombres de tirs concédés par match", min_value=0),
    "duels_defensifs": st.number_input("Duels défensifs gagnés (%)", min_value=0.0, max_value=100.0),
    "possession": st.number_input("Possession moyenne (%)", min_value=0.0, max_value=100.0),
    "passes_reussies": st.number_input("Passes réussies (%)", min_value=0.0, max_value=100.0),
    "touches_surface": st.number_input("Balles touchées dans la surface adverse", min_value=0),
    "forme_recente": st.number_input("Forme récente (points sur les 5 derniers matchs)", min_value=0),
}

# Convertir les statistiques en DataFrame
home_df = pd.DataFrame([home_stats])
away_df = pd.DataFrame([away_stats])

# Prédictions
if st.button("Prédire les résultats"):
    # Exemple de prédiction avec le modèle Poisson
    home_goals = home_stats["moyenne_buts_marques"] + home_stats["xG"] - away_stats["moyenne_buts_encais"]
    away_goals = away_stats["moyenne_buts_marques"] + away_stats["xG"] - home_stats["moyenne_buts_encais"]
    
    home_prob, away_prob = poisson_prediction(home_goals, away_goals)
    
    st.write(f"Probabilité de victoire de {home_team}: {home_prob:.2f}")
    st.write(f"Probabilité de victoire de {away_team}: {away_prob:.2f}")

    # Exemple de prédiction avec régression logistique
    # (Remplacer par des données réelles pour l'entraînement)
    X_train = np.random.rand(100, 15)  # Exemple de données d'entraînement
    y_train = np.random.randint(0, 3, 100)  # Résultats aléatoires (0: nul, 1: domicile, 2: extérieur)
    X_test = np.array([list(home_stats.values()) + list(away_stats.values())])
    
    log_reg_probs = logistic_regression_predict(X_train, y_train, X_test)
    st.write("Probabilités de résultats avec régression logistique:", log_reg_probs)

    # Exemple de prédiction avec Random Forest
    rf_probs = random_forest_predict(X_train, y_train, X_test)
    st.write("Probabilités de résultats avec Random Forest:", rf_probs)

    # Exemple de prédiction avec XGBoost
    xgb_probs = xgboost_predict(X_train, y_train, X_test)
    st.write("Probabilités de résultats avec XGBoost:", xgb_probs)

    # Visualisation des résultats
    st.subheader("Visualisation des résultats")
    fig, ax = plt.subplots()
    ax.bar(["Domicile", "Nul", "Extérieur"], [home_prob, (home_prob + away_prob) / 2, away_prob])
    ax.set_ylabel("Probabilités")
    ax.set_title("Probabilités des résultats")
    st.pyplot(fig)

# Option de téléchargement des résultats
if st.button("Télécharger les résultats"):
    # Code pour générer un fichier DOC avec les résultats
    st.write("Fonctionnalité de téléchargement à implémenter.")

