import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from scipy.stats import poisson

# Fonction pour générer des prédictions avec les modèles
def generate_predictions(team1_data, team2_data):
    # Créer un DataFrame avec les données des deux équipes
    X = pd.DataFrame([team1_data + team2_data], columns=team1_data.keys() + team2_data.keys())
    
    # Variables cibles : issues possibles d'un match (1 = équipe 1 gagne, X = match nul, 2 = équipe 2 gagne)
    y = np.array([1, 0, 0, 0, 1, 0])  # Cible fictive pour entraîner les modèles (exemple)
    
    # Modèle Poisson
    poisson_team1_prob = poisson.pmf(0, team1_data['average_goals'])  # Exemple : probabilité de 0 but pour l'équipe 1
    poisson_team2_prob = poisson.pmf(0, team2_data['average_goals'])  # Exemple : probabilité de 0 but pour l'équipe 2
    
    # Modèle Logistique
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X, y)
    logreg_pred = logreg.predict(X)  # Prédiction logistique
    
    # Modèle Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    rf_pred = rf.predict(X)  # Prédiction Random Forest
    
    # Modèle XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100)
    xgb_model.fit(X, y)
    xgb_pred = xgb_model.predict(X)  # Prédiction XGBoost
    
    # Validation croisée
    rf_cv_score = cross_val_score(rf, X, y, cv=10).mean()  # Validation croisée pour Random Forest
    
    return poisson_team1_prob, poisson_team2_prob, logreg_pred, rf_pred, rf_cv_score, xgb_pred

# Interface utilisateur
st.title("Analyse des Paris Sportifs - Match de Football")

# Saisie des données des équipes
st.header("Saisie des données des équipes")
team1_data = {
    'average_goals': st.number_input("Moyenne de buts de l'équipe 1 par match", min_value=0.0, max_value=5.0, value=1.5),
    'attack_strength': st.number_input("Force d'attaque de l'équipe 1", min_value=0, max_value=100, value=50),
    'defense_strength': st.number_input("Force de défense de l'équipe 1", min_value=0, max_value=100, value=50),
    'home_advantage': st.number_input("Avantage à domicile de l'équipe 1", min_value=0, max_value=100, value=60)
}

team2_data = {
    'average_goals': st.number_input("Moyenne de buts de l'équipe 2 par match", min_value=0.0, max_value=5.0, value=1.2),
    'attack_strength': st.number_input("Force d'attaque de l'équipe 2", min_value=0, max_value=100, value=45),
    'defense_strength': st.number_input("Force de défense de l'équipe 2", min_value=0, max_value=100, value=55),
    'home_advantage': st.number_input("Avantage à domicile de l'équipe 2", min_value=0, max_value=100, value=45)
}

# Calcul des prédictions
if st.button("Générer les prédictions"):
    poisson_team1_prob, poisson_team2_prob, logreg_pred, rf_pred, rf_cv_score, xgb_pred = generate_predictions(team1_data, team2_data)
    
    # Affichage des résultats
    st.subheader("Prédictions des Modèles")
    st.write(f"Probabilité Poisson (Equipe 1) : {poisson_team1_prob:.2f}")
    st.write(f"Probabilité Poisson (Equipe 2) : {poisson_team2_prob:.2f}")
    
    st.write(f"Prédiction Logistique (Equipe 1) : {'Gagne' if logreg_pred[0] == 1 else 'Perd'}")
    st.write(f"Prédiction Random Forest (Equipe 1) : {'Gagne' if rf_pred[0] == 1 else 'Perd'}")
    st.write(f"Prédiction XGBoost (Equipe 1) : {'Gagne' if xgb_pred[0] == 1 else 'Perd'}")
    
    st.write(f"Score de Validation Croisée Random Forest (K=10) : {rf_cv_score:.2f}")

    # Ajout de la fonctionnalité des paris double chance
    st.subheader("Paris Double Chance")
    st.write(f"Double chance pour l'équipe 1 (gagne ou nul) : {poisson_team1_prob + poisson_team2_prob:.2f}")
    st.write(f"Double chance pour l'équipe 2 (gagne ou nul) : {poisson_team2_prob + poisson_team1_prob:.2f}")

# Gestion de la bankroll et système de mise de Kelly (optionnel)
st.sidebar.header("Gestion de la Bankroll")
bet_percentage = st.sidebar.slider("Pourcentage de la bankroll à parier", min_value=1, max_value=10, value=5)
st.sidebar.write(f"Pourcentage à parier : {bet_percentage}%")

# Téléchargement des résultats au format DOC (optionnel)
if st.button("Télécharger les résultats"):
    results = {
        "Equipe 1": team1_data,
        "Equipe 2": team2_data,
        "Poisson Team 1": poisson_team1_prob,
        "Poisson Team 2": poisson_team2_prob,
        "Logistic Regression": logreg_pred[0],
        "Random Forest": rf_pred[0],
        "XGBoost": xgb_pred[0],
        "RF CV Score": rf_cv_score,
        "Double Chance Team 1": poisson_team1_prob + poisson_team2_prob,
        "Double Chance Team 2": poisson_team2_prob + poisson_team1_prob
    }
    
    df = pd.DataFrame([results])
    df.to_csv("match_predictions.csv")
    st.download_button("Télécharger les résultats", data=df.to_csv(), file_name="match_predictions.csv", mime="text/csv")
