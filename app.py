import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from scipy.stats import poisson

# Fonction pour générer les prédictions
def generate_predictions(team1_data, team2_data):
    # Combinaison des données des deux équipes
    combined_data = {**team1_data, **team2_data}
    
    # Créer un DataFrame avec les données combinées
    X = pd.DataFrame([combined_data], columns=list(combined_data.keys()))
    
    # Cibles possibles : 0 -> équipe 1 gagne, 1 -> match nul, 2 -> équipe 2 gagne
    y = [0, 1, 2]
    
    # Modèle de régression logistique
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X, y)
    logreg_prediction = logreg.predict_proba(X)
    logreg_predicted_issue = logreg.predict(X)[0]

    # Modèle Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_prediction = rf.predict_proba(X)
    rf_predicted_issue = rf.predict(X)[0]
    rf_cv_score = cross_val_score(rf, X, y, cv=5).mean()

    # Modèle XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X, y)
    xgb_prediction = xgb_model.predict_proba(X)
    xgb_predicted_issue = xgb_model.predict(X)[0]
    xgb_cv_score = cross_val_score(xgb_model, X, y, cv=5).mean()

    # Double Chance (probabilités pour 1X, X2, 12)
    double_chance_1x = logreg_prediction[0][0] + logreg_prediction[0][1]
    double_chance_x2 = logreg_prediction[0][1] + logreg_prediction[0][2]
    double_chance_12 = logreg_prediction[0][0] + logreg_prediction[0][2]

    return {
        'logreg_prediction': logreg_prediction[0],
        'logreg_predicted_issue': logreg_predicted_issue,
        'rf_prediction': rf_prediction[0],
        'rf_predicted_issue': rf_predicted_issue,
        'rf_cv_score': rf_cv_score,
        'xgb_prediction': xgb_prediction[0],
        'xgb_predicted_issue': xgb_predicted_issue,
        'xgb_cv_score': xgb_cv_score,
        'double_chance_1x': double_chance_1x,
        'double_chance_x2': double_chance_x2,
        'double_chance_12': double_chance_12
    }

# Interface utilisateur Streamlit
st.title("Prédiction de résultats de matchs de football")

# Saisie des données des équipes
team1_name = st.text_input("Nom de l'Équipe 1", "Équipe A")
team2_name = st.text_input("Nom de l'Équipe 2", "Équipe B")

# Variables quantitatives pour chaque équipe
attack1 = st.number_input(f"Force d'attaque de {team1_name}", min_value=0, max_value=100, value=50)
defense1 = st.number_input(f"Force de défense de {team1_name}", min_value=0, max_value=100, value=50)
attack2 = st.number_input(f"Force d'attaque de {team2_name}", min_value=0, max_value=100, value=50)
defense2 = st.number_input(f"Force de défense de {team2_name}", min_value=0, max_value=100, value=50)
home_advantage = st.number_input("Avantage à domicile (en %)", min_value=0, max_value=100, value=0)

# Variables supplémentaires que l'utilisateur peut ajouter
team1_goals_scored = st.number_input(f"Buts marqués par {team1_name}", min_value=0, max_value=100, value=1)
team1_goals_conceded = st.number_input(f"Buts encaissés par {team1_name}", min_value=0, max_value=100, value=1)
team2_goals_scored = st.number_input(f"Buts marqués par {team2_name}", min_value=0, max_value=100, value=1)
team2_goals_conceded = st.number_input(f"Buts encaissés par {team2_name}", min_value=0, max_value=100, value=1)

# Rassembler les données des équipes dans des dictionnaires
team1_data = {
    'attack': attack1,
    'defense': defense1,
    'goals_scored': team1_goals_scored,
    'goals_conceded': team1_goals_conceded,
    'home_advantage': home_advantage
}

team2_data = {
    'attack': attack2,
    'defense': defense2,
    'goals_scored': team2_goals_scored,
    'goals_conceded': team2_goals_conceded,
    'home_advantage': home_advantage
}

# Générer les prédictions
if st.button("Prédire le résultat du match"):
    results = generate_predictions(team1_data, team2_data)

    st.write(f"Prédictions pour le match entre {team1_name} et {team2_name} :")
    
    # Affichage des résultats des modèles
    st.write(f"Régression Logistique : {results['logreg_predicted_issue']} avec probabilités {results['logreg_prediction']}")
    st.write(f"Random Forest : {results['rf_predicted_issue']} avec probabilités {results['rf_prediction']}")
    st.write(f"XGBoost : {results['xgb_predicted_issue']} avec probabilités {results['xgb_prediction']}")
    
    # Affichage de la validation croisée pour Random Forest et XGBoost
    st.write(f"Score CV Random Forest : {results['rf_cv_score']}")
    st.write(f"Score CV XGBoost : {results['xgb_cv_score']}")
    
    # Affichage des résultats Double Chance
    st.write(f"Double Chance 1X : {results['double_chance_1x']}")
    st.write(f"Double Chance X2 : {results['double_chance_x2']}")
    st.write(f"Double Chance 12 : {results['double_chance_12']}")
