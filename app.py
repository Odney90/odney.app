import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Collecte des données des équipes
team1_name = st.text_input("Nom de l'équipe 1", "Équipe 1")
team2_name = st.text_input("Nom de l'équipe 2", "Équipe 2")

# Variables pour l'équipe 1
attack1 = st.number_input(f"Force d'attaque de {team1_name}", min_value=0, max_value=100, value=50)
defense1 = st.number_input(f"Force de défense de {team1_name}", min_value=0, max_value=100, value=50)
goals_scored1 = st.number_input(f"Buts marqués par {team1_name}", min_value=0, max_value=100, value=1)
goals_conceded1 = st.number_input(f"Buts encaissés par {team1_name}", min_value=0, max_value=100, value=1)
shots1 = st.number_input(f"Tirs de {team1_name}", min_value=0, max_value=100, value=10)
shots_on_target1 = st.number_input(f"Tirs cadrés de {team1_name}", min_value=0, max_value=100, value=5)
pass_success1 = st.number_input(f"Passes réussies par {team1_name}", min_value=0, max_value=100, value=50)
fouls1 = st.number_input(f"Fautes commises par {team1_name}", min_value=0, max_value=100, value=5)
yellow_cards1 = st.number_input(f"Cartons jaunes reçus par {team1_name}", min_value=0, max_value=100, value=1)
red_cards1 = st.number_input(f"Cartons rouges reçus par {team1_name}", min_value=0, max_value=100, value=0)
home_advantage1 = st.number_input(f"Avantage à domicile pour {team1_name} (%)", min_value=0, max_value=100, value=50)

# Forme actuelle de l'équipe 1 (qualitative)
form1 = st.selectbox(f"Forme actuelle de {team1_name} (1: Mauvaise, 4: Excellente)", [1, 2, 3, 4])

# Conditions météorologiques pour l'équipe 1
weather1 = st.selectbox(f"Conditions météorologiques pour {team1_name} (0: Pas de pluie, 1: Pluie)", [0, 1])

# Absences pour l'équipe 1 (qualitative)
absences1 = st.number_input(f"Absences de joueurs clés de {team1_name} (0: Pas d'absent, 100: Tous absents)", min_value=0, max_value=100, value=0)

# Variables pour l'équipe 2
attack2 = st.number_input(f"Force d'attaque de {team2_name}", min_value=0, max_value=100, value=50)
defense2 = st.number_input(f"Force de défense de {team2_name}", min_value=0, max_value=100, value=50)
goals_scored2 = st.number_input(f"Buts marqués par {team2_name}", min_value=0, max_value=100, value=1)
goals_conceded2 = st.number_input(f"Buts encaissés par {team2_name}", min_value=0, max_value=100, value=1)
shots2 = st.number_input(f"Tirs de {team2_name}", min_value=0, max_value=100, value=10)
shots_on_target2 = st.number_input(f"Tirs cadrés de {team2_name}", min_value=0, max_value=100, value=5)
pass_success2 = st.number_input(f"Passes réussies par {team2_name}", min_value=0, max_value=100, value=50)
fouls2 = st.number_input(f"Fautes commises par {team2_name}", min_value=0, max_value=100, value=5)
yellow_cards2 = st.number_input(f"Cartons jaunes reçus par {team2_name}", min_value=0, max_value=100, value=1)
red_cards2 = st.number_input(f"Cartons rouges reçus par {team2_name}", min_value=0, max_value=100, value=0)
home_advantage2 = st.number_input(f"Avantage à domicile pour {team2_name} (%)", min_value=0, max_value=100, value=50)

# Forme actuelle de l'équipe 2 (qualitative)
form2 = st.selectbox(f"Forme actuelle de {team2_name} (1: Mauvaise, 4: Excellente)", [1, 2, 3, 4])

# Conditions météorologiques pour l'équipe 2
weather2 = st.selectbox(f"Conditions météorologiques pour {team2_name} (0: Pas de pluie, 1: Pluie)", [0, 1])

# Absences pour l'équipe 2 (qualitative)
absences2 = st.number_input(f"Absences de joueurs clés de {team2_name} (0: Pas d'absent, 100: Tous absents)", min_value=0, max_value=100, value=0)

# Fonction de prédiction
def generate_predictions(team1_data, team2_data):
    # Fusionner les données des deux équipes
    team1_data_prefixed = {f"team1_{key}": value for key, value in team1_data.items()}
    team2_data_prefixed = {f"team2_{key}": value for key, value in team2_data.items()}
    combined_data = {**team1_data_prefixed, **team2_data_prefixed}
    
    X = pd.DataFrame([combined_data], columns=list(combined_data.keys()))
    
    # Prédiction avec Logistic Regression, Random Forest, XGBoost
    y = [0]  # Remplacer par des données réelles pour l'issue du match (0: match nul, 1: victoire équipe 1, 2: victoire équipe 2)
    
    logreg = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=100)
    xgb_model = xgb.XGBClassifier()
    
    logreg.fit(X, y)
    rf.fit(X, y)
    xgb_model.fit(X, y)
    
    logreg_pred = logreg.predict(X)[0]
    rf_pred = rf.predict(X)[0]
    xgb_pred = xgb_model.predict(X)[0]
    
    return logreg_pred, rf_pred, xgb_pred

# Affichage des résultats
if st.button("Prédire le résultat du match"):
    logreg_pred, rf_pred, xgb_pred = generate_predictions(team1_data, team2_data)
    st.write(f"Prédiction avec Logistic Regression: {logreg_pred}")
    st.write(f"Prédiction avec Random Forest: {rf_pred}")
    st.write(f"Prédiction avec XGBoost: {xgb_pred}")
