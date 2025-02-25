import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Fonction pour générer les prédictions
def generate_predictions(team1_data, team2_data):
    # Créer un DataFrame avec les données des deux équipes
    X = pd.DataFrame([team1_data + team2_data], columns=team1_data.keys() + team2_data.keys())
    
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
st.title("Prédiction des Résultats des Matchs de Football")

# Variables modifiables pour l'équipe 1
st.header("Données Équipe 1")
attack1 = st.number_input("Force d'attaque", min_value=0, max_value=100, value=50, key="attack1")
defense1 = st.number_input("Force de défense", min_value=0, max_value=100, value=50, key="defense1")
injuries1 = st.number_input("Blessures", min_value=0, max_value=10, value=0, key="injuries1")
motivation1 = st.number_input("Motivation", min_value=0, max_value=10, value=5, key="motivation1")
tactic1 = st.number_input("Tactique (0-10)", min_value=0, max_value=10, value=5, key="tactic1")
h2h1 = st.number_input("Historique face-à-face", min_value=0, max_value=10, value=5, key="h2h1")
xg1 = st.number_input("xG (Expected Goals)", min_value=0.0, max_value=5.0, value=1.5, key="xg1")
corners1 = st.number_input("Nombre de corners", min_value=0, max_value=10, value=5, key="corners1")
yellow_cards1 = st.number_input("Cartons jaunes", min_value=0, max_value=10, value=2, key="yellow_cards1")
red_cards1 = st.number_input("Cartons rouges", min_value=0, max_value=10, value=0, key="red_cards1")
possession1 = st.number_input("Possession (%)", min_value=0, max_value=100, value=50, key="possession1")
away_advantage1 = st.number_input("Avantage extérieur (0-10)", min_value=0, max_value=10, value=5, key="away_advantage1")
home_advantage1 = st.number_input("Avantage domicile (0-10)", min_value=0, max_value=10, value=5, key="home_advantage1")

# Variables modifiables pour l'équipe 2
st.header("Données Équipe 2")
attack2 = st.number_input("Force d'attaque", min_value=0, max_value=100, value=50, key="attack2")
defense2 = st.number_input("Force de défense", min_value=0, max_value=100, value=50, key="defense2")
injuries2 = st.number_input("Blessures", min_value=0, max_value=10, value=0, key="injuries2")
motivation2 = st.number_input("Motivation", min_value=0, max_value=10, value=5, key="motivation2")
tactic2 = st.number_input("Tactique (0-10)", min_value=0, max_value=10, value=5, key="tactic2")
h2h2 = st.number_input("Historique face-à-face", min_value=0, max_value=10, value=5, key="h2h2")
xg2 = st.number_input("xG (Expected Goals)", min_value=0.0, max_value=5.0, value=1.5, key="xg2")
corners2 = st.number_input("Nombre de corners", min_value=0, max_value=10, value=5, key="corners2")
yellow_cards2 = st.number_input("Cartons jaunes", min_value=0, max_value=10, value=2, key="yellow_cards2")
red_cards2 = st.number_input("Cartons rouges", min_value=0, max_value=10, value=0, key="red_cards2")
possession2 = st.number_input("Possession (%)", min_value=0, max_value=100, value=50, key="possession2")
away_advantage2 = st.number_input("Avantage extérieur (0-10)", min_value=0, max_value=10, value=5, key="away_advantage2")
home_advantage2 = st.number_input("Avantage domicile (0-10)", min_value=0, max_value=10, value=5, key="home_advantage2")

# Données des équipes sous forme de dictionnaires
team1_data = {
    'Force d\'attaque': attack1,
    'Force de défense': defense1,
    'Blessures': injuries1,
    'Motivation': motivation1,
    'Tactique': tactic1,
    'Historique face-à-face': h2h1,
    'xG': xg1,
    'Nombre de corners': corners1,
    'Cartons jaunes': yellow_cards1,
    'Cartons rouges': red_cards1,
    'Possession (%)': possession1,
    'Avantage extérieur': away_advantage1,
    'Avantage domicile': home_advantage1
}

team2_data = {
    'Force d\'attaque': attack2,
    'Force de défense': defense2,
    'Blessures': injuries2,
    'Motivation': motivation2,
    'Tactique': tactic2,
    'Historique face-à-face': h2h2,
    'xG': xg2,
    'Nombre de corners': corners2,
    'Cartons jaunes': yellow_cards2,
    'Cartons rouges': red_cards2,
    'Possession (%)': possession2,
    'Avantage extérieur': away_advantage2,
    'Avantage domicile': home_advantage2
}

# Bouton pour générer les prédictions
if st.button("Générer les Prédictions"):
    results = generate_predictions(team1_data, team2_data)
    
    if results:
        st.write("### Prédictions des Modèles :")
        
        st.write("#### Régression Logistique:")
        st.write(f"Prédiction Issue: {'1' if results['logreg_predicted_issue'] == 0 else 'X' if results['logreg_predicted_issue'] == 1 else '2'}")
        st.write(f"Probabilités : {results['logreg_prediction']}")
        
        st.write("#### Random Forest:")
        st.write(f"Prédiction Issue: {'1' if results['rf_predicted_issue'] == 0 else 'X' if results['rf_predicted_issue'] == 1 else '2'}")
        st.write(f"Probabilités : {results['rf_prediction']}")
        
        st.write("#### XGBoost:")
        st.write(f"Prédiction Issue: {'1' if results['xgb_predicted_issue'] == 0 else 'X' if results['xgb_predicted_issue'] == 1 else '2'}")
        st.write(f"Probabilités : {results['xgb_prediction']}")
        
        st.write("#### Double Chance:")
        st.write(f"1X : {results['double_chance_1x']}")
        st.write(f"X2 : {results['double_chance_x2']}")
        st.write(f"12 : {results['double_chance_12']}")
