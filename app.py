import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Fonction pour générer les prédictions avec les modèles : Régression Logistique, Random Forest, XGBoost
def generate_predictions(team1_data, team2_data):
    # Extraire les valeurs et les clés de chaque dictionnaire
    team1_values = list(team1_data.values())
    team2_values = list(team2_data.values())
    columns = list(team1_data.keys()) + list(team2_data.keys())
    
    # Créer un DataFrame avec ces valeurs combinées
    X = pd.DataFrame([team1_values + team2_values], columns=columns)
    
    # Cible : 1 = victoire équipe 1, X = match nul, 2 = victoire équipe 2
    # Pour simplifier, je vais faire un exemple avec une cible binaire, mais en réalité il vous faut des étiquettes pour chaque issue (1, X, 2, 1X, X2, 12)
    y = np.array([1])  # Exemple fictif de cible, vous devez remplacer ceci par des données réelles
    
    # Régression Logistique
    logreg = LogisticRegression(max_iter=10000, multi_class='ovr')
    logreg.fit(X, y)
    logreg_prediction = logreg.predict_proba(X)[0]  # Renvoie la probabilité pour chaque classe
    
    # Random Forest avec validation croisée K=5
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    rf.fit(X, y)
    rf_prediction = rf.predict_proba(X)[0]
    rf_cv_score = cv_scores_rf.mean()
    
    # XGBoost avec validation croisée K=5
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, 
                                  use_label_encoder=False, eval_metric='logloss', random_state=42)
    cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=5)
    xgb_model.fit(X, y)
    xgb_prediction = xgb_model.predict_proba(X)[0]
    xgb_cv_score = cv_scores_xgb.mean()
    
    # Prédictions des issues les plus probables
    logreg_predicted_issue = np.argmax(logreg_prediction)
    rf_predicted_issue = np.argmax(rf_prediction)
    xgb_predicted_issue = np.argmax(xgb_prediction)
    
    # Option Double Chance (1X, X2, 12)
    double_chance_1x = logreg_prediction[0] + logreg_prediction[1]  # 1X
    double_chance_x2 = logreg_prediction[1] + logreg_prediction[2]  # X2
    double_chance_12 = logreg_prediction[0] + logreg_prediction[2]  # 12
    
    # Retourner les résultats
    return {
        'logreg_predicted_issue': logreg_predicted_issue,
        'rf_predicted_issue': rf_predicted_issue,
        'xgb_predicted_issue': xgb_predicted_issue,
        'logreg_prediction': logreg_prediction,
        'rf_prediction': rf_prediction,
        'xgb_prediction': xgb_prediction,
        'double_chance_1x': double_chance_1x,
        'double_chance_x2': double_chance_x2,
        'double_chance_12': double_chance_12,
        'rf_cv_score': rf_cv_score,
        'xgb_cv_score': xgb_cv_score
    }

# Interface utilisateur Streamlit pour saisir les données des équipes
st.title("Analyse des Paris Sportifs ⚽")

# Saisie des données pour l'équipe 1
st.write("### Données pour l'Équipe 1:")
team1_name = st.text_input("Nom de l'Équipe 1", "Équipe 1")
attack1 = st.slider("Force d'attaque Équipe 1", 0.0, 5.0, 1.2)
defense1 = st.slider("Force de défense Équipe 1", 0.0, 5.0, 1.0)
recent_form1 = st.slider("Forme récente Équipe 1 (0 à 5)", 0.0, 5.0, 3.5)
injuries1 = st.slider("Blessures Équipe 1 (0 à 5)", 0.0, 5.0, 1.5)
motivation1 = st.slider("Motivation Équipe 1 (0 à 5)", 0.0, 5.0, 4.0)
tactic1 = st.slider("Tactique Équipe 1 (0 à 5)", 0.0, 5.0, 3.5)
h2h1 = st.slider("Historique face-à-face Équipe 1 (0 à 5)", 0.0, 5.0, 3.0)
xg1 = st.slider("xG Équipe 1", 0.0, 5.0, 1.4)
corners1 = st.slider("Nombre de corners Équipe 1", 0, 20, 5)
yellow_cards1 = st.slider("Cartons jaunes Équipe 1", 0, 10, 2)
red_cards1 = st.slider("Cartons rouges Équipe 1", 0, 5, 0)
possession1 = st.slider("Possession (%) Équipe 1", 0.0, 100.0, 55.0)
away_advantage1 = st.slider("Avantage extérieur Équipe 1 (0 à 5)", 0.0, 5.0, 2.0)
home_advantage1 = st.slider("Avantage domicile Équipe 1 (0 à 5)", 0.0, 5.0, 3.5)

# Saisie des données pour l'équipe 2
st.write("### Données pour l'Équipe 2:")
team2_name = st.text_input("Nom de l'Équipe 2", "Équipe 2")
attack2 = st.slider("Force d'attaque Équipe 2", 0.0, 5.0, 0.8)
defense2 = st.slider("Force de défense Équipe 2", 0.0, 5.0, 1.0)
recent_form2 = st.slider("Forme récente Équipe 2 (0 à 5)", 0.0, 5.0, 3.0)
injuries2 = st.slider("Blessures Équipe 2 (0 à 5)", 0.0, 5.0, 2.0)
motivation2 = st.slider("Motivation Équipe 2 (0 à 5)", 0.0, 5.0, 3.5)
tactic2 = st.slider("Tactique Équipe 2 (0 à 5)", 0.0, 5.0, 3.0)
h2h2 = st.slider("Historique face-à-face Équipe 2 (0 à 5)", 0.0, 5.0, 2.5)
xg2 = st.slider("xG Équipe 2", 0.0, 5.0, 1.1)
corners2 = st.slider("Nombre de corners Équipe 2", 0, 20, 6)
yellow_cards2 = st.slider("Cartons jaunes Équipe 2", 0, 10, 3)
red_cards2 = st.slider("Cartons rouges Équipe 2", 0, 5, 1)
possession2 = st.slider("Possession (%) Équipe 2", 0.0, 100.0, 45.0)
away_advantage2 = st.slider("Avantage extérieur Équipe 2 (0 à 5)", 0.0, 5.0, 1.5)
home_advantage2 = st.slider("Avantage domicile Équipe 2 (0 à 5)", 0.0, 5.0, 2.5)

# Constitution des dictionnaires pour chaque équipe
team1_data = {
    '🧑‍💼 Nom de l\'équipe': team1_name,
    '⚽ Attaque': attack1,
    '🛡️ Défense': defense1,
    '🔥 Forme récente': recent_form1,
    '💔 Blessures': injuries1,
    '💪 Motivation': motivation1,
    '🔄 Tactique': tactic1,
    '📊 Historique face-à-face': h2h1,
    '⚽xG': xg1,
    '🔝 Nombre de corners': corners1,
    '🟨 Cartons jaunes': yellow_cards1,
    '🟥 Cartons rouges': red_cards1,
    '📉 Possession (%)': possession1,
    '⚔️ Avantage extérieur': away_advantage1,
    '🏠 Avantage domicile': home_advantage1
}

team2_data = {
    '🧑‍💼 Nom de l\'équipe': team2_name,
    '⚽ Attaque': attack2,
    '🛡️ Défense': defense2,
    '🔥 Forme récente': recent_form2,
    '💔 Blessures': injuries2,
    '💪 Motivation': motivation2,
    '🔄 Tactique': tactic2,
    '📊 Historique face-à-face': h2h2,
    '⚽xG': xg2,
    '🔝 Nombre de corners': corners2,
    '🟨 Cartons jaunes': yellow_cards2,
    '🟥 Cartons rouges': red_cards2,
    '📉 Possession (%)': possession2,
    '⚔️ Avantage extérieur': away_advantage2,
    '🏠 Avantage domicile': home_advantage2
}

# Calculer les prédictions
results = generate_predictions(team1_data, team2_data)

# Afficher les résultats
st.write("### Prédictions :")
st.write(f"**Régression Logistique - Issue la plus probable** : {results['logreg_predicted_issue']}")
st.write(f"**Random Forest - Issue la plus probable** : {results['rf_predicted_issue']}")
st.write(f"**XGBoost - Issue la plus probable** : {results['xgb_predicted_issue']}")

st.write(f"**Double Chance 1X** : {results['double_chance_1x']}")
st.write(f"**Double Chance X2** : {results['double_chance_x2']}")
st.write(f"**Double Chance 12** : {results['double_chance_12']}")

st.write(f"**Validation croisée RF** : {results['rf
