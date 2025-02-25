import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy.stats import poisson

# Fonction pour générer les prédictions avec les modèles
def generate_predictions(team1_data, team2_data):
    # Extraire les valeurs et les clés de chaque dictionnaire
    team1_values = list(team1_data.values())
    team2_values = list(team2_data.values())
    columns = list(team1_data.keys()) + list(team2_data.keys())
    
    # Créer un DataFrame avec ces valeurs combinées
    X = pd.DataFrame([team1_values + team2_values], columns=columns)
    
    # Cible fictive (à ajuster selon vos données réelles)
    y = np.array([1])
    
    # 1. Régression Logistique
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)[0]
    
    # 2. Random Forest avec validation croisée K=5
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    rf.fit(X, y)
    rf_prediction = rf.predict(X)[0]
    rf_cv_score = cv_scores_rf.mean()
    
    # 3. XGBoost avec validation croisée K=5
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                  use_label_encoder=False, eval_metric='logloss', random_state=42)
    cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=5)
    xgb_model.fit(X, y)
    xgb_prediction = xgb_model.predict(X)[0]
    xgb_cv_score = cv_scores_xgb.mean()
    
    # 4. Calcul des probabilités Poisson pour les buts marqués (utilise la variable '⚽ Attaque')
    poisson_team1_prob = poisson.pmf(2, team1_data['⚽ Attaque'])
    poisson_team2_prob = poisson.pmf(2, team2_data['⚽ Attaque'])
    
    return (poisson_team1_prob, poisson_team2_prob, logreg_prediction, 
            rf_prediction, rf_cv_score, xgb_prediction, xgb_cv_score)

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
    '🏠 Avantage domicile': home_advantage1,
    '🚶‍♂️ Avantage extérieur': away_advantage1
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
    '🏠 Avantage domicile': home_advantage2,
    '🚶‍♂️ Avantage extérieur': away_advantage2
}

# Génération des prédictions
results = generate_predictions(team1_data, team2_data)
(poisson_team1_prob, poisson_team2_prob, logreg_prediction, rf_prediction, rf_cv_score, xgb_prediction) = results

# Affichage des résultats
st.write("### Résultats des Prédictions:")

st.write(f"⚽ **Probabilité Poisson pour {team1_name}** : {poisson_team1_prob:.4f}")
st.write(f"⚽ **Probabilité Poisson pour {team2_name}** : {poisson_team2_prob:.4f}")

st.write(f"📊 **Prédiction de la régression logistique** : {'Victoire ' + team1_name if logreg_prediction == 1 else 'Victoire ' + team2_name}")
st.write(f"🌳 **Prédiction de Random Forest** : {'Victoire ' + team1_name if rf_prediction == 1 else 'Victoire ' + team2_name}")
st.write(f"🌟 **CV Score moyen (Random Forest)** : {rf_cv_score:.2f}")
st.write(f"🚀 **Prédiction de XGBoost** : {'Victoire ' + team1_name if xgb_prediction == 1 else 'Victoire ' + team2_name}")

# Option de téléchargement des résultats (exemple de texte)
download_text = f"""
Résultats du match : {team1_name} vs {team2_name}

Variables Équipe 1 :
{team1_data}

Variables Équipe 2 :
{team2_data}

Prédictions :
- Poisson {team1_name} (2 buts) : {poisson_team1_prob:.4f}
- Poisson {team2_name} (2 buts) : {poisson_team2_prob:.4f}
- Régression Logistique : {'Victoire ' + team1_name if logreg_prediction == 1 else 'Victoire ' + team2_name}
- Random Forest : {'Victoire ' + team1_name if rf_prediction == 1 else 'Victoire ' + team2_name}
- XGBoost : {'Victoire ' + team1_name if xgb_prediction == 1 else 'Victoire ' + team2_name}

CV Score (RF) : {rf_cv_score:.2f}
"""

st.download_button(
    label="📥 Télécharger les résultats",
    data=download_text,
    file_name="predictions.txt",
    mime="text/plain"
)
