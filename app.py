import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import poisson
import xgboost as xgb

st.set_page_config(page_title="Prédictions de Paris Sportifs", layout="centered")

# --- Fonction de génération des prédictions ---
def generate_predictions(team1_data, team2_data):
    # Fusionner les données des deux équipes dans un DataFrame
    team1_values = list(team1_data.values())
    team2_values = list(team2_data.values())
    columns = list(team1_data.keys()) + list(team2_data.keys())
    X = pd.DataFrame([team1_values + team2_values], columns=columns)
    
    # Cible fictive (à remplacer par une vraie cible si vous disposez d'un dataset d'entraînement)
    y = np.array([1])
    
    # 1. Régression Logistique
    logreg = LogisticRegression(max_iter=10000)
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)
    
    # 2. Random Forest avec validation croisée K=10
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores_rf = cross_val_score(rf, X, y, cv=10)
    rf.fit(X, y)
    rf_prediction = rf.predict(X)
    
    # 3. XGBoost avec validation croisée K=10
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, 
                                  use_label_encoder=False, eval_metric='logloss', random_state=42)
    cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=10)
    xgb_model.fit(X, y)
    xgb_prediction = xgb_model.predict(X)
    
    # 4. Modèle Poisson :
    # On utilisera ici la valeur de "⚽xG" comme proxy pour l'expected goals (λ)
    poisson_team1_prob = poisson.pmf(2, team1_data['⚽xG'])  # Probabilité de marquer exactement 2 buts
    poisson_team2_prob = poisson.pmf(2, team2_data['⚽xG'])
    
    return (poisson_team1_prob, poisson_team2_prob, 
            logreg_prediction, rf_prediction, xgb_prediction, 
            cv_scores_rf.mean(), cv_scores_xgb.mean())

# --- Interface Utilisateur ---
st.title("🔍 Prédictions de Paris Sportifs")

st.sidebar.header("📋 Saisir les statistiques des équipes")

# Saisie du nom des équipes
team1_name = st.sidebar.text_input("Nom de l'Équipe 1", "Équipe A")
team2_name = st.sidebar.text_input("Nom de l'Équipe 2", "Équipe B")

# Variables (12 par équipe)
variables = {
    '⚽ Attaque': 1.0,
    '🛡️ Défense': 1.0,
    '🔥 Forme récente': 1.0,
    '💔 Blessures': 1.0,
    '💪 Motivation': 1.0,
    '🔄 Tactique': 1.0,
    '📊 Historique face à face': 1.0,
    '⚽xG': 1.0,
    '🔝 Nombre de corners': 1.0,
    '🏠 Buts à domicile': 1.0,
    '✈️ Buts à l\'extérieur': 1.0,
    '📈 Possession moyenne (%)': 1.0
}

st.sidebar.write("### Variables pour Équipe 1")
team1_data = {}
for var, default in variables.items():
    team1_data[var] = st.sidebar.number_input(f"{var} (Équipe 1)", value=default, format="%.2f")

st.sidebar.write("### Variables pour Équipe 2")
team2_data = {}
for var, default in variables.items():
    team2_data[var] = st.sidebar.number_input(f"{var} (Équipe 2)", value=default, format="%.2f")

# Pour la fonction Poisson, nous avons besoin des données sur les buts marqués, matchs joués et buts subis.
# Nous allons ajouter ces valeurs fictives (elles peuvent être saisies ou calculées à partir d'autres variables).
if 'buts_marques' not in team1_data:
    team1_data['buts_marques'] = st.sidebar.number_input("Buts marqués (Équipe 1)", value=30, step=1)
if 'matchs_joues' not in team1_data:
    team1_data['matchs_joues'] = st.sidebar.number_input("Matchs joués (Équipe 1)", value=15, step=1)
if 'buts_subis' not in team1_data:
    team1_data['buts_subis'] = st.sidebar.number_input("Buts subis (Équipe 1)", value=18, step=1)

if 'buts_marques' not in team2_data:
    team2_data['buts_marques'] = st.sidebar.number_input("Buts marqués (Équipe 2)", value=25, step=1)
if 'matchs_joues' not in team2_data:
    team2_data['matchs_joues'] = st.sidebar.number_input("Matchs joués (Équipe 2)", value=14, step=1)
if 'buts_subis' not in team2_data:
    team2_data['buts_subis'] = st.sidebar.number_input("Buts subis (Équipe 2)", value=20, step=1)

# Bouton pour lancer la prédiction
if st.sidebar.button("🔮 Générer les prédictions"):
    results = generate_predictions(team1_data, team2_data)
    (poisson_team1_prob, poisson_team2_prob, logreg_pred, rf_pred, xgb_pred, cv_rf_mean, cv_xgb_mean) = results
    
    st.subheader(f"Résultats : {team1_name} vs {team2_name}")
    
    st.write("### Prédictions des modèles :")
    st.write(f"⚽ Probabilité Poisson (2 buts) pour {team1_name} : {poisson_team1_prob:.2f}")
    st.write(f"⚽ Probabilité Poisson (2 buts) pour {team2_name} : {poisson_team2_prob:.2f}")
    st.write(f"📊 Régression Logistique : {'Victoire ' + team1_name if logreg_pred[0] == 1 else 'Victoire ' + team2_name}")
    st.write(f"🌳 Random Forest : {'Victoire ' + team1_name if rf_pred[0] == 1 else 'Victoire ' + team2_name}")
    st.write(f"🚀 XGBoost : {'Victoire ' + team1_name if xgb_pred[0] == 1 else 'Victoire ' + team2_name}")
    st.write(f"📊 CV Score moyen (Random Forest) : {cv_rf_mean:.2f}")
    st.write(f"📊 CV Score moyen (XGBoost) : {cv_xgb_mean:.2f}")
    
    # Option de téléchargement des résultats (exemple texte)
    download_text = f"""
    Résultats du match : {team1_name} vs {team2_name}

    Variables Équipe 1 :
    {team1_data}

    Variables Équipe 2 :
    {team2_data}

    Prédictions :
    - Poisson {team1_name} (2 buts) : {poisson_team1_prob:.2f}
    - Poisson {team2_name} (2 buts) : {poisson_team2_prob:.2f}
    - Régression Logistique : {'Victoire ' + team1_name if logreg_pred[0] == 1 else 'Victoire ' + team2_name}
    - Random Forest : {'Victoire ' + team1_name if rf_pred[0] == 1 else 'Victoire ' + team2_name}
    - XGBoost : {'Victoire ' + team1_name if xgb_pred[0] == 1 else 'Victoire ' + team2_name}

    CV Score (RF) : {cv_rf_mean:.2f}
    CV Score (XGBoost) : {cv_xgb_mean:.2f}
    """
    st.download_button(
        label="📥 Télécharger les résultats",
        data=download_text,
        file_name="predictions.txt",
        mime="text/plain"
    )
