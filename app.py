import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import poisson

# ---------------------- Initialisation de la persistance ----------------------
if "team1_data" not in st.session_state:
    st.session_state.team1_data = [0.0] * 14
if "team2_data" not in st.session_state:
    st.session_state.team2_data = [0.0] * 14
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# ---------------------- Fonction pour calculer les forces ----------------------
def calculate_forces(buts_marques, buts_subis, matchs_joues):
    if matchs_joues == 0:
        return 0, 0
    return buts_marques / matchs_joues, buts_subis / matchs_joues

# ---------------------- Fonction de conversion des cotes ----------------------
def convert_odds_to_prob(odds):
    return 1 / odds if odds > 1 else 0

# ---------------------- Fonction pour générer les prédictions ----------------------
def generate_predictions(team1_data, team2_data):
    X = pd.DataFrame([team1_data + team2_data])
    y = np.array([1])

    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)[0]

    rf = RandomForestClassifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_rf = cross_val_score(rf, X, y, cv=skf)

    poisson_team1_prob = poisson.pmf(2, team1_data[0])
    poisson_team2_prob = poisson.pmf(2, team2_data[0])

    return poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf.mean()

# ---------------------- Interface Streamlit ----------------------
st.title("🔍 Analyse et Prédiction des Matchs de Football ⚽")

variables = [
    "Buts marqués", "Buts subis", "xG", "xGA", "Possession moyenne (%)", "Passes réussies (%)",
    "Balles touchées dans la surface adverse", "Tirs cadrés", "Corners obtenus", "Forme récente",
    "Blessures clés", "Motivation", "Historique face à face", "Tactique"
]

# Saisie utilisateur avec persistance
st.sidebar.header("📋 Saisir les statistiques des équipes")
team1_name = st.sidebar.text_input("Nom de l'équipe 1", "Équipe A")
team2_name = st.sidebar.text_input("Nom de l'équipe 2", "Équipe B")

st.sidebar.write("### 🔵 Statistiques Équipe 1")
for i, var in enumerate(variables):
    st.session_state.team1_data[i] = st.sidebar.number_input(f"{var} (Équipe A)", value=st.session_state.team1_data[i])

st.sidebar.write("### 🔴 Statistiques Équipe 2")
for i, var in enumerate(variables):
    st.session_state.team2_data[i] = st.sidebar.number_input(f"{var} (Équipe B)", value=st.session_state.team2_data[i])

# Calcul des forces d'attaque et de défense
attack_strength_1, defense_strength_1 = calculate_forces(st.session_state.team1_data[0], st.session_state.team1_data[1], 10)
attack_strength_2, defense_strength_2 = calculate_forces(st.session_state.team2_data[0], st.session_state.team2_data[1], 10)

# Bouton de prédiction
if st.sidebar.button("🔮 Générer la prédiction"):
    st.session_state.predictions = {}
    st.session_state.predictions["poisson_team1"], st.session_state.predictions["poisson_team2"], \
    st.session_state.predictions["logreg"], st.session_state.predictions["cv_rf"] = generate_predictions(
        st.session_state.team1_data, st.session_state.team2_data
    )

# Affichage des résultats avec persistance
if st.session_state.predictions:
    st.subheader(f"📊 Comparaison : {team1_name} vs {team2_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**⚽ {team1_name}**")
        for i, var in enumerate(variables):
            st.write(f"• {var} : {st.session_state.team1_data[i]}")
        st.write(f"**💪 Force d'attaque :** {attack_strength_1:.2f}")
        st.write(f"**🛡️ Force de défense :** {defense_strength_1:.2f}")

    with col2:
        st.write(f"**⚽ {team2_name}**")
        for i, var in enumerate(variables):
            st.write(f"• {var} : {st.session_state.team2_data[i]}")
        st.write(f"**💪 Force d'attaque :** {attack_strength_2:.2f}")
        st.write(f"**🛡️ Force de défense :** {defense_strength_2:.2f}")

    st.write("### 🔥 Prédictions des Modèles")
    st.write(f"⚽ **Poisson {team1_name} :** {st.session_state.predictions['poisson_team1']:.2f}")
    st.write(f"⚽ **Poisson {team2_name} :** {st.session_state.predictions['poisson_team2']:.2f}")
    st.write(f"🧠 **Régression Logistique :** {st.session_state.predictions['logreg']}")
    st.write(f"🌳 **Random Forest (CrossVal 5) :** {st.session_state.predictions['cv_rf']:.2f}")

    # Détection de value bet
    st.subheader("🎯 Détection de Value Bet")
    odds_team1 = st.number_input(f"Cote {team1_name}", min_value=1.01, step=0.01)
    odds_team2 = st.number_input(f"Cote {team2_name}", min_value=1.01, step=0.01)
    odds_draw = st.number_input("Cote Match Nul", min_value=1.01, step=0.01)

    prob_team1 = convert_odds_to_prob(odds_team1)
    prob_team2 = convert_odds_to_prob(odds_team2)
    prob_draw = convert_odds_to_prob(odds_draw)

    value_bet_team1 = prob_team1 > st.session_state.predictions["poisson_team1"]
    value_bet_team2 = prob_team2 > st.session_state.predictions["poisson_team2"]
    value_bet_draw = prob_draw > (st.session_state.predictions["poisson_team1"] + st.session_state.predictions["poisson_team2"]) / 2

    st.write(f"💰 **Value Bet {team1_name} :** {'✅ Oui' if value_bet_team1 else '❌ Non'}")
    st.write(f"💰 **Value Bet {team2_name} :** {'✅ Oui' if value_bet_team2 else '❌ Non'}")
    st.write(f"💰 **Value Bet Match Nul :** {'✅ Oui' if value_bet_draw else '❌ Non'}")
