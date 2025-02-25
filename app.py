import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import poisson

# ---------------------- Fonction pour calculer les forces ----------------------
def calculate_forces(buts_marques, buts_subis, matchs_joues):
    if matchs_joues == 0:  
        return 0, 0  
    attack_strength = buts_marques / matchs_joues
    defense_strength = buts_subis / matchs_joues
    return attack_strength, defense_strength

# ---------------------- Fonction de conversion des cotes ----------------------
def convert_odds_to_prob(odds):
    return 1 / odds if odds > 1 else 0

# ---------------------- Fonction pour générer les prédictions ----------------------
def generate_predictions(team1_data, team2_data):
    # Transformation des données en DataFrame
    X = pd.DataFrame([team1_data + team2_data])
    y = np.array([1])  # Label fictif (à remplacer avec des labels réels)

    # Modèle de régression logistique
    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)[0]

    # Modèle Random Forest
    rf = RandomForestClassifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_rf = cross_val_score(rf, X, y, cv=skf)

    # Probabilités Poisson
    poisson_team1_prob = poisson.pmf(2, team1_data[0])  # "Buts marqués"
    poisson_team2_prob = poisson.pmf(2, team2_data[0])

    return poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf.mean()

# ---------------------- Interface Streamlit ----------------------
st.title("🔍 Analyse et Prédiction des Matchs de Football ⚽")

# Variables des équipes (12+ par équipe)
variables = [
    "Buts marqués", "Buts subis", "xG", "xGA", "Possession moyenne (%)", "Passes réussies (%)",
    "Balles touchées dans la surface adverse", "Tirs cadrés", "Corners obtenus", "Forme récente",
    "Blessures clés", "Motivation", "Historique face à face", "Tactique"
]

# Input utilisateur
st.sidebar.header("📋 Saisir les statistiques des équipes")
team1_name = st.sidebar.text_input("Nom de l'équipe 1", "Équipe A")
team2_name = st.sidebar.text_input("Nom de l'équipe 2", "Équipe B")

team1_data = [st.sidebar.number_input(f"{var} (Équipe A)", value=0.0) for var in variables]
team2_data = [st.sidebar.number_input(f"{var} (Équipe B)", value=0.0) for var in variables]

# Calcul des forces d'attaque et de défense
attack_strength_1, defense_strength_1 = calculate_forces(team1_data[0], team1_data[1], 10)
attack_strength_2, defense_strength_2 = calculate_forces(team2_data[0], team2_data[1], 10)

# Bouton de prédiction
if st.sidebar.button("🔮 Générer la prédiction"):
    poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_rf = generate_predictions(team1_data, team2_data)

    # Affichage des résultats
    st.subheader(f"📊 Comparaison : {team1_name} vs {team2_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**⚽ {team1_name}**")
        for i, var in enumerate(variables):
            st.write(f"• {var} : {team1_data[i]}")

        st.write(f"**💪 Force d'attaque :** {attack_strength_1:.2f}")
        st.write(f"**🛡️ Force de défense :** {defense_strength_1:.2f}")

    with col2:
        st.write(f"**⚽ {team2_name}**")
        for i, var in enumerate(variables):
            st.write(f"• {var} : {team2_data[i]}")

        st.write(f"**💪 Force d'attaque :** {attack_strength_2:.2f}")
        st.write(f"**🛡️ Force de défense :** {defense_strength_2:.2f}")

    # Résultats des modèles
    st.write("### 🔥 Prédictions des Modèles")
    st.write(f"⚽ **Poisson {team1_name} :** {poisson_team1_prob:.2f}")
    st.write(f"⚽ **Poisson {team2_name} :** {poisson_team2_prob:.2f}")
    st.write(f"🧠 **Régression Logistique :** {logreg_prediction}")
    st.write(f"🌳 **Random Forest (CrossVal 5) :** {cv_rf:.2f}")

    # Détection de value bet
    st.subheader("🎯 Détection de Value Bet")
    odds_team1 = st.number_input(f"Cote {team1_name}", min_value=1.01, step=0.01)
    odds_team2 = st.number_input(f"Cote {team2_name}", min_value=1.01, step=0.01)
    odds_draw = st.number_input("Cote Match Nul", min_value=1.01, step=0.01)

    prob_team1 = convert_odds_to_prob(odds_team1)
    prob_team2 = convert_odds_to_prob(odds_team2)
    prob_draw = convert_odds_to_prob(odds_draw)

    value_bet_team1 = prob_team1 > poisson_team1_prob
    value_bet_team2 = prob_team2 > poisson_team2_prob
    value_bet_draw = prob_draw > (poisson_team1_prob + poisson_team2_prob) / 2

    st.write(f"💰 **Value Bet {team1_name} :** {'✅ Oui' if value_bet_team1 else '❌ Non'}")
    st.write(f"💰 **Value Bet {team2_name} :** {'✅ Oui' if value_bet_team2 else '❌ Non'}")
    st.write(f"💰 **Value Bet Match Nul :** {'✅ Oui' if value_bet_draw else '❌ Non'}")

    # Option de téléchargement des résultats
    data_to_download = f"""
    **Match :** {team1_name} vs {team2_name}
    
    **Prédictions :**
    - Poisson {team1_name} : {poisson_team1_prob:.2f}
    - Poisson {team2_name} : {poisson_team2_prob:.2f}
    - Régression Logistique : {logreg_prediction}
    - Random Forest (CrossVal 5) : {cv_rf:.2f}

    **Value Bets :**
    - {team1_name} : {'✅ Oui' if value_bet_team1 else '❌ Non'}
    - {team2_name} : {'✅ Oui' if value_bet_team2 else '❌ Non'}
    - Match Nul : {'✅ Oui' if value_bet_draw else '❌ Non'}
    """

    st.download_button(
        label="📥 Télécharger les résultats",
        data=data_to_download,
        file_name="prediction_match.docx",
        mime="text/plain"
    )
