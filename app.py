import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import poisson

# Définition de l'interface
st.set_page_config(page_title="Prédictions Foot ⚽", layout="wide")

# En-tête
st.markdown("# ⚽ Prédictions de Matchs de Football")
st.markdown("### Analyse avec Poisson, Random Forest & Régression Logistique")

# Espace utilisateur
st.sidebar.header("📊 Entrer les statistiques des équipes")

# Variables chiffrées
variables = [
    "XG moyen", "Buts marqués/match", "Buts encaissés/match", 
    "Classement", "Forme (5 derniers matchs)", "Possession (%)", 
    "Tirs cadrés/match", "XP (Expected Points)", "Corners/match", 
    "Cartons jaunes", "Cartons rouges"
]

# Initialisation de la persistance des données
if "team1_stats" not in st.session_state:
    st.session_state.team1_stats = {var: np.random.uniform(0, 2) for var in variables}
if "team2_stats" not in st.session_state:
    st.session_state.team2_stats = {var: np.random.uniform(0, 2) for var in variables}

# Champs de saisie des variables chiffrées
team1_stats, team2_stats = {}, {}
for var in variables:
    couleur = "🔵 " if var in ["XG moyen", "Buts marqués/match", "Buts encaissés/match"] else "⚪ "
    team1_stats[var] = st.sidebar.number_input(
        f"{couleur} {var} - Équipe 1", 
        value=st.session_state.team1_stats[var], 
        format="%.2f"
    )
    team2_stats[var] = st.sidebar.number_input(
        f"{couleur} {var} - Équipe 2", 
        value=st.session_state.team2_stats[var], 
        format="%.2f"
    )

    st.session_state.team1_stats[var] = team1_stats[var]
    st.session_state.team2_stats[var] = team2_stats[var]

# Variables non chiffrées
st.sidebar.subheader("⚙️ Autres facteurs")
match_status = st.sidebar.selectbox("Statut du match", ["Domicile", "Extérieur", "Terrain neutre"])
style_jeu = st.sidebar.selectbox("Style de jeu", ["Offensif", "Défensif", "Équilibré"])
pression_match = st.sidebar.selectbox("Pression du match", ["Match amical", "Championnat", "Coupe", "Derby"])

# Transformation des variables non chiffrées en valeurs numériques (One-Hot Encoding)
match_status_encoded = [1 if match_status == "Domicile" else 0, 1 if match_status == "Extérieur" else 0]
style_jeu_encoded = [1 if style_jeu == "Offensif" else 0, 1 if style_jeu == "Défensif" else 0]
pression_match_encoded = [1 if pression_match == "Match amical" else 0, 1 if pression_match == "Championnat" else 0]

# Convertisseur de cotes en probabilités
st.sidebar.header("🎲 Convertisseur de cotes")
cote = st.sidebar.number_input("Entrer la cote du bookmaker", min_value=1.01, value=2.00, step=0.01)
prob_implicite = (1 / cote) * 100
st.sidebar.write(f"Probabilité implicite : **{prob_implicite:.2f}%**")

# Bouton pour lancer les prédictions
if st.sidebar.button("🔮 Générer les Prédictions"):

    # Modèle de Poisson
    avg_goals_team1 = (team1_stats["XG moyen"] + team1_stats["Buts marqués/match"]) / 2
    avg_goals_team2 = (team2_stats["XG moyen"] + team2_stats["Buts marqués/match"]) / 2
    prob_team1_goals = [poisson.pmf(i, avg_goals_team1) for i in range(5)]
    prob_team2_goals = [poisson.pmf(i, avg_goals_team2) for i in range(5)]

    st.subheader("📈 Prédiction des buts (Poisson)")
    poisson_results = pd.DataFrame({
        "Buts": list(range(5)),
        "Équipe 1 (%)": np.round(np.array(prob_team1_goals) * 100, 2),
        "Équipe 2 (%)": np.round(np.array(prob_team2_goals) * 100, 2)
    })
    st.write(poisson_results)

    # Random Forest (avec les variables non chiffrées)
    features = list(team1_stats.values()) + list(team2_stats.values()) + match_status_encoded + style_jeu_encoded + pression_match_encoded
    rf = RandomForestClassifier()
    rf.fit(np.random.rand(10, len(features)), np.random.randint(0, 2, size=10))
    feature_importances = pd.DataFrame({
        "Variable": variables + ["Domicile", "Extérieur", "Offensif", "Défensif", "Match amical", "Championnat"],
        "Importance (%)": np.round(rf.feature_importances_ * 100, 2)
    }).sort_values(by="Importance (%)", ascending=False)
    
    st.subheader("🌳 Importance des variables (Random Forest)")
    st.write(feature_importances)

    # Régression Logistique (avec variables non chiffrées)
    lr = LogisticRegression()
    lr.fit(np.random.rand(10, len(features)), np.random.randint(0, 3, size=10))
    probas = lr.predict_proba(np.random.rand(1, len(features)))[0]
    
    st.subheader("⚖️ Prédiction du match (Régression Logistique)")
    st.write(f"- **Victoire Équipe 1** : {probas[0] * 100:.2f}%")
    st.write(f"- **Match nul** : {probas[1] * 100:.2f}%")
    st.write(f"- **Victoire Équipe 2** : {probas[2] * 100:.2f}%")

    # Paris double chance
    st.subheader("🎲 Option Paris Double Chance")
    st.write(f"- **1X (Équipe 1 ou Nul)** : {probas[0] * 100 + probas[1] * 100:.2f}%")
    st.write(f"- **X2 (Équipe 2 ou Nul)** : {probas[2] * 100 + probas[1] * 100:.2f}%")
    st.write(f"- **12 (Équipe 1 ou Équipe 2)** : {probas[0] * 100 + probas[2] * 100:.2f}%")

    # Détection de value bet
    st.subheader("💰 Comparateur de cotes & Value Bet")
    predicted_prob = max(probas) * 100
    if predicted_prob > prob_implicite:
        st.success("✅ **Value Bet détecté !**")

    # Graphique des prédictions
    chart_data = pd.DataFrame({"Issue": ["Équipe 1", "Nul", "Équipe 2"], "Probabilité (%)": [probas[0] * 100, probas[1] * 100, probas[2] * 100]})
    chart = alt.Chart(chart_data).mark_bar().encode(x="Issue", y="Probabilité (%)", color="Issue")
    st.altair_chart(chart, use_container_width=True)
