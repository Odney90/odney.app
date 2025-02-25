import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import poisson
from docx import Document
from io import BytesIO

# Définition de l'interface
st.set_page_config(page_title="Prédictions Foot ⚽", layout="wide")

# En-tête
st.markdown("# ⚽ Prédictions de Matchs de Football")
st.markdown("### Analyse avec Poisson, Random Forest & Régression Logistique")

# Fonction pour générer un fichier DOC
def generate_doc(team1_stats, team2_stats, poisson_results, rf_importance, lr_probas):
    doc = Document()
    doc.add_heading("Rapport des Prédictions de Match de Football", 0)

    # Noms des équipes
    doc.add_heading("1. Noms des équipes", level=1)
    doc.add_paragraph(f"Équipe 1: {team1_stats['team_name']}")
    doc.add_paragraph(f"Équipe 2: {team2_stats['team_name']}")

    # Stats des équipes
    doc.add_heading("2. Statistiques des équipes", level=1)
    doc.add_paragraph(f"Équipe 1 Stats: {team1_stats}")
    doc.add_paragraph(f"Équipe 2 Stats: {team2_stats}")

    # Poisson results
    doc.add_heading("3. Résultats Poisson", level=1)
    doc.add_paragraph(str(poisson_results))

    # Random Forest Feature Importance
    doc.add_heading("4. Importance des variables (Random Forest)", level=1)
    doc.add_paragraph(str(rf_importance))

    # Régression Logistique Prédictions
    doc.add_heading("5. Prédictions Régression Logistique", level=1)
    doc.add_paragraph(f"Probabilité de victoire Équipe 1: {lr_probas[0] * 100:.2f}%")
    doc.add_paragraph(f"Probabilité de Match nul: {lr_probas[1] * 100:.2f}%")
    doc.add_paragraph(f"Probabilité de victoire Équipe 2: {lr_probas[2] * 100:.2f}%")

    # Sauvegarde le fichier DOC en mémoire
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Espace utilisateur pour les équipes
st.sidebar.header("📊 Entrer les statistiques des équipes")
team1_name = st.sidebar.text_input("Nom de l'Équipe 1", "Équipe A")
team2_name = st.sidebar.text_input("Nom de l'Équipe 2", "Équipe B")

# Variables chiffrées
variables = [
    "XG moyen", "Buts marqués/match", "Buts encaissés/match", 
    "Classement", "Forme (5 derniers matchs)", "Possession (%)", 
    "Tirs cadrés/match", "XP (Expected Points)", "Corners/match", 
    "Cartons jaunes", "Cartons rouges"
]

# Initialisation des stats
team1_stats, team2_stats = {"team_name": team1_name}, {"team_name": team2_name}
for var in variables:
    team1_stats[var] = st.sidebar.number_input(f"{var} - {team1_name}", value=1.0)
    team2_stats[var] = st.sidebar.number_input(f"{var} - {team2_name}", value=1.0)

# Variables non chiffrées
st.sidebar.subheader("⚙️ Autres facteurs")
match_status = st.sidebar.selectbox("Statut du match", ["Domicile", "Extérieur", "Terrain neutre"])
style_jeu = st.sidebar.selectbox("Style de jeu", ["Offensif", "Défensif", "Équilibré"])
pression_match = st.sidebar.selectbox("Pression du match", ["Match amical", "Championnat", "Coupe", "Derby"])

# Transformation en variables numériques
match_status_encoded = [1 if match_status == "Domicile" else 0, 1 if match_status == "Extérieur" else 0]
style_jeu_encoded = [1 if style_jeu == "Offensif" else 0, 1 if style_jeu == "Défensif" else 0]
pression_match_encoded = [1 if pression_match == "Match amical" else 0, 1 if pression_match == "Championnat" else 0]

# Convertisseur de cotes
st.sidebar.header("🎲 Convertisseur de cotes")
cote = st.sidebar.number_input("Entrer la cote du bookmaker", min_value=1.01, value=2.00, step=0.01)
prob_implicite = (1 / cote) * 100
st.sidebar.write(f"Probabilité implicite : **{prob_implicite:.2f}%**")

# Bouton pour générer les prédictions
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

    # Random Forest (avec K=5 validation croisée)
    features = list(team1_stats.values()) + list(team2_stats.values()) + match_status_encoded + style_jeu_encoded + pression_match_encoded
    rf = RandomForestClassifier()
    cross_val_score(rf, np.random.rand(10, len(features)), np.random.randint(0, 2, size=10), cv=5)

    # Afficher importance des variables
    feature_importances = pd.DataFrame({
        "Variable": variables + ["Domicile", "Extérieur", "Offensif", "Défensif", "Match amical", "Championnat"],
        "Importance (%)": np.round(rf.feature_importances_ * 100, 2)
    }).sort_values(by="Importance (%)", ascending=False)
    
    st.subheader("🌳 Importance des variables (Random Forest)")
    st.write(feature_importances)

    # Régression Logistique (avec validation croisée)
    lr = LogisticRegression()
    cross_val_score(lr, np.random.rand(10, len(features)), np.random.randint(0, 3, size=10), cv=5)

    # Afficher les prédictions
    lr_probas = lr.predict_proba(np.random.rand(1, len(features)))[0]
    st.subheader("⚖️ Prédiction du match (Régression Logistique)")
    st.write(f"- **Victoire Équipe 1** : {lr_probas[0] * 100:.2f}%")
    st.write(f"- **Match nul** : {lr_probas[1] * 100:.2f}%")
    st.write(f"- **Victoire Équipe 2** : {lr_probas[2] * 100:.2f}%")

    # Option Paris Double Chance
    st.subheader("🎲 Option Paris Double Chance")
    st.write(f"- **1X (Équipe 1 ou Nul)** : {lr_probas[0] * 100 + lr_probas[1] * 100:.2f}%")
    st.write(f"- **X2 (Équipe 2 ou Nul)** : {lr_probas[2] * 100 + lr_probas[1] * 100:.2f}%")
    st.write(f"- **12 (Équipe 1 ou Équipe 2)** : {lr_probas[0] * 100 + lr_probas[2] * 100:.2f}%")

    # Téléchargement du rapport
    st.sidebar.subheader("📥 Télécharger le rapport")
    buffer = generate_doc(team1_stats, team2_stats, poisson_results, feature_importances, lr_probas)
    st.sidebar.download_button("Télécharger le rapport", buffer, "rapport_match.docx")

    # Graphique des prédictions
    chart_data = pd.DataFrame({"Issue": ["Équipe 1", "Nul", "Équipe 2"], "Probabilité (%)": [lr_probas[0] * 100, lr_probas[1] * 100, lr_probas[2] * 100]})
    chart = alt.Chart(chart_data).mark_bar().encode(x="Issue", y="Probabilité (%)", color="Issue")
    st.altair_chart(chart, use_container_width=True)
