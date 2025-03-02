import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import altair as alt
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# -------------------------------
# Initialisation de la session state
# -------------------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# -------------------------------
# Fonctions Utilitaires
# -------------------------------
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def calculer_value_bet(prob, cote):
    """Calcule la valeur espérée et fournit une recommandation de pari."""
    ev = (prob * cote) - 1
    return ev, "✅ Value Bet" if ev > 0 else "❌ Pas de Value Bet"

# -------------------------------
# Chargement des Données d'Entraînement
# -------------------------------
st.sidebar.header("📊 Données d'Entraînement")
fichier_csv = st.sidebar.file_uploader("📂 Importer un fichier CSV d'entraînement", type=["csv"])

# Exemple de jeu de données par défaut avec 15 variables par équipe + colonne résultat
donnees_defaut = """
xG_A,Tirs_cadrés_A,Taux_conversion_A,Touches_surface_A,Passes_clés_A,Tirs_tentés_A,Interceptions_A,Duels_defensifs_A,xGA_A,Arrêts_gardien_A,Fautes_A,Possession_A,Corners_A,Forme_recente_A,Points_5_matchs_A,xG_B,Tirs_cadrés_B,Taux_conversion_B,Touches_surface_B,Passes_clés_B,Tirs_tentés_B,Interceptions_B,Duels_defensifs_B,xGA_B,Arrêts_gardien_B,Fautes_B,Possession_B,Corners_B,Forme_recente_B,Points_5_matchs_B,resultat
1.3,5,30,25,6,10,8,18,1.2,4,12,55,5,10,8,1.0,4,25,20,4,9,7,15,1.5,5,11,50,4,8,6,1
1.0,4,28,22,5,9,7,16,1.4,5,10,52,3,8,7,1.2,3,27,18,3,8,6,14,1.3,4,9,48,3,7,5,0
1.4,6,32,27,7,11,9,20,1.1,4,13,57,6,11,9,1.1,5,26,21,5,10,8,16,1.4,6,10,54,5,9,7,1
1.2,5,29,24,6,10,8,19,1.3,5,12,53,4,9,8,0.9,4,30,22,4,9,7,15,1.5,5,10,50,4,8,6,0
"""

if fichier_csv:
    df_entrainement = pd.read_csv(fichier_csv)
else:
    st.sidebar.info("Aucun fichier chargé, utilisation des données d'entraînement par défaut.")
    df_entrainement = pd.read_csv(StringIO(donnees_defaut))

if df_entrainement is not None:
    st.sidebar.write("🔍 Aperçu des données d'entraînement :", df_entrainement.head())
    # On suppose que le fichier comporte déjà toutes les colonnes attendues.
    X = df_entrainement.drop(columns=["resultat"])
    y = df_entrainement["resultat"]

    # -------------------------------
    # Entraînement des Modèles avec Validation Croisée
    # -------------------------------
    @st.cache_resource(show_spinner=False)
    def entrainer_modeles(X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = {}
        models = {
            "📊 Régression Logistique": LogisticRegression(),
            "🌲 Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "⚡ XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        for nom, model in models.items():
            accuracy_list = []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                accuracy_list.append(accuracy_score(y_test, model.predict(X_test)))
            scores[nom] = (model, np.mean(accuracy_list))
        return scores

    if not st.session_state.trained_models:
        st.session_state.trained_models = entrainer_modeles(X, y)

    st.sidebar.markdown("### 📈 Performances des Modèles")
    for nom, (modele, acc) in st.session_state.trained_models.items():
        st.sidebar.markdown(f"**{nom} :** {acc*100:.2f}%")

# -------------------------------
# Interface Utilisateur : Saisie des Données de Match
# -------------------------------
st.title("⚽ Prédiction de Match & Analyse Value Bet")
st.markdown("### 📊 Entrez les statistiques moyennes par match pour chaque équipe")

col1, col2 = st.columns(2)

# Variables pour Équipe A (Domicile) – 15 variables
with col1:
    st.header("🏠 **Équipe A (Domicile)**")
    xG_A = st.number_input("⚽ xG", value=1.5, format="%.2f", step=0.01)
    tirs_cadrés_A = st.number_input("🎯 Tirs cadrés", value=5.0, format="%.2f", step=0.01)
    taux_conversion_A = st.number_input("📊 Taux de conversion (%)", value=30.0, format="%.2f", step=0.01)
    touches_surface_A = st.number_input("🔥 Touches dans la surface", value=25.0, format="%.2f", step=0.01)
    passes_cles_A = st.number_input("🔑 Passes clés", value=5.0, format="%.2f", step=0.01)
    tirs_tentés_A = st.number_input("🎯 Tirs tentés", value=10.0, format="%.2f", step=0.01)
    interceptions_A = st.number_input("🛡️ Interceptions", value=8.0, format="%.2f", step=0.01)
    duels_defensifs_A = st.number_input("💪 Duels défensifs gagnés", value=18.0, format="%.2f", step=0.01)
    xGA_A = st.number_input("🚫 xGA", value=1.2, format="%.2f", step=0.01)
    arrets_gardien_A = st.number_input("🧤 Arrêts du gardien", value=4.0, format="%.2f", step=0.01)
    fautes_A = st.number_input("⚠️ Fautes commises", value=12.0, format="%.2f", step=0.01)
    possession_A = st.number_input("📌 Possession (%)", value=55.0, format="%.2f", step=0.01)
    corners_A = st.number_input("⚠️ Corners", value=5.0, format="%.2f", step=0.01)
    forme_recente_A = st.number_input("📊 Forme récente (Points sur 5 matchs)", value=10.0, format="%.2f", step=0.01)
    points_5_matchs_A = st.number_input("⭐ Points sur 5 matchs", value=8.0, format="%.2f", step=0.01)

# Variables pour Équipe B (Extérieur) – 15 variables
with col2:
    st.header("🏟️ **Équipe B (Extérieur)**")
    xG_B = st.number_input("⚽ xG", value=1.2, format="%.2f", step=0.01)
    tirs_cadrés_B = st.number_input("🎯 Tirs cadrés", value=4.0, format="%.2f", step=0.01)
    taux_conversion_B = st.number_input("📊 Taux de conversion (%)", value=25.0, format="%.2f", step=0.01)
    touches_surface_B = st.number_input("🔥 Touches dans la surface", value=20.0, format="%.2f", step=0.01)
    passes_cles_B = st.number_input("🔑 Passes clés", value=4.0, format="%.2f", step=0.01)
    tirs_tentés_B = st.number_input("🎯 Tirs tentés", value=9.0, format="%.2f", step=0.01)
    interceptions_B = st.number_input("🛡️ Interceptions", value=7.0, format="%.2f", step=0.01)
    duels_defensifs_B = st.number_input("💪 Duels défensifs gagnés", value=15.0, format="%.2f", step=0.01)
    xGA_B = st.number_input("🚫 xGA", value=1.5, format="%.2f", step=0.01)
    arrets_gardien_B = st.number_input("🧤 Arrêts du gardien", value=5.0, format="%.2f", step=0.01)
    fautes_B = st.number_input("⚠️ Fautes commises", value=11.0, format="%.2f", step=0.01)
    possession_B = st.number_input("📌 Possession (%)", value=50.0, format="%.2f", step=0.01)
    corners_B = st.number_input("⚠️ Corners", value=4.0, format="%.2f", step=0.01)
    forme_recente_B = st.number_input("📊 Forme récente (Points sur 5 matchs)", value=8.0, format="%.2f", step=0.01)
    points_5_matchs_B = st.number_input("⭐ Points sur 5 matchs", value=6.0, format="%.2f", step=0.01)

# -------------------------------
# Sélection des cotes bookmakers pour l'analyse Value Bet
# -------------------------------
st.markdown("### 🎲 **Analyse Value Bet**")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    cote_A = st.number_input("💰 Cote - Victoire Équipe A", value=2.00, format="%.2f", step=0.01)
with col_odds2:
    cote_N = st.number_input("💰 Cote - Match Nul", value=3.00, format="%.2f", step=0.01)
with col_odds3:
    cote_B = st.number_input("💰 Cote - Victoire Équipe B", value=2.50, format="%.2f", step=0.01)

# -------------------------------
# Prédiction et Affichage du Résultat
# -------------------------------
st.markdown("### 🔮 **Prédiction du Match**")
if st.button("Prédire le Résultat"):
    # Construction de l'input avec 30 variables (15 par équipe)
    input_features = np.array([[
        xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A, tirs_tentés_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, fautes_A, possession_A, corners_A,
        forme_recente_A, points_5_matchs_A,
        xG_B, tirs_cadrés_B, taux_conversion_B, touches_surface_B, passes_cles_B, tirs_tentés_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, fautes_B, possession_B, corners_B,
        forme_recente_B, points_5_matchs_B
    ]])
    
    # Sélection du meilleur modèle (celui avec la meilleure précision)
    best_model = max(st.session_state.trained_models.items(), key=lambda x: x[1][1])[1][0]
    prediction = best_model.predict(input_features)[0]
    
    st.markdown("#### 🔎 **Résultat Prévu :**")
    if prediction == 1:
        st.success("🏆 **Victoire Équipe A**")
    else:
        st.error("🏟️ **Victoire Équipe B**")
    
    # Affichage des probabilités de chaque issue selon le modèle choisi
    proba = best_model.predict_proba(input_features)[0]
    st.markdown("#### 📊 **Probabilités Prédites :**")
    st.write(f"🏆 Victoire Équipe A : {proba[1]*100:.2f}%")
    st.write(f"🏟️ Victoire Équipe B : {proba[0]*100:.2f}%")
    
    # Calcul et affichage de l'analyse Value Bet pour chaque issue
    outcomes = ["Victoire Équipe A", "Match Nul", "Victoire Équipe B"]
    # Ici, nous supposons pour simplifier que la probabilité de victoire pour A est proba[1] et pour B est proba[0].
    # Pour un match nul, nous prenons une moyenne.
    prob_pred = [proba[1], (proba[0] + proba[1]) / 2, proba[0]]
    
    st.markdown("#### 🎲 **Analyse Value Bet**")
    for outcome, cote, prob_val in zip(outcomes, [cote_A, cote_N, cote_B], prob_pred):
        ev, rec = calculer_value_bet(prob_val, cote)
        st.write(f"**{outcome}** - Cote : {cote:.2f}, Prob. Prévue : {prob_val*100:.2f}%, EV : {ev:.2f} → {rec}")
