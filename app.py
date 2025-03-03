import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import altair as alt
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# =====================================
# Fonctions de base et optimisations
# =====================================
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def predire_resultat_match(
    # Variables Équipe A (Domicile) – 18 variables (13 initiales + 5 nouvelles)
    xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A,
    interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
    possession_A, corners_A,
    fautes_commises_A, cartons_jaunes_A, passes_decisives_A, taux_reussite_passes_A, coups_frais_A,
    # Variables Équipe B (Extérieur) – 18 variables (13 initiales + 5 nouvelles)
    xG_B, tirs_cadrés_B, taux_conversion_B, touches_surface_B, passes_cles_B,
    interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B,
    possession_B, corners_B,
    fautes_commises_B, cartons_jaunes_B, passes_decisives_B, taux_reussite_passes_B, coups_frais_B,
    max_buts=5
):
    # Calcul du score offensif pour l'Équipe A
    note_offensive_A = (
        xG_A * 0.2 +
        tirs_cadrés_A * 0.15 +
        touches_surface_A * 0.1 +
        (taux_conversion_A / 100) * 0.15 +
        passes_cles_A * 0.15 +
        passes_decisives_A * 0.1 +
        (taux_reussite_passes_A / 100) * 0.1 +
        coups_frais_A * 0.05 +
        (possession_A / 100) * 0.1 +
        (corners_A / 10) * 0.15
    )
    note_defensive_B = (
        xGA_B * 0.2 +
        arrets_gardien_B * 0.15 +
        interceptions_B * 0.15 +
        duels_defensifs_B * 0.15 +
        fautes_commises_B * 0.05 +
        cartons_jaunes_B * 0.05 +
        ((100 - possession_B) / 100) * 0.1 +
        (corners_B / 10) * 0.15
    )
    multiplicateur_A = 1 + (forme_recente_A / 10) + (points_5_matchs_A / 15)
    adj_xG_A = (note_offensive_A * multiplicateur_A) / (note_defensive_B + 1)
    
    # Calcul du score offensif pour l'Équipe B
    note_offensive_B = (
        xG_B * 0.2 +
        tirs_cadrés_B * 0.15 +
        touches_surface_B * 0.1 +
        (taux_conversion_B / 100) * 0.15 +
        passes_cles_B * 0.15 +
        passes_decisives_B * 0.1 +
        (taux_reussite_passes_B / 100) * 0.1 +
        coups_frais_B * 0.05 +
        (possession_B / 100) * 0.1 +
        (corners_B / 10) * 0.15
    )
    note_defensive_A = (
        xGA_A * 0.2 +
        arrets_gardien_A * 0.15 +
        interceptions_A * 0.15 +
        duels_defensifs_A * 0.15 +
        fautes_commises_A * 0.05 +
        cartons_jaunes_A * 0.05 +
        ((100 - possession_A) / 100) * 0.1 +
        (corners_A / 10) * 0.15
    )
    multiplicateur_B = 1 + (forme_recente_B / 10) + (points_5_matchs_B / 15)
    adj_xG_B = (note_offensive_B * multiplicateur_B) / (note_defensive_A + 1)
    
    # Calcul vectorisé de la distribution de buts via np.outer
    prob_A = np.array([poisson_prob(adj_xG_A, i) for i in range(max_buts+1)])
    prob_B = np.array([poisson_prob(adj_xG_B, i) for i in range(max_buts+1)])
    matrice = np.outer(prob_A, prob_B)
    victoire_A = np.sum(np.triu(matrice, k=1))
    victoire_B = np.sum(np.tril(matrice, k=-1))
    match_nul = np.sum(np.diag(matrice))
    expected_buts_A = np.sum(np.arange(max_buts+1) * prob_A)
    expected_buts_B = np.sum(np.arange(max_buts+1) * prob_B)
    
    return victoire_A, victoire_B, match_nul, expected_buts_A, expected_buts_B

def calculer_value_bet(prob, cote):
    """Calcule la valeur espérée et fournit une recommandation de pari."""
    ev = (prob * cote) - 1
    recommendation = "✅ Value Bet" if ev > 0 else "❌ Pas de Value Bet"
    return ev, recommendation

# =====================================
# Sauvegarde et chargement des modèles sur disque
# =====================================
def save_models_to_disk(models):
    with open("trained_models.pkl", "wb") as f:
        pickle.dump(models, f)

def load_models_from_disk():
    if os.path.exists("trained_models.pkl"):
        with open("trained_models.pkl", "rb") as f:
            return pickle.load(f)
    return None

# =====================================
# Génération de données d'entraînement synthétiques réalistes
# =====================================
def generer_donnees_foot(n_samples=200):
    data = {}
    # Variables initiales pour l'Équipe A (Domicile)
    data["xG_A"] = np.random.uniform(0.5, 2.5, n_samples)
    data["Tirs_cadrés_A"] = np.random.randint(2, 11, n_samples)
    data["Taux_conversion_A"] = np.random.uniform(20, 40, n_samples)
    data["Touches_surface_A"] = np.random.randint(15, 41, n_samples)
    data["Passes_clés_A"] = np.random.randint(3, 9, n_samples)
    data["Interceptions_A"] = np.random.randint(5, 16, n_samples)
    data["Duels_defensifs_A"] = np.random.randint(10, 31, n_samples)
    data["xGA_A"] = np.random.uniform(1.0, 2.5, n_samples)
    data["Arrêts_gardien_A"] = np.random.randint(3, 8, n_samples)
    data["Forme_recente_A"] = np.random.randint(5, 16, n_samples)
    data["Points_5_matchs_A"] = np.random.randint(5, 16, n_samples)
    data["possession_A"] = np.random.uniform(45, 70, n_samples)
    data["corners_A"] = np.random.randint(3, 11, n_samples)
    # Nouvelles variables pour l'Équipe A
    data["Fautes_commises_A"] = np.random.randint(8, 21, n_samples)
    data["Cartons_jaunes_A"] = np.random.randint(0, 4, n_samples)
    data["Passes_decisives_A"] = np.random.randint(0, 6, n_samples)
    data["Taux_reussite_passes_A"] = np.random.uniform(70, 90, n_samples)
    data["Coups_frais_A"] = np.random.randint(0, 6, n_samples)
    
    # Variables initiales pour l'Équipe B (Extérieur)
    data["xG_B"] = np.random.uniform(0.5, 2.5, n_samples)
    data["Tirs_cadrés_B"] = np.random.randint(2, 11, n_samples)
    data["Taux_conversion_B"] = np.random.uniform(20, 40, n_samples)
    data["Touches_surface_B"] = np.random.randint(15, 41, n_samples)
    data["Passes_clés_B"] = np.random.randint(3, 9, n_samples)
    data["Interceptions_B"] = np.random.randint(5, 16, n_samples)
    data["Duels_defensifs_B"] = np.random.randint(10, 31, n_samples)
    data["xGA_B"] = np.random.uniform(1.0, 2.5, n_samples)
    data["arrets_gardien_B"] = np.random.randint(3, 8, n_samples)
    data["Forme_recente_B"] = np.random.randint(5, 16, n_samples)
    data["Points_5_matchs_B"] = np.random.randint(5, 16, n_samples)
    data["possession_B"] = np.random.uniform(45, 70, n_samples)
    data["corners_B"] = np.random.randint(3, 11, n_samples)
    # Nouvelles variables pour l'Équipe B
    data["Fautes_commises_B"] = np.random.randint(8, 21, n_samples)
    data["Cartons_jaunes_B"] = np.random.randint(0, 4, n_samples)
    data["Passes_decisives_B"] = np.random.randint(0, 6, n_samples)
    data["Taux_reussite_passes_B"] = np.random.uniform(70, 90, n_samples)
    data["Coups_frais_B"] = np.random.randint(0, 6, n_samples)
    
    df = pd.DataFrame(data)
    # Génération de la cible 'resultat'
    results = []
    for _, row in df.iterrows():
        victoire_A, victoire_B, match_nul, _, _ = predire_resultat_match(
            row["xG_A"], row["Tirs_cadrés_A"], row["Taux_conversion_A"], row["Touches_surface_A"], row["Passes_clés_A"],
            row["Interceptions_A"], row["Duels_defensifs_A"], row["xGA_A"], row["Arrêts_gardien_A"], row["Forme_recente_A"], row["Points_5_matchs_A"],
            row["possession_A"], row["corners_A"],
            row["Fautes_commises_A"], row["Cartons_jaunes_A"], row["Passes_decisives_A"], row["Taux_reussite_passes_A"], row["Coups_frais_A"],
            row["xG_B"], row["Tirs_cadrés_B"], row["Taux_conversion_B"], row["Touches_surface_B"], row["Passes_clés_B"],
            row["Interceptions_B"], row["Duels_defensifs_B"], row["xGA_B"], row["arrets_gardien_B"], row["Forme_recente_B"], row["Points_5_matchs_B"],
            row["possession_B"], row["corners_B"],
            row["Fautes_commises_B"], row["Cartons_jaunes_B"], row["Passes_decisives_B"], row["Taux_reussite_passes_B"], row["Coups_frais_B"]
        )
        results.append(1 if victoire_A >= victoire_B else 0)
    df["resultat"] = results
    return df

# =====================================
# Section d'entrée des données d'entraînement
# =====================================
st.sidebar.header("📊 Données d'Entraînement")
st.sidebar.markdown(
    """
    **Format du CSV attendu :**

    - Pour l'Équipe A (Domicile) (18 variables) :
      `xG_A`, `Tirs_cadrés_A`, `Taux_conversion_A`, `Touches_surface_A`, `Passes_clés_A`,
      `Interceptions_A`, `Duels_defensifs_A`, `xGA_A`, `Arrêts_gardien_A`, `Forme_recente_A`, `Points_5_matchs_A`,
      `possession_A`, `corners_A`,
      `Fautes_commises_A`, `Cartons_jaunes_A`, `Passes_decisives_A`, `Taux_reussite_passes_A`, `Coups_frais_A`
      
    - Pour l'Équipe B (Extérieur) (18 variables) :
      `xG_B`, `Tirs_cadrés_B`, `Taux_conversion_B`, `Touches_surface_B`, `Passes_clés_B`,
      `Interceptions_B`, `Duels_defensifs_B`, `xGA_B`, `arrets_gardien_B`, `Forme_recente_B`, `Points_5_matchs_B`,
      `possession_B`, `corners_B`,
      `Fautes_commises_B`, `Cartons_jaunes_B`, `Passes_decisives_B`, `Taux_reussite_passes_B`, `Coups_frais_B`
      
    - Colonne cible : `resultat` (0 = victoire B, 1 = victoire A)
    """
)
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = pd.read_csv(fichier_entrainement)
    st.sidebar.write("📝 Aperçu :", df_entrainement.head())
else:
    st.sidebar.info("Aucun fichier chargé, utilisation de données synthétiques.")
    df_entrainement = generer_donnees_foot(n_samples=200)

# Définition de la liste complète des features
features = [
    "xG_A", "Tirs_cadrés_A", "Taux_conversion_A", "Touches_surface_A", "Passes_clés_A",
    "Interceptions_A", "Duels_defensifs_A", "xGA_A", "Arrêts_gardien_A", "Forme_recente_A", "Points_5_matchs_A",
    "possession_A", "corners_A", "Fautes_commises_A", "Cartons_jaunes_A", "Passes_decisives_A", "Taux_reussite_passes_A", "Coups_frais_A",
    "xG_B", "Tirs_cadrés_B", "Taux_conversion_B", "Touches_surface_B", "Passes_clés_B",
    "Interceptions_B", "Duels_defensifs_B", "xGA_B", "arrets_gardien_B", "Forme_recente_B", "Points_5_matchs_B",
    "possession_B", "corners_B", "Fautes_commises_B", "Cartons_jaunes_B", "Passes_decisives_B", "Taux_reussite_passes_B", "Coups_frais_B"
]
X_reel = df_entrainement[features]
y_reel = df_entrainement["resultat"]

# =====================================
# Pré-chargement et sauvegarde des modèles
# =====================================
@st.cache_resource(show_spinner=False)
def load_models(X, y):
    saved = load_models_from_disk()
    if saved is not None:
        return saved
    else:
        model_log, prec_log, cv_scores_log = entrainer_modele_logistique(X, y)
        model_xgb, prec_xgb, cv_scores_xgb = entrainer_modele_xgb(X, y)
        model_rf, prec_rf, cv_scores_rf = entrainer_modele_rf(X, y)
        models = {
            "log": (model_log, prec_log, cv_scores_log),
            "xgb": (model_xgb, prec_xgb, cv_scores_xgb),
            "rf": (model_rf, prec_rf, cv_scores_rf)
        }
        save_models_to_disk(models)
        return models

@st.cache_resource(show_spinner=False)
def entrainer_modele_logistique(X, y):
    model = LogisticRegression(max_iter=1000)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    model.fit(X, y)
    return model, np.mean(cv_scores), cv_scores

@st.cache_resource(show_spinner=False)
def entrainer_modele_xgb(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    model.fit(X, y)
    return model, np.mean(cv_scores), cv_scores

@st.cache_resource(show_spinner=False)
def entrainer_modele_rf(X, y):
    model = RandomForestClassifier(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    model.fit(X, y)
    return model, np.mean(cv_scores), cv_scores

models = load_models(X_reel, y_reel)
modele_logistique, precision_logistique, cv_scores_log = models["log"]
modele_xgb, precision_xgb, cv_scores_xgb = models["xgb"]
modele_rf, precision_rf, cv_scores_rf = models["rf"]

# Affichage détaillé des métriques de validation croisée
cv_data = pd.DataFrame({
    "Modèle": ["Régression Logistique", "XGBoost", "Random Forest"],
    "Précision Moyenne": [f"{precision_logistique*100:.2f}%", f"{precision_xgb*100:.2f}%", f"{precision_rf*100:.2f}%"],
    "CV Scores": [str(cv_scores_log), str(cv_scores_xgb), str(cv_scores_rf)]
})
st.sidebar.markdown("### 📈 Métriques de Validation Croisée")
st.sidebar.table(cv_data)

# =====================================
# Interface principale de saisie des données de match
# =====================================
st.title("⚽ Prédiction de Match de Football & Analyse Value Bet")
st.markdown("### Entrez les statistiques du match pour chaque équipe")

use_fictives = st.checkbox("Utiliser des données fictives", value=False)

col1, col2 = st.columns(2)
with col1:
    st.header("🏠 Équipe A (Domicile)")
    if use_fictives:
        xG_A = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_A = float(random.randint(2, 10))
        taux_conversion_A = round(random.uniform(20, 40), 2)
        touches_surface_A = float(random.randint(15, 40))
        passes_cles_A = float(random.randint(3, 8))
        interceptions_A = float(random.randint(5, 15))
        duels_defensifs_A = float(random.randint(10, 30))
        xGA_A = round(random.uniform(1, 2.5), 2)
        arrets_gardien_A = float(random.randint(3, 7))
        forme_recente_A = float(random.randint(5, 15))
        points_5_matchs_A = float(random.randint(5, 15))
        possession_A = round(random.uniform(45, 70), 2)
        corners_A = float(random.randint(3, 10))
        fautes_commises_A = float(random.randint(8, 20))
        cartons_jaunes_A = float(random.randint(0, 3))
        passes_decisives_A = float(random.randint(0, 5))
        taux_reussite_passes_A = round(random.uniform(70, 90), 2)
        coups_frais_A = float(random.randint(0, 5))
        st.markdown("**🧪 Données fictives générées pour l'Équipe A**")
    else:
        xG_A = st.number_input("⚽ xG (Équipe A)", value=1.50, format="%.2f")
        tirs_cadrés_A = st.number_input("🎯 Tirs cadrés (Équipe A)", value=5.00, format="%.2f")
        taux_conversion_A = st.number_input("🔥 Taux de conversion (Équipe A)", value=30.00, format="%.2f")
        touches_surface_A = st.number_input("🤾‍♂️ Touches dans la surface (Équipe A)", value=25.00, format="%.2f")
        passes_cles_A = st.number_input("🔑 Passes clés (Équipe A)", value=5.00, format="%.2f")
        interceptions_A = st.number_input("🛡️ Interceptions (Équipe A)", value=8.00, format="%.2f")
        duels_defensifs_A = st.number_input("⚔️ Duels défensifs (Équipe A)", value=18.00, format="%.2f")
        xGA_A = st.number_input("🚫 xGA (Équipe A)", value=1.20, format="%.2f")
        arrets_gardien_A = st.number_input("🧤 Arrêts du gardien (Équipe A)", value=4.00, format="%.2f")
        forme_recente_A = st.number_input("💪 Forme récente (Équipe A)", value=10.00, format="%.2f")
        points_5_matchs_A = st.number_input("📊 Points 5 derniers matchs (A)", value=8.00, format="%.2f")
        possession_A = st.number_input("📈 Possession (Équipe A)", value=55.00, format="%.2f")
        corners_A = st.number_input("⚽ Corners (Équipe A)", value=5.00, format="%.2f")
        fautes_commises_A = st.number_input("⚽ Fautes commises (Équipe A)", value=12, step=1)
        cartons_jaunes_A = st.number_input("🟡 Cartons jaunes (Équipe A)", value=1, step=1)
        passes_decisives_A = st.number_input("🎯 Passes décisives (Équipe A)", value=2, step=1)
        taux_reussite_passes_A = st.number_input("✅ Taux réussite passes (Équipe A)", value=80.00, format="%.2f")
        coups_frais_A = st.number_input("🎯 Coups francs (Équipe A)", value=1, step=1)

with col2:
    st.header("🏟️ Équipe B (Extérieur)")
    if use_fictives:
        xG_B = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_B = float(random.randint(2, 10))
        taux_conversion_B = round(random.uniform(20, 40), 2)
        touches_surface_B = float(random.randint(15, 40))
        passes_cles_B = float(random.randint(3, 8))
        interceptions_B = float(random.randint(5, 15))
        duels_defensifs_B = float(random.randint(10, 30))
        xGA_B = round(random.uniform(1, 2.5), 2)
        arrets_gardien_B = float(random.randint(3, 7))
        forme_recente_B = float(random.randint(5, 15))
        points_5_matchs_B = float(random.randint(5, 15))
        possession_B = round(random.uniform(45, 70), 2)
        corners_B = float(random.randint(3, 10))
        fautes_commises_B = float(random.randint(8, 20))
        cartons_jaunes_B = float(random.randint(0, 3))
        passes_decisives_B = float(random.randint(0, 5))
        taux_reussite_passes_B = round(random.uniform(70, 90), 2)
        coups_frais_B = float(random.randint(0, 5))
        st.markdown("**🧪 Données fictives générées pour l'Équipe B**")
    else:
        xG_B = st.number_input("⚽ xG (Équipe B)", value=1.00, format="%.2f")
        tirs_cadrés_B = st.number_input("🎯 Tirs cadrés (Équipe B)", value=3.00, format="%.2f")
        taux_conversion_B = st.number_input("🔥 Taux de conversion (Équipe B)", value=25.00, format="%.2f")
        touches_surface_B = st.number_input("🤾‍♂️ Touches dans la surface (Équipe B)", value=20.00, format="%.2f")
        passes_cles_B = st.number_input("🔑 Passes clés (Équipe B)", value=4.00, format="%.2f")
        interceptions_B = st.number_input("🛡️ Interceptions (Équipe B)", value=7.00, format="%.2f")
        duels_defensifs_B = st.number_input("⚔️ Duels défensifs (Équipe B)", value=15.00, format="%.2f")
        xGA_B = st.number_input("🚫 xGA (Équipe B)", value=1.50, format="%.2f")
        arrets_gardien_B = st.number_input("🧤 Arrêts du gardien (Équipe B)", value=5.00, format="%.2f")
        forme_recente_B = st.number_input("💪 Forme récente (Équipe B)", value=8.00, format="%.2f")
        points_5_matchs_B = st.number_input("📊 Points 5 derniers matchs (B)", value=6.00, format="%.2f")
        possession_B = st.number_input("📈 Possession (Équipe B)", value=50.00, format="%.2f")
        corners_B = st.number_input("⚽ Corners (Équipe B)", value=4.00, format="%.2f")
        fautes_commises_B = st.number_input("⚽ Fautes commises (Équipe B)", value=12, step=1)
        cartons_jaunes_B = st.number_input("🟡 Cartons jaunes (Équipe B)", value=1, step=1)
        passes_decisives_B = st.number_input("🎯 Passes décisives (Équipe B)", value=2, step=1)
        taux_reussite_passes_B = st.number_input("✅ Taux réussite passes (Équipe B)", value=80.00, format="%.2f")
        coups_frais_B = st.number_input("🎯 Coups francs (Équipe B)", value=1, step=1)

st.markdown("### 🎲 Analyse Value Bet")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    cote_A = st.number_input("🏆 Cote - Victoire A", value=2.00, format="%.2f")
with col_odds2:
    cote_N = st.number_input("🤝 Cote - Match Nul", value=3.00, format="%.2f")
with col_odds3:
    cote_B = st.number_input("🏟️ Cote - Victoire B", value=2.50, format="%.2f")

# =====================================
# Prédictions et affichage des résultats
# =====================================
if st.button("🔮 Prédire le Résultat"):
    # Prédiction via le modèle de Poisson (en passant 36 variables)
    victoire_A, victoire_B, match_nul, expected_buts_A, expected_buts_B = predire_resultat_match(
        xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
        possession_A, corners_A,
        fautes_commises_A, cartons_jaunes_A, passes_decisives_A, taux_reussite_passes_A, coups_frais_A,
        xG_B, tirs_cadrés_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B,
        possession_B, corners_B,
        fautes_commises_B, cartons_jaunes_B, passes_decisives_B, taux_reussite_passes_B, coups_frais_B
    )
    
    st.markdown("## 📊 Résultats du Modèle de Poisson")
    data_poisson = {
        "Mesure": [
            "🏆 Prob. Victoire A", "🤝 Prob. Match Nul", "🏟️ Prob. Victoire B",
            "⚽ Buts attendus A", "⚽ Buts attendus B"
        ],
        "Valeur": [
            f"{victoire_A*100:.2f}%", f"{match_nul*100:.2f}%", f"{victoire_B*100:.2f}%",
            f"{expected_buts_A:.2f}", f"{expected_buts_B:.2f}"
        ]
    }
    st.table(pd.DataFrame(data_poisson))
    
    st.markdown("## 🔍 Résultats des Modèles de Classification")
    input_features = np.array([[ 
        xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
        possession_A, corners_A, fautes_commises_A, cartons_jaunes_A, passes_decisives_A, taux_reussite_passes_A, coups_frais_A,
        xG_B, tirs_cadrés_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B,
        possession_B, corners_B, fautes_commises_B, cartons_jaunes_B, passes_decisives_B, taux_reussite_passes_B, coups_frais_B
    ]])
    st.write("**📐 Forme de l'input :**", input_features.shape)
    
    proba_log = modele_logistique.predict_proba(input_features)[0][1]
    prediction_log = "🏆 Victoire A" if proba_log > 0.5 else "🏟️ Victoire B"
    
    proba_xgb = modele_xgb.predict_proba(input_features)[0][1]
    prediction_xgb = "🏆 Victoire A" if proba_xgb > 0.5 else "🏟️ Victoire B"
    
    proba_rf = modele_rf.predict_proba(input_features)[0][1]
    prediction_rf = "🏆 Victoire A" if proba_rf > 0.5 else "🏟️ Victoire B"
    
    data_classif = {
        "Modèle": ["Régression Logistique", "XGBoost", "Random Forest"],
        "Prédiction": [prediction_log, prediction_xgb, prediction_rf],
        "Probabilité": [f"{proba_log*100:.2f}%", f"{proba_xgb*100:.2f}%", f"{proba_rf*100:.2f}%"],
        "Précision (moyenne / CV scores)": [
            f"{precision_logistique*100:.2f}% / {cv_scores_log}",
            f"{precision_xgb*100:.2f}% / {cv_scores_xgb}",
            f"{precision_rf*100:.2f}% / {cv_scores_rf}"
        ]
    }
    st.table(pd.DataFrame(data_classif))
    
    st.markdown("## 🎲 Analyse Value Bet")
    outcomes = ["🏆 Victoire A", "🤝 Match Nul", "🏟️ Victoire B"]
    bookmaker_cotes = [cote_A, cote_N, cote_B]
    predicted_probs = [victoire_A, match_nul, victoire_B]
    
    value_bet_data = {"Issue": [], "Cote Bookmaker": [], "Prob. Impliquée": [], "Prob. Prédite": [], "Valeur Espérée": [], "Value Bet ?": []}
    for outcome, cote, prob in zip(outcomes, bookmaker_cotes, predicted_probs):
        prob_implied = 1 / cote
        ev, recommendation = calculer_value_bet(prob, cote)
        value_bet_data["Issue"].append(outcome)
        value_bet_data["Cote Bookmaker"].append(cote)
        value_bet_data["Prob. Impliquée"].append(f"{prob_implied*100:.2f}%")
        value_bet_data["Prob. Prédite"].append(f"{prob*100:.2f}%")
        value_bet_data["Valeur Espérée"].append(f"{ev:.2f}")
        value_bet_data["Value Bet ?"].append(recommendation)
    st.table(pd.DataFrame(value_bet_data))
    
    st.markdown("## 🎯 Analyse Double Chance")
    dc_option = st.selectbox("Sélectionnez l'option Double Chance", ["1X (🏠 ou 🤝)", "X2 (🤝 ou 🏟️)", "12 (🏠 ou 🏟️)"])
    if dc_option == "1X (🏠 ou 🤝)":
        dc_odds = st.number_input("💰 Cote - Double Chance 1X", value=1.50, format="%.2f")
        dc_prob = victoire_A + match_nul
    elif dc_option == "X2 (🤝 ou 🏟️)":
        dc_odds = st.number_input("💰 Cote - Double Chance X2", value=1.60, format="%.2f")
        dc_prob = match_nul + victoire_B
    else:
        dc_odds = st.number_input("💰 Cote - Double Chance 12", value=1.40, format="%.2f")
        dc_prob = victoire_A + victoire_B

    dc_implied = 1 / dc_odds
    dc_ev, dc_recommendation = calculer_value_bet(dc_prob, dc_odds)
    
    dc_data = {
        "Option": [dc_option],
        "Cote": [dc_odds],
        "Prob. Impliquée": [f"{dc_implied*100:.2f}%"],
        "Prob. Prédite": [f"{dc_prob*100:.2f}%"],
        "Valeur Espérée": [f"{dc_ev:.2f}"],
        "Double Chance": [dc_recommendation]
    }
    st.table(pd.DataFrame(dc_data))
    
    buts_data = pd.DataFrame({
        "Buts": list(range(0, 6)),
        "Prob Équipe A": [poisson_prob(expected_buts_A, i) for i in range(6)],
        "Prob Équipe B": [poisson_prob(expected_buts_B, i) for i in range(6)]
    })
    chart = alt.Chart(buts_data).mark_bar().encode(
        x=alt.X("Buts:O", title="Nombre de buts"),
        y=alt.Y("Prob Équipe A:Q", title="Probabilité"),
        color=alt.value("#4CAF50")
    ).properties(title="Distribution des buts attendus - Équipe A")
    st.altair_chart(chart, use_container_width=True)
