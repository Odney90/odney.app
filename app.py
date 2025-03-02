import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# -------------------------------
# Initialisation du session state (optionnel)
# -------------------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

# -------------------------------
# Fonctions de base
# -------------------------------
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def predire_resultat_match(
    # Variables Équipe A (Domicile, moyennes par match)
    xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A,
    interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
    possession_A, corners_A,
    # Variables Équipe B (Extérieur, moyennes par match)
    xG_B, tirs_cadrés_B, taux_conversion_B, touches_surface_B, passes_cles_B,
    interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B,
    possession_B, corners_B,
    max_buts=5
):
    note_offensive_A = (
        xG_A * 0.2 +
        tirs_cadrés_A * 0.15 +
        touches_surface_A * 0.1 +
        (taux_conversion_A / 100) * 0.15 +
        passes_cles_A * 0.15 +
        (possession_A / 100) * 0.1 +
        (corners_A / 10) * 0.15
    )
    note_defensive_B = (
        xGA_B * 0.2 +
        arrets_gardien_B * 0.15 +
        interceptions_B * 0.15 +
        duels_defensifs_B * 0.15 +
        ((100 - possession_B) / 100) * 0.1 +
        (corners_B / 10) * 0.15
    )
    multiplicateur_A = 1 + (forme_recente_A / 10) + (points_5_matchs_A / 15)
    adj_xG_A = (note_offensive_A * multiplicateur_A) / (note_defensive_B + 1)
    
    note_offensive_B = (
        xG_B * 0.2 +
        tirs_cadrés_B * 0.15 +
        touches_surface_B * 0.1 +
        (taux_conversion_B / 100) * 0.15 +
        passes_cles_B * 0.15 +
        (possession_B / 100) * 0.1 +
        (corners_B / 10) * 0.15
    )
    note_defensive_A = (
        xGA_A * 0.2 +
        arrets_gardien_A * 0.15 +
        interceptions_A * 0.15 +
        duels_defensifs_A * 0.15 +
        ((100 - possession_A) / 100) * 0.1 +
        (corners_A / 10) * 0.15
    )
    multiplicateur_B = 1 + (forme_recente_B / 10) + (points_5_matchs_B / 15)
    adj_xG_B = (note_offensive_B * multiplicateur_B) / (note_defensive_A + 1)
    
    prob_A = [poisson_prob(adj_xG_A, i) for i in range(max_buts+1)]
    prob_B = [poisson_prob(adj_xG_B, i) for i in range(max_buts+1)]
    
    victoire_A, victoire_B, match_nul = 0, 0, 0
    for i in range(max_buts+1):
        for j in range(max_buts+1):
            p = prob_A[i] * prob_B[j]
            if i > j:
                victoire_A += p
            elif i < j:
                victoire_B += p
            else:
                match_nul += p
                
    expected_buts_A = sum(i * prob_A[i] for i in range(max_buts+1))
    expected_buts_B = sum(i * prob_B[i] for i in range(max_buts+1))
    
    return victoire_A, victoire_B, match_nul, expected_buts_A, expected_buts_B

def calculer_value_bet(prob, cote):
    """Calcule la valeur espérée et fournit une recommandation de pari."""
    ev = (prob * cote) - 1
    recommendation = "✅ Value Bet" if ev > 0 else "❌ Pas de Value Bet"
    return ev, recommendation

# -------------------------------
# Section d'entrée des données d'entraînement
# -------------------------------
st.sidebar.header("📊 Données d'Entraînement")
st.sidebar.markdown(
    """
    **Format du fichier CSV attendu (données par match ou moyennes par match) :**

    Votre fichier doit contenir les colonnes suivantes dans cet ordre :

    - Pour l'Équipe A (Domicile) :
      `xG_A`, `Tirs_cadrés_A`, `Taux_conversion_A`, `Touches_surface_A`, `Passes_clés_A`,
      `Interceptions_A`, `Duels_defensifs_A`, `xGA_A`, `Arrêts_gardien_A`, `Forme_recente_A`, `Points_5_matchs_A`,
      `possession_A`, `corners_A`

    - Pour l'Équipe B (Extérieur) :
      `xG_B`, `Tirs_cadrés_B`, `Taux_conversion_B`, `Touches_surface_B`, `Passes_clés_B`,
      `Interceptions_B`, `Duels_defensifs_B`, `xGA_B`, `Arrêts_du_gardien_B`, `Forme_recente_B`, `Points_5_matchs_B`,
      `possession_B`, `corners_B`

    - Et la colonne cible : `resultat` (0 pour victoire de l'Équipe B, 1 pour victoire de l'Équipe A).

    **Remarque :** Les statistiques doivent être des moyennes par match ou par période récente.
    """
)
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = pd.read_csv(fichier_entrainement)
    st.sidebar.write("Aperçu des données d'entraînement :", df_entrainement.head())
    features = [
        "xG_A", "Tirs_cadrés_A", "Taux_conversion_A", "Touches_surface_A", "Passes_clés_A",
        "Interceptions_A", "Duels_defensifs_A", "xGA_A", "Arrêts_gardien_A", "Forme_recente_A", "Points_5_matchs_A",
        "possession_A", "corners_A",
        "xG_B", "Tirs_cadrés_B", "Taux_conversion_B", "Touches_surface_B", "Passes_clés_B",
        "Interceptions_B", "Duels_defensifs_B", "xGA_B", "Arrêts_du_gardien_B", "Forme_recente_B", "Points_5_matchs_B",
        "possession_B", "corners_B"
    ]
    X_reel = df_entrainement[features]
    y_reel = df_entrainement["resultat"]

# -------------------------------
# Entraînement des modèles (avec validation croisée)
# -------------------------------
@st.cache_resource(show_spinner=False)
def entrainer_modele_cv(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return model, np.mean(scores)

if fichier_entrainement is not None:
    modele_logistique, score_logistique = entrainer_modele_cv(LogisticRegression(), X_reel, y_reel)
    modele_xgb, score_xgb = entrainer_modele_cv(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'), X_reel, y_reel
    )
    modele_rf, score_rf = entrainer_modele_cv(RandomForestClassifier(random_state=42), X_reel, y_reel)

    st.sidebar.markdown(f"**Score Régression Logistique :** {score_logistique*100:.2f}%")
    st.sidebar.markdown(f"**Score XGBoost :** {score_xgb*100:.2f}%")
    st.sidebar.markdown(f"**Score Random Forest :** {score_rf*100:.2f}%")

# -------------------------------
# Interface principale et résultats
# -------------------------------
# L’interface utilisateur pour saisir les données et afficher les résultats reste inchangée.
# Les résultats incluent désormais les scores moyens de la validation croisée.
...
