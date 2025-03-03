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
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# =====================================
# Fonction wrapper pour saisir des nombres avec virgule
# =====================================
def number_input_locale(label, value, key=None):
    """
    Permet de saisir des nombres en utilisant la virgule comme séparateur décimal.
    On utilise st.text_input pour récupérer une chaîne que l'on convertit ensuite en float.
    """
    if key is None:
        user_input = st.text_input(label, value=str(value))
    else:
        user_input = st.text_input(label, value=str(value), key=key)
    try:
        return float(user_input.replace(",", "."))
    except:
        return value

# =====================================
# Fonctions de base et optimisations
# =====================================
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def predire_resultat_match(
    # Variables Équipe Home (Domicile) – 18 variables
    xG_home, tirs_cadrés_home, taux_conversion_home, touches_surface_home, passes_cles_home,
    interceptions_home, duels_defensifs_home, xGA_home, arrets_gardien_home, forme_recente_home, points_5_matchs_home,
    possession_home, corners_home,
    fautes_commises_home, cartons_jaunes_home, passes_decisives_home, taux_reussite_passes_home, coups_frais_home,
    # Variables Équipe Away (Extérieur) – 18 variables
    xG_away, tirs_cadrés_away, taux_conversion_away, touches_surface_away, passes_cles_away,
    interceptions_away, duels_defensifs_away, xGA_away, arrets_gardien_away, forme_recente_away, points_5_matchs_away,
    possession_away, corners_away,
    fautes_commises_away, cartons_jaunes_away, passes_decisives_away, taux_reussite_passes_away, coups_frais_away,
    max_buts=5
):
    # Calcul du score offensif pour l'équipe Home (à domicile)
    note_offensive_home = (
        xG_home * 0.2 +
        tirs_cadrés_home * 0.15 +
        touches_surface_home * 0.1 +
        (taux_conversion_home / 100) * 0.15 +
        passes_cles_home * 0.15 +
        passes_decisives_home * 0.1 +
        (taux_reussite_passes_home / 100) * 0.1 +
        coups_frais_home * 0.05 +
        (possession_home / 100) * 0.1 +
        (corners_home / 10) * 0.15
    )
    # Pour l'adversaire, on utilise ses stats à l'extérieur
    note_defensive_away = (
        xGA_away * 0.2 +
        arrets_gardien_away * 0.15 +
        interceptions_away * 0.15 +
        duels_defensifs_away * 0.15 +
        fautes_commises_away * 0.05 +
        cartons_jaunes_away * 0.05 +
        ((100 - possession_away) / 100) * 0.1 +
        (corners_away / 10) * 0.15
    )
    multiplicateur_home = 1 + (forme_recente_home / 10) + (points_5_matchs_home / 15)
    adj_xG_home = (note_offensive_home * multiplicateur_home) / (note_defensive_away + 1)
    
    # Calcul du score offensif pour l'équipe Away (à l'extérieur)
    note_offensive_away = (
        xG_away * 0.2 +
        tirs_cadrés_away * 0.15 +
        touches_surface_away * 0.1 +
        (taux_conversion_away / 100) * 0.15 +
        passes_cles_away * 0.15 +
        passes_decisives_away * 0.1 +
        (taux_reussite_passes_away / 100) * 0.1 +
        coups_frais_away * 0.05 +
        (possession_away / 100) * 0.1 +
        (corners_away / 10) * 0.15
    )
    # Pour l'équipe Home, on utilise ses stats à domicile pour l'aspect défensif
    note_defensive_home = (
        xGA_home * 0.2 +
        arrets_gardien_home * 0.15 +
        interceptions_home * 0.15 +
        duels_defensifs_home * 0.15 +
        fautes_commises_home * 0.05 +
        cartons_jaunes_home * 0.05 +
        ((100 - possession_home) / 100) * 0.1 +
        (corners_home / 10) * 0.15
    )
    multiplicateur_away = 1 + (forme_recente_away / 10) + (points_5_matchs_away / 15)
    adj_xG_away = (note_offensive_away * multiplicateur_away) / (note_defensive_home + 1)
    
    # Calcul vectorisé de la distribution de buts via np.outer
    prob_home = np.array([poisson_prob(adj_xG_home, i) for i in range(max_buts+1)])
    prob_away = np.array([poisson_prob(adj_xG_away, i) for i in range(max_buts+1)])
    matrice = np.outer(prob_home, prob_away)
    victoire_home = np.sum(np.triu(matrice, k=1))
    victoire_away = np.sum(np.tril(matrice, k=-1))
    match_nul = np.sum(np.diag(matrice))
    expected_buts_home = np.sum(np.arange(max_buts+1) * prob_home)
    expected_buts_away = np.sum(np.arange(max_buts+1) * prob_away)
    
    return victoire_home, victoire_away, match_nul, expected_buts_home, expected_buts_away

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
    # Variables pour l'équipe Home (Domicile)
    data["xG_home"] = np.random.uniform(0.5, 2.5, n_samples)
    data["Tirs_cadrés_home"] = np.random.randint(2, 11, n_samples)
    data["Taux_conversion_home"] = np.random.uniform(20, 40, n_samples)
    data["Touches_surface_home"] = np.random.randint(15, 41, n_samples)
    data["Passes_clés_home"] = np.random.randint(3, 9, n_samples)
    data["Interceptions_home"] = np.random.randint(5, 16, n_samples)
    data["Duels_defensifs_home"] = np.random.randint(10, 31, n_samples)
    data["xGA_home"] = np.random.uniform(1.0, 2.5, n_samples)
    data["Arrêts_gardien_home"] = np.random.randint(3, 8, n_samples)
    data["Forme_recente_home"] = np.random.randint(5, 16, n_samples)
    data["Points_5_matchs_home"] = np.random.randint(5, 16, n_samples)
    data["possession_home"] = np.random.uniform(45, 70, n_samples)
    data["corners_home"] = np.random.randint(3, 11, n_samples)
    data["Fautes_commises_home"] = np.random.randint(8, 21, n_samples)
    data["Cartons_jaunes_home"] = np.random.randint(0, 4, n_samples)
    data["Passes_decisives_home"] = np.random.randint(0, 6, n_samples)
    data["Taux_reussite_passes_home"] = np.random.uniform(70, 90, n_samples)
    data["Coups_frais_home"] = np.random.randint(0, 6, n_samples)
    
    # Variables pour l'équipe Away (Extérieur)
    data["xG_away"] = np.random.uniform(0.5, 2.5, n_samples)
    data["Tirs_cadrés_away"] = np.random.randint(2, 11, n_samples)
    data["Taux_conversion_away"] = np.random.uniform(20, 40, n_samples)
    data["Touches_surface_away"] = np.random.randint(15, 41, n_samples)
    data["Passes_clés_away"] = np.random.randint(3, 9, n_samples)
    data["Interceptions_away"] = np.random.randint(5, 16, n_samples)
    data["Duels_defensifs_away"] = np.random.randint(10, 31, n_samples)
    data["xGA_away"] = np.random.uniform(1.0, 2.5, n_samples)
    data["arrets_gardien_away"] = np.random.randint(3, 8, n_samples)
    data["Forme_recente_away"] = np.random.randint(5, 16, n_samples)
    data["Points_5_matchs_away"] = np.random.randint(5, 16, n_samples)
    data["possession_away"] = np.random.uniform(45, 70, n_samples)
    data["corners_away"] = np.random.randint(3, 11, n_samples)
    data["Fautes_commises_away"] = np.random.randint(8, 21, n_samples)
    data["Cartons_jaunes_away"] = np.random.randint(0, 4, n_samples)
    data["Passes_decisives_away"] = np.random.randint(0, 6, n_samples)
    data["Taux_reussite_passes_away"] = np.random.uniform(70, 90, n_samples)
    data["Coups_frais_away"] = np.random.randint(0, 6, n_samples)
    
    df = pd.DataFrame(data)
    # Génération de la cible 'resultat' en comparant les performances Home vs Away
    results = []
    for _, row in df.iterrows():
        victoire_home, victoire_away, match_nul, _, _ = predire_resultat_match(
            row["xG_home"], row["Tirs_cadrés_home"], row["Taux_conversion_home"], row["Touches_surface_home"], row["Passes_clés_home"],
            row["Interceptions_home"], row["Duels_defensifs_home"], row["xGA_home"], row["Arrêts_gardien_home"], row["Forme_recente_home"], row["Points_5_matchs_home"],
            row["possession_home"], row["corners_home"],
            row["Fautes_commises_home"], row["Cartons_jaunes_home"], row["Passes_decisives_home"], row["Taux_reussite_passes_home"], row["Coups_frais_home"],
            row["xG_away"], row["Tirs_cadrés_away"], row["Taux_conversion_away"], row["Touches_surface_away"], row["Passes_clés_away"],
            row["Interceptions_away"], row["Duels_defensifs_away"], row["xGA_away"], row["arrets_gardien_away"], row["Forme_recente_away"], row["Points_5_matchs_away"],
            row["possession_away"], row["corners_away"],
            row["Fautes_commises_away"], row["Cartons_jaunes_away"], row["Passes_decisives_away"], row["Taux_reussite_passes_away"], row["Coups_frais_away"]
        )
        results.append(1 if victoire_home >= victoire_away else 0)
    df["resultat"] = results
    return df

# =====================================
# Section d'entrée des données d'entraînement
# =====================================
st.sidebar.header("📊 Données d'Entraînement")
st.sidebar.markdown(
    """
    **Format du CSV attendu :**

    - Pour l'équipe Home (Domicile) (18 variables) :
      `xG_home`, `Tirs_cadrés_home`, `Taux_conversion_home`, `Touches_surface_home`, `Passes_clés_home`,
      `Interceptions_home`, `Duels_defensifs_home`, `xGA_home`, `Arrêts_gardien_home`, `Forme_recente_home`, `Points_5_matchs_home`,
      `possession_home`, `corners_home`,
      `Fautes_commises_home`, `Cartons_jaunes_home`, `Passes_decisives_home`, `Taux_reussite_passes_home`, `Coups_frais_home`
      
    - Pour l'équipe Away (Extérieur) (18 variables) :
      `xG_away`, `Tirs_cadrés_away`, `Taux_conversion_away`, `Touches_surface_away`, `Passes_clés_away`,
      `Interceptions_away`, `Duels_defensifs_away`, `xGA_away`, `arrets_gardien_away`, `Forme_recente_away`, `Points_5_matchs_away`,
      `possession_away`, `corners_away`,
      `Fautes_commises_away`, `Cartons_jaunes_away`, `Passes_decisives_away`, `Taux_reussite_passes_away`, `Coups_frais_away`
      
    - Colonne cible : `resultat` (0 = victoire Away, 1 = victoire Home)
    """
)
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = pd.read_csv(fichier_entrainement)
    st.sidebar.write("📝 Aperçu :", df_entrainement.head())
else:
    st.sidebar.info("Aucun fichier chargé, utilisation de données synthétiques.")
    df_entrainement = generer_donnees_foot(n_samples=200)

# Liste complète des features
features = [
    "xG_home", "Tirs_cadrés_home", "Taux_conversion_home", "Touches_surface_home", "Passes_clés_home",
    "Interceptions_home", "Duels_defensifs_home", "xGA_home", "Arrêts_gardien_home", "Forme_recente_home", "Points_5_matchs_home",
    "possession_home", "corners_home", "Fautes_commises_home", "Cartons_jaunes_home", "Passes_decisives_home", "Taux_reussite_passes_home", "Coups_frais_home",
    "xG_away", "Tirs_cadrés_away", "Taux_conversion_away", "Touches_surface_away", "Passes_clés_away",
    "Interceptions_away", "Duels_defensifs_away", "xGA_away", "arrets_gardien_away", "Forme_recente_away", "Points_5_matchs_away",
    "possession_away", "corners_away", "Fautes_commises_away", "Cartons_jaunes_away", "Passes_decisives_away", "Taux_reussite_passes_away", "Coups_frais_away"
]
X_reel = df_entrainement[features]
y_reel = df_entrainement["resultat"]

# =====================================
# Pré-chargement et optimisation automatique des modèles via GridSearchCV
# =====================================
@st.cache_resource(show_spinner=False)
def load_models(X, y):
    saved = load_models_from_disk()
    if saved is not None:
        return saved
    else:
        model_log, best_score_log, cv_results_log = entrainer_modele_logistique(X, y)
        model_xgb, best_score_xgb, cv_results_xgb = entrainer_modele_xgb(X, y)
        model_rf, best_score_rf, cv_results_rf = entrainer_modele_rf(X, y)
        models = {
            "log": (model_log, best_score_log, cv_results_log),
            "xgb": (model_xgb, best_score_xgb, cv_results_xgb),
            "rf": (model_rf, best_score_rf, cv_results_rf)
        }
        save_models_to_disk(models)
        return models

@st.cache_resource(show_spinner=False)
def entrainer_modele_logistique(X, y):
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_, grid.cv_results_

@st.cache_resource(show_spinner=False)
def entrainer_modele_xgb(X, y):
    from sklearn.model_selection import GridSearchCV
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.7, 1.0]}
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_, grid.cv_results_

@st.cache_resource(show_spinner=False)
def entrainer_modele_rf(X, y):
    from sklearn.model_selection import GridSearchCV
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30],
                  'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]}
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_, grid.cv_results_

models = load_models(X_reel, y_reel)
modele_logistique, precision_logistique, cv_results_log = models["log"]
modele_xgb, precision_xgb, cv_results_xgb = models["xgb"]
modele_rf, precision_rf, cv_results_rf = models["rf"]

# Affichage détaillé des métriques de validation croisée
cv_data = pd.DataFrame({
    "Modèle": ["Régression Logistique", "XGBoost", "Random Forest"],
    "Précision Moyenne": [
        f"{precision_logistique*100:.2f}%", 
        f"{precision_xgb*100:.2f}%", 
        f"{precision_rf*100:.2f}%"
    ],  
    
def get_best_params(cv_results):
    """Renvoie les meilleurs hyperparamètres en se basant sur 'mean_test_score' si disponible."""
    mean_scores = cv_results.get('mean_test_score', None)
    if mean_scores is None:
        return "N/A"
    else:
        best_index = np.argmax(mean_scores)
        return cv_results['params'][best_index]

st.sidebar.markdown("### 📈 Métriques de Validation Croisée et Meilleurs Hyperparamètres")
st.sidebar.table(cv_data)

# =====================================
# Interface principale de saisie des données de match
# =====================================
st.title("⚽ Prédiction de Match de Football & Analyse Value Bet")
st.markdown("### Entrez les statistiques du match pour chaque équipe")

use_fictives = st.checkbox("Utiliser des données fictives", value=False)

col1, col2 = st.columns(2)
with col1:
    st.header("🏠 Équipe Home (Domicile)")
    if use_fictives:
        xG_home = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_home = float(random.randint(2, 10))
        taux_conversion_home = round(random.uniform(20, 40), 2)
        touches_surface_home = float(random.randint(15, 40))
        passes_cles_home = float(random.randint(3, 8))
        interceptions_home = float(random.randint(5, 15))
        duels_defensifs_home = float(random.randint(10, 30))
        xGA_home = round(random.uniform(1, 2.5), 2)
        arrets_gardien_home = float(random.randint(3, 7))
        forme_recente_home = float(random.randint(5, 15))
        points_5_matchs_home = float(random.randint(5, 15))
        possession_home = round(random.uniform(45, 70), 2)
        corners_home = float(random.randint(3, 10))
        fautes_commises_home = float(random.randint(8, 20))
        cartons_jaunes_home = float(random.randint(0, 3))
        passes_decisives_home = float(random.randint(0, 5))
        taux_reussite_passes_home = round(random.uniform(70, 90), 2)
        coups_frais_home = float(random.randint(0, 5))
        st.markdown("**🧪 Données fictives générées pour l'équipe Home**")
    else:
        xG_home = number_input_locale("⚽ xG (Home)", 1.50, key="xg_home")
        tirs_cadrés_home = number_input_locale("🎯 Tirs cadrés (Home)", 5.00, key="tc_home")
        taux_conversion_home = number_input_locale("🔥 Taux de conversion (Home)", 30.00, key="tcvt_home")
        touches_surface_home = number_input_locale("🤾‍♂️ Touches dans la surface (Home)", 25.00, key="ts_home")
        passes_cles_home = number_input_locale("🔑 Passes clés (Home)", 5.00, key="pc_home")
        interceptions_home = number_input_locale("🛡️ Interceptions (Home)", 8.00, key="int_home")
        duels_defensifs_home = number_input_locale("⚔️ Duels défensifs (Home)", 18.00, key="dd_home")
        xGA_home = number_input_locale("🚫 xGA (Home)", 1.20, key="xga_home")
        arrets_gardien_home = number_input_locale("🧤 Arrêts du gardien (Home)", 4.00, key="ag_home")
        forme_recente_home = number_input_locale("💪 Forme récente (Home)", 10.00, key="fr_home")
        points_5_matchs_home = number_input_locale("📊 Points 5 derniers matchs (Home)", 8.00, key="p5_home")
        possession_home = number_input_locale("📈 Possession (Home)", 55.00, key="pos_home")
        corners_home = number_input_locale("⚽ Corners (Home)", 5.00, key="corners_home")
        fautes_commises_home = number_input_locale("⚽ Fautes commises (Home)", 12, key="fc_home")
        cartons_jaunes_home = number_input_locale("🟡 Cartons jaunes (Home)", 1, key="cj_home")
        passes_decisives_home = number_input_locale("🎯 Passes décisives (Home)", 2, key="pd_home")
        taux_reussite_passes_home = number_input_locale("✅ Taux réussite passes (Home)", 80.00, key="trp_home")
        coups_frais_home = number_input_locale("🎯 Coups francs (Home)", 1, key="cf_home")

with col2:
    st.header("🏟️ Équipe Away (Extérieur)")
    if use_fictives:
        xG_away = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_away = float(random.randint(2, 10))
        taux_conversion_away = round(random.uniform(20, 40), 2)
        touches_surface_away = float(random.randint(15, 40))
        passes_cles_away = float(random.randint(3, 8))
        interceptions_away = float(random.randint(5, 15))
        duels_defensifs_away = float(random.randint(10, 30))
        xGA_away = round(random.uniform(1, 2.5), 2)
        arrets_gardien_away = float(random.randint(3, 7))
        forme_recente_away = float(random.randint(5, 15))
        points_5_matchs_away = float(random.randint(5, 15))
        possession_away = round(random.uniform(45, 70), 2)
        corners_away = float(random.randint(3, 10))
        fautes_commises_away = float(random.randint(8, 20))
        cartons_jaunes_away = float(random.randint(0, 3))
        passes_decisives_away = float(random.randint(0, 5))
        taux_reussite_passes_away = round(random.uniform(70, 90), 2)
        coups_frais_away = float(random.randint(0, 5))
        st.markdown("**🧪 Données fictives générées pour l'équipe Away**")
    else:
        xG_away = number_input_locale("⚽ xG (Away)", 1.00, key="xg_away")
        tirs_cadrés_away = number_input_locale("🎯 Tirs cadrés (Away)", 3.00, key="tc_away")
        taux_conversion_away = number_input_locale("🔥 Taux de conversion (Away)", 25.00, key="tcvt_away")
        touches_surface_away = number_input_locale("🤾‍♂️ Touches dans la surface (Away)", 20.00, key="ts_away")
        passes_cles_away = number_input_locale("🔑 Passes clés (Away)", 4.00, key="pc_away")
        interceptions_away = number_input_locale("🛡️ Interceptions (Away)", 7.00, key="int_away")
        duels_defensifs_away = number_input_locale("⚔️ Duels défensifs (Away)", 15.00, key="dd_away")
        xGA_away = number_input_locale("🚫 xGA (Away)", 1.50, key="xga_away")
        arrets_gardien_away = number_input_locale("🧤 Arrêts du gardien (Away)", 5.00, key="ag_away")
        forme_recente_away = number_input_locale("💪 Forme récente (Away)", 8.00, key="fr_away")
        points_5_matchs_away = number_input_locale("📊 Points 5 derniers matchs (Away)", 6.00, key="p5_away")
        possession_away = number_input_locale("📈 Possession (Away)", 50.00, key="pos_away")
        corners_away = number_input_locale("⚽ Corners (Away)", 4.00, key="corners_away")
        fautes_commises_away = number_input_locale("⚽ Fautes commises (Away)", 12, key="fc_away")
        cartons_jaunes_away = number_input_locale("🟡 Cartons jaunes (Away)", 1, key="cj_away")
        passes_decisives_away = number_input_locale("🎯 Passes décisives (Away)", 2, key="pd_away")
        taux_reussite_passes_away = number_input_locale("✅ Taux réussite passes (Away)", 80.00, key="trp_away")
        coups_frais_away = number_input_locale("🎯 Coups francs (Away)", 1, key="cf_away")

st.markdown("### 🎲 Analyse Value Bet")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    cote_A = number_input_locale("🏆 Cote - Victoire Home", 2.00, key="cote_home")
with col_odds2:
    cote_N = number_input_locale("🤝 Cote - Match Nul", 3.00, key="cote_n")
with col_odds3:
    cote_B = number_input_locale("🏟️ Cote - Victoire Away", 2.50, key="cote_away")

# =====================================
# Prédictions et affichage des résultats
# =====================================
if st.button("🔮 Prédire le Résultat"):
    # Prédiction via le modèle de Poisson en passant 36 variables
    victoire_home, victoire_away, match_nul, expected_buts_home, expected_buts_away = predire_resultat_match(
        xG_home, tirs_cadrés_home, taux_conversion_home, touches_surface_home, passes_cles_home,
        interceptions_home, duels_defensifs_home, xGA_home, arrets_gardien_home, forme_recente_home, points_5_matchs_home,
        possession_home, corners_home,
        fautes_commises_home, cartons_jaunes_home, passes_decisives_home, taux_reussite_passes_home, coups_frais_home,
        xG_away, tirs_cadrés_away, taux_conversion_away, touches_surface_away, passes_cles_away,
        interceptions_away, duels_defensifs_away, xGA_away, arrets_gardien_away, forme_recente_away, points_5_matchs_away,
        possession_away, corners_away,
        fautes_commises_away, cartons_jaunes_away, passes_decisives_away, taux_reussite_passes_away, coups_frais_away
    )
    
    st.markdown("## 📊 Résultats du Modèle de Poisson")
    data_poisson = {
        "Mesure": [
            "🏆 Prob. Victoire Home", "🤝 Prob. Match Nul", "🏟️ Prob. Victoire Away",
            "⚽ Buts attendus Home", "⚽ Buts attendus Away"
        ],
        "Valeur": [
            f"{victoire_home*100:.2f}%", f"{match_nul*100:.2f}%", f"{victoire_away*100:.2f}%",
            f"{expected_buts_home:.2f}", f"{expected_buts_away:.2f}"
        ]
    }
    st.table(pd.DataFrame(data_poisson))
    
    st.markdown("## 🔍 Résultats des Modèles de Classification")
    input_features = np.array([[ 
        xG_home, tirs_cadrés_home, taux_conversion_home, touches_surface_home, passes_cles_home,
        interceptions_home, duels_defensifs_home, xGA_home, arrets_gardien_home, forme_recente_home, points_5_matchs_home,
        possession_home, corners_home, fautes_commises_home, cartons_jaunes_home, passes_decisives_home, taux_reussite_passes_home, coups_frais_home,
        xG_away, tirs_cadrés_away, taux_conversion_away, touches_surface_away, passes_cles_away,
        interceptions_away, duels_defensifs_away, xGA_away, arrets_gardien_away, forme_recente_away, points_5_matchs_away,
        possession_away, corners_away, fautes_commises_away, cartons_jaunes_away, passes_decisives_away, taux_reussite_passes_away, coups_frais_away
    ]])
    st.write("**📐 Forme de l'input :**", input_features.shape)
    
    proba_log = modele_logistique.predict_proba(input_features)[0][1]
    prediction_log = "🏆 Victoire Home" if proba_log > 0.5 else "🏟️ Victoire Away"
    
    proba_xgb = modele_xgb.predict_proba(input_features)[0][1]
    prediction_xgb = "🏆 Victoire Home" if proba_xgb > 0.5 else "🏟️ Victoire Away"
    
    proba_rf = modele_rf.predict_proba(input_features)[0][1]
    prediction_rf = "🏆 Victoire Home" if proba_rf > 0.5 else "🏟️ Victoire Away"
    
    data_classif = {
        "Modèle": ["Régression Logistique", "XGBoost", "Random Forest"],
        "Prédiction": [prediction_log, prediction_xgb, prediction_rf],
        "Probabilité": [f"{proba_log*100:.2f}%", f"{proba_xgb*100:.2f}%", f"{proba_rf*100:.2f}%"],
        "Précision (moyenne / CV scores)": [
            f"{precision_logistique*100:.2f}% / {cv_results_log['mean_test_score'][np.argmax(cv_results_log['mean_test_score'])]}",
            f"{precision_xgb*100:.2f}% / {cv_results_xgb['mean_test_score'][np.argmax(cv_results_xgb['mean_test_score'])]}",
            f"{precision_rf*100:.2f}% / {cv_results_rf['mean_test_score'][np.argmax(cv_results_rf['mean_test_score'])]}"
        ]
    }
    st.table(pd.DataFrame(data_classif))
    
    st.markdown("## 🎲 Analyse Value Bet")
    outcomes = ["🏆 Victoire Home", "🤝 Match Nul", "🏟️ Victoire Away"]
    bookmaker_cotes = [cote_A, cote_N, cote_B]
    predicted_probs = [victoire_home, match_nul, victoire_away]
    
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
        dc_odds = number_input_locale("💰 Cote - Double Chance 1X", 1.50, key="dc1")
        dc_prob = victoire_home + match_nul
    elif dc_option == "X2 (🤝 ou 🏟️)":
        dc_odds = number_input_locale("💰 Cote - Double Chance X2", 1.60, key="dc2")
        dc_prob = match_nul + victoire_away
    else:
        dc_odds = number_input_locale("💰 Cote - Double Chance 12", 1.40, key="dc3")
        dc_prob = victoire_home + victoire_away

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
        "Prob Équipe Home": [poisson_prob(expected_buts_home, i) for i in range(6)],
        "Prob Équipe Away": [poisson_prob(expected_buts_away, i) for i in range(6)]
    })
    chart = alt.Chart(buts_data).mark_bar().encode(
        x=alt.X("Buts:O", title="Nombre de buts"),
        y=alt.Y("Prob Équipe Home:Q", title="Probabilité"),
        color=alt.value("#4CAF50")
    ).properties(title="Distribution des buts attendus - Équipe Home")
    st.altair_chart(chart, use_container_width=True)
