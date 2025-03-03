import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import altair as alt
import os
import pickle
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# =====================================
# Saisie utilisateur avec gestion de la virgule
# =====================================
def number_input_locale(label, value, key=None):
    """Permet de saisir des nombres avec une virgule comme séparateur décimal."""
    user_input = st.text_input(label, value=str(value), key=key)
    try:
        return float(user_input.replace(",", "."))
    except:
        return value

# =====================================
# Récupération des meilleurs hyperparamètres
# =====================================
def get_best_params(cv_results):
    """Extrait les meilleurs hyperparamètres à partir de GridSearchCV."""
    if 'mean_test_score' in cv_results:
        mean_scores = cv_results['mean_test_score']
    elif 'mean_validation_score' in cv_results:
        mean_scores = cv_results['mean_validation_score']
    else:
        return "N/A"
    best_index = np.argmax(mean_scores)
    return cv_results['params'][best_index]

# =====================================
# Calcul de la probabilité avec la loi de Poisson (utilise SciPy)
# =====================================
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts en utilisant scipy.stats.poisson.pmf."""
    return poisson.pmf(k, lam)

# =====================================
# Prédiction du résultat du match (modèle Poisson amélioré)
# =====================================
def predire_resultat_match(
    # Variables Home (13 variables)
    xG_home, tirs_cadrés_home, taux_conversion_home, touches_surface_home, passes_decisives_home,
    interceptions_home, duels_defensifs_home, xGA_home, arrets_gardien_home, forme_recente_home,
    possession_home, corners_home, fautes_commises_home,
    # Variables Away (13 variables)
    xG_away, tirs_cadrés_away, taux_conversion_away, touches_surface_away, passes_decisives_away,
    interceptions_away, duels_defensifs_away, xGA_away, arrets_gardien_away, forme_recente_away,
    possession_away, corners_away, fautes_commises_away,
    max_buts=5, home_advantage=1.05
):
    # Ajustement du xG Home avec l'avantage du terrain
    xG_home_adj = xG_home * home_advantage

    # Calcul de la note offensive pour Home (à domicile)
    Ro_home = (0.25 * xG_home_adj +
               0.20 * tirs_cadrés_home +
               0.10 * (taux_conversion_home / 100) +
               0.10 * touches_surface_home +
               0.10 * passes_decisives_home +
               0.05 * forme_recente_home +
               0.05 * (possession_home / 100) +
               0.05 * (corners_home / 10))
    
    # Calcul de la note défensive pour Away (en déplacement)
    Rd_away = (0.25 * xGA_away +
               0.20 * arrets_gardien_away +
               0.10 * interceptions_away +
               0.10 * duels_defensifs_away +
               0.05 * (corners_away / 10) +
               0.05 * fautes_commises_away)
    
    adj_xG_home = Ro_home / (Rd_away + 1)
    
    # Calcul de la note offensive pour Away (à l'extérieur)
    Ro_away = (0.25 * xG_away +
               0.20 * tirs_cadrés_away +
               0.10 * (taux_conversion_away / 100) +
               0.10 * touches_surface_away +
               0.10 * passes_decisives_away +
               0.05 * forme_recente_away +
               0.05 * (possession_away / 100) +
               0.05 * (corners_away / 10))
    
    # Calcul de la note défensive pour Home
    Rd_home = (0.25 * xGA_home +
               0.20 * arrets_gardien_home +
               0.10 * interceptions_home +
               0.10 * duels_defensifs_home +
               0.05 * (corners_home / 10) +
               0.05 * fautes_commises_home)
    
    adj_xG_away = Ro_away / (Rd_home + 1)
    
    # Calcul des distributions de buts via np.outer
    prob_home = np.array([poisson_prob(adj_xG_home, i) for i in range(max_buts+1)])
    prob_away = np.array([poisson_prob(adj_xG_away, i) for i in range(max_buts+1)])
    matrice = np.outer(prob_home, prob_away)
    victoire_home = np.sum(np.triu(matrice, k=1))
    victoire_away = np.sum(np.tril(matrice, k=-1))
    match_nul = np.sum(np.diag(matrice))
    expected_buts_home = np.sum(np.arange(max_buts+1) * prob_home)
    expected_buts_away = np.sum(np.arange(max_buts+1) * prob_away)
    
    return victoire_home, victoire_away, match_nul, expected_buts_home, expected_buts_away

# =====================================
# Prédiction Double Chance améliorée
# =====================================
def predire_double_chance(prob_home, match_nul, prob_away, option, dc_factor=1.0):
    """
    Calcule la probabilité double chance.
    - Option "1X (🏠 ou 🤝)" : Victoire Home ou Match Nul.
    - Option "X2 (🤝 ou 🏟️)" : Match Nul ou Victoire Away.
    - Option "12 (🏠 ou 🏟️)" : Victoire Home ou Victoire Away.
    Applique un facteur correctif dc_factor.
    """
    if option == "1X (🏠 ou 🤝)":
        base_prob = prob_home + match_nul
    elif option == "X2 (🤝 ou 🏟️)":
        base_prob = match_nul + prob_away
    elif option == "12 (🏠 ou 🏟️)":
        base_prob = prob_home + prob_away
    else:
        base_prob = 0
    return dc_factor * base_prob

# =====================================
# Calcul de la Value Bet
# =====================================
def calculer_value_bet(prob, cote):
    """Calcule la valeur espérée et renvoie la recommandation."""
    ev = (prob * cote) - 1
    recommendation = "✅ Value Bet" if ev > 0 else "❌ Pas de Value Bet"
    return ev, recommendation

# =====================================
# Sauvegarde et chargement des modèles
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
# Génération de données d'entraînement synthétiques réalistes (moyennes par match)
# =====================================
def generer_donnees_foot(n_samples=200):
    data = {}
    # Home (13 variables)
    data["xG_home"] = np.random.uniform(0.5, 2.5, n_samples)
    data["Tirs_cadrés_home"] = np.random.randint(2, 11, n_samples)
    data["Taux_conversion_home"] = np.random.uniform(20, 40, n_samples)
    data["Touches_surface_home"] = np.random.randint(15, 41, n_samples)
    data["Passes_decisives_home"] = np.random.randint(1, 8, n_samples)
    data["Interceptions_home"] = np.random.randint(4, 16, n_samples)
    data["Duels_defensifs_home"] = np.random.randint(8, 26, n_samples)
    data["xGA_home"] = np.random.uniform(0.5, 2.5, n_samples)
    data["Arrêts_gardien_home"] = np.random.randint(3, 9, n_samples)
    data["Forme_recente_home"] = np.random.randint(5, 16, n_samples)
    data["possession_home"] = np.random.uniform(45, 70, n_samples)
    data["corners_home"] = np.random.randint(3, 11, n_samples)
    data["Fautes_commises_home"] = np.random.randint(8, 21, n_samples)
    
    # Away (13 variables)
    data["xG_away"] = np.random.uniform(0.5, 2.5, n_samples)
    data["Tirs_cadrés_away"] = np.random.randint(2, 11, n_samples)
    data["Taux_conversion_away"] = np.random.uniform(20, 40, n_samples)
    data["Touches_surface_away"] = np.random.randint(15, 41, n_samples)
    data["Passes_decisives_away"] = np.random.randint(1, 8, n_samples)
    data["Interceptions_away"] = np.random.randint(4, 16, n_samples)
    data["Duels_defensifs_away"] = np.random.randint(8, 26, n_samples)
    data["xGA_away"] = np.random.uniform(0.5, 2.5, n_samples)
    data["arrets_gardien_away"] = np.random.randint(3, 9, n_samples)
    data["Forme_recente_away"] = np.random.randint(5, 16, n_samples)
    data["possession_away"] = np.random.uniform(45, 70, n_samples)
    data["corners_away"] = np.random.randint(3, 11, n_samples)
    data["Fautes_commises_away"] = np.random.randint(8, 21, n_samples)
    
    df = pd.DataFrame(data)
    # Génération de la cible "resultat" (1 = victoire Home, 0 = victoire Away)
    results = []
    for _, row in df.iterrows():
        victoire_home, victoire_away, match_nul, _, _ = predire_resultat_match(
            row["xG_home"], row["Tirs_cadrés_home"], row["Taux_conversion_home"], row["Touches_surface_home"], row["Passes_decisives_home"],
            row["Interceptions_home"], row["Duels_defensifs_home"], row["xGA_home"], row["Arrêts_gardien_home"], row["Forme_recente_home"],
            row["possession_home"], row["corners_home"], row["Fautes_commises_home"],
            row["xG_away"], row["Tirs_cadrés_away"], row["Taux_conversion_away"], row["Touches_surface_away"], row["Passes_decisives_away"],
            row["Interceptions_away"], row["Duels_defensifs_away"], row["xGA_away"], row["arrets_gardien_away"], row["Forme_recente_away"],
            row["possession_away"], row["corners_away"], row["Fautes_commises_away"]
        )
        results.append(1 if victoire_home >= victoire_away else 0)
    df["resultat"] = results
    return df

# =====================================
# Données d'entraînement (CSV ou synthétiques)
# =====================================
st.sidebar.header("📊 Données d'Entraînement")
st.sidebar.markdown(
    """
    **Format du CSV attendu :**
    
    *Équipe Home (13 variables)*:
    xG_home, Tirs_cadrés_home, Taux_conversion_home, Touches_surface_home, Passes_decisives_home,
    Interceptions_home, Duels_defensifs_home, xGA_home, Arrêts_gardien_home, Forme_recente_home,
    possession_home, corners_home, Fautes_commises_home
    
    *Équipe Away (13 variables)*:
    xG_away, Tirs_cadrés_away, Taux_conversion_away, Touches_surface_away, Passes_decisives_away,
    Interceptions_away, Duels_defensifs_away, xGA_away, arrets_gardien_away, Forme_recente_away,
    possession_away, corners_away, Fautes_commises_away
    
    Colonne cible : resultat (1 = victoire Home, 0 = victoire Away)
    """
)
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = pd.read_csv(fichier_entrainement)
    st.sidebar.write("📝 Aperçu :", df_entrainement.head())
else:
    st.sidebar.info("Aucun fichier chargé, utilisation de données synthétiques.")
    df_entrainement = generer_donnees_foot(n_samples=200)

# Liste des features (26 variables)
features = [
    "xG_home", "Tirs_cadrés_home", "Taux_conversion_home", "Touches_surface_home", "Passes_decisives_home",
    "Interceptions_home", "Duels_defensifs_home", "xGA_home", "Arrêts_gardien_home", "Forme_recente_home",
    "possession_home", "corners_home", "Fautes_commises_home",
    "xG_away", "Tirs_cadrés_away", "Taux_conversion_away", "Touches_surface_away", "Passes_decisives_away",
    "Interceptions_away", "Duels_defensifs_away", "xGA_away", "arrets_gardien_away", "Forme_recente_away",
    "possession_away", "corners_away", "Fautes_commises_away"
]
X_reel = df_entrainement[features]
y_reel = df_entrainement["resultat"]

# =====================================
# Chargement et entraînement des modèles
# =====================================
@st.cache_resource(show_spinner=False)
def load_models(X, y):
    saved = load_models_from_disk()
    if saved is not None:
        if hasattr(saved["log"][0], "n_features_in_") and saved["log"][0].n_features_in_ != len(features):
            os.remove("trained_models.pkl")
            saved = None
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
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_, grid.cv_results_

@st.cache_resource(show_spinner=False)
def entrainer_modele_xgb(X, y):
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.1, 0.2],
                  'subsample': [0.7, 1.0]}
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_, grid.cv_results_

@st.cache_resource(show_spinner=False)
def entrainer_modele_rf(X, y):
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [None, 10, 20, 30],
                  'max_features': ['sqrt', 'log2'],
                  'min_samples_split': [2, 5, 10]}
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_, grid.cv_results_

models = load_models(X_reel, y_reel)
modele_logistique, precision_logistique, cv_results_log = models["log"]
modele_xgb, precision_xgb, cv_results_xgb = models["xgb"]
modele_rf, precision_rf, cv_results_rf = models["rf"]

# Affichage dans la sidebar
cv_data = pd.DataFrame({
    "Modèle": ["Régression Logistique", "XGBoost", "Random Forest"],
    "Précision Moyenne": [
        f"{precision_logistique*100:.2f}%",
        f"{precision_xgb*100:.2f}%",
        f"{precision_rf*100:.2f}%"
    ],
    "Meilleurs Paramètres": [
        get_best_params(cv_results_log),
        get_best_params(cv_results_xgb),
        get_best_params(cv_results_rf)
    ]
})
st.sidebar.dataframe(cv_data)

# =====================================
# Interface principale de saisie des données de match
# =====================================
st.title("⚽ Prédiction de Match de Football & Analyse Value Bet")
st.markdown("### Entrez les statistiques par match pour chaque équipe")

use_fictives = st.checkbox("Utiliser des données fictives", value=False)

col1, col2 = st.columns(2)
with col1:
    st.header("🏠 Équipe Home (Domicile)")
    if use_fictives:
        xG_home = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_home = float(random.randint(2, 10))
        taux_conversion_home = round(random.uniform(20, 40), 2)
        touches_surface_home = float(random.randint(15, 40))
        passes_decisives_home = float(random.randint(1, 7))
        interceptions_home = float(random.randint(4, 16))
        duels_defensifs_home = float(random.randint(8, 26))
        xGA_home = round(random.uniform(0.5, 2.5), 2)
        arrets_gardien_home = float(random.randint(3, 9))
        forme_recente_home = float(random.randint(5, 15))
        possession_home = round(random.uniform(45, 70), 2)
        corners_home = float(random.randint(3, 11))
        fautes_commises_home = float(random.randint(8, 21))
        st.markdown("**🧪 Données fictives générées pour l'équipe Home**")
    else:
        xG_home = number_input_locale("⚽ xG (Home, par match)", 1.50, key="xg_home")
        tirs_cadrés_home = number_input_locale("🎯 Tirs cadrés (Home, par match)", 5.00, key="tc_home")
        taux_conversion_home = number_input_locale("🔥 Taux de conversion (Home, % par match)", 30.00, key="tcvt_home")
        touches_surface_home = number_input_locale("🤾‍♂️ Touches dans la surface (Home, par match)", 25.00, key="ts_home")
        passes_decisives_home = number_input_locale("🔑 Passes décisives (Home, par match)", 5.00, key="pd_home")
        interceptions_home = number_input_locale("🛡️ Interceptions (Home, par match)", 8.00, key="int_home")
        duels_defensifs_home = number_input_locale("⚔️ Duels défensifs (Home, par match)", 16.00, key="dd_home")
        xGA_home = number_input_locale("🚫 xGA (Home, par match)", 1.20, key="xga_home")
        arrets_gardien_home = number_input_locale("🧤 Arrêts du gardien (Home, par match)", 4.00, key="ag_home")
        forme_recente_home = number_input_locale("💪 Forme récente (Home, par match)", 10.00, key="fr_home")
        possession_home = number_input_locale("📈 Possession (Home, % par match)", 55.00, key="pos_home")
        corners_home = number_input_locale("⚽ Corners (Home, par match)", 5.00, key="corners_home")
        fautes_commises_home = number_input_locale("⚽ Fautes commises (Home, par match)", 12, key="fc_home")

with col2:
    st.header("🏟️ Équipe Away (Extérieur)")
    if use_fictives:
        xG_away = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_away = float(random.randint(2, 10))
        taux_conversion_away = round(random.uniform(20, 40), 2)
        touches_surface_away = float(random.randint(15, 40))
        passes_decisives_away = float(random.randint(1, 7))
        interceptions_away = float(random.randint(4, 16))
        duels_defensifs_away = float(random.randint(8, 26))
        xGA_away = round(random.uniform(0.5, 2.5), 2)
        arrets_gardien_away = float(random.randint(3, 9))
        forme_recente_away = float(random.randint(5, 15))
        possession_away = round(random.uniform(45, 70), 2)
        corners_away = float(random.randint(3, 11))
        fautes_commises_away = float(random.randint(8, 21))
        st.markdown("**🧪 Données fictives générées pour l'équipe Away**")
    else:
        xG_away = number_input_locale("⚽ xG (Away, par match)", 1.00, key="xg_away")
        tirs_cadrés_away = number_input_locale("🎯 Tirs cadrés (Away, par match)", 3.00, key="tc_away")
        taux_conversion_away = number_input_locale("🔥 Taux de conversion (Away, % par match)", 25.00, key="tcvt_away")
        touches_surface_away = number_input_locale("🤾‍♂️ Touches dans la surface (Away, par match)", 20.00, key="ts_away")
        passes_decisives_away = number_input_locale("🔑 Passes décisives (Away, par match)", 4.00, key="pd_away")
        interceptions_away = number_input_locale("🛡️ Interceptions (Away, par match)", 7.00, key="int_away")
        duels_defensifs_away = number_input_locale("⚔️ Duels défensifs (Away, par match)", 14.00, key="dd_away")
        xGA_away = number_input_locale("🚫 xGA (Away, par match)", 1.50, key="xga_away")
        arrets_gardien_away = number_input_locale("🧤 Arrêts du gardien (Away, par match)", 5.00, key="ag_away")
        forme_recente_away = number_input_locale("💪 Forme récente (Away, par match)", 8.00, key="fr_away")
        possession_away = number_input_locale("📈 Possession (Away, % par match)", 50.00, key="pos_away")
        corners_away = number_input_locale("⚽ Corners (Away, par match)", 4.00, key="corners_away")
        fautes_commises_away = number_input_locale("⚽ Fautes commises (Away, par match)", 12, key="fc_away")

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
    # Préparation de l'input_features (forme (1, 26))
    input_features = np.array([[
        xG_home, tirs_cadrés_home, taux_conversion_home, touches_surface_home, passes_decisives_home,
        interceptions_home, duels_defensifs_home, xGA_home, arrets_gardien_home, forme_recente_home,
        possession_home, corners_home, fautes_commises_home,
        xG_away, tirs_cadrés_away, taux_conversion_away, touches_surface_away, passes_decisives_away,
        interceptions_away, duels_defensifs_away, xGA_away, arrets_gardien_away, forme_recente_away,
        possession_away, corners_away, fautes_commises_away
    ]])
    
    # Vérification de la dimension d'input
    if input_features.shape[1] != modele_logistique.n_features_in_:
        st.error(f"Erreur: Le modèle attend {modele_logistique.n_features_in_} caractéristiques, mais l'input possède {input_features.shape[1]}. Veuillez supprimer le fichier 'trained_models.pkl' pour réentraîner les modèles.")
    else:
        st.write("**📐 Forme de l'input :**", input_features.shape)
    
        # Prédiction via la méthode Poisson améliorée
        victoire_home, victoire_away, match_nul, expected_buts_home, expected_buts_away = predire_resultat_match(
            xG_home, tirs_cadrés_home, taux_conversion_home, touches_surface_home, passes_decisives_home,
            interceptions_home, duels_defensifs_home, xGA_home, arrets_gardien_home, forme_recente_home,
            possession_home, corners_home, fautes_commises_home,
            xG_away, tirs_cadrés_away, taux_conversion_away, touches_surface_away, passes_decisives_away,
            interceptions_away, duels_defensifs_away, xGA_away, arrets_gardien_away, forme_recente_away,
            possession_away, corners_away, fautes_commises_away
        )
    
        st.markdown("## 📊 Résultats du Modèle de Poisson")
        data_poisson = pd.DataFrame({
            "Mesure": ["🏆 Prob. Victoire Home", "🤝 Prob. Match Nul", "🏟️ Prob. Victoire Away", "⚽ Buts attendus Home", "⚽ Buts attendus Away"],
            "Valeur": [f"{victoire_home*100:.2f}%", f"{match_nul*100:.2f}%", f"{victoire_away*100:.2f}%",
                       f"{expected_buts_home:.2f}", f"{expected_buts_away:.2f}"]
        })
        st.table(data_poisson)
    
        st.markdown("## 🔍 Résultats des Modèles de Classification")
        proba_log = modele_logistique.predict_proba(input_features)[0][1]
        prediction_log = "🏆 Victoire Home" if proba_log > 0.5 else "🏟️ Victoire Away"
    
        proba_xgb = modele_xgb.predict_proba(input_features)[0][1]
        prediction_xgb = "🏆 Victoire Home" if proba_xgb > 0.5 else "🏟️ Victoire Away"
    
        proba_rf = modele_rf.predict_proba(input_features)[0][1]
        prediction_rf = "🏆 Victoire Home" if proba_rf > 0.5 else "🏟️ Victoire Away"
    
        classif_data = pd.DataFrame({
            "Modèle": ["Régression Logistique", "XGBoost", "Random Forest"],
            "Prédiction": [prediction_log, prediction_xgb, prediction_rf],
            "Probabilité": [f"{proba_log*100:.2f}%", f"{proba_xgb*100:.2f}%", f"{proba_rf*100:.2f}%"],
            "Précision Moyenne": [
                f"{precision_logistique*100:.2f}%",
                f"{precision_xgb*100:.2f}%",
                f"{precision_rf*100:.2f}%"
            ],
            "Meilleurs Paramètres": [
                get_best_params(cv_results_log),
                get_best_params(cv_results_xgb),
                get_best_params(cv_results_rf)
            ]
        })
        st.dataframe(classif_data)
    
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
        st.dataframe(pd.DataFrame(value_bet_data))
    
        st.markdown("## 🎯 Analyse Double Chance")
        dc_option = st.selectbox("Sélectionnez l'option Double Chance", ["1X (🏠 ou 🤝)", "X2 (🤝 ou 🏟️)", "12 (🏠 ou 🏟️)"])
        dc_factor = 1.0  # Facteur correctif ajustable
        dc_prob = predire_double_chance(victoire_home, match_nul, victoire_away, option=dc_option, dc_factor=dc_factor)
    
        dc_odds = number_input_locale(f"💰 Cote - Double Chance ({dc_option})", 
                                       1.50 if dc_option=="1X (🏠 ou 🤝)" else 1.60 if dc_option=="X2 (🤝 ou 🏟️)" else 1.40, 
                                       key="dc_option")
        dc_implied = 1 / dc_odds
        dc_ev, dc_recommendation = calculer_value_bet(dc_prob, dc_odds)
    
        dc_data = pd.DataFrame({
            "Option": [dc_option],
            "Cote": [dc_odds],
            "Prob. Impliquée": [f"{dc_implied*100:.2f}%"],
            "Prob. Prédite": [f"{dc_prob*100:.2f}%"],
            "Valeur Espérée": [f"{dc_ev:.2f}"],
            "Double Chance": [dc_recommendation]
        })
        st.dataframe(dc_data)
    
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

# =====================================
# Fonction pour prédire la Double Chance
# =====================================
def predire_double_chance(prob_home, match_nul, prob_away, option, dc_factor=1.0):
    """
    Calcule la probabilité pour l'option Double Chance choisie.
      - "1X (🏠 ou 🤝)" : somme des probabilités Home et Match Nul.
      - "X2 (🤝 ou 🏟️)" : somme des probabilités Match Nul et Away.
      - "12 (🏠 ou 🏟️)" : somme des probabilités Home et Away.
    Applique ensuite un facteur correctif dc_factor.
    """
    if option == "1X (🏠 ou 🤝)":
        base_prob = prob_home + match_nul
    elif option == "X2 (🤝 ou 🏟️)":
        base_prob = match_nul + prob_away
    elif option == "12 (🏠 ou 🏟️)":
        base_prob = prob_home + prob_away
    else:
        base_prob = 0
    return dc_factor * base_prob
