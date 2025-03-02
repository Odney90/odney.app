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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# Jeu de données d'entraînement par défaut
# ===============================
default_training_data = """
xG_A,Tirs_cadrés_A,Taux_conversion_A,Touches_surface_A,Passes_clés_A,Interceptions_A,Duels_defensifs_A,xGA_A,Arrêts_gardien_A,Forme_recente_A,Points_5_matchs_A,possession_A,corners_A,xG_B,Tirs_cadrés_B,Taux_conversion_B,Touches_surface_B,Passes_clés_B,Interceptions_B,Duels_defensifs_B,xGA_B,Arrêts_du_gardien_B,Forme_recente_B,Points_5_matchs_B,possession_B,corners_B,resultat
1.3,5,28,27,6,7,20,1.1,4,9,8,57,5,1.0,4,30,25,8,19,15,1.3,5,7,9,50,4,1
0.9,4,30,23,5,8,19,1.4,5,8,7,52,3,1.5,6,32,28,7,21,1.0,4,10,9,60,6,0
1.0,5,27,26,5,6,22,1.2,3,10,8,54,4,0.8,3,22,20,10,15,1.6,6,6,5,48,3,1
1.2,6,29,28,6,7,21,1.0,4,11,9,56,5,1.1,4,28,24,9,18,1.2,5,8,8,53,4,1
0.8,4,31,24,5,8,18,1.3,5,7,7,50,3,1.3,5,31,27,8,20,1.1,4,9,7,55,4,0
1.4,5,28,27,6,7,20,1.1,4,9,8,57,5,1.0,4,30,25,8,19,15,1.3,5,7,9,50,4,1
1.1,5,26,25,5,7,21,1.2,4,10,8,54,4,1.2,4,29,26,7,18,1.3,5,8,8,52,4,1
1.0,4,30,24,5,8,19,1.4,5,8,7,52,3,1.5,6,32,28,7,21,1.0,4,10,9,60,6,0
1.3,6,28,27,6,7,20,1.1,4,9,8,57,5,1.0,4,30,25,8,19,15,1.3,5,7,9,50,4,1
0.9,5,29,25,5,8,20,1.3,5,8,7,53,4,1.2,5,28,24,8,20,1.1,4,9,7,55,4,0
"""

# ===============================
# Fonctions Utilitaires
# ===============================
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def load_training_data(file):
    """Charge le fichier CSV d'entraînement, nettoie les données et vérifie que toutes les colonnes requises sont présentes."""
    required_columns = [
        "xG_A", "Tirs_cadrés_A", "Taux_conversion_A", "Touches_surface_A", "Passes_clés_A",
        "Interceptions_A", "Duels_defensifs_A", "xGA_A", "Arrêts_gardien_A", "Forme_recente_A", "Points_5_matchs_A",
        "possession_A", "corners_A",
        "xG_B", "Tirs_cadrés_B", "Taux_conversion_B", "Touches_surface_B", "Passes_clés_B",
        "Interceptions_B", "Duels_defensifs_B", "xGA_B", "Arrêts_du_gardien_B", "Forme_recente_B", "Points_5_matchs_B",
        "possession_B", "corners_B", "resultat"
    ]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error("Erreur lors du chargement du fichier CSV.")
        return None

    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna()
    # Convertir la colonne "resultat" en numérique, forçant les erreurs à NaN et supprimer ensuite
    df["resultat"] = pd.to_numeric(df["resultat"], errors="coerce")
    df = df.dropna(subset=["resultat"])
    df["resultat"] = df["resultat"].astype(int)
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Les colonnes suivantes sont manquantes dans le fichier CSV : {', '.join(missing)}")
        return None

    st.sidebar.write("Exemple de 'resultat':", df["resultat"].head())
    return df

def validate_numeric_input(value, min_val, max_val, var_name):
    """Vérifie si la valeur est dans l'intervalle [min_val, max_val] et affiche un avertissement le cas échéant."""
    if not (min_val <= value <= max_val):
        st.warning(f"⚠️ La valeur de '{var_name}' ({value}) doit être comprise entre {min_val} et {max_val}.")
    return value

# ===============================
# Entraînement des modèles (avec cache et session_state)
# ===============================
@st.cache_resource(show_spinner=False)
def train_models(X, y):
    """Entraîne les modèles de classification et renvoie un dictionnaire contenant les modèles et leur précision."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Régression Logistique
    model_log = LogisticRegression()
    model_log.fit(X_train, y_train)
    prec_log = accuracy_score(y_test, model_log.predict(X_test))
    # XGBoost
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    prec_xgb = accuracy_score(y_test, model_xgb.predict(X_test))
    # Random Forest
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train, y_train)
    prec_rf = accuracy_score(y_test, model_rf.predict(X_test))
    return {"logistic": (model_log, prec_log), "xgb": (model_xgb, prec_xgb), "rf": (model_rf, prec_rf)}

if 'models' not in st.session_state:
    st.session_state.models = None

# ===============================
# Modèle de prédiction par Poisson
# ===============================
def predict_match(
    xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A,
    interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
    possession_A, corners_A,
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

# ===============================
# Interface pour les données d'entraînement
# ===============================
st.sidebar.header("📊 Données d'Entraînement")
st.sidebar.markdown(
    """
    **Format du fichier CSV attendu (moyennes par match) :**

    - **Pour l'Équipe A (Domicile)** :
      ⚽ `xG_A`, 🎯 `Tirs_cadrés_A`, 📈 `Taux_conversion_A`, 🔥 `Touches_surface_A`, 🔑 `Passes_clés_A`,
      🛡️ `Interceptions_A`, 💪 `Duels_defensifs_A`, 🚫 `xGA_A`, 🧤 `Arrêts_gardien_A`, 📊 `Forme_recente_A`, 
      ⭐ `Points_5_matchs_A`, ⏱️ `possession_A`, ⚽ `corners_A`

    - **Pour l'Équipe B (Extérieur)** :
      ⚽ `xG_B`, 🎯 `Tirs_cadrés_B`, 📈 `Taux_conversion_B`, 🔥 `Touches_surface_B`, 🔑 `Passes_clés_B`,
      🛡️ `Interceptions_B`, 💪 `Duels_defensifs_B`, 🚫 `xGA_B`, 🧤 `Arrêts_du_gardien_B`, 📊 `Forme_recente_B`, 
      ⭐ `Points_5_matchs_B`, ⏱️ `possession_B`, ⚽ `corners_B`

    - **Colonne cible** :
      `resultat` (1 = victoire de l'Équipe A, 0 = victoire de l'Équipe B)

    **Remarque :** Les statistiques doivent être des moyennes par match.
    """
)
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = load_training_data(fichier_entrainement)
else:
    st.sidebar.info("Aucun fichier chargé, utilisation des données d'entraînement par défaut.")
    df_entrainement = pd.read_csv(StringIO(default_training_data))
    
if df_entrainement is not None:
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
else:
    st.error("Aucune donnée d'entraînement disponible.")

# ===============================
# Entraînement des modèles (avec cache et session_state)
# ===============================
if y_reel is not None:
    if st.session_state.models is None:
        st.session_state.models = train_models(X_reel, y_reel)
    model_log, prec_log = st.session_state.models["logistic"]
    model_xgb, prec_xgb = st.session_state.models["xgb"]
    model_rf, prec_rf = st.session_state.models["rf"]

# ===============================
# Interface principale pour la saisie des données de match
# ===============================
st.title("⚽ Prédiction de Match de Football & Analyse Value Bet")
st.markdown("### Entrez les statistiques du match (moyennes par match) pour chaque équipe")

with st.expander("📖 Comment remplir les données ?"):
    st.markdown("""
    - **Les statistiques sont des moyennes par match.**
    - Par exemple, entrez le xG moyen, le nombre moyen de tirs cadrés, etc.
    - Pour les taux (conversion, possession), saisissez des valeurs décimales (ex: 30.00, 55.00).
    - Assurez-vous que les valeurs respectent des plages raisonnables (ex: possession entre 0 et 100).
    """)

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
        st.markdown("**📊 Données fictives générées pour l'Équipe A**")
    else:
        xG_A = st.number_input("⚽ xG (Équipe A, par match)", value=1.50, format="%.2f")
        tirs_cadrés_A = st.number_input("🎯 Tirs cadrés (Équipe A, par match)", value=5.00, format="%.2f")
        taux_conversion_A = st.number_input("📈 Taux de conversion (%) (Équipe A, par match)", value=30.00, format="%.2f")
        touches_surface_A = st.number_input("🔥 Touches dans la surface (Équipe A, par match)", value=25.00, format="%.2f")
        passes_cles_A = st.number_input("🔑 Passes clés (Équipe A, par match)", value=5.00, format="%.2f")
        interceptions_A = st.number_input("🛡️ Interceptions (Équipe A, par match)", value=8.00, format="%.2f")
        duels_defensifs_A = st.number_input("💪 Duels défensifs gagnés (Équipe A, par match)", value=18.00, format="%.2f")
        xGA_A = st.number_input("🚫 xGA (Équipe A, par match)", value=1.20, format="%.2f")
        arrets_gardien_A = st.number_input("🧤 Arrêts du gardien (Équipe A, par match)", value=4.00, format="%.2f")
        forme_recente_A = st.number_input("📊 Forme récente (points cumulés, Équipe A)", value=10.00, format="%.2f")
        points_5_matchs_A = st.number_input("⭐ Points sur les 5 derniers matchs (Équipe A)", value=8.00, format="%.2f")
        possession_A = st.number_input("⏱️ Possession moyenne (%) (Équipe A)", value=55.00, format="%.2f")
        corners_A = st.number_input("⚽ Corners (Équipe A, par match)", value=5.00, format="%.2f")

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
        st.markdown("**📊 Données fictives générées pour l'Équipe B**")
    else:
        xG_B = st.number_input("⚽ xG (Équipe B, par match)", value=1.00, format="%.2f")
        tirs_cadrés_B = st.number_input("🎯 Tirs cadrés (Équipe B, par match)", value=3.00, format="%.2f")
        taux_conversion_B = st.number_input("📈 Taux de conversion (%) (Équipe B, par match)", value=25.00, format="%.2f")
        touches_surface_B = st.number_input("🔥 Touches dans la surface (Équipe B, par match)", value=20.00, format="%.2f")
        passes_cles_B = st.number_input("🔑 Passes clés (Équipe B, par match)", value=4.00, format="%.2f")
        interceptions_B = st.number_input("🛡️ Interceptions (Équipe B, par match)", value=7.00, format="%.2f")
        duels_defensifs_B = st.number_input("💪 Duels défensifs gagnés (Équipe B, par match)", value=15.00, format="%.2f")
        xGA_B = st.number_input("🚫 xGA (Équipe B, par match)", value=1.50, format="%.2f")
        arrets_gardien_B = st.number_input("🧤 Arrêts du gardien (Équipe B, par match)", value=5.00, format="%.2f")
        forme_recente_B = st.number_input("📊 Forme récente (points cumulés, Équipe B)", value=8.00, format="%.2f")
        points_5_matchs_B = st.number_input("⭐ Points sur les 5 derniers matchs (Équipe B)", value=6.00, format="%.2f")
        possession_B = st.number_input("⏱️ Possession moyenne (%) (Équipe B)", value=50.00, format="%.2f")
        corners_B = st.number_input("⚽ Corners (Équipe B, par match)", value=4.00, format="%.2f")

st.markdown("### 🎲 Analyse Value Bet")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    cote_A = st.number_input("💰 Cote Bookmaker - Victoire Équipe A", value=2.00, format="%.2f")
with col_odds2:
    cote_N = st.number_input("💰 Cote Bookmaker - Match Nul", value=3.00, format="%.2f")
with col_odds3:
    cote_B = st.number_input("💰 Cote Bookmaker - Victoire Équipe B", value=2.50, format="%.2f")

# ===============================
# Chargement des données d'entraînement (ou utilisation par défaut)
# ===============================
st.sidebar.header("📊 Données d'Entraînement")
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = load_training_data(fichier_entrainement)
else:
    st.sidebar.info("Aucun fichier chargé, utilisation des données d'entraînement par défaut.")
    df_entrainement = pd.read_csv(StringIO(default_training_data))
    
if df_entrainement is not None:
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
else:
    st.error("Aucune donnée d'entraînement disponible.")

# ===============================
# Entraînement des modèles (avec cache et session_state)
# ===============================
if y_reel is not None:
    if st.session_state.models is None:
        st.session_state.models = train_models(X_reel, y_reel)
    model_log, prec_log = st.session_state.models["logistic"]
    model_xgb, prec_xgb = st.session_state.models["xgb"]
    model_rf, prec_rf = st.session_state.models["rf"]

# ===============================
# Interface principale pour la saisie des données de match
# ===============================
st.title("⚽ Prédiction de Match de Football & Analyse Value Bet")
st.markdown("### Entrez les statistiques du match (moyennes par match) pour chaque équipe")

with st.expander("📖 Comment remplir les données ?"):
    st.markdown("""
    - **Les statistiques sont des moyennes par match.**
    - Par exemple, entrez le xG moyen, le nombre moyen de tirs cadrés, etc.
    - Pour les taux (conversion, possession), saisissez des valeurs décimales (ex: 30.00, 55.00).
    - Assurez-vous que les valeurs respectent des plages raisonnables (ex: possession entre 0 et 100).
    """)

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
        st.markdown("**📊 Données fictives générées pour l'Équipe A**")
    else:
        xG_A = st.number_input("⚽ xG (Équipe A, par match)", value=1.50, format="%.2f")
        tirs_cadrés_A = st.number_input("🎯 Tirs cadrés (Équipe A, par match)", value=5.00, format="%.2f")
        taux_conversion_A = st.number_input("📈 Taux de conversion (%) (Équipe A, par match)", value=30.00, format="%.2f")
        touches_surface_A = st.number_input("🔥 Touches dans la surface (Équipe A, par match)", value=25.00, format="%.2f")
        passes_cles_A = st.number_input("🔑 Passes clés (Équipe A, par match)", value=5.00, format="%.2f")
        interceptions_A = st.number_input("🛡️ Interceptions (Équipe A, par match)", value=8.00, format="%.2f")
        duels_defensifs_A = st.number_input("💪 Duels défensifs gagnés (Équipe A, par match)", value=18.00, format="%.2f")
        xGA_A = st.number_input("🚫 xGA (Équipe A, par match)", value=1.20, format="%.2f")
        arrets_gardien_A = st.number_input("🧤 Arrêts du gardien (Équipe A, par match)", value=4.00, format="%.2f")
        forme_recente_A = st.number_input("📊 Forme récente (points cumulés, Équipe A)", value=10.00, format="%.2f")
        points_5_matchs_A = st.number_input("⭐ Points sur les 5 derniers matchs (Équipe A)", value=8.00, format="%.2f")
        possession_A = st.number_input("⏱️ Possession moyenne (%) (Équipe A)", value=55.00, format="%.2f")
        corners_A = st.number_input("⚽ Corners (Équipe A, par match)", value=5.00, format="%.2f")

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
        st.markdown("**📊 Données fictives générées pour l'Équipe B**")
    else:
        xG_B = st.number_input("⚽ xG (Équipe B, par match)", value=1.00, format="%.2f")
        tirs_cadrés_B = st.number_input("🎯 Tirs cadrés (Équipe B, par match)", value=3.00, format="%.2f")
        taux_conversion_B = st.number_input("📈 Taux de conversion (%) (Équipe B, par match)", value=25.00, format="%.2f")
        touches_surface_B = st.number_input("🔥 Touches dans la surface (Équipe B, par match)", value=20.00, format="%.2f")
        passes_cles_B = st.number_input("🔑 Passes clés (Équipe B, par match)", value=4.00, format="%.2f")
        interceptions_B = st.number_input("🛡️ Interceptions (Équipe B, par match)", value=7.00, format="%.2f")
        duels_defensifs_B = st.number_input("💪 Duels défensifs gagnés (Équipe B, par match)", value=15.00, format="%.2f")
        xGA_B = st.number_input("🚫 xGA (Équipe B, par match)", value=1.50, format="%.2f")
        arrets_gardien_B = st.number_input("🧤 Arrêts du gardien (Équipe B, par match)", value=5.00, format="%.2f")
        forme_recente_B = st.number_input("📊 Forme récente (points cumulés, Équipe B)", value=8.00, format="%.2f")
        points_5_matchs_B = st.number_input("⭐ Points sur les 5 derniers matchs (Équipe B)", value=6.00, format="%.2f")
        possession_B = st.number_input("⏱️ Possession moyenne (%) (Équipe B)", value=50.00, format="%.2f")
        corners_B = st.number_input("⚽ Corners (Équipe B, par match)", value=4.00, format="%.2f")

st.markdown("### 🎲 Analyse Value Bet")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    cote_A = st.number_input("💰 Cote Bookmaker - Victoire Équipe A", value=2.00, format="%.2f")
with col_odds2:
    cote_N = st.number_input("💰 Cote Bookmaker - Match Nul", value=3.00, format="%.2f")
with col_odds3:
    cote_B = st.number_input("💰 Cote Bookmaker - Victoire Équipe B", value=2.50, format="%.2f")

# ===============================
# Chargement des données d'entraînement (ou utilisation par défaut)
# ===============================
st.sidebar.header("📊 Données d'Entraînement")
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = load_training_data(fichier_entrainement)
else:
    st.sidebar.info("Aucun fichier chargé, utilisation des données d'entraînement par défaut.")
    df_entrainement = pd.read_csv(StringIO(default_training_data))
    
if df_entrainement is not None:
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
else:
    st.error("Aucune donnée d'entraînement disponible.")

# ===============================
# Entraînement des modèles (avec cache et session_state)
# ===============================
if y_reel is not None:
    if st.session_state.models is None:
        st.session_state.models = train_models(X_reel, y_reel)
    model_log, prec_log = st.session_state.models["logistic"]
    model_xgb, prec_xgb = st.session_state.models["xgb"]
    model_rf, prec_rf = st.session_state.models["rf"]

# ===============================
# Prédictions et affichage des résultats
# ===============================
if st.button("🔮 Prédire le Résultat"):
    # Calcul du modèle de Poisson
    victoire_A, victoire_B, match_nul, expected_buts_A, expected_buts_B = predict_match(
        xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
        possession_A, corners_A,
        xG_B, tirs_cadrés_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B,
        possession_B, corners_B
    )
    
    st.markdown("#### 📊 Résultats du Modèle de Poisson")
    data_poisson = {
        "Mesure": [
            "Prob. Victoire Équipe A", "Prob. Match Nul", "Prob. Victoire Équipe B",
            "Buts attendus Équipe A", "Buts attendus Équipe B"
        ],
        "Valeur": [
            f"{victoire_A*100:.2f}%", f"{match_nul*100:.2f}%", f"{victoire_B*100:.2f}%",
            f"{expected_buts_A:.2f}", f"{expected_buts_B:.2f}"
        ]
    }
    st.table(pd.DataFrame(data_poisson))
    
    # Préparation de l'input pour la prédiction
    input_features = np.array([[
        xG_A, tirs_cadrés_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
        possession_A, corners_A,
        xG_B, tirs_cadrés_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B,
        possession_B, corners_B
    ]])
    
    st.write("Forme de l'input pour la prédiction :", input_features.shape)
    
    # Prédiction via Régression Logistique
    proba_log = model_log.predict_proba(input_features)[0][1]
    prediction_log = "🏆 Victoire Équipe A" if proba_log > 0.5 else "🏟️ Victoire Équipe B"
    
    # Prédiction via XGBoost
    proba_xgb = model_xgb.predict_proba(input_features)[0][1]
    prediction_xgb = "🏆 Victoire Équipe A" if proba_xgb > 0.5 else "🏟️ Victoire Équipe B"
    
    # Prédiction via Random Forest
    proba_rf = model_rf.predict_proba(input_features)[0][1]
    prediction_rf = "🏆 Victoire Équipe A" if proba_rf > 0.5 else "🏟️ Victoire Équipe B"
    
    st.markdown("#### 🔍 Résultats des Modèles de Classification")
    data_classif = {
        "Modèle": ["Régression Logistique", "XGBoost Classifier", "Random Forest"],
        "Prédiction": [prediction_log, prediction_xgb, prediction_rf],
        "Probabilité": [f"{proba_log*100:.2f}%", f"{proba_xgb*100:.2f}%", f"{proba_rf*100:.2f}%"],
        "Précision": [f"{prec_log*100:.2f}%", f"{prec_xgb*100:.2f}%", f"{prec_rf*100:.2f}%"]
    }
    st.table(pd.DataFrame(data_classif))
    
    st.markdown("#### 🎲 Analyse Value Bet")
    outcomes = ["Victoire Équipe A", "Match Nul", "Victoire Équipe B"]
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
    
    # Graphique interactif : Distribution des buts attendus pour l'Équipe A
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
