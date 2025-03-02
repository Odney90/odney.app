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
# 🎯 Initialisation du session state
# -------------------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

# -------------------------------
# 🛠️ Fonctions Utilitaires
# -------------------------------
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def predire_resultat_match(
    # Variables Équipe A (Domicile, moyennes par match) – 15 variables
    xG_A, tirs_cadrés_A, taux_conversion_A, Touches_surface_A, Passes_cles_A, Tirs_tentés_A,
    Interceptions_A, Duels_defensifs_A, xGA_A, Arrêts_gardien_A, Fautes_A, Possession_A, Corners_A,
    Forme_recente_A, Points_5_matchs_A,
    # Variables Équipe B (Extérieur, moyennes par match) – 15 variables
    xG_B, tirs_cadrés_B, taux_conversion_B, Touches_surface_B, Passes_cles_B, Tirs_tentés_B,
    Interceptions_B, Duels_defensifs_B, xGA_B, Arrêts_du_gardien_B, Fautes_B, Possession_B, Corners_B,
    Forme_recente_B, Points_5_matchs_B,
    max_buts=5
):
    note_offensive_A = (
        xG_A * 0.2 +
        tirs_cadrés_A * 0.15 +
        Touches_surface_A * 0.1 +
        (taux_conversion_A / 100) * 0.15 +
        Passes_cles_A * 0.15 +
        Tirs_tentés_A * 0.05 +      # Ajouté pour atteindre 15 variables
        (Possession_A / 100) * 0.1 +
        (Corners_A / 10) * 0.15
    )
    note_defensive_B = (
        xGA_B * 0.2 +
        Arrêts_du_gardien_B * 0.15 +
        Interceptions_B * 0.15 +
        Duels_defensifs_B * 0.15 +
        Fautes_B * 0.05 +          # Ajouté pour atteindre 15 variables
        ((100 - Possession_B) / 100) * 0.1 +
        (Corners_B / 10) * 0.15
    )
    multiplicateur_A = 1 + (Forme_recente_A / 10) + (Points_5_matchs_A / 15)
    adj_xG_A = (note_offensive_A * multiplicateur_A) / (note_defensive_B + 1)
    
    note_offensive_B = (
        xG_B * 0.2 +
        tirs_cadrés_B * 0.15 +
        Touches_surface_B * 0.1 +
        (taux_conversion_B / 100) * 0.15 +
        Passes_cles_B * 0.15 +
        Tirs_tentés_B * 0.05 +      # Ajouté pour atteindre 15 variables
        (Possession_B / 100) * 0.1 +
        (Corners_B / 10) * 0.15
    )
    note_defensive_A = (
        xGA_A * 0.2 +
        Arrêts_gardien_A * 0.15 +
        Interceptions_A * 0.15 +
        Duels_defensifs_A * 0.15 +
        Fautes_A * 0.05 +          # Ajouté pour atteindre 15 variables
        ((100 - Possession_A) / 100) * 0.1 +
        (Corners_A / 10) * 0.15
    )
    multiplicateur_B = 1 + (Forme_recente_B / 10) + (Points_5_matchs_B / 15)
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
# 📊 Section d'entrée des données d'entraînement
# -------------------------------
st.sidebar.header("📊 Données d'Entraînement")
st.sidebar.markdown(
    """
    **Format du fichier CSV attendu (données par match ou moyennes par match) :**

    Votre fichier doit contenir les colonnes suivantes dans cet ordre :

    - Pour l'Équipe A (Domicile) – 15 variables :
      `xG_A`, `Tirs_cadrés_A`, `Taux_conversion_A`, `Touches_surface_A`, `Passes_clés_A`, `Tirs_tentés_A`, `Interceptions_A`, `Duels_defensifs_A`, `xGA_A`, `Arrêts_gardien_A`, `Fautes_A`, `Possession_A`, `Corners_A`, `Forme_recente_A`, `Points_5_matchs_A`

    - Pour l'Équipe B (Extérieur) – 15 variables :
      `xG_B`, `Tirs_cadrés_B`, `Taux_conversion_B`, `Touches_surface_B`, `Passes_clés_B`, `Tirs_tentés_B`, `Interceptions_B`, `Duels_defensifs_B`, `xGA_B`, `Arrêts_du_gardien_B`, `Fautes_B`, `Possession_B`, `Corners_B`, `Forme_recente_B`, `Points_5_matchs_B`

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
        "Tirs_tentés_A", "Interceptions_A", "Duels_defensifs_A", "xGA_A", "Arrêts_gardien_A",
        "Fautes_A", "Possession_A", "Corners_A", "Forme_recente_A", "Points_5_matchs_A",
        "xG_B", "Tirs_cadrés_B", "Taux_conversion_B", "Touches_surface_B", "Passes_clés_B",
        "Tirs_tentés_B", "Interceptions_B", "Duels_defensifs_B", "xGA_B", "Arrêts_du_gardien_B",
        "Fautes_B", "Possession_B", "Corners_B", "Forme_recente_B", "Points_5_matchs_B"
    ]
    X_reel = df_entrainement[features]
    y_reel = df_entrainement["resultat"]

# -------------------------------
# 🏆 Entraînement des modèles (avec validation croisée)
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

    st.sidebar.markdown("### 🌟 Performances des modèles (CV)")
    st.sidebar.markdown(f"- 🧮 Régression Logistique : {score_logistique*100:.2f}%")
    st.sidebar.markdown(f"- ⚡ XGBoost : {score_xgb*100:.2f}%")
    st.sidebar.markdown(f"- 🌲 Random Forest : {score_rf*100:.2f}%")

# -------------------------------
# 🎯 Affichage des Résultats de la Validation Croisée
# -------------------------------
st.markdown("## 📊 Résultats des Modèles (Validation Croisée)")
df_scores = pd.DataFrame([
    ["🧮 Régression Logistique", f"{score_logistique*100:.2f}%", modele_logistique],
    ["⚡ XGBoost", f"{score_xgb*100:.2f}%", modele_xgb],
    ["🌲 Random Forest", f"{score_rf*100:.2f}%", modele_rf]
], columns=["Modèle", "Précision (CV)", "Modèle entraîné"])
st.table(df_scores[["Modèle", "Précision (CV)"]])

# -------------------------------
# 🏟️ Interface Principale pour la Saisie des Données de Match
# -------------------------------
st.title("⚽ Prédiction de Match de Football & Analyse Value Bet")
st.markdown("### Entrez les statistiques du match (moyennes par match) pour chaque équipe")

use_fictives = st.checkbox("Utiliser des données fictives", value=False)

col1, col2 = st.columns(2)
with col1:
    st.header("🏠 Équipe A (Domicile)")
    if use_fictives:
        xG_A = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_A = float(random.randint(2, 10))
        taux_conversion_A = round(random.uniform(20, 40), 2)
        Touches_surface_A = float(random.randint(15, 40))
        Passes_cles_A = float(random.randint(3, 8))
        Tirs_tentés_A = float(random.randint(8, 16))
        Interceptions_A = float(random.randint(5, 15))
        Duels_defensifs_A = float(random.randint(10, 30))
        xGA_A = round(random.uniform(1, 2.5), 2)
        Arrêts_gardien_A = float(random.randint(3, 7))
        Fautes_A = float(random.randint(10, 20))
        Forme_recente_A = float(random.randint(5, 15))
        Points_5_matchs_A = float(random.randint(5, 15))
        Possession_A = round(random.uniform(45, 70), 2)
        Corners_A = float(random.randint(3, 10))
        st.markdown("**Données fictives générées pour l'Équipe A**")
    else:
        xG_A = st.number_input("xG (Équipe A)", value=1.50, format="%.2f")
        tirs_cadrés_A = st.number_input("Tirs cadrés (Équipe A)", value=5.00, format="%.2f")
        taux_conversion_A = st.number_input("Taux de conversion (%) (Équipe A)", value=30.00, format="%.2f")
        Touches_surface_A = st.number_input("Touches dans la surface (Équipe A)", value=25.00, format="%.2f")
        Passes_cles_A = st.number_input("Passes clés (Équipe A)", value=5.00, format="%.2f")
        Tirs_tentés_A = st.number_input("Tirs tentés (Équipe A)", value=10.00, format="%.2f")
        Interceptions_A = st.number_input("Interceptions (Équipe A)", value=8.00, format="%.2f")
        Duels_defensifs_A = st.number_input("Duels défensifs gagnés (Équipe A)", value=18.00, format="%.2f")
        xGA_A = st.number_input("xGA (Équipe A)", value=1.20, format="%.2f")
        Arrêts_gardien_A = st.number_input("Arrêts du gardien (Équipe A)", value=4.00, format="%.2f")
        Fautes_A = st.number_input("Fautes commises (Équipe A)", value=12.00, format="%.2f")
        Forme_recente_A = st.number_input("Forme récente (Équipe A)", value=10.00, format="%.2f")
        Points_5_matchs_A = st.number_input("Points sur 5 matchs (Équipe A)", value=8.00, format="%.2f")
        Possession_A = st.number_input("Possession (%) (Équipe A)", value=55.00, format="%.2f")
        Corners_A = st.number_input("Corners (Équipe A)", value=5.00, format="%.2f")
        
with col2:
    st.header("🏟️ Équipe B (Extérieur)")
    if use_fictives:
        xG_B = round(random.uniform(0.5, 2.5), 2)
        tirs_cadrés_B = float(random.randint(2, 10))
        taux_conversion_B = round(random.uniform(20, 40), 2)
        Touches_surface_B = float(random.randint(15, 40))
        Passes_cles_B = float(random.randint(3, 8))
        Tirs_tentés_B = float(random.randint(8, 16))
        Interceptions_B = float(random.randint(5, 15))
        Duels_defensifs_B = float(random.randint(10, 30))
        xGA_B = round(random.uniform(1, 2.5), 2)
        Arrêts_du_gardien_B = float(random.randint(3, 7))
        Fautes_B = float(random.randint(10, 20))
        Forme_recente_B = float(random.randint(5, 15))
        Points_5_matchs_B = float(random.randint(5, 15))
        Possession_B = round(random.uniform(45, 70), 2)
        Corners_B = float(random.randint(3, 10))
        st.markdown("**Données fictives générées pour l'Équipe B**")
    else:
        xG_B = st.number_input("xG (Équipe B)", value=1.00, format="%.2f")
        tirs_cadrés_B = st.number_input("Tirs cadrés (Équipe B)", value=3.00, format="%.2f")
        taux_conversion_B = st.number_input("Taux de conversion (%) (Équipe B)", value=25.00, format="%.2f")
        Touches_surface_B = st.number_input("Touches dans la surface (Équipe B)", value=20.00, format="%.2f")
        Passes_cles_B = st.number_input("Passes clés (Équipe B)", value=4.00, format="%.2f")
        Tirs_tentés_B = st.number_input("Tirs tentés (Équipe B)", value=9.00, format="%.2f")
        Interceptions_B = st.number_input("Interceptions (Équipe B)", value=7.00, format="%.2f")
        Duels_defensifs_B = st.number_input("Duels défensifs gagnés (Équipe B)", value=15.00, format="%.2f")
        xGA_B = st.number_input("xGA (Équipe B)", value=1.50, format="%.2f")
        Arrêts_du_gardien_B = st.number_input("Arrêts du gardien (Équipe B)", value=5.00, format="%.2f")
        Fautes_B = st.number_input("Fautes commises (Équipe B)", value=11.00, format="%.2f")
        Forme_recente_B = st.number_input("Forme récente (Équipe B)", value=8.00, format="%.2f")
        Points_5_matchs_B = st.number_input("Points sur 5 matchs (Équipe B)", value=6.00, format="%.2f")
        Possession_B = st.number_input("Possession (%) (Équipe B)", value=50.00, format="%.2f")
        Corners_B = st.number_input("Corners (Équipe B)", value=4.00, format="%.2f")

st.markdown("### 🎲 Analyse Value Bet")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    cote_A = st.number_input("Cote Bookmaker - Victoire Équipe A", value=2.00, format="%.2f")
with col_odds2:
    cote_N = st.number_input("Cote Bookmaker - Match Nul", value=3.00, format="%.2f")
with col_odds3:
    cote_B = st.number_input("Cote Bookmaker - Victoire Équipe B", value=2.50, format="%.2f")

# -------------------------------
# 🔮 Prédictions et affichage des résultats
# -------------------------------
if st.button("🔮 Prédire le Résultat"):
    # Prédiction via le modèle de Poisson
    victoire_A, victoire_B, match_nul, expected_buts_A, expected_buts_B = predire_resultat_match(
        xG_A, tirs_cadrés_A, taux_conversion_A, Touches_surface_A, Passes_cles_A, Tirs_tentés_A,
        Interceptions_A, Duels_defensifs_A, xGA_A, Arrêts_gardien_A, Fautes_A, Possession_A, Corners_A, Forme_recente_A, Points_5_matchs_A,
        xG_B, tirs_cadrés_B, taux_conversion_B, Touches_surface_B, Passes_cles_B, Tirs_tentés_B,
        Interceptions_B, Duels_defensifs_B, xGA_B, Arrêts_du_gardien_B, Fautes_B, Possession_B, Corners_B, Forme_recente_B, Points_5_matchs_B
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
    
    # Entraînement / Utilisation des modèles de classification
    if fichier_entrainement is not None:
        model_log = modele_logistique
        prec_log = precision_logistique
        model_xgb = modele_xgb
        prec_xgb = precision_xgb
        model_rf = modele_rf
        prec_rf = precision_rf
    else:
        X_data = np.random.rand(200, 30)
        y_data = np.random.randint(0, 2, 200)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
        model_log = LogisticRegression()
        model_log.fit(X_train, y_train)
        prec_log = accuracy_score(y_test, model_log.predict(X_test))
        model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model_xgb.fit(X_train, y_train)
        prec_xgb = accuracy_score(y_test, model_xgb.predict(X_test))
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)
        prec_rf = accuracy_score(y_test, model_rf.predict(X_test))
    
    input_features = np.array([[
        xG_A, tirs_cadrés_A, taux_conversion_A, Touches_surface_A, Passes_cles_A, Tirs_tentés_A,
        Interceptions_A, Duels_defensifs_A, xGA_A, Arrêts_gardien_A, Fautes_A, Possession_A, Corners_A, Forme_recente_A, Points_5_matchs_A,
        xG_B, tirs_cadrés_B, taux_conversion_B, Touches_surface_B, Passes_cles_B, Tirs_tentés_B,
        Interceptions_B, Duels_defensifs_B, xGA_B, Arrêts_du_gardien_B, Fautes_B, Possession_B, Corners_B, Forme_recente_B, Points_5_matchs_B
    ]])
    
    st.write("Forme de l'input pour la prédiction :", input_features.shape)
    
    # Régression Logistique
    proba_log = model_log.predict_proba(input_features)[0][1]
    prediction_log = "🏆 Victoire Équipe A" if proba_log > 0.5 else "🏟️ Victoire Équipe B"
    
    # XGBoost
    proba_xgb = model_xgb.predict_proba(input_features)[0][1]
    prediction_xgb = "🏆 Victoire Équipe A" if proba_xgb > 0.5 else "🏟️ Victoire Équipe B"
    
    # Random Forest
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
    
    # Graphique pour visualiser la distribution des buts attendus pour l'équipe A
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
