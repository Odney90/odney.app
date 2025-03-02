import streamlit as st
import numpy as np
import pandas as pd
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Fonctions de base
# -------------------------------

# Fonction de probabilité de Poisson
def poisson_prob(lam, k):
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

# Fonction de prédiction du résultat du match avec le modèle de Poisson
def predire_resultat_match(
    # Variables de l'Équipe A (Domicile)
    xG_A, tirs_cadres_A, taux_conversion_A, touches_surface_A, passes_cles_A,
    interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
    # Variables de l'Équipe B (Extérieur)
    xG_B, tirs_cadres_B, taux_conversion_B, touches_surface_B, passes_cles_B,
    interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B,
    max_buts=5
):
    # Calcul des indices offensifs et défensifs pour chaque équipe
    note_offensive_A = (xG_A * 0.3 + tirs_cadres_A * 0.2 + touches_surface_A * 0.1 +
                        (taux_conversion_A / 100) * 0.2 + passes_cles_A * 0.2)
    note_defensive_B = (xGA_B * 0.3 + arrets_gardien_B * 0.2 + interceptions_B * 0.25 + duels_defensifs_B * 0.25)
    multiplicateur_A = 1 + (forme_recente_A / 10) + (points_5_matchs_A / 15)
    adj_xG_A = (note_offensive_A * multiplicateur_A) / (note_defensive_B + 1)
    
    note_offensive_B = (xG_B * 0.3 + tirs_cadres_B * 0.2 + touches_surface_B * 0.1 +
                        (taux_conversion_B / 100) * 0.2 + passes_cles_B * 0.2)
    note_defensive_A = (xGA_A * 0.3 + arrets_gardien_A * 0.2 + interceptions_A * 0.25 + duels_defensifs_A * 0.25)
    multiplicateur_B = 1 + (forme_recente_B / 10) + (points_5_matchs_B / 15)
    adj_xG_B = (note_offensive_B * multiplicateur_B) / (note_defensive_A + 1)
    
    # Calcul des probabilités de buts selon la loi de Poisson pour chaque équipe
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
                
    # Calcul des buts attendus pour chaque équipe
    expected_buts_A = sum(i * prob_A[i] for i in range(max_buts+1))
    expected_buts_B = sum(i * prob_B[i] for i in range(max_buts+1))
    
    return victoire_A, victoire_B, match_nul, expected_buts_A, expected_buts_B

# Fonction pour calculer la Value Bet
def calculer_value_bet(prob, cote):
    # Valeur espérée = (probabilité prédite * cote) - 1
    ev = (prob * cote) - 1
    recommendation = "Value Bet" if ev > 0 else "Pas de Value Bet"
    return ev, recommendation

# -------------------------------
# Chargement des données d'entraînement réelles
# -------------------------------

st.sidebar.header("Données d'Entraînement")
fichier_entrainement = st.sidebar.file_uploader("Charger le CSV d'entraînement", type=["csv"])
if fichier_entrainement is not None:
    df_entrainement = pd.read_csv(fichier_entrainement)
    st.sidebar.write("Aperçu des données d'entraînement :", df_entrainement.head())
    
    # Sélection des variables explicatives et de la cible
    features = [
        "xG_A", "Tirs_cadrés_A", "Taux_conversion_A", "Touches_surface_A", "Passes_clés_A",
        "Interceptions_A", "Duels_défensifs_A", "xGA_A", "Arrêts_gardien_A", "Forme_recente_A", "Points_5_matchs_A",
        "xG_B", "Tirs_cadrés_B", "Taux_conversion_B", "Touches_surface_B", "Passes_clés_B",
        "Interceptions_B", "Duels_défensifs_B", "xGA_B", "Arrêts_gardien_B", "Forme_recente_B", "Points_5_matchs_B"
    ]
    # La colonne "resultat" doit contenir la cible (0 ou 1)
    X_reel = df_entrainement[features]
    y_reel = df_entrainement["resultat"]

# -------------------------------
# Entraînement des modèles avec les données réelles (mise en cache)
# -------------------------------

@st.cache_resource(show_spinner=False)
def entrainer_modele_logistique(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

@st.cache_resource(show_spinner=False)
def entrainer_modele_xgb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

@st.cache_resource(show_spinner=False)
def entrainer_modele_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

if fichier_entrainement is not None:
    modele_logistique, precision_logistique = entrainer_modele_logistique(X_reel, y_reel)
    modele_xgb, precision_xgb = entrainer_modele_xgb(X_reel, y_reel)
    modele_rf, precision_rf = entrainer_modele_rf(X_reel, y_reel)
    st.sidebar.write(f"Précision Régression Logistique : {precision_logistique*100:.2f}%")
    st.sidebar.write(f"Précision XGBoost : {precision_xgb*100:.2f}%")
    st.sidebar.write(f"Précision Random Forest : {precision_rf*100:.2f}%")

# -------------------------------
# Interface principale pour la saisie des données de match
# -------------------------------

st.title("Prédiction de Match de Football et Analyse Value Bet")

# Option pour utiliser des données fictives (si pas de données réelles)
use_fictives = st.checkbox("Utiliser des données fictives pour tester le code", value=False)

# Saisie des données pour l'Équipe A (Domicile) et l'Équipe B (Extérieur)
col1, col2 = st.columns(2)

with col1:
    st.header("Équipe A (Domicile)")
    if use_fictives:
        xG_A = round(random.uniform(0.5, 3.0), 2)
        tirs_cadres_A = random.randint(1, 10)
        taux_conversion_A = round(random.uniform(20, 50), 1)
        touches_surface_A = random.randint(5, 20)
        passes_cles_A = random.randint(1, 8)
        interceptions_A = random.randint(3, 15)
        duels_defensifs_A = random.randint(10, 30)
        xGA_A = round(random.uniform(0.5, 2.0), 2)
        arrets_gardien_A = random.randint(1, 6)
        forme_recente_A = random.randint(5, 15)
        points_5_matchs_A = random.randint(5, 15)
        st.write("Données fictives pour l'Équipe A générées.")
    else:
        xG_A = st.number_input("xG (Équipe A)", value=1.5)
        tirs_cadres_A = st.number_input("Tirs cadrés (Équipe A)", value=5)
        taux_conversion_A = st.number_input("Taux de conversion (%) (Équipe A)", value=30.0)
        touches_surface_A = st.number_input("Touches dans la surface (Équipe A)", value=15)
        passes_cles_A = st.number_input("Passes clés (Équipe A)", value=5)
        interceptions_A = st.number_input("Interceptions (Équipe A)", value=8)
        duels_defensifs_A = st.number_input("Duels défensifs gagnés (Équipe A)", value=18)
        xGA_A = st.number_input("xGA (Équipe A)", value=1.2)
        arrets_gardien_A = st.number_input("Arrêts du gardien (Équipe A)", value=4)
        forme_recente_A = st.number_input("Forme récente (points cumulés) (Équipe A)", value=10)
        points_5_matchs_A = st.number_input("Points sur les 5 derniers matchs (Équipe A)", value=8)

with col2:
    st.header("Équipe B (Extérieur)")
    if use_fictives:
        xG_B = round(random.uniform(0.5, 3.0), 2)
        tirs_cadres_B = random.randint(1, 10)
        taux_conversion_B = round(random.uniform(20, 50), 1)
        touches_surface_B = random.randint(5, 20)
        passes_cles_B = random.randint(1, 8)
        interceptions_B = random.randint(3, 15)
        duels_defensifs_B = random.randint(10, 30)
        xGA_B = round(random.uniform(0.5, 2.0), 2)
        arrets_gardien_B = random.randint(1, 6)
        forme_recente_B = random.randint(5, 15)
        points_5_matchs_B = random.randint(5, 15)
        st.write("Données fictives pour l'Équipe B générées.")
    else:
        xG_B = st.number_input("xG (Équipe B)", value=1.0)
        tirs_cadres_B = st.number_input("Tirs cadrés (Équipe B)", value=3)
        taux_conversion_B = st.number_input("Taux de conversion (%) (Équipe B)", value=25.0)
        touches_surface_B = st.number_input("Touches dans la surface (Équipe B)", value=10)
        passes_cles_B = st.number_input("Passes clés (Équipe B)", value=4)
        interceptions_B = st.number_input("Interceptions (Équipe B)", value=7)
        duels_defensifs_B = st.number_input("Duels défensifs gagnés (Équipe B)", value=15)
        xGA_B = st.number_input("xGA (Équipe B)", value=1.5)
        arrets_gardien_B = st.number_input("Arrêts du gardien (Équipe B)", value=5)
        forme_recente_B = st.number_input("Forme récente (points cumulés) (Équipe B)", value=8)
        points_5_matchs_B = st.number_input("Points sur les 5 derniers matchs (Équipe B)", value=6)

# Saisie des cotes bookmakers pour l'analyse Value Bet
st.subheader("Analyse Value Bet")
col_odds1, col_odds2, col_odds3 = st.columns(3)
with col_odds1:
    cote_A = st.number_input("Cote Bookmaker - Victoire Équipe A", value=2.0)
with col_odds2:
    cote_N = st.number_input("Cote Bookmaker - Match Nul", value=3.0)
with col_odds3:
    cote_B = st.number_input("Cote Bookmaker - Victoire Équipe B", value=2.5)

# -------------------------------
# Prédictions et affichage des résultats
# -------------------------------

if st.button("🔮 Prédire le Résultat"):
    # Prédiction via le modèle de Poisson
    victoire_A, victoire_B, match_nul, expected_buts_A, expected_buts_B = predire_resultat_match(
        xG_A, tirs_cadres_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
        xG_B, tirs_cadres_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B
    )
    
    # Affichage des résultats du modèle de Poisson dans un tableau
    st.write("### Modèle de Poisson")
    data_poisson = {
        "Mesure": ["Probabilité Victoire Équipe A", "Probabilité Match Nul", "Probabilité Victoire Équipe B", 
                   "Buts attendus Équipe A", "Buts attendus Équipe B"],
        "Valeur": [f"{victoire_A*100:.2f}%", f"{match_nul*100:.2f}%", f"{victoire_B*100:.2f}%",
                   f"{expected_buts_A:.2f}", f"{expected_buts_B:.2f}"]
    }
    df_poisson = pd.DataFrame(data_poisson)
    st.table(df_poisson)
    
    # Si des données d'entraînement réelles ont été chargées, utiliser les modèles entraînés sur celles-ci ;
    # sinon, utiliser des données fictives pour l'entraînement.
    if fichier_entrainement is not None:
        # Les modèles sont déjà entraînés et mis en cache
        model_log = modele_logistique
        prec_log = precision_logistique
        model_xgb = modele_xgb
        prec_xgb = precision_xgb
        model_rf = modele_rf
        prec_rf = precision_rf
    else:
        # Génération de données fictives pour l'entraînement
        X_data = np.random.rand(200, 22)
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
    
    # Préparation des variables saisies pour la prédiction des modèles de classification
    input_features = np.array([[
        xG_A, tirs_cadres_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A, points_5_matchs_A,
        xG_B, tirs_cadres_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B, points_5_matchs_B
    ]])
    
    # Prédiction via Régression Logistique
    proba_log = model_log.predict_proba(input_features)[0][1]
    prediction_log = "Victoire Équipe A" if proba_log > 0.5 else "Victoire Équipe B"
    
    # Prédiction via XGBoost
    proba_xgb = model_xgb.predict_proba(input_features)[0][1]
    prediction_xgb = "Victoire Équipe A" if proba_xgb > 0.5 else "Victoire Équipe B"
    
    # Prédiction via Random Forest
    proba_rf = model_rf.predict_proba(input_features)[0][1]
    prediction_rf = "Victoire Équipe A" if proba_rf > 0.5 else "Victoire Équipe B"
    
    # Affichage des résultats des modèles de classification dans un tableau
    st.write("### Modèles de Classification")
    data_classif = {
        "Modèle": ["Régression Logistique", "XGBoost Classifier", "Random Forest"],
        "Prédiction": [prediction_log, prediction_xgb, prediction_rf],
        "Probabilité Prédite": [f"{proba_log*100:.2f}%", f"{proba_xgb*100:.2f}%", f"{proba_rf*100:.2f}%"],
        "Précision du Modèle": [f"{prec_log*100:.2f}%", f"{prec_xgb*100:.2f}%", f"{prec_rf*100:.2f}%"]
    }
    df_classif = pd.DataFrame(data_classif)
    st.table(df_classif)
    
    # Analyse Value Bet pour chaque issue (en utilisant les probabilités issues du modèle de Poisson)
    st.write("### Analyse Value Bet")
    outcomes = ["Victoire Équipe A", "Match Nul", "Victoire Équipe B"]
    bookmaker_cotes = [cote_A, cote_N, cote_B]
    predicted_probs = [victoire_A, match_nul, victoire_B]
    
    value_bet_data = {"Issue": [], "Cote Bookmaker": [], "Probabilité Impliquée": [], "Probabilité Prédite": [], "Valeur Espérée": [], "Value Bet ?": []}
    
    for outcome, cote, prob in zip(outcomes, bookmaker_cotes, predicted_probs):
        prob_implied = 1 / cote
        ev, recommendation = calculer_value_bet(prob, cote)
        value_bet_data["Issue"].append(outcome)
        value_bet_data["Cote Bookmaker"].append(cote)
        value_bet_data["Probabilité Impliquée"].append(f"{prob_implied*100:.2f}%")
        value_bet_data["Probabilité Prédite"].append(f"{prob*100:.2f}%")
        value_bet_data["Valeur Espérée"].append(f"{ev:.2f}")
        value_bet_data["Value Bet ?"].append(recommendation)
    
    df_value_bet = pd.DataFrame(value_bet_data)
    st.table(df_value_bet)
