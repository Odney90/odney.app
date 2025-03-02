import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import random

# Fonction de probabilité de Poisson
def poisson_prob(lam, k):
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

# Fonction de prédiction du résultat du match en utilisant 22 variables (11 par équipe)
def predire_resultat_match(
    # Variables de l'Équipe A (Domicile)
    xG_A, tirs_cadres_A, taux_conversion_A, touches_surface_A, passes_cles_A,
    interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A,
    points_5_matchs_A,
    # Variables de l'Équipe B (Extérieur)
    xG_B, tirs_cadres_B, taux_conversion_B, touches_surface_B, passes_cles_B,
    interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B,
    points_5_matchs_B,
    max_buts=5
):
    # Évaluation offensive pour l'équipe A
    note_offensive_A = (
        xG_A * 0.3 +
        tirs_cadres_A * 0.2 +
        touches_surface_A * 0.1 +
        (taux_conversion_A / 100) * 0.2 +
        passes_cles_A * 0.2
    )
    # Évaluation défensive pour l'équipe B
    note_defensive_B = (
        xGA_B * 0.3 +
        arrets_gardien_B * 0.2 +
        interceptions_B * 0.25 +
        duels_defensifs_B * 0.25
    )
    # Calcul du multiplicateur de forme pour l'équipe A
    multiplicateur_A = 1 + (forme_recente_A / 10) + (points_5_matchs_A / 15)
    adj_xG_A = (note_offensive_A * multiplicateur_A) / (note_defensive_B + 1)
    
    # Même démarche pour l'équipe B
    note_offensive_B = (
        xG_B * 0.3 +
        tirs_cadres_B * 0.2 +
        touches_surface_B * 0.1 +
        (taux_conversion_B / 100) * 0.2 +
        passes_cles_B * 0.2
    )
    note_defensive_A = (
        xGA_A * 0.3 +
        arrets_gardien_A * 0.2 +
        interceptions_A * 0.25 +
        duels_defensifs_A * 0.25
    )
    multiplicateur_B = 1 + (forme_recente_B / 10) + (points_5_matchs_B / 15)
    adj_xG_B = (note_offensive_B * multiplicateur_B) / (note_defensive_A + 1)
    
    # Calcul des probabilités de buts de 0 à max_buts pour chaque équipe à l'aide de la loi de Poisson
    prob_A = [poisson_prob(adj_xG_A, i) for i in range(max_buts+1)]
    prob_B = [poisson_prob(adj_xG_B, i) for i in range(max_buts+1)]
    
    victoire_A, victoire_B, match_nul = 0, 0, 0
    for buts_A in range(max_buts+1):
        for buts_B in range(max_buts+1):
            proba = prob_A[buts_A] * prob_B[buts_B]
            if buts_A > buts_B:
                victoire_A += proba
            elif buts_A < buts_B:
                victoire_B += proba
            else:
                match_nul += proba
    return victoire_A, victoire_B, match_nul

# Titre de l'application Streamlit
st.title("⚽ Prédiction de Match de Football avec Données Fictives")

# Chargement optionnel des données historiques via un fichier CSV
st.sidebar.header("Données Historiques")
fichier_historique = st.sidebar.file_uploader("Charger le fichier CSV des données historiques", type=["csv"])
if fichier_historique is not None:
    df_historique = pd.read_csv(fichier_historique)
    st.sidebar.write("Aperçu des données historiques :", df_historique.head())

# Affichage d'un tableau récapitulatif des variables utilisées
st.subheader("Variables Impactantes Utilisées pour la Prédiction")
data = {
    "Équipe": ["Équipe A"]*11 + ["Équipe B"]*11,
    "Variable": [
        "xG", "Tirs cadrés", "Taux de conversion", "Touches dans la surface", "Passes clés",
        "Interceptions", "Duels défensifs", "xGA", "Arrêts du gardien", "Forme récente", "Points sur 5 matchs",
        "xG", "Tirs cadrés", "Taux de conversion", "Touches dans la surface", "Passes clés",
        "Interceptions", "Duels défensifs", "xGA", "Arrêts du gardien", "Forme récente", "Points sur 5 matchs"
    ],
    "Description": [
        "Buts attendus par l'équipe.",
        "Nombre de tirs cadrés.",
        "Pourcentage de conversion des tirs.",
        "Nombre de touches dans la surface de réparation.",
        "Passes menant à une situation dangereuse.",
        "Nombre d'interceptions réalisées.",
        "Duels défensifs gagnés.",
        "Buts attendus encaissés.",
        "Arrêts réalisés par le gardien.",
        "Évaluation de la forme récente (points cumulés).",
        "Points obtenus sur les 5 derniers matchs.",
        "Buts attendus par l'équipe.",
        "Nombre de tirs cadrés.",
        "Pourcentage de conversion des tirs.",
        "Nombre de touches dans la surface de réparation.",
        "Passes menant à une situation dangereuse.",
        "Nombre d'interceptions réalisées.",
        "Duels défensifs gagnés.",
        "Buts attendus encaissés.",
        "Arrêts réalisés par le gardien.",
        "Évaluation de la forme récente (points cumulés).",
        "Points obtenus sur les 5 derniers matchs."
    ]
}
df_variables = pd.DataFrame(data)
st.dataframe(df_variables)

# Option pour utiliser des données fictives pour tester le code
use_fictives = st.checkbox("Utiliser des données fictives pour tester le code", value=False)

# Mise en page pour la saisie des données (deux colonnes)
col1, col2 = st.columns(2)

# Saisie des variables pour l'Équipe A (Domicile)
with col1:
    st.header("🏠 Équipe A (Domicile)")
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
        st.write("Données fictives pour l'Équipe A :")
        st.write(f"xG: {xG_A}, Tirs cadrés: {tirs_cadres_A}, Taux conversion: {taux_conversion_A}%, Touches: {touches_surface_A}, "
                 f"Passes clés: {passes_cles_A}, Interceptions: {interceptions_A}, Duels défensifs: {duels_defensifs_A}, "
                 f"xGA: {xGA_A}, Arrêts gardien: {arrets_gardien_A}, Forme récente: {forme_recente_A}, Points sur 5 matchs: {points_5_matchs_A}")
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

# Saisie des variables pour l'Équipe B (Extérieur)
with col2:
    st.header("🏟️ Équipe B (Extérieur)")
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
        st.write("Données fictives pour l'Équipe B :")
        st.write(f"xG: {xG_B}, Tirs cadrés: {tirs_cadres_B}, Taux conversion: {taux_conversion_B}%, Touches: {touches_surface_B}, "
                 f"Passes clés: {passes_cles_B}, Interceptions: {interceptions_B}, Duels défensifs: {duels_defensifs_B}, "
                 f"xGA: {xGA_B}, Arrêts gardien: {arrets_gardien_B}, Forme récente: {forme_recente_B}, Points sur 5 matchs: {points_5_matchs_B}")
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

# Bouton de prédiction
if st.button("🔮 Prédire le Résultat"):
    # Prédiction basée sur la fonction utilisant les 22 variables
    victoire_A, victoire_B, match_nul = predire_resultat_match(
        xG_A, tirs_cadres_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A,
        points_5_matchs_A,
        xG_B, tirs_cadres_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B,
        points_5_matchs_B
    )
    st.write(f"🏠 Probabilité de victoire de l'Équipe A : {victoire_A:.2%}")
    st.write(f"🏟️ Probabilité de victoire de l'Équipe B : {victoire_B:.2%}")
    st.write(f"🤝 Probabilité de match nul : {match_nul:.2%}")
    
    # Entraînement d'un modèle de régression logistique sur des données fictives
    # (ici, nous générons 100 exemples avec 22 variables aléatoires)
    X_train = np.random.rand(100, 22)
    y_train = np.random.randint(0, 2, 100)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    
    # Utilisation des variables saisies pour la prédiction via régression logistique
    input_features = np.array([[
        xG_A, tirs_cadres_A, taux_conversion_A, touches_surface_A, passes_cles_A,
        interceptions_A, duels_defensifs_A, xGA_A, arrets_gardien_A, forme_recente_A,
        points_5_matchs_A,
        xG_B, tirs_cadres_B, taux_conversion_B, touches_surface_B, passes_cles_B,
        interceptions_B, duels_defensifs_B, xGA_B, arrets_gardien_B, forme_recente_B,
        points_5_matchs_B
    ]])
    proba_logistic = logistic_model.predict_proba(input_features)[0][1]
    prediction = "Victoire de l'Équipe A" if proba_logistic > 0.5 else "Victoire de l'Équipe B"
    st.write(f"📊 Prédiction par régression logistique : {prediction} avec une confiance de {proba_logistic*100:.2f}%.")
