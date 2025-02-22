import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt

# Fonction pour sauvegarder les données localement
DATA_FILE = "saved_data.json"

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Charger les données sauvegardées
data = load_data()

st.title("⚽ Analyse de Matchs de Football 📊")

# Saisie des équipes
team1 = st.text_input("🏆 Équipe 1", data.get("team1", ""))
team2 = st.text_input("🏆 Équipe 2", data.get("team2", ""))

# Forme récente des équipes
st.header("📈 Forme Récente")
forme_team1 = st.text_area(f"📊 Forme de {team1} (5 derniers matchs, ex: V,N,D,V,V)", data.get("forme_team1", ""))
forme_team2 = st.text_area(f"📊 Forme de {team2} (5 derniers matchs, ex: D,V,N,N,V)", data.get("forme_team2", ""))

# Face-à-face
st.header("⚔️ Face-à-Face")
face_a_face = st.text_area("📅 Résultats des dernières confrontations", data.get("face_a_face", ""))

# Organisation des statistiques
st.header("🔥 Top Statistiques")
scores = st.number_input("📊 Score Rating Équipe", value=data.get("scores", 0.0))
buts_totals = st.number_input("⚽ Buts Totals", value=data.get("buts_totals", 0))
buts_par_match = st.number_input("⚽ Buts par Match", value=data.get("buts_par_match", 0.0))
buts_concedes_par_match = st.number_input("🛑 Buts Concédés par Match", value=data.get("buts_concedes_par_match", 0.0))

st.header("⚔️ Critères Attaque")
xG = st.number_input("🎯 Expected Goals (xG)", value=data.get("xG", 0.0))
tirs_cadres = st.number_input("🎯 Tirs Cadrés par Match", value=data.get("tirs_cadres", 0))
grandes_chances = st.number_input("🔥 Grandes Chances", value=data.get("grandes_chances", 0))
grandes_chances_manquees = st.number_input("💨 Grandes Chances Manquées", value=data.get("grandes_chances_manquees", 0))
passes_reussies = st.number_input("✅ Passes Réussies par Match", value=data.get("passes_reussies", 0))
passes_longues = st.number_input("🎯 Passes Longues Précises par Match", value=data.get("passes_longues", 0))
centres_reussis = st.number_input("🎯 Centres Réussis par Match", value=data.get("centres_reussis", 0))
penalities_obtenues = st.number_input("⚡ Pénalties Obtenues", value=data.get("penalities_obtenues", 0))
balles_touchees_surface = st.number_input("⚽ Balles Touchées dans la Surface Adverse", value=data.get("balles_touchees_surface", 0))
corners = st.number_input("🟦 Corners", value=data.get("corners", 0))

st.header("🛡️ Critères Défense")
xGA = st.number_input("🚧 Expected Goals Against (xGA)", value=data.get("xGA", 0.0))
interceptions = st.number_input("🛑 Interceptions par Match", value=data.get("interceptions", 0))
tacles = st.number_input("⚔️ Tacles Réussis par Match", value=data.get("tacles", 0))
degagements = st.number_input("💨 Dégagements par Match", value=data.get("degagements", 0))
penalities_concedees = st.number_input("⚠️ Pénalties Concédées", value=data.get("penalities_concedees", 0))
possession_remportee_tiers = st.number_input("🔄 Possession Remportée Dernier Tiers", value=data.get("possession_remportee_tiers", 0))
arrets = st.number_input("🧤 Arrêts par Match", value=data.get("arrets", 0))

st.header("🚨 Critères Discipline")
fautes = st.number_input("⚠️ Fautes par Match", value=data.get("fautes", 0))
cartons_jaunes = st.number_input("🟨 Cartons Jaunes", value=data.get("cartons_jaunes", 0))
cartons_rouges = st.number_input("🟥 Cartons Rouges", value=data.get("cartons_rouges", 0))

# Sauvegarde des données
data.update({
    "forme_team1": forme_team1,
    "forme_team2": forme_team2,
    "face_a_face": face_a_face
})
save_data(data)

st.success("💾 Données sauvegardées automatiquement !")

# Chargement des modèles de prédiction
try:
    model_logistic = joblib.load("logistic_model.pkl")
    model_rf = joblib.load("random_forest_model.pkl")
    model_poisson = joblib.load("poisson_model.pkl")
except FileNotFoundError:
    model_logistic = model_rf = model_poisson = None

# Faire une prédiction
if model_logistic and model_rf and model_poisson:
    features = np.array([
        buts_par_match, buts_concedes_par_match, xG, tirs_cadres,
        grandes_chances, passes_reussies, corners, xGA, interceptions,
        tacles, degagements, fautes, cartons_jaunes, cartons_rouges
    ]).reshape(1, -1)
    
    prediction_logistic = model_logistic.predict(features)[0]
    prediction_rf = model_rf.predict(features)[0]
    prediction_poisson = model_poisson.predict(features)[0]
    
    st.header("🔮 Prédictions des Modèles")
    st.write(f"📈 Régression Logistique: {prediction_logistic}")
    st.write(f"📊 Random Forest: {prediction_rf}")
    st.write(f"📉 Modèle de Poisson: {prediction_poisson}")
