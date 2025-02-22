import streamlit as st  
import json  
import os  
from scipy.stats import poisson  

# Chemin du fichier de sauvegarde  
data_file = 'data.json'  

def load_data():  
    """Charge les données à partir d'un fichier JSON."""  
    if os.path.exists(data_file):  
        with open(data_file, 'r') as f:  
            return json.load(f)  
    return {}  

def save_data(data):  
    """Sauvegarde les données dans un fichier JSON."""  
    with open(data_file, 'w') as f:  
        json.dump(data, f)  

# Chargement des données  
data = load_data()  

# --- Initialisation des valeurs par défaut ---  
default_values_A = {  
    "score_rating_A": 70.0, "buts_totaux_A": 50.0, "buts_par_match_A": 1.5,  
    "buts_concedes_par_match_A": 1.0, "buts_concedes_totaux_A": 30.0, "possession_moyenne_A": 55.0,  
    "aucun_but_encaisse_A": 10, "expected_but_A": 1.5, "tirs_cadres_A": 10.0,  
    "grandes_chances_A": 5.0, "grandes_chances_manquees_A": 2.0, "passes_reussies_A": 300.0,  
    "passes_longues_A": 15.0, "centres_reussis_A": 5.0, "penalties_obtenues_A": 1.0,  
    "balles_surface_A": 10.0, "corners_A": 4.0, "expected_concedes_A": 1.0,  
    "interceptions_A": 10.0, "tacles_reussis_A": 5.0, "degagements_A": 8.0,  
    "penalties_concedes_A": 0.0, "arrets_A": 3.0  
}  

default_values_B = {  
    "score_rating_B": 65.0, "buts_totaux_B": 40.0, "buts_par_match_B": 1.0,  
    "buts_concedes_par_match_B": 1.5, "buts_concedes_totaux_B": 35.0, "possession_moyenne_B": 45.0,  
    "aucun_but_encaisse_B": 8, "expected_but_B": 1.0, "tirs_cadres_B": 8.0,  
    "grandes_chances_B": 3.0, "grandes_chances_manquees_B": 1.0, "passes_reussies_B": 250.0,  
    "passes_longues_B": 10.0, "centres_reussis_B": 3.0, "penalties_obtenues_B": 0.0,  
    "balles_surface_B": 8.0, "corners_B": 3.0, "expected_concedes_B": 1.5,  
    "interceptions_B": 8.0, "tacles_reussis_B": 4.0, "degagements_B": 6.0,  
    "penalties_concedes_B": 1.0, "arrets_B": 2.0  
}  

# Initialisation des valeurs pour l'équipe A  
for key, value in default_values_A.items():  
    data.setdefault(key, value)  

# Initialisation des valeurs pour l'équipe B  
for key, value in default_values_B.items():  
    data.setdefault(key, value)  

# --- Interface Utilisateur ---  
st.header("⚽ Statistiques des Équipes")  

# --- Critères pour l'équipe A ---  
st.subheader("🎯 Critères pour l'équipe A")  
col1, col2 = st.columns(2)  
with col1:  
    data["score_rating_A"] = st.number_input("⭐ Score Rating", min_value=0.0, value=data["score_rating_A"], key="score_rating_A")  
    data["buts_totaux_A"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=data["buts_totaux_A"], key="buts_totaux_A")  
    data["buts_par_match_A"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=data["buts_par_match_A"], key="buts_par_match_A")  
with col2:  
    data["buts_concedes_par_match_A"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=data["buts_concedes_par_match_A"], key="buts_concedes_par_match_A")  
    data["buts_concedes_totaux_A"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=data["buts_concedes_totaux_A"], key="buts_concedes_totaux_A")  
    data["possession_moyenne_A"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=data["possession_moyenne_A"], key="possession_moyenne_A")  
data["aucun_but_encaisse_A"] = st.number_input("🔒 Aucun But Encaissé", min_value=0, value=data["aucun_but_encaisse_A"], key="aucun_but_encaisse_A")  

# --- Critères pour l'équipe B ---  
st.subheader("🎯 Critères pour l'équipe B")  
col1, col2 = st.columns(2)  
with col1:  
    data["score_rating_B"] = st.number_input("⭐ Score Rating", min_value=0.0, value=data.get("score_rating_B", 65.0), key="score_rating_B")  
    data["buts_totaux_B"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=data.get("buts_totaux_B", 40.0), key="buts_totaux_B")  
    data["buts_par_match_B"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=data.get("buts_par_match_B", 1.0), key="buts_par_match_B")  
with col2:  
    data["buts_concedes_par_match_B"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=data.get("buts_concedes_par_match_B", 1.5), key="buts_concedes_par_match_B")  
    data["buts_concedes_totaux_B"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=data.get("buts_concedes_totaux_B", 35.0), key="buts_concedes_totaux_B")  
    data["possession_moyenne_B"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=data.get("possession_moyenne_B", 45.0), key="possession_moyenne_B")  
data["aucun_but_encaisse_B"] = st.number_input("🔒 Aucun But Encaissé", min_value=0, value=data.get("aucun_but_encaisse_B", 8), key="aucun_but_encaisse_B")  

# --- Attaque pour l'équipe A ---  
st.subheader("⚔️ Attaque pour l'équipe A")  
col1, col2 = st.columns(2)  
with col1:  
    data["expected_but_A"] = st.number_input("⭐ Expected But (xG)", min_value=0.0, value=data["expected_but_A"], key="expected_but_A")  
    data["tirs_cadres_A"] = st.number_input("🎯 Tirs Cadrés par Match", min_value=0.0, value=data["tirs_cadres_A"], key="tirs_cadres_A")  
    data["grandes_chances_A"] = st.number_input("✨ Grandes Chances", min_value=0.0, value=data["grandes_chances_A"], key="grandes_chances_A")  
with col2:  
    data["grandes_chances_manquees_A"] = st.number_input("💨 Grandes Chances Manquées", min_value=0.0, value=data["grandes_chances_manquees_A"], key="grandes_chances_manquees_A")  
    data["passes_reussies_A"] = st.number_input("✅ Passes Réussies par Match", min_value=0.0, value=data["passes_reussies_A"], key="passes_reussies_A")  
    data["passes_longues_A"] = st.number_input("➡️ Passes Longues Précises par Match", min_value=0.0, value=data["passes_longues_A"], key="passes_longues_A")  

# --- Défense pour l'équipe A ---  
st.subheader("🛡️ Défense pour l'équipe A")  
col1, col2 = st.columns(2)  
with col1:  
    data["expected_concedes_A"] = st.number_input("⭐ xG Concédés", min_value=0.0, value=data["expected_concedes_A"], key="expected_concedes_A")  
    data["interceptions_A"] = st.number_input("✋ Interceptions par Match", min_value=0.0, value=data["interceptions_A"], key="interceptions_A")  
    data["tacles_reussis_A"] = st.number_input("Tacles Réussis par Match", min_value=0.0, value=data["tacles_reussis_A"], key="tacles_reussis_A")  
with col2:  
    data["degagements_A"] = st.number_input("Dégagements par Match", min_value=0.0, value=data["degagements_A"], key="degagements_A")  
    data["penalties_concedes_A"] = st.number_input("🎁 Pénalties Concédés", min_value=0.0, value=data["penalties_concedes_A"], key="penalties_concedes_A")  
    data["arrets_A"] = st.number_input("🛑 Arrêts par Match", min_value=0.0, value=data["arrets_A"], key="arrets_A")  

# --- Attaque pour l'équipe B ---  
st.subheader("⚔️ Attaque pour l'équipe B")  
col1, col2 = st.columns(2)  
with col1:  
    data["expected_but_B"] = st.number_input("⭐ Expected But (xG)", min_value=0.0, value=data["expected_but_B"], key="expected_but_B")  
    data["tirs_cadres_B"] = st.number_input("🎯 Tirs Cadrés par Match", min_value=0.0, value=data["tirs_cadres_B"], key="tirs_cadres_B")  
    data["grandes_chances_B"] = st.number_input("✨ Grandes Chances", min_value=0.0, value=data["grandes_chances_B"], key="grandes_chances_B")  
with col2:  
    data["grandes_chances_manquees_B"] = st.number_input("💨 Grandes Chances Manquées", min_value=0.0, value=data["grandes_chances_manquees_B"], key="grandes_chances_manquees_B")  
    data["passes_reussies_B"] = st.number_input("✅ Passes Réussies par Match", min_value=0.0, value=data["passes_reussies_B"], key="passes_reussies_B")  
    data["passes_longues_B"] = st.number_input("➡️ Passes Longues Précises par Match", min_value=0.0, value=data["passes_longues_B"], key="passes_longues_B")  

# --- Défense pour l'équipe B ---  
st.subheader("🛡️ Défense pour l'équipe B")  
col1, col2 = st.columns(2)  
with col1:  
    data["expected_concedes_B"] = st.number_input("⭐ xG Concédés", min_value=0.0, value=data["expected_concedes_B"], key="expected_concedes_B")  
    data["interceptions_B"] = st.number_input("✋ Interceptions par Match", min_value=0.0, value=data["interceptions_B"], key="interceptions_B")  
    data["tacles_reussis_B"] = st.number_input("Tacles Réussis par Match", min_value=0.0, value=data["tacles_reussis_B"], key="tacles_reussis_B")  
with col2:  
    data["degagements_B"] = st.number_input("Dégagements par Match", min_value=0.0, value=data["degagements_B"], key="degagements_B")  
    data["penalties_concedes_B"] = st.number_input("🎁 Pénalties Concédés", min_value=0.0, value=data["penalties_concedes_B"], key="penalties_concedes_B")  
    data["arrets_B"] = st.number_input("🛑 Arrêts par Match", min_value=0.0, value=data["arrets_B"], key="arrets_B")  

# Sauvegarde des données après chaque mise à jour  
save_data(data)  

# Affichage des données sauvegardées  
st.subheader("📊 Données Sauvegardées")  
st.write(data)  

# --- Prédiction des buts ---  
if st.button("Prédire les Buts"):  
    # Calcul des buts attendus pour chaque équipe  
    buts_moyens_A = poisson.pmf(range(6), data["expected_but_A"])  
    buts_moyens_B = poisson.pmf(range(6), data["expected_but_B"])  
    
    # Affichage des résultats  
    st.subheader("🎯 Prédiction des Buts")  
    st.write(f"Prédiction des buts pour l'équipe A : {buts_moyens_A}")  
    st.write(f"Prédiction des buts pour l'équipe B : {buts_moyens_B}")
