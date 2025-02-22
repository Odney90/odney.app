import streamlit as st  
import numpy as np  
from scipy.stats import poisson  
import json  
import os  

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

# --- Chargement des données initiales ---  
data = load_data()  

# --- Initialisation des valeurs dans session_state ---  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        "score_rating_A": data.get("score_rating_A", 70.0),  
        "buts_totaux_A": data.get("buts_totaux_A", 50.0),  
        "buts_par_match_A": data.get("buts_par_match_A", 1.5),  
        "buts_concedes_par_match_A": data.get("buts_concedes_par_match_A", 1.0),  
        "buts_concedes_totaux_A": data.get("buts_concedes_totaux_A", 30.0),  
        "possession_moyenne_A": data.get("possession_moyenne_A", 55.0),  
        "aucun_but_encaisse_A": data.get("aucun_but_encaisse_A", 10),  
        "expected_but_A": data.get("expected_but_A", 1.5),  
        "tirs_cadres_A": data.get("tirs_cadres_A", 10.0),  
        "grandes_chances_A": data.get("grandes_chances_A", 5.0),  
        "grandes_chances_manquees_A": data.get("grandes_chances_manquees_A", 2.0),  
        "passes_reussies_A": data.get("passes_reussies_A", 300.0),  
        "passes_longues_A": data.get("passes_longues_A", 15.0),  
        "centres_reussis_A": data.get("centres_reussis_A", 5.0),  
        "penalties_obtenues_A": data.get("penalties_obtenues_A", 1.0),  
        "balles_surface_A": data.get("balles_surface_A", 10.0),  # Assurez-vous que cette clé est initialisée  
        "corners_A": data.get("corners_A", 4.0),  
        "expected_concedes_A": data.get("expected_concedes_A", 1.0),  
        "interceptions_A": data.get("interceptions_A", 10.0),  
        "tacles_reussis_A": data.get("tacles_reussis_A", 5.0),  
        "degagements_A": data.get("degagements_A", 8.0),  
        "penalties_concedes_A": data.get("penalties_concedes_A", 0.0),  
        "possessions_remporte_A": data.get("possessions_remporte_A", 15.0),  
        "arrets_A": data.get("arrets_A", 3.0),  
        "score_rating_B": data.get("score_rating_B", 65.0),  
        "buts_totaux_B": data.get("buts_totaux_B", 40.0),  
        "buts_par_match_B": data.get("buts_par_match_B", 1.0),  
        "buts_concedes_par_match_B": data.get("buts_concedes_par_match_B", 1.5),  
        "buts_concedes_totaux_B": data.get("buts_concedes_totaux_B", 35.0),  
        "possession_moyenne_B": data.get("possession_moyenne_B", 45.0),  
        "aucun_but_encaisse_B": data.get("aucun_but_encaisse_B", 8),  
        "expected_but_B": data.get("expected_but_B", 1.0),  
        "tirs_cadres_B": data.get("tirs_cadres_B", 8.0),  
        "grandes_chances_B": data.get("grandes_chances_B", 3.0),  
        "grandes_chances_manquees_B": data.get("grandes_chances_manquees_B", 1.0),  
        "passes_reussies_B": data.get("passes_reussies_B", 250.0),  
        "passes_longues_B": data.get("passes_longues_B", 10.0),  
        "centres_reussis_B": data.get("centres_reussis_B", 3.0),  
        "penalties_obtenues_B": data.get("penalties_obtenues_B", 0.0),  
        "balles_surface_B": data.get("balles_surface_B", 8.0),  # Assurez-vous que cette clé est initialisée  
        "corners_B": data.get("corners_B", 3.0),  
        "expected_concedes_B": data.get("expected_concedes_B", 1.5),  
        "interceptions_B": data.get("interceptions_B", 8.0),  
        "tacles_reussis_B": data.get("tacles_reussis_B", 4.0),  
        "degagements_B": data.get("degagements_B", 6.0),  
        "penalties_concedes_B": data.get("penalties_concedes_B", 1.0),  
        "possessions_remporte_B": data.get("possessions_remporte_B", 12.0),  
        "arrets_B": data.get("arrets_B", 2.0),  
        "fautes_A": data.get("fautes_A", 5.0),  
        "cartons_jaunes_A": data.get("cartons_jaunes_A", 2.0),  
        "cartons_rouges_A": data.get("cartons_rouges_A", 0.0),  
        "fautes_B": data.get("fautes_B", 4.0),  
        "cartons_jaunes_B": data.get("cartons_jaunes_B", 1.0),  
        "cartons_rouges_B": data.get("cartons_rouges_B", 0.0),  
    }  

# --- Interface Utilisateur ---  
st.header("⚽ Statistiques des Équipes de Football")  

# --- Top Statistiques ---  
st.subheader("📊 Top Statistiques")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["score_rating_A"] = st.number_input("⭐ Score Rating Équipe A", min_value=0.0, value=st.session_state.data["score_rating_A"], key="score_rating_A")  
    st.session_state.data["buts_totaux_A"] = st.number_input("⚽ Buts Totaux Équipe A", min_value=0.0, value=st.session_state.data["buts_totaux_A"], key="buts_totaux_A")  
    st.session_state.data["buts_par_match_A"] = st.number_input("🥅 Buts par Match Équipe A", min_value=0.0, value=st.session_state.data["buts_par_match_A"], key="buts_par_match_A")  
    st.session_state.data["buts_concedes_par_match_A"] = st.number_input("🚫 Buts Concédés par Match Équipe A", min_value=0.0, value=st.session_state.data["buts_concedes_par_match_A"], key="buts_concedes_par_match_A")  
    st.session_state.data["buts_concedes_totaux_A"] = st.number_input("🤕 Buts Concédés Totaux Équipe A", min_value=0.0, value=st.session_state.data["buts_concedes_totaux_A"], key="buts_concedes_totaux_A")  

with col2:  
    st.session_state.data["possession_moyenne_A"] = st.number_input("Ballon Possession Moyenne Équipe A (%)", min_value=0.0, max_value=100.0, value=st.session_state.data["possession_moyenne_A"], key="possession_moyenne_A")  
    st.session_state.data["aucun_but_encaisse_A"] = st.number_input("🔒 Aucun But Encaissé Équipe A", min_value=0, value=st.session_state.data["aucun_but_encaisse_A"], key="aucun_but_encaisse_A")  

# --- Critères d'Attaque ---  
st.subheader("⚔️ Critères d'Attaque Équipe A")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["expected_but_A"] = st.number_input("Expected Buts Équipe A", min_value=0.0, value=st.session_state.data["expected_but_A"], key="expected_but_A")  
    st.session_state.data["tirs_cadres_A"] = st.number_input("🎯 Tirs Cadres Équipe A", min_value=0.0, value=st.session_state.data["tirs_cadres_A"], key="tirs_cadres_A")  
    st.session_state.data["grandes_chances_A"] = st.number_input("🌟 Grandes Chances Équipe A", min_value=0.0, value=st.session_state.data["grandes_chances_A"], key="grandes_chances_A")  
    st.session_state.data["grandes_chances_manquees_A"] = st.number_input("🌟 Grandes Chances Manquées Équipe A", min_value=0.0, value=st.session_state.data["grandes_chances_manquees_A"], key="grandes_chances_manquees_A")  
    st.session_state.data["passes_reussies_A"] = st.number_input("✅ Passes Réussies Équipe A", min_value=0.0, value=st.session_state.data["passes_reussies_A"], key="passes_reussies_A")  

with col2:  
    st.session_state.data["passes_longues_A"] = st.number_input("Passes Longues Précises Équipe A", min_value=0.0, value=st.session_state.data["passes_longues_A"], key="passes_longues_A")  
    st.session_state.data["centres_reussis_A"] = st.number_input("Centres Réussis Équipe A", min_value=0.0, value=st.session_state.data["centres_reussis_A"], key="centres_reussis_A")  
    st.session_state.data["penalties_obtenues_A"] = st.number_input("Pénalités Obtenues Équipe A", min_value=0.0, value=st.session_state.data["penalties_obtenues_A"], key="penalties_obtenues_A")  
    st.session_state.data["balles_surface_A"] = st.number_input("Balles Touchées dans la Surface Équipe A", min_value=0.0, value=st.session_state.data["balles_surface_A"], key="balles_surface_A")  
    st.session_state.data["corners_A"] = st.number_input("⚽ Corners Équipe A", min_value=0.0, value=st.session_state.data["corners_A"], key="corners_A")  

# --- Critères de Défense ---  
st.subheader("🛡️ Critères de Défense Équipe A")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["expected_concedes_A"] = st.number_input("Expected Buts Concédés Équipe A", min_value=0.0, value=st.session_state.data["expected_concedes_A"], key="expected_concedes_A")  
    st.session_state.data["interceptions_A"] = st.number_input("Interceptions Équipe A", min_value=0.0, value=st.session_state.data["interceptions_A"], key="interceptions_A")  
    st.session_state.data["tacles_reussis_A"] = st.number_input("Tacles Réussis Équipe A", min_value=0.0, value=st.session_state.data["tacles_reussis_A"], key="tacles_reussis_A")  
    st.session_state.data["degagements_A"] = st.number_input("Dégagements Équipe A", min_value=0.0, value=st.session_state.data["degagements_A"], key="degagements_A")  

with col2:  
    st.session_state.data["penalties_concedes_A"] = st.number_input("Pénalités Concédées Équipe A", min_value=0.0, value=st.session_state.data["penalties_concedes_A"], key="penalties_concedes_A")  
    st.session_state.data["possessions_remporte_A"] = st.number_input("Possessions Remportées Équipe A", min_value=0.0, value=st.session_state.data["possessions_remporte_A"], key="possessions_remporte_A")  
    st.session_state.data["arrets_A"] = st.number_input("Arrêts Équipe A", min_value=0.0, value=st.session_state.data["arrets_A"], key="arrets_A")  

# --- Discipline ---  
st.subheader("📜 Discipline Équipe A")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["fautes_A"] = st.number_input("Fautes par Match Équipe A", min_value=0.0, value=st.session_state.data["fautes_A"], key="fautes_A")  
    st.session_state.data["cartons_jaunes_A"] = st.number_input("Cartons Jaunes Équipe A", min_value=0.0, value=st.session_state.data["cartons_jaunes_A"], key="cartons_jaunes_A")  

with col2:  
    st.session_state.data["cartons_rouges_A"] = st.number_input("Cartons Rouges Équipe A", min_value=0.0, value=st.session_state.data["cartons_rouges_A"], key="cartons_rouges_A")  

# --- Répéter pour l'équipe B ---  
# (Le même code que pour l'équipe A, mais avec les données de l'équipe B)  

# --- Prédictions ---  
st.subheader("🔮 Prédictions")  

# Prédiction des buts  
if st.button("Prédire les Buts"):  
    # Calcul des buts attendus  
    lambda_A = st.session_state.data["expected_but_A"]  
    lambda_B = st.session_state.data["expected_but_B"]  

    # Utilisation de la distribution de Poisson pour prédire le nombre de buts  
    predicted_goals_A = poisson.rvs(lambda_A)  
    predicted_goals_B = poisson.rvs(lambda_B)  

    st.write(f"Prédiction des buts pour l'équipe A : {predicted_goals_A}")  
    st.write(f"Prédiction des buts pour l'équipe B : {predicted_goals_B}")  

    # Prédiction du gagnant  
    if predicted_goals_A > predicted_goals_B:  
        st.write("🏆 L'équipe A gagne !")  
    elif predicted_goals_B > predicted_goals_A:  
        st.write("🏆 L'équipe B gagne !")  
    else:  
        st.write("🤝 Match nul !")  

# Sauvegarde des données après chaque mise à jour  
save_data(st.session_state.data)  

# --- Affichage des données sauvegardées ---  
st.subheader("📊 Données Actuelles")  
st.write(st.session_state.data)
