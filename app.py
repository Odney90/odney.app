import streamlit as st  
import numpy as np  

# --- Initialisation des valeurs dans session_state ---  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        "score_rating_A": 70.0,  
        "buts_totaux_A": 50.0,  
        "buts_par_match_A": 1.5,  
        "buts_concedes_par_match_A": 1.0,  
        "buts_concedes_totaux_A": 30.0,  
        "possession_moyenne_A": 55.0,  
        "aucun_but_encaisse_A": 10,  
        "expected_but_A": 1.5,  
        "tirs_cadres_A": 10.0,  
        "grandes_chances_A": 5.0,  
        "grandes_chances_manquees_A": 2.0,  
        "passes_reussies_A": 300.0,  
        "passes_longues_A": 15.0,  
        "centres_reussis_A": 5.0,  
        "penalties_obtenues_A": 1.0,  
        "corners_A": 4.0,  
        "expected_concedes_A": 1.0,  
        "interceptions_A": 10.0,  
        "tacles_reussis_A": 5.0,  
        "degagements_A": 8.0,  
        "arrets_A": 3.0,  
        "score_rating_B": 65.0,  
        "buts_totaux_B": 40.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "buts_concedes_totaux_B": 35.0,  
        "possession_moyenne_B": 45.0,  
        "aucun_but_encaisse_B": 8,  
        "expected_but_B": 1.0,  
        "tirs_cadres_B": 8.0,  
        "grandes_chances_B": 3.0,  
        "grandes_chances_manquees_B": 1.0,  
        "passes_reussies_B": 250.0,  
        "passes_longues_B": 10.0,  
        "centres_reussis_B": 3.0,  
        "penalties_obtenues_B": 0.0,  
        "corners_B": 3.0,  
        "expected_concedes_B": 1.5,  
        "interceptions_B": 8.0,  
        "tacles_reussis_B": 4.0,  
        "degagements_B": 6.0,  
        "arrets_B": 2.0  
    }  

# --- Interface Utilisateur ---  
st.header("⚽ Statistiques des Équipes de Football")  

# --- Critères pour l'équipe A ---  
st.subheader("🎯 Critères pour l'équipe A")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["score_rating_A"] = st.number_input("⭐ Score Rating", min_value=0.0, value=st.session_state.data["score_rating_A"], key="score_rating_A")  
    st.session_state.data["buts_totaux_A"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=st.session_state.data["buts_totaux_A"], key="buts_totaux_A")  
    st.session_state.data["buts_par_match_A"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=st.session_state.data["buts_par_match_A"], key="buts_par_match_A")  
    st.session_state.data["tirs_cadres_A"] = st.number_input("🎯 Tirs Cadres", min_value=0.0, value=st.session_state.data["tirs_cadres_A"], key="tirs_cadres_A")  
    st.session_state.data["passes_reussies_A"] = st.number_input("✅ Passes Réussies", min_value=0.0, value=st.session_state.data["passes_reussies_A"], key="passes_reussies_A")  

with col2:  
    st.session_state.data["buts_concedes_par_match_A"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=st.session_state.data["buts_concedes_par_match_A"], key="buts_concedes_par_match_A")  
    st.session_state.data["buts_concedes_totaux_A"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=st.session_state.data["buts_concedes_totaux_A"], key="buts_concedes_totaux_A")  
    st.session_state.data["possession_moyenne_A"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=st.session_state.data["possession_moyenne_A"], key="possession_moyenne_A")  
    st.session_state.data["corners_A"] = st.number_input("⚽ Corners", min_value=0.0, value=st.session_state.data["corners_A"], key="corners_A")  
    st.session_state.data["grandes_chances_A"] = st.number_input("🌟 Grandes Chances", min_value=0.0, value=st.session_state.data["grandes_chances_A"], key="grandes_chances_A")  

# --- Critères pour l'équipe B ---  
st.subheader("🎯 Critères pour l'équipe B")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["score_rating_B"] = st.number_input("⭐ Score Rating", min_value=0.0, value=st.session_state.data["score_rating_B"], key="score_rating_B")  
    st.session_state.data["buts_totaux_B"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=st.session_state.data["buts_totaux_B"], key="buts_totaux_B")  
    st.session_state.data["buts_par_match_B"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=st.session_state.data["buts_par_match_B"], key="buts_par_match_B")  
    st.session_state.data["tirs_cadres_B"] = st.number_input("🎯 Tirs Cadres", min_value=0.0, value=st.session_state.data["tirs_cadres_B"], key="tirs_cadres_B")  
    st.session_state.data["passes_reussies_B"] = st.number_input("✅ Passes Réussies", min_value=0.0, value=st.session_state.data["passes_reussies_B"], key="passes_reussies_B")  

with col2:  
    st.session_state.data["buts_concedes_par_match_B"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=st.session_state.data["buts_concedes_par_match_B"], key="buts_concedes_par_match_B")  
    st.session_state.data["buts_concedes_totaux_B"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=st.session_state.data["buts_concedes_totaux_B"], key="buts_concedes_totaux_B")  
    st.session_state.data["possession_moyenne_B"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=st.session_state.data["possession_moyenne_B"], key="possession_moyenne_B")  
    st.session_state.data["corners_B"] = st.number_input("⚽ Corners", min_value=0.0, value=st.session_state.data["corners_B"], key="corners_B")  
    st.session_state.data["grandes_chances_B"] = st.number_input("🌟 Grandes Chances", min_value=0.0, value=st.session_state.data["grandes_chances_B"], key="grandes_chances_B")  

# --- Méthode de Prédiction ---  
st.subheader("🔮 Prédiction des Buts")  
if st.button("Prédire les Buts"):  
    expected_goals_A = st.session_state.data["expected_but_A"]  # Utiliser les données saisies  
    expected_goals_B = st.session_state.data["expected_but_B"]  # Utiliser les données saisies  

    # Simple prédiction basée sur les buts attendus  
    st.write(f"Prédiction des buts pour l'équipe A : {expected_goals_A:.2f}")  
    st.write(f"Prédiction des buts pour l'équipe B : {expected_goals_B:.2f}")  

# --- Affichage des données sauvegardées ---  
st.subheader("📊 Données Actuelles")  
st.write(st.session_state.data)
