import streamlit as st  
import numpy as np  
import pandas as pd  

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
        
        "score_rating_B": 65.0,  
        "buts_totaux_B": 40.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "buts_concedes_totaux_B": 35.0,  
        "possession_moyenne_B": 45.0,  
        "aucun_but_encaisse_B": 8,  
        
        "recent_form_A": [0, 0, 0, 0, 0],  # Forme récente sur 5 matchs  
        "recent_form_B": [0, 0, 0, 0, 0],  # Forme récente sur 5 matchs  
        "head_to_head": [],  # Historique des confrontations  
        "conditions_match": "",  # Conditions du match  
    }  

# --- Interface Utilisateur ---  
st.header("⚽ Statistiques des Équipes de Football")  

# --- Forme récente sur 5 matchs ---  
st.subheader("📈 Forme Récente sur 5 Matchs")  
col1, col2 = st.columns(2)  

with col1:  
    st.write("Équipe A")  
    for i in range(5):  
        st.session_state.data["recent_form_A"][i] = st.number_input(f"Match {i + 1} (Buts marqués)", min_value=0, value=st.session_state.data["recent_form_A"][i], key=f"recent_form_A_{i}")  

with col2:  
    st.write("Équipe B")  
    for i in range(5):  
        st.session_state.data["recent_form_B"][i] = st.number_input(f"Match {i + 1} (Buts marqués)", min_value=0, value=st.session_state.data["recent_form_B"][i], key=f"recent_form_B_{i}")  

# --- Conditions du Match ---  
st.subheader("🌦️ Conditions du Match")  
st.session_state.data["conditions_match"] = st.text_input("Conditions du Match (ex: pluie, terrain sec, etc.)", value=st.session_state.data["conditions_match"])  

# --- Historique des confrontations ---  
st.subheader("📊 Historique des Confrontations Directes")  
if st.button("Ajouter un résultat de confrontation"):  
    equipe_A_buts = st.number_input("Buts Équipe A", min_value=0)  
    equipe_B_buts = st.number_input("Buts Équipe B", min_value=0)  
    st.session_state.data["head_to_head"].append((equipe_A_buts, equipe_B_buts))  
    st.success("Résultat ajouté !")  

# Afficher l'historique  
if st.session_state.data["head_to_head"]:  
    st.write("Historique des Confrontations :")  
    for index, (buts_A, buts_B) in enumerate(st.session_state.data["head_to_head"]):  
        st.write(f"Match {index + 1}: Équipe A {buts_A} - Équipe B {buts_B}")  

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

with col1:  
    st.session_state.data["score_rating_B"] = st.number_input("⭐ Score Rating Équipe B", min_value=0.0, value=st.session_state.data["score_rating_B"], key="score_rating_B")  
    st.session_state.data["buts_totaux_B"] = st.number_input("⚽ Buts Totaux Équipe B", min_value=0.0, value=st.session_state.data["buts_totaux_B"], key="buts_totaux_B")  
    st.session_state.data["buts_par_match_B"] = st.number_input("🥅 Buts par Match Équipe B", min_value=0.0, value=st.session_state.data["buts_par_match_B"], key="buts_par_match_B")  
    st.session_state.data["buts_concedes_par_match_B"] = st.number_input("🚫 Buts Concédés par Match Équipe B", min_value=0.0, value=st.session_state.data["buts_concedes_par_match_B"], key="buts_concedes_par_match_B")  
    st.session_state.data["buts_concedes_totaux_B"] = st.number_input("🤕 Buts Concédés Totaux Équipe B", min_value=0.0, value=st.session_state.data["buts_concedes_totaux_B"], key="buts_concedes_totaux_B")  

with col2:  
    st.session_state.data["possession_moyenne_B"] = st.number_input("Ballon Possession Moyenne Équipe B (%)", min_value=0.0, max_value=100.0, value=st.session_state.data["possession_moyenne_B"], key="possession_moyenne_B")  
    st.session_state.data["aucun_but_encaisse_B"] = st.number_input("🔒 Aucun But Encaissé Équipe B", min_value=0, value=st.session_state.data["aucun_but_encaisse_B"], key="aucun_but_encaisse_B")  

# --- Critères d'Attaque ---  
st.subheader("⚔️ Critères d'Attaque")  
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
st.subheader("🛡️ Critères de Défense")  
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

# --- Critères d'Attaque Équipe B ---  
st.subheader("⚔️ Critères d'Attaque Équipe B")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["expected_but_B"] = st.number_input("Expected Buts Équipe B", min_value=0.0, value=st.session_state.data["expected_but_B"], key="expected_but_B")  
    st.session_state.data["tirs_cadres_B"] = st.number_input("🎯 Tirs Cadres Équipe B", min_value=0.0, value=st.session_state.data["tirs_cadres_B"], key="tirs_cadres_B")  
    st.session_state.data["grandes_chances_B"] = st.number_input("🌟 Grandes Chances Équipe B", min_value=0.0, value=st.session_state.data["grandes_chances_B"], key="grandes_chances_B")  
    st.session_state.data["grandes_chances_manquees_B"] = st.number_input("🌟 Grandes Chances Manquées Équipe B", min_value=0.0, value=st.session_state.data["grandes_chances_manquees_B"], key="grandes_chances_manquees_B")  
    st.session_state.data["passes_reussies_B"] = st.number_input("✅ Passes Réussies Équipe B", min_value=0.0, value=st.session_state.data["passes_reussies_B"], key="passes_reussies_B")  

with col2:  
    st.session_state.data["passes_longues_B"] = st.number_input("Passes Longues Précises Équipe B", min_value=0.0, value=st.session_state.data["passes_longues_B"], key="passes_longues_B")  
    st.session_state.data["centres_reussis_B"] = st.number_input("Centres Réussis Équipe B", min_value=0.0, value=st.session_state.data["centres_reussis_B"], key="centres_reussis_B")  
    st.session_state.data["penalties_obtenues_B"] = st.number_input("Pénalités Obtenues Équipe B", min_value=0.0, value=st.session_state.data["penalties_obtenues_B"], key="penalties_obtenues_B")  
    st.session_state.data["balles_surface_B"] = st.number_input("Balles Touchées dans la Surface Équipe B", min_value=0.0, value=st.session_state.data["balles_surface_B"], key="balles_surface_B")  
    st.session_state.data["corners_B"] = st.number_input("⚽ Corners Équipe B", min_value=0.0, value=st.session_state.data["corners_B"], key="corners_B")  

# --- Critères de Défense Équipe B ---  
st.subheader("🛡️ Critères de Défense Équipe B")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["expected_concedes_B"] = st.number_input("Expected Buts Concédés Équipe B", min_value=0.0, value=st.session_state.data["expected_concedes_B"], key="expected_concedes_B")  
    st.session_state.data["interceptions_B"] = st.number_input("Interceptions Équipe B", min_value=0.0, value=st.session_state.data["interceptions_B"], key="interceptions_B")  
    st.session_state.data["tacles_reussis_B"] = st.number_input("Tacles Réussis Équipe B", min_value=0.0, value=st.session_state.data["tacles_reussis_B"], key="tacles_reussis_B")  
    st.session_state.data["degagements_B"] = st.number_input("Dégagements Équipe B", min_value=0.0, value=st.session_state.data["degagements_B"], key="degagements_B")  

with col2:  
    st.session_state.data["penalties_concedes_B"] = st.number_input("Pénalités Concédées Équipe B", min_value=0.0, value=st.session_state.data["penalties_concedes_B"], key="penalties_concedes_B")  
    st.session_state.data["possessions_remporte_B"] = st.number_input("Possessions Remportées Équipe B", min_value=0.0, value=st.session_state.data["possessions_remporte_B"], key="possessions_remporte_B")  
    st.session_state.data["arrets_B"] = st.number_input("Arrêts Équipe B", min_value=0.0, value=st.session_state.data["arrets_B"], key="arrets_B")  

# --- Discipline ---  
st.subheader("📜 Discipline")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["fautes_A"] = st.number_input("Fautes par Match Équipe A", min_value=0.0, value=st.session_state.data["fautes_A"], key="fautes_A")  
    st.session_state.data["cartons_jaunes_A"] = st.number_input("Cartons Jaunes Équipe A", min_value=0.0, value=st.session_state.data["cartons_jaunes_A"], key="cartons_jaunes_A")  

with col2:  
    st.session_state.data["cartons_rouges_A"] = st.number_input("Cartons Rouges Équipe A", min_value=0.0, value=st.session_state.data["cartons_rouges_A"], key="cartons_rouges_A")  

with col1:  
    st.session_state.data["fautes_B"] = st.number_input("Fautes par Match Équipe B", min_value=0.0, value=st.session_state.data["fautes_B"], key="fautes_B")  
    st.session_state.data["cartons_jaunes_B"] = st.number_input("Cartons Jaunes Équipe B", min_value=0.0, value=st.session_state.data["cartons_jaunes_B"], key="cartons_jaunes_B")
  
with col2:                                                            
   st.session_state.data["cartons_rouges_B"] = st.number_input("Cartons Rouges Équipe B", min_value=0.0, value=st.session_state.data["cartons_rouges_B"], key="cartons_rouges_B")  

# --- Méthode de Prédiction ---  
st.subheader("🔮 Méthode de Prédiction")  
if st.button("Prédire le Résultat"):  
    # Calculer la forme récente  
    forme_A = np.mean(st.session_state.data["recent_form_A"])  
    forme_B = np.mean(st.session_state.data["recent_form_B"])  

    # Calculer les moyennes des buts marqués et concédés  
    moyenne_buts_A = st.session_state.data["buts_totaux_A"] / (len(st.session_state.data["recent_form_A"]) or 1)  
    moyenne_buts_B = st.session_state.data["buts_totaux_B"] / (len(st.session_state.data["recent_form_B"]) or 1)  
    moyenne_buts_concedes_A = st.session_state.data["buts_concedes_totaux_A"] / (len(st.session_state.data["recent_form_A"]) or 1)  
    moyenne_buts_concedes_B = st.session_state.data["buts_concedes_totaux_B"] / (len(st.session_state.data["recent_form_B"]) or 1)  

    # Calculer les buts attendus avec le modèle de Poisson  
    lambda_A = (moyenne_buts_A * forme_A) / (moyenne_buts_concedes_B + 1)  # Ajustement pour la défense de l'équipe B  
    lambda_B = (moyenne_buts_B * forme_B) / (moyenne_buts_concedes_A + 1)  # Ajustement pour la défense de l'équipe A  

    # Prédire le nombre de buts  
    buts_A = np.random.poisson(lambda_A)  
    buts_B = np.random.poisson(lambda_B)  

    st.write(f"Prédiction : Équipe A {buts_A} - Équipe B {buts_B}")  

    # Afficher les probabilités de victoire  
    prob_A = (lambda_A / (lambda_A + lambda_B)) * 100  
    prob_B = (lambda_B / (lambda_A + lambda_B)) * 100  

    st.write(f"Probabilité de victoire Équipe A : {prob_A:.2f}%")  
    st.write(f"Probabilité de victoire Équipe B : {prob_B:.2f}%")  

# --- Fin du Code ---
