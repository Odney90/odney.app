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
        "expected_but_A": 1.5,  
        "tirs_cadres_A": 10.0,  
        "grandes_chances_A": 5.0,  
        "grandes_chances_manquees_A": 2.0,  
        "passes_reussies_A": 300.0,  
        "passes_longues_A": 15.0,  
        "centres_reussis_A": 5.0,  
        "penalties_obtenues_A": 1.0,  
        "balles_surface_A": 10.0,  
        "corners_A": 4.0,  
        "expected_concedes_A": 1.0,  
        "interceptions_A": 10.0,  
        "tacles_reussis_A": 5.0,  
        "degagements_A": 8.0,  
        "penalties_concedes_A": 0.0,  
        "possessions_remporte_A": 15.0,  
        "arrets_A": 3.0,  
        "fautes_A": 5.0,  
        "cartons_jaunes_A": 2.0,  
        "cartons_rouges_A": 0.0,  
        
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
        "balles_surface_B": 8.0,  
        "corners_B": 3.0,  
        "expected_concedes_B": 1.5,  
        "interceptions_B": 8.0,  
        "tacles_reussis_B": 4.0,  
        "degagements_B": 6.0,  
        "penalties_concedes_B": 1.0,  
        "possessions_remporte_B": 12.0,  
        "arrets_B": 2.0,  
        "fautes_B": 4.0,  
        "cartons_jaunes_B": 1.0,  
        "cartons_rouges_B": 0.0,  
        
        "recent_form_A": [0, 0, 0, 0, 0],  # Forme récente sur 5 matchs  
        "recent_form_B": [0, 0, 0, 0, 0],  # Forme récente sur 5 matchs  
        "head_to_head": []  # Historique des confrontations  
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

# --- Critères Équipe A ---  
st.subheader("📊 Critères Équipe A")  
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
    st.session_state.data["expected_but_A"] = st.number_input("Expected Buts Équipe A", min_value=0.0, value=st.session_state.data["expected_but_A"], key="expected_but_A")  
    st.session_state.data["tirs_cadres_A"] = st.number_input("🎯 Tirs Cadres Équipe A", min_value=0.0, value=st.session_state.data["tirs_cadres_A"], key="tirs_cadres_A")  
    st.session_state.data["grandes_chances_A"] = st.number_input("🌟 Grandes Chances Équipe A", min_value=0.0, value=st.session_state.data["grandes_chances_A"], key="grandes_chances_A")  

# --- Critères Équipe B ---  
st.subheader("📊 Critères Équipe B")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state.data["score_rating_B"] = st.number_input("⭐ Score Rating Équipe B", min_value=0.0, value=st.session_state.data["score_rating_B"], key="score_rating_B")  
    st.session_state.data["buts_totaux_B"] = st.number_input("⚽ Buts Totaux Équipe B", min_value=0.0, value=st.session_state.data["buts_totaux_B"], key="buts_totaux_B")  
    st.session_state.data["buts_par_match_B"] = st.number_input("🥅 Buts par Match Équipe B", min_value=0.0, value=st.session_state.data["buts_par_match_B"], key="buts_par_match_B")  
    st.session_state.data["buts_concedes_par_match_B"] = st.number_input("🚫 Buts Concédés par Match Équipe B", min_value=0.0, value=st.session_state.data["buts_concedes_par_match_B"], key="buts_concedes_par_match_B")  
    st.session_state.data["buts_concedes_totaux_B"] = st.number_input("🤕 Buts Concédés Totaux Équipe B", min_value=0.0, value=st.session_state.data["buts_concedes_totaux_B"], key="buts_concedes_totaux_B")  

with col2:  
    st.session_state.data["possession_moyenne_B"] = st.number_input("Ballon Possession Moyenne Équipe B (%)", min_value=0.0, max_value=100.0, value=st.session_state.data["possession_moyenne_B"], key="possession_moyenne_B")  
    st.session_state.data["aucun_but_encaisse_B"] = st.number_input("🔒 Aucun But Encaissé Équipe B", min_value=0, value=st.session_state.data["aucun_but_encaisse_B"], key="aucun_but_encaisse_B")  
    st.session_state.data["expected_but_B"] = st.number_input("Expected Buts Équipe B", min_value=0.0, value=st.session_state.data["expected_but_B"], key="expected_but_B")  
    st.session_state.data["tirs_cadres_B"] = st.number_input("🎯 Tirs Cadres Équipe B", min_value=0.0, value=st.session_state.data["tirs_cadres_B"], key="tirs_cadres_B")  
    st.session_state.data["grandes_chances_B"] = st.number_input("🌟 Grandes Chances Équipe B", min_value=0.0, value=st.session_state.data["grandes_chances_B"], key="grandes_chances_B")  

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

# --- Méthode de Prédiction ---  
st.subheader("🔮 Méthode de Prédiction")  
if st.button("Prédire le Résultat"):  
    # Calculer la forme récente  
    forme_A = np.mean(st.session_state.data["recent_form_A"])  
    forme_B = np.mean(st.session_state.data["recent_form_B"])  

    # Calculer les poids des critères  
    poids_criteres = {  
        "Score Rating": (st.session_state.data["score_rating_A"] + st.session_state.data["score_rating_B"]) / 2,  
        "Buts Totaux": (st.session_state.data["buts_totaux_A"] + st.session_state.data["buts_totaux_B"]) / 2,  
        "Buts Concedes": (st.session_state.data["buts_concedes_totaux_A"] + st.session_state.data["buts_concedes_totaux_B"]) / 2,  
        "Possession": (st.session_state.data["possession_moyenne_A"] + st.session_state.data["possession_moyenne_B"]) / 2,  
        "Forme Récente": (forme_A + forme_B) / 2,  
        "Tirs Cadres": (st.session_state.data["tirs_cadres_A"] + st.session_state.data["tirs_cadres_B"]) / 2,  
        "Grandes Chances": (st.session_state.data["grandes_chances_A"] + st.session_state.data["grandes_chances_B"]) / 2,  
    }  

    # Afficher les poids des critères  
    st.write("Poids des Critères :")  
    df_poids = pd.DataFrame(list(poids_criteres.items()), columns=["Critère", "Poids"])  
    st.table(df_poids)  

    # Prédiction simple basée sur les poids  
    prediction_A = (poids_criteres["Score Rating"] + poids_criteres["Buts Totaux"] - poids_criteres["Buts Concedes"] + poids_criteres["Forme Récente"]) / 4  
    prediction_B = (poids_criteres["Score Rating"] + poids_criteres["Buts Totaux"] - poids_criteres["Buts Concedes"] + poids_criteres["Forme Récente"]) / 4  

    st.write(f"Prédiction : Équipe A {prediction_A:.2f} - Équipe B {prediction_B:.2f}")
