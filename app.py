import streamlit as st  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import poisson  

# Configuration de la page Streamlit  
st.set_page_config(  
    page_title="⚽ Prédiction de Match",  
    page_icon="⚽",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)  

# Ajout de CSS personnalisé pour améliorer l'apparence  
st.markdown(  
    """  
    <style>  
    body {  
        color: #333;  
        background-color: #f4f4f4;  
    }  
    .stApp {  
        background-color: #f0f8ff; /* Fond plus clair */  
    }  
    h1 {  
        color: #e44d26; /* Rouge-orange pour le titre principal */  
        text-align: center;  
    }  
    h2 {  
        color: #4285f4; /* Bleu pour les sous-titres */  
    }  
    h3 {  
        color: #0f9d58; /* Vert pour les sections */  
    }  
    .stNumberInput > label {  
        color: #757575;  
    }  
    .stButton > button {  
        background-color: #4CAF50;  
        color: white;  
    }  
    .sidebar .sidebar-content {  
        background-color: #e8f5e9;  
    }  
    </style>  
    """,  
    unsafe_allow_html=True  
)

init_session_state()
# --- Fonctions Utilitaires ---  
def calculer_forme(historique):  
    """Calcule la forme récente d'une équipe sur les 5 derniers matchs."""  
    return sum(historique) / len(historique) if historique else 0  

def prediction_buts_poisson(xG_A, xG_B):  
    """Prédit le nombre de buts attendus pour chaque équipe en utilisant la distribution de Poisson."""  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  
    buts_attendus_A = sum(i * prob for i, prob in enumerate(buts_A))  
    buts_attendus_B = sum(i * prob for i, prob in enumerate(buts_B))  
    return buts_attendus_A, buts_attendus_B  

def afficher_resultats_poisson(buts_moyens_A, buts_moyens_B):  
    """Affiche les résultats de la prédiction de Poisson."""  
    st.header("⚽ Prédiction des Buts (Méthode de Poisson)")  
    st.write(f"Buts attendus pour l'équipe A : **{buts_moyens_A:.2f}**")  
    st.write(f"Buts attendus pour l'équipe B : **{buts_moyens_B:.2f}**")  

def visualiser_forme_equipes(historique_A, historique_B):  
    """Visualise l'historique des performances des équipes."""  
    fig, ax = plt.subplots()  
    ax.plot(range(1, 6), historique_A, marker='o', label='Équipe A', color='blue')  
    ax.plot(range(1, 6), historique_B, marker='o', label='Équipe B', color='red')  
    ax.set_xticks(range(1, 6))  
    ax.set_xticklabels([f"Match {i}" for i in range(1, 6)], rotation=45)  
    ax.set_ylim(-1.5, 1.5)  
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')  
    ax.set_ylabel('Résultat')  
    ax.set_title('Historique des Performances des Équipes (5 Derniers Matchs)')  
    ax.legend()  
    st.pyplot(fig)
    # --- Sidebar pour les paramètres des équipes ---  
with st.sidebar:  
    st.header("⚙️ Paramètres des Équipes")  

    # Historique des 5 derniers matchs (1 pour victoire, 0 pour défaite, -1 pour nul)  
    st.subheader("📈 Forme Récente")  
    st.write("Entrez les résultats des 5 derniers matchs :")  
    historique_A = []  
    historique_B = []  
    for i in range(5):  
        result_A = st.slider(f"Match {i + 1} - Équipe A", min_value=-1, max_value=1, value=0, step=1, key=f"result_A_{i}")  
        historique_A.append(result_A)  
        result_B = st.slider(f"Match {i + 1} - Équipe B", min_value=-1, max_value=1, value=0, step=1, key=f"result_B_{i}")  
        historique_B.append(result_B)  

    forme_recente_A = calculer_forme(historique_A)  
    forme_recente_B = calculer_forme(historique_B)  

    st.write(f"Forme récente de l'équipe A : **{forme_recente_A:.2f}**")  
    st.write(f"Forme récente de l'équipe B : **{forme_recente_B:.2f}**")  

    visualiser_forme_equipes(historique_A, historique_B)  

    # Historique des confrontations  
    st.subheader("🤝 Historique des Confrontations")  
    victoires_A_contre_B = st.number_input("Victoires de A contre B", min_value=0, value=0, step=1)  
    victoires_B_contre_A = st.number_input("Victoires de B contre A", min_value=0, value=0, step=1)  
    nuls_AB = st.number_input("Matchs nuls entre A et B", min_value=0, value=0, step=1)  

    # Historique général des victoires  
    st.subheader("🏆 Historique Général")  
    victoires_A = st.number_input("Nombre total de victoires de A", min_value=0, value=0, step=1)  
    victoires_B = st.number_input("Nombre total de victoires de B", min_value=0, value=0, step=1)
    # --- Critères pour l'équipe A ---  
st.header("🎯 Critères pour l'équipe A")  

# Top Statistiques  
st.subheader("📊 Top Statistiques")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["score_rating_A"] = st.number_input("⭐ Score Rating", min_value=0.0, value=st.session_state.get("score_rating_A", 70.0), key="score_rating_A")  
    st.session_state["buts_totaux_A"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=st.session_state.get("buts_totaux_A", 50.0), key="buts_totaux_A")  
    st.session_state["buts_par_match_A"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=st.session_state.get("buts_par_match_A", 1.5), key="buts_par_match_A")  
with col2:  
    st.session_state["buts_concedes_par_match_A"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=st.session_state.get("buts_concedes_par_match_A", 1.0), key="buts_concedes_par_match_A")  
    st.session_state["buts_concedes_totaux_A"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=st.session_state.get("buts_concedes_totaux_A", 30.0), key="buts_concedes_totaux_A")  
    st.session_state["possession_moyenne_A"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=st.session_state.get("possession_moyenne_A", 55.0), key="possession_moyenne_A")  
st.session_state["aucun_but_encaisse_A"] = st.number_input("🔒 Aucun But Encaissé", min_value=0, value=st.session_state.get("aucun_but_encaisse_A", 10), key="aucun_but_encaisse_A")  

# Attaque  
st.subheader("⚔️ Attaque")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["expected_but_A"] = st.number_input("⭐ Expected But (xG)", min_value=0.0, value=st.session_state.get("expected_but_A", 1.5), key="expected_but_A")  
    st.session_state["tirs_cadres_A"] = st.number_input("🎯 Tirs Cadrés par Match", min_value=0.0, value=st.session_state.get("tirs_cadres_A", 10.0), key="tirs_cadres_A")  
    st.session_state["grandes_chances_A"] = st.number_input("✨ Grandes Chances", min_value=0.0, value=st.session_state.get("grandes_chances_A", 5.0), key="grandes_chances_A")  
with col2:  
    st.session_state["grandes_chances_manquees_A"] = st.number_input("💨 Grandes Chances Manquées", min_value=0.0, value=st.session_state.get("grandes_chances_manquees_A", 2.0), key="grandes_chances_manquees_A")  
    st.session_state["passes_reussies_A"] = st.number_input("✅ Passes Réussies par Match", min_value=0.0, value=st.session_state.get("passes_reussies_A", 300.0), key="passes_reussies_A")  
    st.session_state["passes_longues_A"] = st.number_input("➡️ Passes Longues Précises par Match", min_value=0.0, value=st.session_state.get("passes_longues_A", 15.0), key="passes_longues_A")  
col3, col4 = st.columns(2)  
with col3:  
    st.session_state["centres_reussis_A"] = st.number_input("↗️ Centres Réussis par Match", min_value=0.0, value=st.session_state.get("centres_reussis_A", 5.0), key="centres_reussis_A")  
    st.session_state["penalties_obtenues_A"] = st.number_input("🎁 Pénalties Obtenues", min_value=0.0, value=st.session_state.get("penalties_obtenues_A", 1.0), key="penalties_obtenues_A")  
with col4:  
    st.session_state["balles_surface_A"] = st.number_input("⚽ Balles Touchées dans la Surface Adverse", min_value=0.0, value=st.session_state.get("balles_surface_A", 10.0), key="balles_surface_A")  
    st.session_state["corners_A"] = st.number_input("Corner Nombre de corners", min_value=0.0, value=st.session_state.get("corners_A", 4.0), key="corners_A")  

# Défense  
st.subheader("🛡️ Défense")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["expected_concedes_A"] = st.number_input("⭐ xG Concédés", min_value=0.0, value=st.session_state.get("expected_concedes_A", 1.0), key="expected_concedes_A")  
    st.session_state["interceptions_A"] = st.number_input("✋ Interceptions par Match", min_value=0.0, value=st.session_state.get("interceptions_A", 10.0), key="interceptions_A")  
    st.session_state["tacles_reussis_A"] = st.number_input("Tacles Réussis par Match", min_value=0.0, value=st.session_state.get("tacles_reussis_A", 5.0), key="tacles_reussis_A")  
with col2:  
    st.session_state["degagements_A"] = st.number_input("Dégagements par Match", min_value=0.0, value=st.session_state.get("degagements_A", 8.0), key="degagements_A")  
    st.session_state["penalties_concedes_A"] = st.number_input("🎁 Pénalties Concédés", min_value=0.0, value=st.session_state.get("penalties_concedes_A", 0.0), key="penalties_concedes_A")  
    st.session_state["arrets_A"] = st.number_input("🛑 Arrêts par Match", min_value=0.0, value=st.session_state.get("arrets_A", 3.0), key="arrets_A")
    # --- Critères pour l'équipe B ---  
st.header("🎯 Critères pour l'équipe B")  

# Top Statistiques  
st.subheader("📊 Top Statistiques")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["score_rating_B"] = st.number_input("⭐ Score Rating", min_value=0.0, value=st.session_state.get("score_rating_B", 65.0), key="score_rating_B")  
    st.session_state["buts_totaux_B"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=st.session_state.get("buts_totaux_B", 40.0), key="buts_totaux_B")  
    st.session_state["buts_par_match_B"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=st.session_state.get("buts_par_match_B", 1.0), key="buts_par_match_B")  
with col2:  
    st.session_state["buts_concedes_par_match_B"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=st.session_state.get("buts_concedes_par_match_B", 1.5), key="buts_concedes_par_match_B")  
    st.session_state["buts_concedes_totaux_B"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=st.session_state.get("buts_concedes_totaux_B", 35.0), key="buts_concedes_totaux_B")  
    st.session_state["possession_moyenne_B"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=st.session_state.get("possession_moyenne_B", 45.0), key="possession_moyenne_B")  
st.session_state["aucun_but_encaisse_B"] = st.number_input("🔒 Aucun But Encaissé", min_value=0, value=st.session_state.get("aucun_but_encaisse_B", 8), key="aucun_but_encaisse_B")  

# Attaque  
st.subheader("⚔️ Attaque")  
col1, col2 = st.columns(2)  
with col1:  
       st.session_state["expected_but_B"] = st.number_input("⭐ Expected But (xG)", min_value=0.0, value=st.session_state.get("expected_but_B", 1.0), key="expected_but_B")  
    st.session_state["tirs_cadres_B"] = st.number_input("🎯 Tirs Cadrés par Match", min_value=0.0, value=st.session_state.get("tirs_cadres_B", 8.0), key="tirs_cadres_B")  
    st.session_state["grandes_chances_B"] = st.number_input("✨ Grandes Chances", min_value=0.0, value=st.session_state.get("grandes_chances_B", 3.0), key="grandes_chances_B")  
with col2:  
    st.session_state["grandes_chances_manquees_B"] = st.number_input("💨 Grandes Chances Manquées", min_value=0.0, value=st.session_state.get("grandes_chances_manquees_B", 1.0), key="grandes_chances_manquees_B")  
    st.session_state["passes_reussies_B"] = st.number_input("✅ Passes Réussies par Match", min_value=0.0, value=st.session_state.get("passes_reussies_B", 250.0), key="passes_reussies_B")  
    st.session_state["passes_longues_B"] = st.number_input("➡️ Passes Longues Précises par Match", min_value=0.0, value=st.session_state.get("passes_longues_B", 10.0), key="passes_longues_B")  
col3, col4 = st.columns(2)  
with col3:  
    st.session_state["centres_reussis_B"] = st.number_input("↗️ Centres Réussis par Match", min_value=0.0, value=st.session_state.get("centres_reussis_B", 3.0), key="centres_reussis_B")  
    st.session_state["penalties_obtenues_B"] = st.number_input("🎁 Pénalties Obtenues", min_value=0.0, value=st.session_state.get("penalties_obtenues_B", 0.0), key="penalties_obtenues_B")  
with col4:  
    st.session_state["balles_surface_B"] = st.number_input("⚽ Balles Touchées dans la Surface Adverse", min_value=0.0, value=st.session_state.get("balles_surface_B", 8.0), key="balles_surface_B")  
    st.session_state["corners_B"] = st.number_input("Corner Nombre de corners", min_value=0.0, value=st.session_state.get("corners_B", 3.0), key="corners_B")  

# Défense  
st.subheader("🛡️ Défense")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["expected_concedes_B"] = st.number_input("⭐ xG Concédés", min_value=0.0, value=st.session_state.get("expected_concedes_B", 1.5), key="expected_concedes_B")  
    st.session_state["interceptions_B"] = st.number_input("✋ Interceptions par Match", min_value=0.0, value=st.session_state.get("interceptions_B", 8.0), key="interceptions_B")  
    st.session_state["tacles_reussis_B"] = st.number_input("Tacles Réussis par Match", min_value=0.0, value=st.session_state.get("tacles_reussis_B", 4.0), key="tacles_reussis_B")  
with col2:  
    st.session_state["degagements_B"] = st.number_input("Dégagements par Match", min_value=0.0, value=st.session_state.get("degagements_B", 6.0), key="degagements_B")  
    st.session_state["penalties_concedes_B"] = st.number_input("🎁 Pénalties Concédés", min_value=0.0, value=st.session_state.get("penalties_concedes_B", 1.0), key="penalties_concedes_B")  
    st.session_state["arrets_B"] = st.number_input("🛑 Arrêts par Match", min_value=0.0, value=st.session_state.get("arrets_B", 2.0), key="arrets_B")
    # --- Calcul et affichage des résultats ---  
if st.button("Prédire les Buts"):  
    buts_moyens_A, buts_moyens_B = prediction_buts_poisson(st.session_state["expected_but_A"], st.session_state["expected_but_B"])  
    afficher_resultats_poisson(buts_moyens_A, buts_moyens_B)
    
