import streamlit as st  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  

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

# Titre principal avec emoji  
st.title("⚽ Prédiction de Match de Football")  

# --- Initialisation de session_state ---  
def init_session_state():  
    default_values_A = {  
        "score_rating_A": 70.0, "buts_totaux_A": 50.0, "buts_par_match_A": 1.5,  
        "buts_concedes_par_match_A": 1.0, "buts_concedes_totaux_A": 30.0, "possession_moyenne_A": 55.0,  
        "aucun_but_encaisse_A": 10, "expected_but_A": 1.5, "tirs_cadres_A": 10.0,  
        "grandes_chances_A": 5.0, "grandes_chances_manquees_A": 2.0, "passes_reussies_A": 300.0,  
        "passes_longues_A": 15.0, "centres_reussis_A": 5.0, "penalties_obtenues_A": 1.0,  
        "balles_surface_A": 10.0, "corners_A": 4.0, "expected_concedes_A": 1.0,  
        "interceptions_A": 10.0, "tacles_reussis_A": 5.0, "degagements_A": 8.0,  
        "possession_remportes_A": 5.0, "penalties_concedes_A": 0.0, "arrets_A": 3.0,  
        "fautes_A": 10.0, "cartons_jaunes_A": 2.0, "cartons_rouges_A": 0.0,  
        "motivation_A": 5.0  
    }  

    default_values_B = {  
        "score_rating_B": 65.0, "buts_totaux_B": 40.0, "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5, "buts_concedes_totaux_B": 35.0, "possession_moyenne_B": 45.0,  
        "aucun_but_encaisse_B": 8, "expected_but_B": 1.0, "tirs_cadres_B": 8.0,  
        "grandes_chances_B": 3.0, "grandes_chances_manquees_B": 1.0, "passes_reussies_B": 250.0,  
        "passes_longues_B": 10.0, "centres_reussis_B": 3.0, "penalties_obtenues_B": 0.0,  
        "balles_surface_B": 8.0, "corners_B": 3.0, "expected_concedes_B": 1.5,  
        "interceptions_B": 8.0, "tacles_reussis_B": 4.0, "degagements_B": 6.0,  
        "possession_remportes_B": 4.0, "penalties_concedes_B": 1.0, "arrets_B": 2.0,  
        "fautes_B": 12.0, "cartons_jaunes_B": 1.0, "cartons_rouges_B": 0.0,  
         "motivation_B": 5.0  
    }  

    for key, value in default_values_A.items():  
        if key not in st.session_state:  
            st.session_state[key] = value  

    for key, value in default_values_B.items():  
        if key not in st.session_state:  
            st.session_state[key] = value  

if not st.session_state:  
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

# --- Historique des équipes et confrontations directes ---  
st.header("📊 Historique des équipes et confrontations directes")  
st.markdown("Entrez le nombre de victoires, nuls et défaites pour chaque équipe :")  

col1, col2 = st.columns(2)  

with col1:  
    st.subheader("Équipe A")  
    victoires_A = st.number_input("🏆 Victoires", min_value=0, value=10, key="victoires_A")  
    nuls_A = st.number_input("🤝 Nuls", min_value=0, value=5, key="nuls_A")  
    defaites_A = st.number_input("❌ Défaites", min_value=0, value=5, key="defaites_A")  

with col2:  
    st.subheader("Équipe B")  
    victoires_B = st.number_input("🏆 Victoires", min_value=0, value=8, key="victoires_B")  
    nuls_B = st.number_input("🤝 Nuls", min_value=0, value=7, key="nuls_B")  
    defaites_B = st.number_input("❌ Défaites", min_value=0, value=5, key="defaites_B")  

# Confrontations directes  
st.subheader("⚔️ Confrontations directes")  
victoires_A_contre_B = st.number_input("Victoires de A contre B", min_value=0, value=3, key="victoires_A_contre_B")  
victoires_B_contre_A = st.number_input("Victoires de B contre A", min_value=0, value=2, key="victoires_B_contre_A")  
nuls_AB = st.number_input("Nuls entre A et B", min_value=0, value=1, key="nuls_AB")  

# --- Historique des 5 derniers matchs ---  
st.header("📅 Historique des 5 derniers matchs")  
st.markdown("Sélectionnez le résultat des 5 derniers matchs pour chaque équipe :")  

col1, col2 = st.columns(2)  

with col1:  
    st.subheader("Équipe A")  
    historique_A = []  
    for i in range(5):  
        resultat_A = st.selectbox(  
            f"Match {i + 1}",  
            options=["Victoire 🟢", "Match Nul 🟡", "Défaite 🔴"],  
            key=f"match_A_{i}"  
        )  
        if resultat_A == "Victoire 🟢":  
            historique_A.append(1)  
        elif resultat_A == "Match Nul 🟡":  
            historique_A.append(0)  
        else:  
            historique_A.append(-1)  

with col2:  
    st.subheader("Équipe B")  
    historique_B = []  
    for i in range(5):  
        resultat_B = st.selectbox(  
            f"Match {i + 1}",  
            options=["Victoire 🟢", "Match Nul 🟡", "Défaite 🔴"],  
            key=f"match_B_{i}"  
        )  
        if resultat_B == "Victoire 🟢":  
            historique_B.append(1)  
        elif resultat_B == "Match Nul 🟡":  
            historique_B.append(0)  
        else:  
            historique_B.append(-1)  

# Calcul de la forme récente sur 5 matchs  
forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  

st.write(f"Forme récente de l'équipe A (moyenne sur 5 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 5 matchs) : **{forme_recente_B:.2f}**")  

# Visualisation de la forme des équipes  
st.header("📈 Visualisation de la forme des équipes")  
visualiser_forme_equipes(historique_A, historique_B)  

# --- Critères pour l'équipe A ---  
st.header("🎯 Critères pour l'équipe A")  

# Top Statistiques  
st.subheader("📊 Top Statistiques")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["score_rating_A"] = st.number_input("⭐ Score Rating", min_value=0.0, value=st.session_state["score_rating_A"], key="score_rating_A")  
    st.session_state["buts_totaux_A"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=st.session_state["buts_totaux_A"], key="buts_totaux_A")  
    st.session_state["buts_par_match_A"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=st.session_state["buts_par_match_A"], key="buts_par_match_A")  
with col2:  
    st.session_state["buts_concedes_par_match_A"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=st.session_state["buts_concedes_par_match_A"], key="buts_concedes_par_match_A")  
    st.session_state["buts_concedes_totaux_A"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=st.session_state["buts_concedes_totaux_A"], key="buts_concedes_totaux_A")  
    st.session_state["possession_moyenne_A"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=st.session_state["possession_moyenne_A"], key="possession_moyenne_A")  
st.session_state["aucun_but_encaisse_A"] = st.number_input("🔒 Aucun But Encaissé", min_value=0, value=st.session_state["aucun_but_encaisse_A"], key="aucun_but_encaisse_A")  

# Attaque  
st.subheader("⚔️ Attaque")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["expected_but_A"] = st.number_input("⭐ Expected But (xG)", min_value=0.0, value=st.session_state["expected_but_A"], key="expected_but_A")  
    st.session_state["tirs_cadres_A"] = st.number_input("🎯 Tirs Cadrés par Match", min_value=0.0, value=st.session_state["tirs_cadres_A"], key="tirs_cadres_A")  
    st.session_state["grandes_chances_A"] = st.number_input("✨ Grandes Chances", min_value=0.0, value=st.session_state["grandes_chances_A"], key="grandes_chances_A")  
with col2:  
    st.session_state["grandes_chances_manquees_A"] = st.number_input("💨 Grandes Chances Manquées", min_value=0.0, value=st.session_state["grandes_chances_manquees_A"], key="grandes_chances_manquees_A")  
    st.session_state["passes_reussies_A"] = st.number_input("✅ Passes Réussies par Match", min_value=0.0, value=st.session_state["passes_reussies_A"], key="passes_reussies_A")  
    st.session_state["passes_longues_A"] = st.number_input("➡️ Passes Longues Précises par Match", min_value=0.0, value=st.session_state["passes_longues_A"], key="passes_longues_A")  
col3, col4 = st.columns(2)  
with col3:  
    st.session_state["centres_reussis_A"] = st.number_input("↗️ Centres Réussis par Match", min_value=0.0, value=st.session_state["centres_reussis_A"], key="centres_reussis_A")  
    st.session_state["penalties_obtenues_A"] = st.number_input("🎁 Pénalties Obtenues", min_value=0.0, value=st.session_state["penalties_obtenues_A"], key="penalties_obtenues_A")  
with col4:  
    st.session_state["balles_surface_A"] = st.number_input("⚽ Balles Touchées dans la Surface Adverse", min_value=0.0, value=st.session_state["balles_surface_A"], key="balles_surface_A")
        st.session_state["corners_A"] = st.number_input("Corner Nombre de corners", min_value=0.0, value=st.session_state["corners_A"], key="corners_A")  

# Défense  
st.subheader("🛡️ Défense")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["expected_concedes_A"] = st.number_input("⭐ xG Concédés", min_value=0.0, value=st.session_state["expected_concedes_A"], key="expected_concedes_A")  
    st.session_state["interceptions_A"] = st.number_input("✋ Interceptions par Match", min_value=0.0, value=st.session_state["interceptions_A"], key="interceptions_A")  
    st.session_state["tacles_reussis_A"] = st.number_input("Tacles Réussis par Match", min_value=0.0, value=st.session_state["tacles_reussis_A"], key="tacles_reussis_A")  
with col2:  
    st.session_state["degagements_A"] = st.number_input("Dégagements par Match", min_value=0.0, value=st.session_state["degagements_A"], key="degagements_A")  
    st.session_state["penalties_concedes_A"] = st.number_input("🎁 Pénalties Concédés", min_value=0.0, value=st.session_state["penalties_concedes_A"], key="penalties_concedes_A")  
    st.session_state["possession_remportes_A"] = st.number_input("Milieu Possessions Remportées au Dernier Tiers", min_value=0.0, value=st.session_state["possession_remportes_A"], key="possession_remportes_A")  
st.session_state["arrets_A"] = st.number_input("🧤 Arrêts par Match", min_value=0.0, value=st.session_state["arrets_A"], key="arrets_A")  

# Discipline  
st.subheader("🟨🟥 Discipline")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["fautes_A"] = st.number_input("Fautes par Match", min_value=0.0, value=st.session_state["fautes_A"], key="fautes_A")  
    st.session_state["cartons_jaunes_A"] = st.number_input("🟨 Cartons Jaunes", min_value=0.0, value=st.session_state["cartons_jaunes_A"], key="cartons_jaunes_A")  
with col2:  
    st.session_state["cartons_rouges_A"] = st.number_input("🟥 Cartons Rouges", min_value=0.0, value=st.session_state["cartons_rouges_A"], key="cartons_rouges_A")  

# Section pour les joueurs clés de l'équipe A  
st.header("⭐ Joueurs Clés de l'équipe A")  
absents_A = st.number_input("Nombre de joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_A")  
ratings_A = []  
for i in range(5):  
    rating = st.number_input(f"Rating du joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_A_{i}", step=0.1)  
    ratings_A.append(rating)  

# --- Critères pour l'équipe B ---  
st.header("🎯 Critères pour l'équipe B")  

# Top Statistiques  
st.subheader("📊 Top Statistiques")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["score_rating_B"] = st.number_input("⭐ Score Rating", min_value=0.0, value=st.session_state["score_rating_B"], key="score_rating_B")  
    st.session_state["buts_totaux_B"] = st.number_input("⚽ Buts Totaux", min_value=0.0, value=st.session_state["buts_totaux_B"], key="buts_totaux_B")  
    st.session_state["buts_par_match_B"] = st.number_input("🥅 Buts par Match", min_value=0.0, value=st.session_state["buts_par_match_B"], key="buts_par_match_B")  
with col2:  
    st.session_state["buts_concedes_par_match_B"] = st.number_input("🚫 Buts Concédés par Match", min_value=0.0, value=st.session_state["buts_concedes_par_match_B"], key="buts_concedes_par_match_B")  
    st.session_state["buts_concedes_totaux_B"] = st.number_input("🤕 Buts Concédés Totaux", min_value=0.0, value=st.session_state["buts_concedes_totaux_B"], key="buts_concedes_totaux_B")  
    st.session_state["possession_moyenne_B"] = st.number_input("Ballon Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=st.session_state["possession_moyenne_B"], key="possession_moyenne_B")  
st.session_state["aucun_but_encaisse_B"] = st.number_input("🔒 Aucun But Encaissé", min_value=0, value=st.session_state["aucun_but_encaisse_B"], key="aucun_but_encaisse_B")  

# Attaque  
st.subheader("⚔️ Attaque")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["expected_but_B"] = st.number_input("⭐ Expected But (xG)", min_value=0.0, value=st.session_state["expected_but_B"], key="expected_but_B")  
    st.session_state["tirs_cadres_B"] = st.number_input("🎯 Tirs Cadrés par Match", min_value=0.0, value=st.session_state["tirs_cadres_B"], key="tirs_cadres_B")  
    st.session_state["grandes_chances_B"] = st.number_input("✨ Grandes Chances", min_value=0.0, value=st.session_state["grandes_chances_B"], key="grandes_chances_B")  
with col2:  
    st.session_state["grandes_chances_manquees_B"] = st.number_input("💨 Grandes Chances Manquées", min_value=0.0, value=st.session_state["grandes_chances_manquees_B"], key="grandes_chances_manquees_B")  
    st.session_state["passes_reussies_B"] = st.number_input("✅ Passes Réussies par Match", min_value=0.0, value=st.session_state["passes_reussies_B"], key="passes_reussies_B")  
    st.session_state["passes_longues_B"] = st.number_input("➡️ Passes Longues Précises par Match", min_value=0.0, value=st.session_state["passes_longues_B"], key="passes_longues_B")  
col3, col4 = st.columns(2)  
with col3:  
    st.session_state["centres_reussis_B"] = st.number_input("↗️ Centres Réussis par Match", min_value=0.0, value=st.session_state["centres_reussis_B"], key="centres_reussis_B")  
    st.session_state["penalties_obtenues_B"] = st.number_input("🎁 Pénalties Obtenues", min_value=0.0, value=st.session_state["penalties_obtenues_B"], key="penalties_obtenues_B")  
with col4:  
    st.session_state["balles_surface_B"] = st.number_input("⚽ Balles Touchées dans la Surface Adverse", min_value=0.0, value=st.session_state["balles_surface_B"], key="balles_surface_B")  
    st.session_state["corners_B"] = st.number_input("Corner Nombre de corners", min_value=0.0, value=st.session_state["corners_B"], key="corners_B")  

# Défense  
st.subheader("🛡️ Défense")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["expected_concedes_B"] = st.number_input("⭐ xG Concédés", min_value=0.0, value=st.session_state["expected_concedes_B"], key="expected_concedes_B")  
    st.session_state["interceptions_B"] = st.number_input("✋ Interceptions par Match", min_value=0.0, value=st.session_state["interceptions_B"], key="interceptions_B")  
    st.session_state["tacles_reussis_B"] = st.number_input("Tacles Réussis par Match", min_value=0.0, value=st.session_state["tacles_reussis_B"], key="tacles_reussis_B")  
with col2:  
    st.session_state["degagements_B"] = st.number_input("Dégagements par Match", min_value=0.0, value=st.session_state["degagements_B"], key="degagements_B")  
    st.session_state["penalties_concedes_B"] = st.number_input("🎁 Pénalties Concédés", min_value=0.0, value=st.session_state["penalties_concedes_B"], key="penalties_concedes_B")  
    st.session_state["possession_remportes_B"] = st.number_input("Milieu Possessions Remportées au Dernier Tiers", min_value=0.0, value=st.session_state["possession_remportes_B"], key="possession_remportes_B")  
st.session_state["arrets_B"] = st.number_input("🧤 Arrêts par Match", min_value=0.0, value=st.session_state["arrets_B"], key="arrets_B")  

# Discipline  
st.subheader("🟨🟥 Discipline")  
col1, col2 = st.columns(2)  
with col1:  
    st.session_state["fautes_B"] = st.number_input("Fautes par Match", min_value=0.0, value=st.session_state["fautes_B"], key="fautes_B")  
    st.session_state["cartons_jaunes_B"] = st.number_input("🟨 Cartons Jaunes", min_value=0.0, value=st.session_state["cartons_jaunes_B"], key="cartons_jaunes_B")  
with col2:  
    st.session_state["cartons_rouges_B"] = st.number_input("🟥 Cartons Rouges", min_value=0.0, value=st.session_state["cartons_rouges_B"], key="cartons_rouges_B")  

# --- Section pour les joueurs clés de l'équipe B ---  
st.header("⭐ Joueurs Clés de l'équipe B")  
absents_B = st.number_input("Nombre de joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_B")  
ratings_B = []  
for i in range(5):  
    rating = st.number_input(f"Rating du joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_B_{i}", step=0.1)  
    ratings_B.append(rating)  

# --- Motivation des Équipes ---  
st.header("🔥 Motivation des Équipes")  
col1, col2 = st.columns(2)  

with col1:  
    st.session_state["motivation_A"] = st.number_input("Motivation de l'équipe A (1 à 10)", min_value=1.0, max_value=10.0, value=st.session_state["motivation_A"], key="motivation_A", step=0.1)  
    st.write(f"Niveau de motivation de l'équipe A : **{st.session_state['motivation_A']}**")  

with col2:  
    st.session_state["motivation_B"] = st.number_input("Motivation de l'équipe B (1 à 10)", min_value=1.0, max_value=10.0, value=st.session_state["motivation_B"], key="motivation_B", step=0.1)  
    st.write(f"Niveau de motivation de l'équipe B : **{st.session_state['motivation_B']}**")  

# --- Prédiction des buts avec la méthode de Poisson ---  
if 'expected_but_A' in st.session_state and 'expected_but_B' in st.session_state:  
    buts_moyens_A, buts_moyens_B = prediction_buts_poisson(st.session_state["expected_but_A"], st.session_state["expected_but_B"])  
    afficher_resultats_poisson(buts_moyens_A, buts_moyens_B)  
else:  
    st.warning("Veuillez remplir les critères des équipes A et B pour activer la prédiction de Poisson.")  

# --- Prédiction multivariable avec régression logistique (version améliorée) ---  
st.header("🔮 Prédiction Multivariable (Régression Logistique)")  

# Poids pour chaque critère (ajuster selon l'importance)  
poids = {  
    'score_rating': 0.20,  
    'buts_totaux': 0.15,  
    'buts_concedes': -0.10,  
    'possession': 0.08,  
    'aucun_but_encaisse': 0.05,  
    'expected_but': 0.10,  
    'tirs_cadres': 0.07,  
    'grandes_chances': 0.06,  
    'passes_reussies': 0.04,  
    'passes_longues': 0.03,  
    'centres_reussis': 0.02,  
    'penalties_obtenues': 0.01,  
    'balles_surface': 0.05,  
    'corners': 0.03,  
    'expected_concedes': -0.08,  
    'interceptions': 0.06,  
    'tacles_reussis': 0.05,  
    'degagements': -0.04,  
    'penalties_concedes': -0.03,  
    'possession_remportes': 0.04,  
    'arrets': 0.03,  
    'fautes': -0.02,  
    'cartons_jaunes': -0.03,  
    'cartons_rouges': -0.05,  
    'motivation': 0.12,  
    'absents': -0.15,  
    'rating_joueurs': 0.20,  
    'forme_recente': 0.20,  
    'historique_confrontations': 0.10,  # Nouveau critère  
    'victoires': 0.05,  # Poids pour l'historique général des victoires  
    'confrontations_directes': 0.10  # Poids pour les confrontations directes  
}  

# Historique des confrontations directes (1 pour victoire de A, -1 pour victoire de B, 0 pour nul)  
historique_confrontations = (victoires_A_contre_B - victoires_B_contre_A) / (victoires_A_contre_B + victoires_B_contre_A + nuls_AB) if (victoires_A_contre_B + victoires_B_contre_A + nuls_AB) > 0 else 0  

# Préparation des données pour le modèle  
X = np.array([  
    [st.session_state["score_rating_A"] * poids['score_rating'],  
     st.session_state["buts_totaux_A"] * poids['buts_totaux'],  
     st.session_state["buts_concedes_par_match_A"] * poids['buts_concedes'],  
     st.session_state["possession_moyenne_A"] * poids['possession'],  
     st.session_state["aucun_but_encaisse_A"] * poids['aucun_but_encaisse'],  
     st.session_state["expected_but_A"] * poids['expected_but'],  
     st.session_state["tirs_cadres_A"] * poids['tirs_cadres'],  
     st.session_state["grandes_chances_A"] * poids['grandes_chances'],  
     st.session_state["passes_reussies_A"] * poids['passes_reussies'],  
     st.session_state["passes_longues_A"] * poids['passes_longues'],  
     st.session_state["centres_reussis_A"] * poids['centres_reussis'],  
     st.session_state["penalties_obtenues_A"] * poids['penalties_obtenues'],  
     st.session_state["balles_surface_A"] * poids['balles_surface'],  
     st.session_state["corners_A"] * poids['corners'],  
     st.session_state["expected_concedes_A"] * poids['expected_concedes'],  
     st.session_state["interceptions_A"] * poids['interceptions'],  
     st.session_state["tacles_reussis_A"] * poids['tacles_reussis'],  
     st.session_state["degagements_A"] * poids['degagements'],  
     st.session_state["penalties_concedes_A"] * poids['penalties_concedes'],  
     st.session_state["possession_remportes_A"] * poids['possession_remportes'],  
     st.session_state["arrets_A"] * poids['arrets'],  
     st.session_state["fautes_A"] * poids['fautes'],  
     st.session_state["cartons_jaunes_A"] * poids['cartons_jaunes'],  
     st.session_state["cartons_rouges_A"] * poids['cartons_rouges'],  
     st.session_state["motivation_A"] * poids['motivation'],  
     absents_A * poids['absents'],  
     np.mean(ratings_A) * poids['rating_joueurs'],  
     forme_recente_A * poids['forme_recente'],  
     historique_confrontations * poids['historique_confrontations'],  
     victoires_A * poids['victoires'],  # Ajout de l'historique des victoires  
     victoires_A_contre_B * poids['confrontations_directes']],  # Ajout des confrontations directes  

    [st.session_state["score_rating_B"] * poids['score_rating'],  
     st.session_state["buts_totaux_B"] * poids['buts_totaux'],  
     st.session_state["buts_concedes_par_match_B"] * poids['buts_concedes'],  
     st.session_state["possession_moyenne_B"] * poids['possession'],  
     st.session_state["aucun_but_encaisse_B"] * poids['aucun_but_encaisse'],  
     st.session_state["expected_but_B"] * poids['expected_but'],  
     st.session_state["tirs_cadres_B"] * poids['tirs_cadres'],  
     st.session_state["grandes_chances_B"] * poids['grandes_chances'],  
     st.session_state["passes_reussies_B"] * poids['passes_reussies'],  
     st.session_state["passes_longues_B"] * poids['passes_longues'],  
     st.session_state["centres_reussis_B"] * poids['centres_reussis'],  
     st.session_state["penalties_obtenues_B"] * poids['penalties_obtenues'],  
     st.session_state["balles_surface_B"] * poids['balles_surface'],  
     st.session_state["corners_B"] * poids['corners'],  
     st.session_state["expected_concedes_B"] * poids['expected_concedes'],  
     st.session_state["interceptions_B"] * poids['interceptions'],  
     st.session_state["tacles_reussis_B"] * poids['tacles_reussis'],  
     st.session_state["degagements_B"] * poids['degagements'],  
     st.session_state["penalties_concedes_B"] * poids['penalties_concedes'],  
     st.session_state["possession_remportes_B"] * poids['possession_remportes'],  
     st.session_state["arrets_B"] * poids['arrets'],  
     st.session_state["fautes_B"] * poids['fautes'],  
     st.session_state["cartons_jaunes_B"] * poids['cartons_jaunes'],  
     st.session_state["cartons_rouges_B"] * poids['cartons_rouges'],  
     st.session_state["motivation_B"] * poids['motivation'],  
     absents_B * poids['absents'],  
     np.mean(ratings_B) * poids['rating_joueurs'],  
     forme_recente_B * poids['forme_recente'],  
     historique_confrontations * poids['historique_confrontations'],  
     victoires_B * poids['victoires'],  # Ajout de l'historique des victoires  
     victoires_B_contre_A * poids['confrontations_directes']]  # Ajout des confrontations directes  
])  

y = np.array([1, 0])  # 1 pour l'équipe A, 0 pour l'équipe B  

# Normalisation des données (important pour la régression logistique)  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Entraînement du modèle avec régularisation pour éviter le surapprentissage  
model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')  # L2 régularisation  
model.fit(X_scaled, y)  

# Prédiction des résultats  
prediction = model.predict_proba(X_scaled)  

# Affichage des probabilités de victoire  
st.write(f"Probabilité de victoire pour l'équipe A : **{prediction[0][1]:.2%}**")  
st.write(f"Probabilité de victoire pour l'équipe B : **{prediction[1][1]:.2%}**")  

# --- Interprétation des coefficients du modèle ---  
st.header("🧐 Interprétation du Modèle")  
feature_names = ['Score Rating', 'Buts Totaux', 'Buts Concédés', 'Possession Moyenne',  
                 'Aucun But Encaissé', 'Expected But (xG)', 'Tirs Cadrés par Match',  
                 'Grandes Chances', 'Passes Réussies par Match', 'Passes Longues Précises par Match',  
                 'Centres Réussis par Match', 'Pénalties Obtenues', 'Balles Touchées dans la Surface Adverse',  
                 'Corners', 'xG Concédés', 'Interceptions par Match', 'Tacles Réussis par Match',  
                 'Dégagements par Match', 'Pénalties Concédés', 'Possessions Remportées au Dernier Tiers',  
                 'Arrêts par Match', 'Fautes par Match', 'Cartons Jaunes', 'Cartons Rouges',  
                 'Motivation', 'Absents', 'Rating Joueurs', 'Forme Récente',  
                 'Historique Confrontations', 'Historique Victoires', 'Confrontations Directes']  
coefficients = model.coef_[0]  

for feature, coef in zip(feature_names, coefficients):  
    st.write(f"{feature}: Coefficient = {coef:.2f}")  

# --- Calcul des paris alternatifs ---  
double_chance_A = prediction[0][1] + (1 - prediction[1][1])  # Équipe A gagne ou match nul  
double_chance_B = prediction[1][1] + (1 - prediction[0][1])  # Équipe B gagne ou match nul  

# --- Affichage des paris alternatifs ---  
st.header("💰 Paris Alternatifs")  
st.write(f"Double chance pour l'équipe A (gagner ou match nul) : **{double_chance_A:.2%}**")  
st.write(f"Double chance pour l'équipe B (gagner ou match nul) : **{double_chance_B:.2%}**")
