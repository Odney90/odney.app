import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
import tools  # Importe le module tools.py  

# Configuration de la page Streamlit  
st.set_page_config(  
    page_title="Prédiction de Match de Football",  
    page_icon="⚽",  
    layout="wide",  
    initial_sidebar_state="expanded",  
)  

# Titre principal de l'application  
st.title("⚽ Prédiction de Match de Football")
# Informations générales sur le match  
st.header("ℹ️ Informations Générales sur le Match")  
st.markdown("Entrez les informations de base sur le match.")  

col1, col2 = st.columns(2)  

with col1:  
    equipe_A = st.text_input("Nom de l'équipe A", value="Équipe A")  
    forme_recente_A = st.slider("Forme récente de l'équipe A (1 à 10)", min_value=1, max_value=10, value=7, key="forme_recente_A")  
    ratio_face_a_face_A = st.slider("Ratio de victoires en face à face (1 à 10)", min_value=1, max_value=10, value=6, key="ratio_face_a_face_A")  

with col2:  
    equipe_B = st.text_input("Nom de l'équipe B", value="Équipe B")  
    forme_recente_B = st.slider("Forme récente de l'équipe B (1 à 10)", min_value=1, max_value=10, value=6, key="forme_recente_B")  
    ratio_face_a_face_B = st.slider("Ratio de victoires en face à face (1 à 10)", min_value=1, max_value=10, value=4, key="ratio_face_a_face_B")
    # Critères détaillés pour les équipes  
st.header("🔍 Critères Détaillés des Équipes")  
st.markdown("Entrez les statistiques détaillées pour chaque équipe.")  

col5, col6 = st.columns(2)  

with col5:  
    st.subheader("Équipe A")  
    buts_produits_A = st.number_input("Buts produits (moyenne/match)", min_value=0.0, value=1.5, step=0.1, key="buts_produits_A")  
    buts_encaisse_A = st.number_input("Buts encaissés (moyenne/match)", min_value=0.0, value=1.0, step=0.1, key="buts_encaisse_A")  
    poss_moyenne_A = st.number_input("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
    xG_A = st.number_input("Buts attendus (xG)", min_value=0.0, value=1.5, step=0.1, key="xG_A")  
    tirs_cadres_A = st.number_input("Tirs cadrés par match", min_value=0, value=10, key="tirs_cadres_A")  
    pourcentage_tirs_convertis_A = st.number_input("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=20, key="pourcentage_tirs_convertis_A")  
    grosses_occasions_A = st.number_input("Grosses occasions", min_value=0, value=5, key="grosses_occasions_A")  
    grosses_occasions_ratees_A = st.number_input("Grosses occasions ratées", min_value=0, value=2, key="grosses_occasions_ratees_A")  
    passes_reussies_A = st.number_input("Passes réussies/match (sur 1000)", min_value=0, value=300, key="passes_reussies_A")  
    passes_longues_precises_A = st.number_input("Passes longues précises/match", min_value=0, value=15, key="passes_longues_precises_A")  
    centres_reussis_A = st.number_input("Centres réussis/match", min_value=0, value=5, key="centres_reussis_A")  
    penalties_obtenus_A = st.number_input("Pénalties obtenus", min_value=0, value=1, key="penalties_obtenus_A")  
    touches_surface_adverse_A = st.number_input("Touches dans la surface adverse", min_value=0, value=10, key="touches_surface_adverse_A")  
    corners_A = st.number_input("Nombre de corners", min_value=0, value=4, key="corners_A")  
    corners_par_match_A = st.number_input("Corners par match", min_value=0, value=2, key="corners_par_match_A")  
    corners_concedes_A = st.number_input("Corners concédés par match", min_value=0, value=3, key="corners_concedes_A")  
    xG_concedes_A = st.number_input("xG concédés", min_value=0.0, value=1.0, step=0.1, key="xG_concedes_A")  
    interceptions_A = st.number_input("Interceptions par match", min_value=0, value=10, key="interceptions_A")  
    tacles_reussis_A = st.number_input("Tacles réussis par match", min_value=0, value=5, key="tacles_reussis_A")  
    degagements_A = st.number_input("Dégagements par match", min_value=0, value=8, key="degagements_A")  
    possessions_recuperees_A = st.number_input("Possessions récupérées au milieu/match", min_value=0, value=5, key="possessions_recuperees_A")  
    penalties_concedes_A = st.number_input("Pénalties concédés", min_value=0, value=0, key="penalties_concedes_A")  
    arrets_A = st.number_input("Arrêts par match", min_value=0, value=3, key="arrets_A")  
    fautes_A = st.number_input("Fautes par match", min_value=0, value=10, key="fautes_A")  
    cartons_jaunes_A = st.number_input("Cartons jaunes", min_value=0, value=2, key="cartons_jaunes_A")  
    cartons_rouges_A = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_A")  
    tactique_A = st.text_input("Tactique de l'équipe A", value="4-3-3", key="tactique_A")  

    st.subheader("Joueurs Clés de l'équipe A")  
    absents_A = st.number_input("Nb joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_A")  
    ratings_A = []  
    for i in range(5):  
        rating = st.number_input(f"Rating joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_A_{i}")  
        ratings_A.append(rating)  

with col6:  
    st.subheader("Équipe B")  
    buts_produits_B = st.number_input("Buts produits (moyenne/match)", min_value=0.0, value=1.0, step=0.1, key="buts_produits_B")  
    buts_encaisse_B = st.number_input("Buts encaissés (moyenne/match)", min_value=0.0, value=1.5, step=0.1, key="buts_encaisse_B")  
    poss_moyenne_B = st.number_input("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
    xG_B = st.number_input("Buts attendus (xG)", min_value=0.0, value=1.0, step=0.1, key="xG_B")  
    tirs_cadres_B = st.number_input("Tirs cadrés par match", min_value=0, value=8, key="tirs_cadres_B")  
    pourcentage_tirs_convertis_B = st.number_input("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=15, key="pourcentage_tirs_convertis_B")  
    grosses_occasions_B = st.number_input("Grosses occasions", min_value=0, value=3, key="grosses_occasions_B")  
    grosses_occasions_ratees_B = st.number_input("Grosses occasions ratées", min_value=0, value=1, key="grosses_occasions_ratees_B")  
    passes_reussies_B = st.number_input("Passes réussies/match (sur 1000)", min_value=0, value=250, key="passes_reussies_B")  
    passes_longues_precises_B = st.number_input("Passes longues précises/match", min_value=0, value=10, key="passes_longues_precises_B")  
    centres_reussis_B = st.number_input("Centres réussis/match", min_value=0, value=3, key="centres_reussis_B")  
    penalties_obtenus_B = st.number_input("Pénalties obtenus", min_value=0, value=0, key="penalties_obtenus_B")  
    touches_surface_adverse_B = st.number_input("Touches dans la surface adverse", min_value=0, value=8, key="touches_surface_adverse_B")  
    corners_B = st.number_input("Nombre de corners", min_value=0, value=3, key="corners_B")  
    corners_par_match_B = st.number_input("Corners par match", min_value=0, value=1, key="corners_par_match_B")  
    corners_concedes_B = st.number_input("Corners concédés par match", min_value=0, value=2, key="corners_concedes_B")  
    xG_concedes_B = st.number_input("xG concédés", min_value=0.0, value=1.5, step=0.1, key="xG_concedes_B")  
    interceptions_B = st.number_input("Interceptions par match", min_value=0, value=8, key="interceptions_B")  
    tacles_reussis_B = st.number_input("Tacles réussis par match", min_value=0, value=4, key="tacles_reussis_B")  
    degagements_B = st.number_input("Dégagements par match", min_value=0, value=6, key="degagements_B")  
    possessions_recuperees_B = st.number_input("Possessions récupérées au milieu/match", min_value=0, value=4, key="possessions_recuperees_B")  
    penalties_concedes_B = st.number_input("Pénalties concédés", min_value=0, value=1, key="penalties_concedes_B")  
    arrets_B = st.number_input("Arrêts par match", min_value=0, value=2, key="arrets_B")  
    fautes_B = st.number_input("Fautes par match", min_value=0, value=12, key="fautes_B")  
    cartons_jaunes_B = st.number_input("Cartons jaunes", min_value=0, value=1, key="cartons_jaunes_B")  
    cartons_rouges_B = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_B")  
    tactique_B = st.text_input("Tactique de l'équipe B", value="4-2-3-1", key="tactique_B")  

    st.subheader("Joueurs Clés de l'équipe B")  
    absents_B = st.number_input("Nb joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_B")  
    ratings_B = []  
    for i in range(5):  
        rating = st.number_input(f"Rating joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_B_{i}")  
        ratings_B.append(rating)
        # Bouton de prédiction  
if st.button("🔮 Prédire le Match"):  
    # Calcul des forces relatives des équipes (exemple simplifié)  
    force_A = (  
        forme_recente_A * 0.3 +  
        ratio_face_a_face_A * 0.2 +  
        buts_produits_A * 0.2 +  
        (10 - buts_encaisse_A) * 0.3  
    )  
    force_B = (  
        forme_recente_B * 0.3 +  
        ratio_face_a_face_B * 0.2 +  
        buts_produits_B * 0.2 +  
        (10 - buts_encaisse_B) * 0.3  
    )  

    # Prédiction basée sur les forces relatives  
    if force_A > force_B:  
        st.success(f"🎉 Victoire probable de {equipe_A} !")  
    elif force_B > force_A:  
        st.success(f"🎉 Victoire probable de {equipe_B} !")  
    else:  
        st.info("🤝 Match nul probable.")  

    # Prédiction du score (exemple simplifié avec Poisson)  
    st.subheader("🎯 Prédiction du Score")  
    st.markdown("Probabilités des scores exacts (exemple simplifié).")  

    # Paramètres pour la loi de Poisson (à ajuster)  
    lambda_A = buts_produits_A  # Nombre moyen de buts marqués par A  
    lambda_B = buts_produits_B  # Nombre moyen de buts marqués par B  

    # Calcul des probabilités pour quelques scores possibles  
    scores_possibles = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]  
    probabilites = []  
    for score_A, score_B in scores_possibles:  
        proba = poisson.pmf(score_A, lambda_A) * poisson.pmf(score_B, lambda_B)  
        probabilites.append(proba)  

    # Affichage des résultats  
    df_scores = pd.DataFrame({  
        "Score": [f"{score_A}-{score_B}" for score_A, score_B in scores_possibles],  
        "Probabilité": [f"{proba:.2%}" for proba in probabilites]  
    })  
    st.dataframe(df_scores, width=500)
    # Barre de navigation  
page = st.sidebar.radio("Navigation", ["Prédiction de Match", "Outils de Paris"])  

# Affichage de la page sélectionnée  
if page == "Prédiction de Match":  
    # Le code de prédiction de match (sections précédentes) est affiché ici  
    pass  # Le code des sections précédentes sera exécuté ici  
elif page == "Outils de Paris":  
    tools.tools_page()  # Affiche la page "Outils de Paris"
    
