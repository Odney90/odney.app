import streamlit as st  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import StratifiedKFold, cross_val_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from scipy.stats import poisson  
import traceback  

# Configuration de la page  
st.set_page_config(page_title="⚽ Analyse de Match de Football", page_icon="⚽", layout="wide")  

# Initialisation de st.session_state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Titre de l'application  
st.title("⚽ Analyse de Match de Football")  

# Fonction pour mettre en couleur les données utilisées par Poisson et Régression Logistique  
def highlight_data(data, key):  
    if key in ['expected_but_A', 'buts_par_match_A', 'tirs_cadres_A', 'grandes_chances_A', 'corners_A',  
               'expected_but_B', 'buts_par_match_B', 'tirs_cadres_B', 'grandes_chances_B', 'corners_B']:  
        return f"<span style='color: #1f77b4; font-size: 16px;'>{data}</span>"  
    elif key in ['score_rating_A', 'possession_moyenne_A', 'motivation_A', 'victoires_domicile_A',  
                 'score_rating_B', 'possession_moyenne_B', 'motivation_B', 'victoires_exterieur_B']:  
        return f"<span style='color: #ff7f0e; font-size: 16px;'>{data}</span>"  
    else:  
        return f"<span style='font-size: 16px;'>{data}</span>"  

# Formulaire de collecte des données  
with st.form("data_form"):  
    st.markdown("### Équipe A")  
    col1, col2 = st.columns(2)  
    with col1:  
        st.session_state.data['score_rating_A'] = st.number_input("⭐ Score Rating", value=70.6, format="%.2f", key="rating_A")  
        st.session_state.data['buts_par_match_A'] = st.number_input("⚽ Buts Marqués", value=1.5, format="%.2f", key="buts_A")  
        st.session_state.data['buts_concedes_par_match_A'] = st.number_input("🥅 Buts Concédés", value=1.0, format="%.2f", key="concedes_A")  
        st.session_state.data['possession_moyenne_A'] = st.number_input("🎯 Possession Moyenne", value=55.0, format="%.2f", key="possession_A")  
        st.session_state.data['expected_but_A'] = st.number_input("📊 Expected Goals (xG)", value=1.5, format="%.2f", key="xG_A")  
        st.session_state.data['expected_concedes_A'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.2, format="%.2f", key="xGA_A")  
        st.session_state.data['tirs_cadres_A'] = st.number_input("🎯 Tirs Cadrés", value=120.0, format="%.2f", key="tirs_A")  
        st.session_state.data['tirs_non_cadres_A'] = st.number_input("🎯 Tirs Non Cadrés", value=80.0, format="%.2f", key="tirs_non_cadres_A")  
        st.session_state.data['grandes_chances_A'] = st.number_input("🔥 Grandes Chances", value=25.0, format="%.2f", key="chances_A")  
        st.session_state.data['passes_reussies_A'] = st.number_input("🔄 Passes Réussies", value=400.0, format="%.2f", key="passes_A")  
        st.session_state.data['corners_A'] = st.number_input("🔄 Corners", value=60.0, format="%.2f", key="corners_A")  
        st.session_state.data['interceptions_A'] = st.number_input("🛡️ Interceptions", value=50.0, format="%.2f", key="interceptions_A")  
        st.session_state.data['tacles_reussis_A'] = st.number_input("🛡️ Tacles Réussis", value=40.0, format="%.2f", key="tacles_A")  
        st.session_state.data['fautes_A'] = st.number_input("⚠️ Fautes", value=15.0, format="%.2f", key="fautes_A")  
        st.session_state.data['cartons_jaunes_A'] = st.number_input("🟨 Cartons Jaunes", value=5.0, format="%.2f", key="jaunes_A")  
        st.session_state.data['cartons_rouges_A'] = st.number_input("🟥 Cartons Rouges", value=1.0, format="%.2f", key="rouges_A")  
        st.session_state.data['joueurs_cles_absents_A'] = st.number_input("🚑 Joueurs Clés Absents", value=0.0, format="%.2f", key="absents_A")  
        st.session_state.data['motivation_A'] = st.number_input("💪 Motivation", value=4.0, format="%.2f", key="motivation_A")  
        st.session_state.data['clean_sheets_gardien_A'] = st.number_input("🧤 Clean Sheets Gardien", value=2.0, format="%.2f", key="clean_sheets_A")  
        st.session_state.data['ratio_tirs_arretes_A'] = st.number_input("🧤 Ratio Tirs Arrêtés", value=0.7, format="%.2f", key="ratio_tirs_A")  
        st.session_state.data['victoires_domicile_A'] = st.number_input("🏠 Victoires Domicile", value=50.0, format="%.2f", key="victoires_A")  
        st.session_state.data['passes_longues_A'] = st.number_input("🎯 Passes Longes", value=60.0, format="%.2f", key="passes_longues_A")  
        st.session_state.data['dribbles_reussis_A'] = st.number_input("🔥 Dribbles Réussis", value=12.0, format="%.2f", key="dribbles_A")  
        st.session_state.data['grandes_chances_manquees_A'] = st.number_input("🔥 Grandes Chances Manquées", value=5.0, format="%.2f", key="chances_manquees_A")  
        st.session_state.data['fautes_zones_dangereuses_A'] = st.number_input("⚠️ Fautes Zones Dangereuses", value=3.0, format="%.2f", key="fautes_zones_A")  
        st.session_state.data['buts_corners_A'] = st.number_input("⚽ Buts Corners", value=2.0, format="%.2f", key="buts_corners_A")  
        st.session_state.data['jours_repos_A'] = st.number_input("⏳ Jours de Repos", value=4.0, format="%.2f", key="repos_A")  
        st.session_state.data['matchs_30_jours_A'] = st.number_input("📅 Matchs (30 jours)", value=8.0, format="%.2f", key="matchs_A")  

    with col2:  
        st.markdown("### Équipe B")  
        st.session_state.data['score_rating_B'] = st.number_input("⭐ Score Rating", value=65.7, format="%.2f", key="rating_B")  
        st.session_state.data['buts_par_match_B'] = st.number_input("⚽ Buts Marqués", value=1.0, format="%.2f", key="buts_B")  
        st.session_state.data['buts_concedes_par_match_B'] = st.number_input("🥅 Buts Concédés", value=1.5, format="%.2f", key="concedes_B")  
        st.session_state.data['possession_moyenne_B'] = st.number_input("🎯 Possession Moyenne", value=45.0, format="%.2f", key="possession_B")  
        st.session_state.data['expected_but_B'] = st.number_input("📊 Expected Goals (xG)", value=1.2, format="%.2f", key="xG_B")  
        st.session_state.data['expected_concedes_B'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.8, format="%.2f", key="xGA_B")  
        st.session_state.data['tirs_cadres_B'] = st.number_input("🎯 Tirs Cadrés", value=100.0, format="%.2f", key="tirs_B")  
        st.session_state.data['tirs_non_cadres_B'] = st.number_input("🎯 Tirs Non Cadrés", value=70.0, format="%.2f", key="tirs_non_cadres_B")  
        st.session_state.data['grandes_chances_B'] = st.number_input("🔥 Grandes Chances", value=20.0, format="%.2f", key="chances_B")  
        st.session_state.data['passes_reussies_B'] = st.number_input("🔄 Passes Réussies", value=350.0, format="%.2f", key="passes_B")  
        st.session_state.data['corners_B'] = st.number_input("🔄 Corners", value=50.0, format="%.2f", key="corners_B")  
        st.session_state.data['interceptions_B'] = st.number_input("🛡️ Interceptions", value=40.0, format="%.2f", key="interceptions_B")  
        st.session_state.data['tacles_reussis_B'] = st.number_input("🛡️ Tacles Réussis", value=35.0, format="%.2f", key="tacles_B")  
        st.session_state.data['fautes_B'] = st.number_input("⚠️ Fautes", value=20.0, format="%.2f", key="fautes_B")  
        st.session_state.data['cartons_jaunes_B'] = st.number_input("🟨 Cartons Jaunes", value=6.0, format="%.2f", key="jaunes_B")  
        st.session_state.data['cartons_rouges_B'] = st.number_input("🟥 Cartons Rouges", value=2.0, format="%.2f", key="rouges_B")  
        st.session_state.data['joueurs_cles_absents_B'] = st.number_input("🚑 Joueurs Clés Absents", value=0.0, format="%.2f", key="absents_B")  
        st.session_state.data['motivation_B'] = st.number_input("💪 Motivation", value=3.0, format="%.2f", key="motivation_B")  
        st.session_state.data['clean_sheets_gardien_B'] = st.number_input("🧤 Clean Sheets Gardien", value=1.0, format="%.2f", key="clean_sheets_B")  
        st.session_state.data['ratio_tirs_arretes_B'] = st.number_input("🧤 Ratio Tirs Arrêtés", value=0.65, format="%.2f", key="ratio_tirs_B")  
        st.session_state.data['victoires_exterieur_B'] = st.number_input("🏠 Victoires Extérieur", value=40.0, format="%.2f", key="victoires_B")  
        st.session_state.data['passes_longues_B'] = st.number_input("🎯 Passes Longes", value=40.0, format="%.2f", key="passes_longues_B")  
        st.session_state.data['dribbles_reussis_B'] = st.number_input("🔥 Dribbles Réussis", value=8.0, format="%.2f", key="dribbles_B")  
        st.session_state.data['grandes_chances_manquees_B'] = st.number_input("🔥 Grandes Chances Manquées", value=6.0, format="%.2f", key="chances_manquees_B")  
        st.session_state.data['fautes_zones_dangereuses_B'] = st.number_input("⚠️ Fautes Zones Dangereuses", value=4.0, format="%.2f", key="fautes_zones_B")  
        st.session_state.data['buts_corners_B'] = st.number_input("⚽ Buts Corners", value=1.0, format="%.2f", key="buts_corners_B")  
        st.session_state.data['jours_repos_B'] = st.number_input("⏳ Jours de Repos", value=3.0, format="%.2f", key="repos_B")  
        st.session_state.data['matchs_30_jours_B'] = st.number_input("📅 Matchs (30 jours)", value=9.0, format="%.2f", key="matchs_B")  
  # Bouton de soumission du formulaire  
    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Section d'analyse et de prédiction (séparée du formulaire)  
if submitted:  
    # Ajoutez ici votre logique d'analyse et de prédiction  
    st.write("Analyse en cours...")  
# Fonction pour mettre en couleur les données utilisées par Poisson et Régression Logistique  
def highlight_data(data, key):  
    if key in ['expected_but_A', 'buts_par_match_A', 'tirs_cadres_A', 'grandes_chances_A', 'corners_A',  
               'expected_but_B', 'buts_par_match_B', 'tirs_cadres_B', 'grandes_chances_B', 'corners_B']:  
        return f"<span style='color: #1f77b4; font-size: 16px;'>{data}</span>"  
    elif key in ['score_rating_A', 'possession_moyenne_A', 'motivation_A', 'victoires_domicile_A',  
                 'score_rating_B', 'possession_moyenne_B', 'motivation_B', 'victoires_exterieur_B']:  
        return f"<span style='color: #ff7f0e; font-size: 16px;'>{data}</span>"  
    else:  
        return f"<span style='font-size: 16px;'>{data}</span>"  

# Formulaire de collecte des données  
with st.form("data_form"):  
    st.markdown("### Équipe A")  
    col1, col2 = st.columns(2)  
    with col1:  
        st.session_state.data['score_rating_A'] = st.number_input("⭐ Score Rating", value=70.6, format="%.2f", key="rating_A")  
        st.session_state.data['buts_par_match_A'] = st.number_input("⚽ Buts Marqués", value=1.5, format="%.2f", key="buts_A")  
        st.session_state.data['buts_concedes_par_match_A'] = st.number_input("🥅 Buts Concédés", value=1.0, format="%.2f", key="concedes_A")  
        st.session_state.data['possession_moyenne_A'] = st.number_input("🎯 Possession Moyenne", value=55, format="%.2f", key="possession_A")  
        st.session_state.data['expected_but_A'] = st.number_input("📊 Expected Goals (xG)", value=1.5, format="%.2f", key="xG_A")  
        st.session_state.data['expected_concedes_A'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.2, format="%.2f", key="xGA_A")  
        st.session_state.data['tirs_cadres_A'] = st.number_input("🎯 Tirs Cadrés", value=120, format="%.2f", key="tirs_A")  
        st.session_state.data['tirs_non_cadres_A'] = st.number_input("🎯 Tirs Non Cadrés", value=80, format="%.2f", key="tirs_non_cadres_A")  
        st.session_state.data['grandes_chances_A'] = st.number_input("🔥 Grandes Chances", value=25, format="%.2f", key="chances_A")  
        st.session_state.data['passes_reussies_A'] = st.number_input("🔄 Passes Réussies", value=400, format="%.2f", key="passes_A")  
        st.session_state.data['corners_A'] = st.number_input("🔄 Corners", value=60, format="%.2f", key="corners_A")  
        st.session_state.data['interceptions_A'] = st.number_input("🛡️ Interceptions", value=50, format="%.2f", key="interceptions_A")  
        st.session_state.data['tacles_reussis_A'] = st.number_input("🛡️ Tacles Réussis", value=40, format="%.2f", key="tacles_A")  
        st.session_state.data['fautes_A'] = st.number_input("⚠️ Fautes", value=15, format="%.2f", key="fautes_A")  
        st.session_state.data['cartons_jaunes_A'] = st.number_input("🟨 Cartons Jaunes", value=5, format="%.2f", key="jaunes_A")  
        st.session_state.data['cartons_rouges_A'] = st.number_input("🟥 Cartons Rouges", value=1, format="%.2f", key="rouges_A")  
        st.session_state.data['joueurs_cles_absents_A'] = st.number_input("🚑 Joueurs Clés Absents", value=0, format="%.2f", key="absents_A")  
        st.session_state.data['motivation_A'] = st.number_input("💪 Motivation", value=4, format="%.2f", key="motivation_A")  
        st.session_state.data['clean_sheets_gardien_A'] = st.number_input("🧤 Clean Sheets Gardien", value=2, format="%.2f", key="clean_sheets_A")  
        st.session_state.data['ratio_tirs_arretes_A'] = st.number_input("🧤 Ratio Tirs Arrêtés", value=0.7, format="%.2f", key="ratio_tirs_A")  
        st.session_state.data['victoires_domicile_A'] = st.number_input("🏠 Victoires Domicile", value=50, format="%.2f", key="victoires_A")  
        st.session_state.data['passes_longues_A'] = st.number_input("🎯 Passes Longes", value=60, format="%.2f", key="passes_longues_A")  
        st.session_state.data['dribbles_reussis_A'] = st.number_input("🔥 Dribbles Réussis", value=12, format="%.2f", key="dribbles_A")  
        st.session_state.data['grandes_chances_manquees_A'] = st.number_input("🔥 Grandes Chances Manquées", value=5, format="%.2f", key="chances_manquees_A")  
        st.session_state.data['fautes_zones_dangereuses_A'] = st.number_input("⚠️ Fautes Zones Dangereuses", value=3, format="%.2f", key="fautes_zones_A")  
        st.session_state.data['buts_corners_A'] = st.number_input("⚽ Buts Corners", value=2, format="%.2f", key="buts_corners_A")  
        st.session_state.data['jours_repos_A'] = st.number_input("⏳ Jours de Repos", value=4, format="%.2f", key="repos_A")  
        st.session_state.data['matchs_30_jours_A'] = st.number_input("📅 Matchs (30 jours)", value=8, format="%.2f", key="matchs_A")  

    with col2:  
        st.markdown("### Équipe B")  
        st.session_state.data['score_rating_B'] = st.number_input("⭐ Score Rating", value=65.7, format="%.2f", key="rating_B")  
        st.session_state.data['buts_par_match_B'] = st.number_input("⚽ Buts Marqués", value=1.0, format="%.2f", key="buts_B")  
        st.session_state.data['buts_concedes_par_match_B'] = st.number_input("🥅 Buts Concédés", value=1.5, format="%.2f", key="concedes_B")  
        st.session_state.data['possession_moyenne_B'] = st.number_input("🎯 Possession Moyenne", value=45, format="%.2f", key="possession_B")  
        st.session_state.data['expected_but_B'] = st.number_input("📊 Expected Goals (xG)", value=1.2, format="%.2f", key="xG_B")  
        st.session_state.data['expected_concedes_B'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.8, format="%.2f", key="xGA_B")  
        st.session_state.data['tirs_cadres_B'] = st.number_input("🎯 Tirs Cadrés", value=100, format="%.2f", key="tirs_B")  
        st.session_state.data['tirs_non_cadres_B'] = st.number_input("🎯 Tirs Non Cadrés", value=70, format="%.2f", key="tirs_non_cadres_B")  
        st.session_state.data['grandes_chances_B'] = st.number_input("🔥 Grandes Chances", value=20, format="%.2f", key="chances_B")  
        st.session_state.data['passes_reussies_B'] = st.number_input("🔄 Passes Réussies", value=350, format="%.2f", key="passes_B")  
        st.session_state.data['corners_B'] = st.number_input("🔄 Corners", value=50, format="%.2f", key="corners_B")  
        st.session_state.data['interceptions_B'] = st.number_input("🛡️ Interceptions", value=40, format="%.2f", key="interceptions_B")  
        st.session_state.data['tacles_reussis_B'] = st.number_input("🛡️ Tacles Réussis", value=35, format="%.2f", key="tacles_B")  
        st.session_state.data['fautes_B'] = st.number_input("⚠️ Fautes", value=20, format="%.2f", key="fautes_B")  
        st.session_state.data['cartons_jaunes_B'] = st.number_input("🟨 Cartons Jaunes", value=6, format="%.2f", key="jaunes_B")  
        st.session_state.data['cartons_rouges_B'] = st.number_input("🟥 Cartons Rouges", value=2, format="%.2f", key="rouges_B")  
        st.session_state.data['joueurs_cles_absents_B'] = st.number_input("🚑 Joueurs Clés Absents", value=0, format="%.2f", key="absents_B")  
        st.session_state.data['motivation_B'] = st.number_input("💪 Motivation", value=3, format="%.2f", key="motivation_B")  
        st.session_state.data['clean_sheets_gardien_B'] = st.number_input("🧤 Clean Sheets Gardien", value=1, format="%.2f", key="clean_sheets_B")  
        st.session_state.data['ratio_tirs_arretes_B'] = st.number_input("🧤 Ratio Tirs Arrêtés", value=0.65, format="%.2f", key="ratio_tirs_B")  
        st.session_state.data['victoires_exterieur_B'] = st.number_input("🏠 Victoires Extérieur", value=40, format="%.2f", key="victoires_B")  
        st.session_state.data['passes_longues_B'] = st.number_input("🎯 Passes Longes", value=40, format="%.2f", key="passes_longues_B")  
        st.session_state.data['dribbles_reussis_B'] = st.number_input("🔥 Dribbles Réussis", value=8, format="%.2f", key="dribbles_B")  
        st.session_state.data['grandes_chances_manquees_B'] = st.number_input("🔥 Grandes Chances Manquées", value=6, format="%.2f", key="chances_manquees_B")  
        st.session_state.data['fautes_zones_dangereuses_B'] = st.number_input("⚠️ Fautes Zones Dangereuses", value=4, format="%.2f", key="fautes_zones_B")  
        st.session_state.data['buts_corners_B'] = st.number_input("⚽ Buts Corners", value=1, format="%.2f", key="buts_corners_B")  
        st.session_state.data['jours_repos_B'] = st.number_input("⏳ Jours de Repos", value=3, format="%.2f", key="repos_B")  
        st.session_state.data['matchs_30_jours_B'] = st.number_input("📅 Matchs (30 jours)", value=9, format="%.2f", key="matchs_B")  

    # Bouton de soumission du formulaire  
    submitted = st.form_submit_button("🔍 Analyser le Match")  
    # Bouton de soumission du formulaire  
    submitted = st.form_submit_button("🔍 Analyser le Match")
	# Section d'analyse et de prédiction (séparée du formulaire)  
if submitted:  
    try:  
        # Préparation des données pour Poisson  
        lambda_A = (  
            st.session_state.data['expected_but_A']  # xG  
            + st.session_state.data['buts_par_match_A']  # Buts marqués  
            + st.session_state.data['tirs_cadres_A'] * 0.1  # Tirs cadrés (pondération)  
            + st.session_state.data['grandes_chances_A'] * 0.2  # Grandes chances (pondération)  
            + st.session_state.data['corners_A'] * 0.05  # Corners (pondération)  
        ) / 3  # Normalisation  

        lambda_B = (  
            st.session_state.data['expected_but_B']  # xG  
            + st.session_state.data['buts_par_match_B']  # Buts marqués  
            + st.session_state.data['tirs_cadres_B'] * 0.1  # Tirs cadrés (pondération)  
            + st.session_state.data['grandes_chances_B'] * 0.2  # Grandes chances (pondération)  
            + st.session_state.data['corners_B'] * 0.05  # Corners (pondération)  
        ) / 3  # Normalisation  

        # Prédiction des buts avec Poisson  
        buts_A = poisson.rvs(mu=lambda_A, size=1000)  # 1000 simulations pour l'équipe A  
        buts_B = poisson.rvs(mu=lambda_B, size=1000)  # 1000 simulations pour l'équipe B  

        # Résultats Poisson  
        st.subheader("📊 Prédiction des Buts (Poisson)")  
        col_poisson_A, col_poisson_B = st.columns(2)  
        with col_poisson_A:  
            st.metric("⚽ Buts Moyens (Équipe A)", f"{np.mean(buts_A):.2f}")  
            st.metric("⚽ Buts Prévus (Équipe A)", f"{np.percentile(buts_A, 75):.2f} (75e percentile)")  
        with col_poisson_B:  
            st.metric("⚽ Buts Moyens (Équipe B)", f"{np.mean(buts_B):.2f}")  
            st.metric("⚽ Buts Prévus (Équipe B)", f"{np.percentile(buts_B, 75):.2f} (75e percentile)")  

        # Convertisseur de score  
        st.subheader("🔄 Analyse des Cotes Implicites")  
        cote_A = st.number_input("Cote de Victoire Équipe A", value=2.0, key="cote_A")  
        cote_B = st.number_input("Cote de Victoire Équipe B", value=2.5, key="cote_B")  
        cote_Nul = st.number_input("Cote de Match Nul", value=3.0, key="cote_Nul")  

        # Calcul des probabilités implicites  
        proba_implicite_A = 1 / cote_A  
        proba_implicite_B = 1 / cote_B  
        proba_implicite_Nul = 1 / cote_Nul  

        # Normalisation des probabilités implicites pour qu'elles somment à 1  
        total_proba_implicite = proba_implicite_A + proba_implicite_B + proba_implicite_Nul  
        proba_implicite_A /= total_proba_implicite  
        proba_implicite_B /= total_proba_implicite  
        proba_implicite_Nul /= total_proba_implicite  

        # Préparation des données pour Régression Logistique et Random Forest  
        X = np.array(list(st.session_state.data.values())).reshape(1, -1)  # Toutes les données des équipes  
        X = np.repeat(X, 1000, axis=0)  # Répéter les données pour correspondre à la taille de y  
        y = np.random.randint(0, 3, 1000)  # 3 classes : 0 (défaite), 1 (victoire A), 2 (match nul)  

        # Modèles  
        modeles = {  
            "Régression Logistique": LogisticRegression(max_iter=1000),  
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  
        }  

        # Validation croisée et prédictions  
        st.markdown("### 🤖 Performance des Modèles")  
        resultats_modeles = {}  
        for nom, modele in modeles.items():  
            # Validation croisée stratifiée  
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
            scores = cross_val_score(modele, X, y, cv=cv, scoring='accuracy')  

            # Affichage des métriques de validation croisée  
            st.markdown(f"#### {nom}")  
            col_accuracy, col_precision, col_recall, col_f1 = st.columns(4)  
            with col_accuracy:  
                st.metric("🎯 Précision Globale", f"{np.mean(scores):.2%}")  

            # Prédiction finale  
            modele.fit(X, y)  
            proba = modele.predict_proba(X)[0]  # Mise à jour de proba  

            # Vérification de la taille de proba  
            if len(proba) >= 3:  # Assurez-vous qu'il y a au moins 3 classes  
                # Affichage des prédictions  
                st.markdown("**📊 Prédictions**")  
                col_victoire_A, col_victoire_B, col_nul = st.columns(3)  
                with col_victoire_A:  
                    st.metric("🏆 Victoire A", f"{proba[1]:.2%}")  
                with col_victoire_B:  
                    st.metric("🏆 Victoire B", f"{proba[0]:.2%}")  
                with col_nul:  
                    st.metric("🤝 Match Nul", f"{proba[2]:.2%}")  
            else:  
                st.error(f"Erreur : La prédiction n'a pas retourné le bon nombre de classes pour {nom}.")  

            # Stockage des résultats pour comparaison  
            resultats_modeles[nom] = {  
                'accuracy': np.mean(scores),  
                'precision': np.mean(cross_val_score(modele, X, y, cv=cv, scoring='precision_macro')),  
                'recall': np.mean(cross_val_score(modele, X, y, cv=cv, scoring='recall_macro')),  
                'f1_score': np.mean(cross_val_score(modele, X, y, cv=cv, scoring='f1_macro'))  
            }  

        # Vérification que proba est défini avant de l'utiliser  
        if proba is not None and len(proba) >= 3:  
            # Récupération des probabilités prédites par le modèle  
            proba_predite_A = proba[1]  # Victoire A  
            proba_predite_B = proba[0]  # Victoire B  
            proba_predite_Nul = proba[2]  # Match Nul  

            # Affichage des résultats  
            st.write("##### Probabilités Implicites (Cotes)")  
            col_implicite_A, col_implicite_B, col_implicite_Nul = st.columns(3)  
            with col_implicite_A:  
                st.metric("Victoire A", f"{proba_implicite_A:.2%}")  
            with col_implicite_B:  
                st.metric("Victoire B", f"{proba_implicite_B:.2%}")  
            with col_implicite_Nul:  
                st.metric("Match Nul", f"{proba_implicite_Nul:.2%}")  

            st.write("##### Probabilités Prédites (Modèle)")  
            col_predite_A, col_predite_B, col_predite_Nul = st.columns(3)  
            with col_predite_A:  
                st.metric("Victoire A", f"{proba_predite_A:.2%}")  
            with col_predite_B:  
                st.metric("Victoire B", f"{proba_predite_B:.2%}")  
            with col_predite_Nul:  
                st.metric("Match Nul", f"{proba_predite_Nul:.2%}")  

            # Comparaison des probabilités  
            st.write("##### Comparaison")  
            col_comparaison_A, col_comparaison_B, col_comparaison_Nul = st.columns(3)  
            with col_comparaison_A:  
                st.metric("Différence Victoire A", f"{(proba_predite_A - proba_implicite_A):.2%}")  
            with col_comparaison_B:  
                st.metric("Différence Victoire B", f"{(proba_predite_B - proba_implicite_B):.2%}")  
            with col_comparaison_Nul:  
                st.metric("Différence Match Nul", f"{(proba_predite_Nul - proba_implicite_Nul):.2%}")  
        else:  
            st.error("Erreur : Les probabilités n'ont pas pu être calculées.")  

    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(traceback.format_exc())  

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  

- **📊 Prédiction des Buts (Poisson)** :   
  - Les buts moyens prévus pour chaque équipe sont calculés à partir des statistiques d'entrée.  
  - Le 75e percentile des buts prédit permet d'évaluer le potentiel maximum de chaque équipe.  

- **🤖 Performance des Modèles** :   
  - **Régression Logistique** : Modèle linéaire simple qui prédit les résultats basés sur les caractéristiques des équipes.  
  - **Random Forest** : Modèle plus complexe qui utilise plusieurs arbres de décision pour améliorer la précision des prédictions.  

- **📈 Comparaison des Performances des Modèles** :   
  - Les métriques affichées (précision, rappel, F1-score) permettent de comparer l'efficacité des modèles.  

- **🔄 Analyse des Cotes Implicites** :  
  - Analyse les cotes fournies pour calculer les probabilités implicites et les compare avec les probabilités prédites par le modèle.  
  - Une différence positive suggère une possible value bet.  

⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""")
