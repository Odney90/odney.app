import streamlit as st  
import numpy as np  
from scipy.stats import poisson  
import traceback  

# Configuration de la page  
st.set_page_config(page_title="⚽ Analyse de Match de Football", page_icon="⚽", layout="wide")  

# Initialisation de st.session_state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Formulaire de collecte des données  
with st.form("data_form"):  
    # Équipe A  
    st.markdown("### Équipe A")  
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

    # Équipe B  
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

# Section d'analyse et de prédiction  
if submitted:  
    try:  
        # Calcul des lambda pour Poisson  
        lambda_A = (  
            st.session_state.data['expected_but_A'] +  
            st.session_state.data['buts_par_match_A'] +  
            st.session_state.data['tirs_cadres_A'] * 0.1 +  
            st.session_state.data['grandes_chances_A'] * 0.2 +  
            st.session_state.data['corners_A'] * 0.05  
        ) / 3  # Normalisation  

        lambda_B = (  
            st.session_state.data['expected_but_B'] +  
            st.session_state.data['buts_par_match_B'] +  
            st.session_state.data['tirs_cadres_B'] * 0.1 +  
            st.session_state.data['grandes_chances_B'] * 0.2 +  
            st.session_state.data['corners_B'] * 0.05  
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

        # Calcul des probabilités de victoire  
        proba_implicite_A = 1 / (1 + (st.session_state.data['buts_concedes_par_match_B'] / st.session_state.data['buts_par_match_A']))  
        proba_implicite_B = 1 / (1 + (st.session_state.data['buts_concedes_par_match_A'] / st.session_state.data['buts_par_match_B']))  
        proba_implicite_Nul = 1 - (proba_implicite_A + proba_implicite_B)  

        proba_predite_A = np.mean(buts_A) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_predite_B = np.mean(buts_B) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_predite_Nul = 1 - (proba_predite_A + proba_predite_B)  

        # Affichage des probabilités  
        st.write("##### Probabilités Prédites (Modèle)")  
        col_predite_A, col_predite_B, col_predite_Nul = st.columns(3)  
        with col_predite_A:  
            st.metric("Victoire A", f"{proba_predite_A:.2%}")  
        with col_predite_B:  
            st.metric("Victoire B", f"{proba_predite_B:.2%}")  
        with col_predite_Nul:  
            st.metric("Match Nul", f"{proba_predite_Nul:.2%}")  

        # Comparaison des probabilités  
        st.write("##### Comparaison des Probabilités")  
        col_comparaison_A, col_comparaison_B, col_comparaison_Nul = st.columns(3)  
        with col_comparaison_A:  
            st.metric("Différence Victoire A", f"{(proba_predite_A - proba_implicite_A):.2%}")  
        with col_comparaison_B:  
            st.metric("Différence Victoire B", f"{(proba_predite_B - proba_implicite_B):.2%}")  
        with col_comparaison_Nul:  
            st.metric("Différence Match Nul", f"{(proba_predite_Nul - proba_implicite_Nul):.2%}")  

        # Calcul des cotes implicites  
        cote_implicite_A = 1 / proba_implicite_A if proba_implicite_A > 0 else float('inf')  
        cote_implicite_B = 1 / proba_implicite_B if proba_implicite_B > 0 else float('inf')  
        cote_implicite_Nul = 1 / proba_implicite_Nul if proba_implicite_Nul > 0 else float('inf')  

        # Calcul des cotes prédites  
        cote_predite_A = 1 / proba_predite_A if proba_predite_A > 0 else float('inf')  
        cote_predite_B = 1 / proba_predite_B if proba_predite_B > 0 else float('inf')  
        cote_predite_Nul = 1 / proba_predite_Nul if proba_predite_Nul > 0 else float('inf')  

        # Affichage des cotes  
        st.write("##### Cotes Implicites et Prédites")  
        col_cotes_A, col_cotes_B, col_cotes_Nul = st.columns(3)  
        with col_cotes_A:  
            st.metric("Cote Implicite A", f"{cote_implicite_A:.2f}")  
            st.metric("Cote Prédite A", f"{cote_predite_A:.2f}")  
        with col_cotes_B:  
            st.metric("Cote Implicite B", f"{cote_implicite_B:.2f}")  
            st.metric("Cote Prédite B", f"{cote_predite_B:.2f}")  
        with col_cotes_Nul:  
            st.metric("Cote Implicite Nul", f"{cote_implicite_Nul:.2f}")  
            st.metric("Cote Prédite Nul", f"{cote_predite_Nul:.2f}")  

    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(traceback.format_exc())  

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  
- **📊 Prédiction des Buts (Poisson)** : Les buts moyens prévus pour chaque équipe sont calculés à partir des statistiques d'entrée.  
- **🤖 Performance des Modèles** : Mod
