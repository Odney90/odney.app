import streamlit as st  
import numpy as np  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  

# Initialisation des données dans st.session_state  
if "data" not in st.session_state:  
    st.session_state.data = {  
        # Équipe A  
        "score_rating_A": 70.0,  
        "buts_par_match_A": 1.5,  
        "buts_concedes_par_match_A": 1.0,  
        "possession_moyenne_A": 55.0,  
        "expected_but_A": 1.8,  
        "expected_concedes_A": 1.2,  
        "tirs_cadres_A": 120.0,  
        "grandes_chances_A": 25.0,  
        "passes_reussies_A": 400.0,  
        "corners_A": 60.0,  
        "interceptions_A": 50.0,  
        "tacles_reussis_A": 40.0,  
        "fautes_A": 15.0,  
        "cartons_jaunes_A": 5.0,  
        "cartons_rouges_A": 1.0,  
        "joueurs_cles_absents_A": 0,  
        "motivation_A": 3,  
        "clean_sheets_gardien_A": 2.0,  
        "ratio_tirs_arretes_A": 0.75,  
        "victoires_domicile_A": 60.0,  
        "passes_longues_A": 50.0,  
        "dribbles_reussis_A": 10.0,  
        "ratio_tirs_cadres_A": 0.4,  
        "grandes_chances_manquees_A": 5.0,  
        "fautes_zones_dangereuses_A": 3.0,  
        "buts_corners_A": 2.0,  
        "jours_repos_A": 4.0,  
        "matchs_30_jours_A": 8.0,  
        # Équipe B  
        "score_rating_B": 65.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "possession_moyenne_B": 45.0,  
        "expected_but_B": 1.2,  
        "expected_concedes_B": 1.8,  
        "tirs_cadres_B": 100.0,  
        "grandes_chances_B": 20.0,  
        "passes_reussies_B": 350.0,  
        "corners_B": 50.0,  
        "interceptions_B": 40.0,  
        "tacles_reussis_B": 35.0,  
        "fautes_B": 20.0,  
        "cartons_jaunes_B": 6.0,  
        "cartons_rouges_B": 2.0,  
        "joueurs_cles_absents_B": 0,  
        "motivation_B": 3,  
        "clean_sheets_gardien_B": 1.0,  
        "ratio_tirs_arretes_B": 0.65,  
        "victoires_exterieur_B": 40.0,  
        "passes_longues_B": 40.0,  
        "dribbles_reussis_B": 8.0,  
        "ratio_tirs_cadres_B": 0.35,  
        "grandes_chances_manquees_B": 6.0,  
        "fautes_zones_dangereuses_B": 4.0,  
        "buts_corners_B": 1.0,  
        "jours_repos_B": 3.0,  
        "matchs_30_jours_B": 9.0,  
        # Conditions du Match  
        "conditions_match": "",  
    }  

# Création des onglets  
tab1, tab2, tab3 = st.tabs(["📊 Statistiques", "🌦️ Conditions et Motivation", "🔮 Prédictions"])  

# Onglet 1 : Statistiques  
with tab1:  
    st.header("📊 Statistiques des Équipes")  
    col_a, col_b = st.columns(2)  

    # Statistiques de l'Équipe A  
    with col_a:  
        st.subheader("Équipe A")  
        st.session_state.data["score_rating_A"] = st.number_input(  
            "Score Rating A", value=st.session_state.data["score_rating_A"]  
        )  
        st.session_state.data["buts_par_match_A"] = st.number_input(  
            "Buts par Match A", value=st.session_state.data["buts_par_match_A"]  
        )  
        st.session_state.data["possession_moyenne_A"] = st.number_input(  
            "Possession Moyenne A", value=st.session_state.data["possession_moyenne_A"]  
        )  
        st.session_state.data["motivation_A"] = st.slider(  
            "Motivation A (1 à 5)", min_value=1, max_value=5, value=int(st.session_state.data["motivation_A"])  
        )  
        st.session_state.data["tirs_cadres_A"] = st.number_input(  
            "Tirs Cadrés A", value=st.session_state.data["tirs_cadres_A"]  
        )  
        st.session_state.data["grandes_chances_A"] = st.number_input(  
            "Grandes Chances A", value=st.session_state.data["grandes_chances_A"]  
        )  
        st.session_state.data["passes_reussies_A"] = st.number_input(  
            "Passes Réussies A", value=st.session_state.data["passes_reussies_A"]  
        )  
        st.session_state.data["corners_A"] = st.number_input(  
            "Corners A", value=st.session_state.data["corners_A"]  
        )  
        st.session_state.data["interceptions_A"] = st.number_input(  
            "Interceptions A", value=st.session_state.data["interceptions_A"]  
        )  
        st.session_state.data["tacles_reussis_A"] = st.number_input(  
            "Tacles Réussis A", value=st.session_state.data["tacles_reussis_A"]  
        )  
        st.session_state.data["fautes_A"] = st.number_input(  
            "Fautes A", value=st.session_state.data["fautes_A"]  
        )  
        st.session_state.data["cartons_jaunes_A"] = st.number_input(  
            "Cartons Jaunes A", value=st.session_state.data["cartons_jaunes_A"]  
        )  
        st.session_state.data["cartons_rouges_A"] = st.number_input(  
            "Cartons Rouges A", value=st.session_state.data["cartons_rouges_A"]  
        )  
        st.session_state.data["joueurs_cles_absents_A"] = st.number_input(  
            "Joueurs Clés Absents A", value=st.session_state.data["joueurs_cles_absents_A"]  
        )  
        st.session_state.data["clean_sheets_gardien_A"] = st.number_input(  
            "Clean Sheets Gardien A", value=st.session_state.data["clean_sheets_gardien_A"]  
        )  
        st.session_state.data["ratio_tirs_arretes_A"] = st.number_input(  
            "Ratio Tirs Arrêtés A", value=st.session_state.data["ratio_tirs_arretes_A"]  
        )  
        st.session_state.data["victoires_domicile_A"] = st.number_input(  
            "Victoires Domicile A", value=st.session_state.data["victoires_domicile_A"]  
        )  
        st.session_state.data["passes_longues_A"] = st.number_input(  
            "Passes Longues A", value=st.session_state.data["passes_longues_A"]  
        )  
        st.session_state.data["dribbles_reussis_A"] = st.number_input(  
            "Dribbles Réussis A", value=st.session_state.data["dribbles_reussis_A"]  
        )  
        st.session_state.data["ratio_tirs_cadres_A"] = st.number_input(  
            "Ratio Tirs Cadrés A", value=st.session_state.data["ratio_tirs_cadres_A"]  
        )  
        st.session_state.data["grandes_chances_manquees_A"] = st.number_input(  
            "Grandes Chances Manquées A", value=st.session_state.data["grandes_chances_manquees_A"]  
        )  
        st.session_state.data["fautes_zones_dangereuses_A"] = st.number_input(  
            "Fautes Zones Dangereuses A", value=st.session_state.data["fautes_zones_dangereuses_A"]  
        )  
        st.session_state.data["buts_corners_A"] = st.number_input(  
            "Buts Corners A", value=st.session_state.data["buts_corners_A"]  
        )  
        st.session_state.data["jours_repos_A"] = st.number_input(  
            "Jours de Repos A", value=st.session_state.data["jours_repos_A"]  
        )  
        st.session_state.data["matchs_30_jours_A"] = st.number_input(  
            "Matchs (30 jours) A", value=st.session_state.data["matchs_30_jours_A"]  
        )  

    # Statistiques de l'Équipe B  
    with col_b:  
        st.subheader("Équipe B")  
        st.session_state.data["score_rating_B"] = st.number_input(  
            "Score Rating B", value=st.session_state.data["score_rating_B"]  
        )  
        st.session_state.data["buts_par_match_B"] = st.number_input(  
            "Buts par Match B", value=st.session_state.data["buts_par_match_B"]  
        )  
        st.session_state.data["possession_moyenne_B"] = st.number_input(  
            "Possession Moyenne B", value=st.session_state.data["possession_moyenne_B"]  
        )  
        st.session_state.data["motivation_B"] = st.slider(  
            "Motivation B (1 à 5)", min_value=1, max_value=5, value=int(st.session_state.data["motivation_B"])  
        )  
        st.session_state.data["tirs_cadres_B"] = st.number_input(  
            "Tirs Cadrés B", value=st.session_state.data["tirs_cadres_B"]  
        )  
        st.session_state.data["grandes_chances_B"] = st.number_input(  
            "Grandes Chances B", value=st.session_state.data["grandes_chances_B"]  
        )  
        st.session_state.data["passes_reussies_B"] = st.number_input(  
            "Passes Réussies B", value=st.session_state.data["passes_reussies_B"]  
        )  
        st.session_state.data["corners_B"] = st.number_input(  
            "Corners B", value=st.session_state.data["corners_B"]  
        )  
        st.session_state.data["interceptions_B"] = st.number_input(  
            "Interceptions B", value=st.session_state.data["interceptions_B"]  
        )  
        st.session_state.data["tacles_reussis_B"] = st.number_input(  
            "Tacles Réussis B", value=st.session_state.data["tacles_reussis_B"]  
        )  
        st.session_state.data["fautes_B"] = st.number_input(  
            "Fautes B", value=st.session_state.data["fautes_B"]  
        )  
        st.session_state.data["cartons_jaunes_B"] = st.number_input(  
            "Cartons Jaunes B", value=st.session_state.data["cartons_jaunes_B"]  
        )  
        st.session_state.data["cartons_rouges_B"] = st.number_input(  
            "Cartons Rouges B", value=st.session_state.data["cartons_rouges_B"]  
        )  
        st.session_state.data["joueurs_cles_absents_B"] = st.number_input(  
            "Joueurs Clés Absents B", value=st.session_state.data["joueurs_cles_absents_B"]  
        )  
        st.session_state.data["clean_sheets_gardien_B"] = st.number_input(  
            "Clean Sheets Gardien B", value=st.session_state.data["clean_sheets_gardien_B"]  
        )  
        st.session_state.data["ratio_tirs_arretes_B"] = st.number_input(  
            "Ratio Tirs Arrêtés B", value=st.session_state.data["ratio_tirs_arretes_B"]  
        )  
        st.session_state.data["victoires_exterieur_B"] = st.number_input(  
            "Victoires Extérieur B", value=st.session_state.data["victoires_exterieur_B"]  
        )  
        st.session_state.data["passes_longues_B"] = st.number_input(  
            "Passes Longues B", value=st.session_state.data["passes_longues_B"]  
        )  
        st.session_state.data["dribbles_reussis_B"] = st.number_input(  
            "Dribbles Réussis B", value=st.session_state.data["dribbles_reussis_B"]  
        )  
        st.session_state.data["ratio_tirs_cadres_B"] = st.number_input(  
            "Ratio Tirs Cadrés B", value=st.session_state.data["ratio_tirs_cadres_B"]  
        )  
        st.session_state.data["grandes_chances_manquees_B"] = st.number_input(  
            "Grandes Chances Manquées B", value=st.session_state.data["grandes_chances_manquees_B"]  
        )  
        st.session_state.data["fautes_zones_dangereuses_B"] = st.number_input(  
            "Fautes Zones Dangereuses B", value=st.session_state.data["fautes_zones_dangereuses_B"]  
        )  
        st.session_state.data["buts_corners_B"] = st.number_input(  
            "Buts Corners B", value=st.session_state.data["buts_corners_B"]  
        )  
        st.session_state.data["jours_repos_B"] = st.number_input(  
            "Jours de Repos B", value=st.session_state.data["jours_repos_B"]  
        )  
        st.session_state.data["matchs_30_jours_B"] = st.number_input(  
            "Matchs (30 jours) B", value=st.session_state.data["matchs_30_jours_B"]  
        )  

# Onglet 2 : Conditions et Motivation  
with tab2:  
    st.header("🌦️ Conditions du Match et Motivation")  
    st.session_state.data["conditions_match"] = st.text_input(  
        "🌧️ Conditions du Match (ex : pluie, terrain sec)",  
        value=st.session_state.data["conditions_match"],  
        key="conditions_match_input"  
    )  

# Onglet 3 : Prédictions  
with tab3:  
    st.header("🔮 Prédictions")  
    if st.button("Prédire le résultat"):  
        try:  
            # Prédiction des Buts avec Poisson  
            avg_goals_A = st.session_state.data["buts_par_match_A"]  
            avg_goals_B = st.session_state.data["buts_par_match_B"]  
            prob_0_0 = poisson.pmf(0, avg_goals_A) * poisson.pmf(0, avg_goals_B)  
            prob_1_1 = poisson.pmf(1, avg_goals_A) * poisson.pmf(1, avg_goals_B)  
            prob_2_2 = poisson.pmf(2, avg_goals_A) * poisson.pmf(2, avg_goals_B)  

            # Régression Logistique  
            X_lr = np.array([  
                [  
                    st.session_state.data["score_rating_A"],  
                    st.session_state.data["score_rating_B"],  
                    st.session_state.data["possession_moyenne_A"],  
                    st.session_state.data["possession_moyenne_B"],  
                    st.session_state.data["motivation_A"],  
                    st.session_state.data["motivation_B"],  
                ]  
            ])  
            y_lr = np.array([1])  # Label factice pour correspondre à X_lr  
            model_lr = LogisticRegression()  
            model_lr.fit(X_lr, y_lr)  
            prediction_lr = model_lr.predict(X_lr)  

            # Random Forest  
            X_rf = np.array([  
                [  
                    st.session_state.data[key]  
                    for key in st.session_state.data  
                    if (key.endswith("_A") or key.endswith("_B")) and isinstance(st.session_state.data[key], (int, float))  
                ]  
            ])  
            y_rf = np.array([1])  # Label factice pour correspondre à X_rf  
            model_rf = RandomForestClassifier()  
            model_rf.fit(X_rf, y_rf)  
            prediction_rf = model_rf.predict(X_rf)  

            # Affichage des résultats  
            st.subheader("Résultats des Prédictions")  
            st.write(f"📊 **Probabilité de 0-0 (Poisson)** : {prob_0_0:.2%}")  
            st.write(f"📊 **Probabilité de 1-1 (Poisson)** : {prob_1_1:.2%}")  
            st.write(f"📊 **Probabilité de 2-2 (Poisson)** : {prob_2_2:.2%}")  
            st.write(f"📊 **Régression Logistique** : {'Équipe A' if prediction_lr[0] == 1 else 'Équipe B'}")  
            st.write(f"🌲 **Random Forest** : {'Équipe A' if prediction_rf[0] == 1 else 'Équipe B'}")  

        except Exception as e:  
            st.error(f"Une erreur s'est produite lors de la prédiction : {e}")
