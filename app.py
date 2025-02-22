import streamlit as st  
import numpy as np  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
import pandas as pd  
import matplotlib.pyplot as plt  

# Initialisation des données par défaut  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        # Statistiques de l'Équipe A  
        "score_rating_A": 70.0,  
        "buts_par_match_A": 1.5,  
        "buts_concedes_par_match_A": 1.0,  
        "possession_moyenne_A": 55.0,  
        "expected_but_A": 1.8,  
        "expected_concedes_A": 1.2,  
        "tirs_cadres_A": 120,  
        "grandes_chances_A": 25,  
        "passes_reussies_A": 400,  
        "corners_A": 60,  
        "interceptions_A": 50,  
        "tacles_reussis_A": 40,  
        "fautes_A": 15,  
        "cartons_jaunes_A": 5,  
        "cartons_rouges_A": 1,  

        # Statistiques de l'Équipe B  
        "score_rating_B": 65.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "possession_moyenne_B": 45.0,  
        "expected_but_B": 1.2,  
        "expected_concedes_B": 1.8,  
        "tirs_cadres_B": 100,  
        "grandes_chances_B": 20,  
        "passes_reussies_B": 350,  
        "corners_B": 50,  
        "interceptions_B": 40,  
        "tacles_reussis_B": 35,  
        "fautes_B": 20,  
        "cartons_jaunes_B": 6,  
        "cartons_rouges_B": 2,  

        # Forme récente  
        "recent_form_A": [1, 2, 1, 0, 3],  
        "recent_form_B": [0, 1, 2, 1, 1],  

        # Historique des confrontations  
        "head_to_head": {"victoires_A": 0, "nuls": 0, "victoires_B": 0},  

        # Nouveaux critères  
        "joueurs_cles_absents_A": 0,  # Nombre de joueurs clés absents  
        "score_rating_joueurs_cles_A": 0.0,  # Score rating des joueurs clés  
        "motivation_A": 3,  # Motivation sur une échelle de 5  
        "clean_sheets_gardien_A": 2,  # Nombre de clean sheets du gardien  
        "ratio_tirs_arretes_A": 0.75,  # Ratio de tirs arrêtés par le gardien  
        "victoires_domicile_A": 60.0,  # Pourcentage de victoires à domicile  
        "passes_longues_A": 50,  # Nombre de passes longues par match  
        "dribbles_reussis_A": 10,  # Nombre de dribbles réussis par match  
        "ratio_tirs_cadres_A": 0.4,  # Ratio de tirs cadrés  
        "grandes_chances_manquees_A": 5,  # Nombre de grandes chances manquées  
        "fautes_zones_dangereuses_A": 3,  # Fautes dans des zones dangereuses  
        "buts_corners_A": 2,  # Nombre de buts marqués sur corners  
        "jours_repos_A": 4,  # Nombre de jours de repos  
        "matchs_30_jours_A": 8,  # Nombre de matchs sur les 30 derniers jours  

        # Nouveaux critères pour l'Équipe B  
        "joueurs_cles_absents_B": 0,  # Nombre de joueurs clés absents  
        "score_rating_joueurs_cles_B": 0.0,  # Score rating des joueurs clés  
        "motivation_B": 3,  # Motivation sur une échelle de 5  
        "clean_sheets_gardien_B": 1,  # Nombre de clean sheets du gardien  
        "ratio_tirs_arretes_B": 0.65,  # Ratio de tirs arrêtés par le gardien  
        "victoires_exterieur_B": 40.0,  # Pourcentage de victoires à l'extérieur  
        "passes_longues_B": 40,  # Nombre de passes longues par match  
        "dribbles_reussis_B": 8,  # Nombre de dribbles réussis par match  
        "ratio_tirs_cadres_B": 0.35,  # Ratio de tirs cadrés  
        "grandes_chances_manquees_B": 6,  # Nombre de grandes chances manquées  
        "fautes_zones_dangereuses_B": 4,  # Fautes dans des zones dangereuses  
        "buts_corners_B": 1,  # Nombre de buts marqués sur corners  
        "jours_repos_B": 3,  # Nombre de jours de repos  
        "matchs_30_jours_B": 9,  # Nombre de matchs sur les 30 derniers jours  
    }  

# Fonctions pour les outils de paris  
def cotes_vers_probabilite(cote):  
    """Convertit une cote décimale en probabilité implicite."""  
    return 1 / cote  

def enlever_marge(probabilites, marge):  
    """Enlève la marge du bookmaker des probabilités."""  
    total_probabilite = sum(probabilites)  
    facteur_correction = (1 - marge) / total_probabilite  
    return [p * facteur_correction for p in probabilites]  

def kelly_criterion(cote, probabilite, bankroll, kelly_fraction):  
    """Calcule la mise de Kelly."""  
    probabilite_avantage = probabilite * cote - 1  
    fraction = kelly_fraction * probabilite_avantage / (cote - 1)  
    return bankroll * fraction  

# Configuration de la page  
st.set_page_config(page_title="Prédiction de Matchs de Football", page_icon="⚽", layout="wide")  

# En-tête  
st.header("⚽ Analyse de Prédiction de Matchs de Football")  

# Onglets pour les différentes sections  
tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistiques des Équipes", "🌦️ Conditions du Match", "🔮 Prédictions", "🛠️ Outils de Paris"])  

with tab1:  
    st.subheader("📊 Statistiques des Équipes")  

    col_a, col_b = st.columns(2)  

    with col_a:  
        st.subheader("Équipe A 🟡")  
        for key in st.session_state.data:  
            if key.endswith("_A") and isinstance(st.session_state.data[key], (int, float)):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    min_value=0.0,  
                    value=float(st.session_state.data[key]),  
                    key=key  
                )  

    with col_b:  
        st.subheader("Équipe B 🔴")  
        for key in st.session_state.data:  
            if key.endswith("_B") and isinstance(st.session_state.data[key], (int, float)):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    min_value=0.0,  
                    value=float(st.session_state.data[key]),  
                    key=key  
                )  

with tab2:  
    st.subheader("🌦️ Conditions du Match")  

    # Conditions du match  
    st.session_state.data["conditions_match"] = st.text_input("🌧️ Conditions du Match (ex : pluie, terrain sec, etc.)", value="", key="conditions_match")  

    # Joueurs clés absents et score rating des joueurs clés  
    st.subheader("👥 Joueurs Clés Absents")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["joueurs_cles_absents_A"] = st.number_input(  
            "🚑 Nombre de joueurs clés absents (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["joueurs_cles_absents_A"]),  
            key="joueurs_cles_absents_A"  
        )  
        st.session_state.data["score_rating_joueurs_cles_A"] = st.number_input(  
            "⭐ Score rating des joueurs clés (Équipe A)",  
            min_value=0.0,  
            value=float(st.session_state.data["score_rating_joueurs_cles_A"]),  
            key="score_rating_joueurs_cles_A"  
        )  
    with col_b:  
        st.session_state.data["joueurs_cles_absents_B"] = st.number_input(  
            "🚑 Nombre de joueurs clés absents (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["joueurs_cles_absents_B"]),  
            key="joueurs_cles_absents_B"  
        )  
        st.session_state.data["score_rating_joueurs_cles_B"] = st.number_input(  
            "⭐ Score rating des joueurs clés (Équipe B)",  
            min_value=0.0,  
            value=float(st.session_state.data["score_rating_joueurs_cles_B"]),  
            key="score_rating_joueurs_cles_B"  
        )  

    # Motivation des équipes  
    st.subheader("🔥 Motivation des Équipes")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["motivation_A"] = st.slider(  
            "💪 Motivation de l'Équipe A (1 à 5)",  
            min_value=1,  
            max_value=5,  
            value=int(st.session_state.data["motivation_A"]),  
            key="motivation_A"  
        )  
    with col_b:  
        st.session_state.data["motivation_B"] = st.slider(  
            "💪 Motivation de l'Équipe B (1 à 5)",  
            min_value=1,  
            max_value=5,  
            value=int(st.session_state.data["motivation_B"]),  
            key="motivation_B"  
        )  

    # Forme des gardiens  
    st.subheader("🧤 Forme des Gardiens")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["clean_sheets_gardien_A"] = st.number_input(  
            "🛡️ Clean sheets du gardien (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["clean_sheets_gardien_A"]),  
            key="clean_sheets_gardien_A"  
        )  
        st.session_state.data["ratio_tirs_arretes_A"] = st.number_input(  
            "🎯 Ratio de tirs arrêtés (Équipe A)",  
            min_value=0.0,  
            max_value=1.0,  
            value=float(st.session_state.data["ratio_tirs_arretes_A"]),  
            key="ratio_tirs_arretes_A"  
        )  
    with col_b:  
        st.session_state.data["clean_sheets_gardien_B"] = st.number_input(  
            "🛡️ Clean sheets du gardien (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["clean_sheets_gardien_B"]),  
            key="clean_sheets_gardien_B"  
        )  
        st.session_state.data["ratio_tirs_arretes_B"] = st.number_input(  
            "🎯 Ratio de tirs arrêtés (Équipe B)",  
            min_value=0.0,  
            max_value=1.0,  
            value=float(st.session_state.data["ratio_tirs_arretes_B"]),  
            key="ratio_tirs_arretes_B"  
        )  

    # Performances à domicile et à l'extérieur  
    st.subheader("🏠 Performances à Domicile et à l'Extérieur")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["victoires_domicile_A"] = st.number_input(  
            "🏆 Pourcentage de victoires à domicile (Équipe A)",  
            min_value=0.0,  
            max_value=100.0,  
            value=float(st.session_state.data["victoires_domicile_A"]),  
            key="victoires_domicile_A"  
        )  
    with col_b:  
        st.session_state.data["victoires_exterieur_B"] = st.number_input(  
            "🏆 Pourcentage de victoires à l'extérieur (Équipe B)",  
            min_value=0.0,  
            max_value=100.0,  
            value=float(st.session_state.data["victoires_exterieur_B"]),  
            key="victoires_exterieur_B"  
        )  

    # Style de jeu  
    st.subheader("🎯 Style de Jeu")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["passes_longues_A"] = st.number_input(  
            "📏 Passes longues par match (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["passes_longues_A"]),  
            key="passes_longues_A"  
        )  
        st.session_state.data["dribbles_reussis_A"] = st.number_input(  
            "🏃 Dribbles réussis par match (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["dribbles_reussis_A"]),  
            key="dribbles_reussis_A"  
        )  
    with col_b:  
        st.session_state.data["passes_longues_B"] = st.number_input(  
            "📏 Passes longues par match (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["passes_longues_B"]),  
            key="passes_longues_B"  
        )  
        st.session_state.data["dribbles_reussis_B"] = st.number_input(  
            "🏃 Dribbles réussis par match (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["dribbles_reussis_B"]),  
            key="dribbles_reussis_B"  
        )  

    # Efficacité offensive  
    st.subheader("⚽ Efficacité Offensive")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["ratio_tirs_cadres_A"] = st.number_input(  
            "🎯 Ratio de tirs cadrés (Équipe A)",  
            min_value=0.0,  
            max_value=1.0,  
            value=float(st.session_state.data["ratio_tirs_cadres_A"]),  
            key="ratio_tirs_cadres_A"  
        )  
        st.session_state.data["grandes_chances_manquees_A"] = st.number_input(  
            "❌ Grandes chances manquées (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["grandes_chances_manquees_A"]),  
            key="grandes_chances_manquees_A"  
        )  
    with col_b:  
        st.session_state.data["ratio_tirs_cadres_B"] = st.number_input(  
            "🎯 Ratio de tirs cadrés (Équipe B)",  
            min_value=0.0,  
            max_value=1.0,  
            value=float(st.session_state.data["ratio_tirs_cadres_B"]),  
            key="ratio_tirs_cadres_B"  
        )  
        st.session_state.data["grandes_chances_manquees_B"] = st.number_input(  
            "❌ Grandes chances manquées (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["grandes_chances_manquees_B"]),  
            key="grandes_chances_manquees_B"  
        )  

    # Efficacité défensive  
    st.subheader("🛡️ Efficacité Défensive")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["fautes_zones_dangereuses_A"] = st.number_input(  
            "⚠️ Fautes dans des zones dangereuses (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["fautes_zones_dangereuses_A"]),  
            key="fautes_zones_dangereuses_A"  
        )  
    with col_b:  
        st.session_state.data["fautes_zones_dangereuses_B"] = st.number_input(  
            "⚠️ Fautes dans des zones dangereuses (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["fautes_zones_dangereuses_B"]),  
            key="fautes_zones_dangereuses_B"  
        )  

    # Impact des corners  
    st.subheader("🎯 Impact des Corners")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["buts_corners_A"] = st.number_input(  
            "⚽ Buts marqués sur corners (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["buts_corners_A"]),  
            key="buts_corners_A"  
        )  
    with col_b:  
        st.session_state.data["buts_corners_B"] = st.number_input(  
            "⚽ Buts marqués sur corners (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["buts_corners_B"]),  
            key="buts_corners_B"  
        )  

    # Fatigue de l'équipe  
    st.subheader("⏱️ Fatigue de l'Équipe")  
    col_a, col_b = st.columns(2)  
    with col_a:  
        st.session_state.data["jours_repos_A"] = st.number_input(  
            "⏳ Jours de repos (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["jours_repos_A"]),  
            key="jours_repos_A"  
        )  
        st.session_state.data["matchs_30_jours_A"] = st.number_input(  
            "📅 Matchs sur les 30 derniers jours (Équipe A)",  
            min_value=0,  
            value=int(st.session_state.data["matchs_30_jours_A"]),  
            key="matchs_30_jours_A"  
        )  
    with col_b:  
        st.session_state.data["jours_repos_B"] = st.number_input(  
            "⏳ Jours de repos (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["jours_repos_B"]),  
            key="jours_repos_B"  
        )  
        st.session_state.data["matchs_30_jours_B"] = st.number_input(  
            "📅 Matchs sur les 30 derniers jours (Équipe B)",  
            min_value=0,  
            value=int(st.session_state.data["matchs_30_jours_B"]),  
            key="matchs_30_jours_B"  
        )  

with tab3:  
    st.subheader("🔮 Prédictions")  

    # Ajustement des buts attendus pour la méthode de Poisson  
    avg_goals_A = st.session_state.data["expected_but_A"]  
    avg_goals_B = st.session_state.data["expected_but_B"]  

    # Ajustement en fonction des blessures et des clean sheets  
    avg_goals_A *= (1 - st.session_state.data["joueurs_cles_absents_A"] * 0.1) * (st.session_state.data["clean_sheets_gardien_A"] / 5)  
    avg_goals_B *= (1 - st.session_state.data["joueurs_cles_absents_B"] * 0.1) * (st.session_state.data["clean_sheets_gardien_B"] / 5)  

    # Ajustement en fonction des jours de repos  
    avg_goals_A *= (st.session_state.data["jours_repos_A"] / 7)  
    avg_goals_B *= (st.session_state.data["jours_repos_B"] / 7)  

    # Calcul des probabilités avec la méthode de Poisson  
    prob_A = poisson.pmf(0, avg_goals_A) * poisson.pmf(0, avg_goals_B)  # Probabilité de 0-0  
    prob_B = poisson.pmf(1, avg_goals_A) * poisson.pmf(1, avg_goals_B)  # Probabilité de 1-1  
    prob_C = poisson.pmf(2, avg_goals_A) * poisson.pmf(2, avg_goals_B)  # Probabilité de 2-2  

    # Affichage des résultats  
    st.write(f"📊 Probabilité de 0-0 : {prob_A:.2%}")  
    st.write(f"📊 Probabilité de 1-1 : {prob_B:.2%}")  
    st.write(f"📊 Probabilité de 2-2 : {prob_C:.2%}")  

    # Prédiction avec la régression logistique  
    X = np.array([  
        [  
            st.session_state.data["score_rating_A"],  
            st.session_state.data["score_rating_B"],  
            st.session_state.data["buts_par_match_A"],  
            st.session_state.data["buts_par_match_B"],  
            st.session_state.data["possession_moyenne_A"],  
            st.session_state.data["possession_moyenne_B"],  
            st.session_state.data["expected_but_A"],  
            st.session_state.data["expected_but_B"],  
            st.session_state.data["tirs_cadres_A"],  
            st.session_state.data["tirs_cadres_B"],  
            st.session_state.data["grandes_chances_A"],  
            st.session_state.data["grandes_chances_B"],  
            st.session_state.data["passes_reussies_A"],  
            st.session_state.data["passes_reussies_B"],  
            st.session_state.data["corners_A"],  
            st.session_state.data["corners_B"],  
            st.session_state.data["interceptions_A"],  
            st.session_state.data["interceptions_B"],  
            st.session_state.data["tacles_reussis_A"],  
            st.session_state.data["tacles_reussis_B"],  
            st.session_state.data["fautes_A"],  
            st.session_state.data["fautes_B"],  
            st.session_state.data["cartons_jaunes_A"],  
            st.session_state.data["cartons_jaunes_B"],  
            st.session_state.data["cartons_rouges_A"],  
            st.session_state.data["cartons_rouges_B"],  
            st.session_state.data["joueurs_cles_absents_A"],  
            st.session_state.data["joueurs_cles_absents_B"],  
            st.session_state.data["motivation_A"],  
            st.session_state.data["motivation_B"],  
            st.session_state.data["clean_sheets_gardien_A"],  
            st.session_state.data["clean_sheets_gardien_B"],  
            st.session_state.data["ratio_tirs_arretes_A"],  
            st.session_state.data["ratio_tirs_arretes_B"],  
            st.session_state.data["victoires_domicile_A"],  
            st.session_state.data["victoires_exterieur_B"],  
            st.session_state.data["passes_longues_A"],  
            st.session_state.data["passes_longues_B"],  
            st.session_state.data["dribbles_reussis_A"],  
            st.session_state.data["dribbles_reussis_B"],  
            st.session_state.data["ratio_tirs_cadres_A"],  
            st.session_state.data["ratio_tirs_cadres_B"],  
            st.session_state.data["grandes_chances_manquees_A"],  
            st.session_state.data["grandes_chances_manquees_B"],  
            st.session_state.data["fautes_zones_dangereuses_A"],  
            st.session_state.data["fautes_zones_dangereuses_B"],  
            st.session_state.data["buts_corners_A"],  
            st.session_state.data["buts_corners_B"],  
            st.session_state.data["jours_repos_A"],  
            st.session_state.data["jours_repos_B"],  
            st.session_state.data["matchs_30_jours_A"],  
            st.session_state.data["matchs_30_jours_B"],  
        ]  
    ])  

    # Modèle de régression logistique  
    model_lr = LogisticRegression()  
    model_lr.fit(X, np.array([1]))  # Exemple d'entraînement (à adapter avec des données réelles)  
    prediction_lr = model_lr.predict(X)  

    # Modèle Random Forest  
    model_rf = RandomForestClassifier()  
    model_rf.fit(X, np.array([1]))  # Exemple d'entraînement (à adapter avec des données réelles)  
    prediction_rf = model_rf.predict(X)  

    # Affichage des prédictions  
    if prediction_lr[0] == 1 and prediction_rf[0] == 1:  
        st.success("🎉 Prédiction : L'Équipe A gagne !")  
    elif prediction_lr[0] == 0 and prediction_rf[0] == 0:  
        st.success("🎉 Prédiction : L'Équipe B gagne !")  
    else:  
        st.warning("🤔 Prédiction : Match Nul ou Résultat Incertain")  

with tab4:  
    st.subheader("🛠️ Outils de Paris")  

    # Convertisseur de cotes  
    st.header("🧮 Convertisseur de Cotes")  
    cote_decimale = st.number_input("Cote décimale", min_value=1.01, value=2.0)  
    probabilite_implicite = cotes_vers_probabilite(cote_decimale)  
    st.write(f"📊 Probabilité implicite : {probabilite_implicite:.2%}")  

    # Calcul de la mise de Kelly  
    st.header("💰 Calcul de la Mise de Kelly")  
    bankroll = st.number_input("Bankroll (€)", min_value=1.0, value=100.0)  
    kelly_fraction = st.number_input("Fraction de Kelly (0 à 1)", min_value=0.0, max_value=1.0, value=0.5)  
    mise_kelly = kelly_criterion(cote_decimale, probabilite_implicite, bankroll, kelly_fraction)  
    st.write(f"💶 Mise de Kelly recommandée : {mise_kelly:.2f} €")
