import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  

# Fonction pour convertir les valeurs en float de manière sécurisée  
def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

# Fonction pour la Mise Kelly  
def mise_kelly(probabilite, cote, bankroll):  
    if cote <= 1:  
        return 0  
    return (probabilite * (cote - 1) - (1 - probabilite)) / (cote - 1) * bankroll  

# Initialisation des données dans st.session_state  
if "data" not in st.session_state:  
    st.session_state.data = {  
        # Critères d'attaque  
        "tirs_cadres_par_match_A": 0.45,  
        "grandes_chances_A": 25,  
        "passes_reussies_par_match_A": 15.0,  
        "centres_reussies_par_match_A": 5.5,  
        "dribbles_reussis_par_match_A": 15.0,  
        "tirs_cadres_par_match_B": 0.35,  
        "grandes_chances_B": 20,  
        "passes_reussies_par_match_B": 12.0,  
        "centres_reussies_par_match_B": 4.8,  
        "dribbles_reussis_par_match_B": 13.9,  
        # Critères de défense  
        "buts_attendus_concedes_A": 1.5,  
        "interceptions_A": 45.0,  
        "tacles_reussis_par_match_A": 40.0,  
        "penalties_concedees_A": 3,  
        "fautes_par_match_A": 18.0,  
        "buts_attendus_concedes_B": 1.8,  
        "interceptions_B": 40.0,  
        "tacles_reussis_par_match_B": 35.0,  
        "penalties_concedees_B": 5,  
        "fautes_par_match_B": 20.0,  
        # Face-à-face  
        "face_a_face_A": 3,  
        "face_a_face_B": 2,  
        # Motivation  
        "motivation_A": 8,  
        "motivation_B": 7,  
        # Cotes et bankroll  
        "cote_victoire_X": 2.0,  
        "cote_nul": 3.0,  
        "cote_victoire_Z": 4.0,  
        "bankroll": 1000.0,  
        # Nouveaux critères pour Poisson  
        "buts_marques_A": 1.8,  
        "buts_marques_B": 1.5,  
        "buts_concedes_A": 1.2,  
        "buts_concedes_B": 1.5,  
        "tirs_par_match_A": 12.5,  
        "tirs_par_match_B": 11.0,  
        "possession_moyenne_A": 55.0,  
        "possession_moyenne_B": 48.0,  
        "corners_par_match_A": 5.5,  
        "corners_par_match_B": 4.8,  
        # Forme récente des équipes (victoires, nuls, défaites sur les 5 derniers matchs)  
        "forme_recente_A_victoires": 3,  
        "forme_recente_A_nuls": 1,  
        "forme_recente_A_defaites": 1,  
        "forme_recente_B_victoires": 2,  
        "forme_recente_B_nuls": 2,  
        "forme_recente_B_defaites": 1,  
    }  

# Création des onglets  
tab1, tab2, tab3, tab4 = st.tabs(  
    ["📊 Statistiques", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"]  
)  

# Onglet 1 : Statistiques  
with tab1:  
    st.header("📊 Statistiques des Équipes")  
    st.write("Veuillez saisir les données pour chaque critère.")  

    # Formulaire pour les Équipes A et B  
    with st.form("Statistiques des Équipes"):  
        col1, col2 = st.columns(2)  

        # Équipe A  
        with col1:  
            st.subheader("🏆 Équipe A")  
            st.session_state.data["tirs_cadres_par_match_A"] = st.number_input(  
                "🎯 Tirs Cadres par Match (A)",  
                value=st.session_state.data["tirs_cadres_par_match_A"],  
                key="tirs_cadres_A_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_A"] = st.number_input(  
                "🔥 Grandes Chances (A)",  
                value=int(st.session_state.data["grandes_chances_A"]),  
                key="grandes_chances_A_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_A"] = st.number_input(  
                "🔄 Passes Réussies par Match (A)",  
                value=st.session_state.data["passes_reussies_par_match_A"],  
                key="passes_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_A"] = st.number_input(  
                "🎯 Centres Réussies par Match (A)",  
                value=st.session_state.data["centres_reussies_par_match_A"],  
                key="centres_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_A"] = st.number_input(  
                "🏃‍♂️ Dribbles Réussis par Match (A)",  
                value=st.session_state.data["dribbles_reussis_par_match_A"],  
                key="dribbles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_A"] = st.number_input(  
                "🚫 Buts Attendus Concédés (A)",  
                value=st.session_state.data["buts_attendus_concedes_A"],  
                key="buts_attendus_concedes_A_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_A"] = st.number_input(  
                "🛑 Interceptions (A)",  
                value=st.session_state.data["interceptions_A"],  
                key="interceptions_A_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_A"] = st.number_input(  
                "🦶 Tacles Réussis par Match (A)",  
                value=st.session_state.data["tacles_reussis_par_match_A"],  
                key="tacles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_A"] = st.number_input(  
                "⚠️ Penalties Concédées (A)",  
                value=int(st.session_state.data["penalties_concedees_A"]),  
                key="penalties_concedees_A_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_A"] = st.number_input(  
                "🚩 Fautes par Match (A)",  
                value=st.session_state.data["fautes_par_match_A"],  
                key="fautes_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_A"] = st.number_input(  
                "💪 Motivation (A)",  
                value=int(st.session_state.data["motivation_A"]),  
                key="motivation_A_input",  
                step=1  
            )  
            # Nouveaux critères pour Poisson  
            st.session_state.data["buts_marques_A"] = st.number_input(  
                "⚽ Buts Marqués par Match (A)",  
                value=st.session_state.data["buts_marques_A"],  
                key="buts_marques_A_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_A"] = st.number_input(  
                "🎯 Tirs par Match (A)",  
                value=st.session_state.data["tirs_par_match_A"],  
                key="tirs_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["possession_moyenne_A"] = st.number_input(  
                "⏳ Possession (%) (A)",  
                value=st.session_state.data["possession_moyenne_A"],  
                key="possession_moyenne_A_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_A"] = st.number_input(  
                "🔄 Corners par Match (A)",  
                value=st.session_state.data["corners_par_match_A"],  
                key="corners_par_match_A_input",  
                step=0.01  
            )  
            # Forme récente de l'Équipe A  
            st.subheader("📅 Forme Récente (Équipe A)")  
            st.session_state.data["forme_recente_A_victoires"] = st.number_input(  
                "✅ Victoires (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_victoires"],  
                key="forme_recente_A_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_nuls"] = st.number_input(  
                "➖ Nuls (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_nuls"],  
                key="forme_recente_A_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_defaites"] = st.number_input(  
                "❌ Défaites (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_defaites"],  
                key="forme_recente_A_defaites_input",  
                step=1  
            )  

        # Équipe B  
        with col2:  
            st.subheader("🏆 Équipe B")  
            st.session_state.data["tirs_cadres_par_match_B"] = st.number_input(  
                "🎯 Tirs Cadres par Match (B)",  
                value=st.session_state.data["tirs_cadres_par_match_B"],  
                key="tirs_cadres_B_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_B"] = st.number_input(  
                "🔥 Grandes Chances (B)",  
                value=int(st.session_state.data["grandes_chances_B"]),  
                key="grandes_chances_B_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_B"] = st.number_input(  
                "🔄 Passes Réussies par Match (B)",  
                value=st.session_state.data["passes_reussies_par_match_B"],  
                key="passes_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_B"] = st.number_input(  
                "🎯 Centres Réussies par Match (B)",  
                value=st.session_state.data["centres_reussies_par_match_B"],  
                key="centres_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_B"] = st.number_input(  
                "🏃‍♂️ Dribbles Réussis par Match (B)",  
                value=st.session_state.data["dribbles_reussis_par_match_B"],  
                key="dribbles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_B"] = st.number_input(  
                "🚫 Buts Attendus Concédés (B)",  
                value=st.session_state.data["buts_attendus_concedes_B"],  
                key="buts_attendus_concedes_B_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_B"] = st.number_input(  
                "🛑 Interceptions (B)",  
                value=st.session_state.data["interceptions_B"],  
                key="interceptions_B_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_B"] = st.number_input(  
                "🦶 Tacles Réussis par Match (B)",  
                value=st.session_state.data["tacles_reussis_par_match_B"],  
                key="tacles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_B"] = st.number_input(  
                "⚠️ Penalties Concédées (B)",  
                value=int(st.session_state.data["penalties_concedees_B"]),  
                key="penalties_concedees_B_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_B"] = st.number_input(  
                "🚩 Fautes par Match (B)",  
                value=st.session_state.data["fautes_par_match_B"],  
                key="fautes_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_B"] = st.number_input(  
                "💪 Motivation (B)",  
                value=int(st.session_state.data["motivation_B"]),  
                key="motivation_B_input",  
                step=1  
            )  
            # Nouveaux critères pour Poisson  
            st.session_state.data["buts_marques_B"] = st.number_input(  
                "⚽ Buts Marqués par Match (B)",  
                value=st.session_state.data["buts_marques_B"],  
                key="buts_marques_B_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_B"] = st.number_input(  
                "🎯 Tirs par Match (B)",  
                value=st.session_state.data["tirs_par_match_B"],  
                key="tirs_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["possession_moyenne_B"] = st.number_input(  
                "⏳ Possession (%) (B)",  
                value=st.session_state.data["possession_moyenne_B"],  
                key="possession_moyenne_B_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_B"] = st.number_input(  
                "🔄 Corners par Match (B)",  
                value=st.session_state.data["corners_par_match_B"],  
                key="corners_par_match_B_input",  
                step=0.01  
            )  
            # Forme récente de l'Équipe B  
            st.subheader("📅 Forme Récente (Équipe B)")  
            st.session_state.data["forme_recente_B_victoires"] = st.number_input(  
                "✅ Victoires (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_victoires"],  
                key="forme_recente_B_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_nuls"] = st.number_input(  
                "➖ Nuls (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_nuls"],  
                key="forme_recente_B_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_defaites"] = st.number_input(  
                "❌ Défaites (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_defaites"],  
                key="forme_recente_B_defaites_input",  
                step=1  
            )  

        # Face-à-face  
        st.subheader("🤝 Face-à-Face")  
        st.session_state.data["face_a_face_A"] = st.number_input(  
            "✅ Victoires de l'Équipe A (Face-à-Face)",  
            value=int(st.session_state.data["face_a_face_A"]),  
            key="face_a_face_A_input",  
            step=1  
        )  
        st.session_state.data["face_a_face_B"] = st.number_input(  
            "✅ Victoires de l'Équipe B (Face-à-Face)",  
            value=int(st.session_state.data["face_a_face_B"]),  
            key="face_a_face_B_input",  
            step=1  
        )  

        # Bouton de soumission  
        st.form_submit_button("💾 Enregistrer les données")  

# Onglet 2 : Prédictions  
with tab2:  
    st.header("🔮 Prédictions")  
    st.write("Cette section prédit le gagnant entre l'Équipe A et l'Équipe B en utilisant plusieurs méthodes de prédiction.")  

    # Calcul des scores de forme récente  
    score_forme_A = (  
        safe_float(st.session_state.data["forme_recente_A_victoires"]) * 3  
        + safe_float(st.session_state.data["forme_recente_A_nuls"]) * 1  
        - safe_float(st.session_state.data["forme_recente_A_defaites"]) * 2  
    )  
    score_forme_B = (  
        safe_float(st.session_state.data["forme_recente_B_victoires"]) * 3  
        + safe_float(st.session_state.data["forme_recente_B_nuls"]) * 1  
        - safe_float(st.session_state.data["forme_recente_B_defaites"]) * 2  
    )  

    # Préparation des données pour la Régression Logistique  
    X_lr = np.array([  
        [  
            safe_float(st.session_state.data["score_rating_A"]),  
            safe_float(st.session_state.data["score_rating_B"]),  
            safe_float(st.session_state.data["possession_moyenne_A"]),  
            safe_float(st.session_state.data["possession_moyenne_B"]),  
            safe_float(st.session_state.data["motivation_A"]),  
            safe_float(st.session_state.data["motivation_B"]),  
            safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
            safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
            safe_float(st.session_state.data["interceptions_A"]),  
            safe_float(st.session_state.data["interceptions_B"]),  
            score_forme_A,  # Intégration de la forme récente  
            score_forme_B,  # Intégration de la forme récente  
        ],  
        [  
            safe_float(st.session_state.data["score_rating_B"]),  
            safe_float(st.session_state.data["score_rating_A"]),  
            safe_float(st.session_state.data["possession_moyenne_B"]),  
            safe_float(st.session_state.data["possession_moyenne_A"]),  
            safe_float(st.session_state.data["motivation_B"]),  
            safe_float(st.session_state.data["motivation_A"]),  
            safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
            safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
            safe_float(st.session_state.data["interceptions_B"]),  
            safe_float(st.session_state.data["interceptions_A"]),  
            score_forme_B,  # Intégration de la forme récente  
            score_forme_A,  # Intégration de la forme récente  
        ]  
    ])  

    # Étiquettes pour la Régression Logistique  
    y_lr = np.array([0, 1])  # 0 pour l'Équipe A, 1 pour l'Équipe B  

    # Méthode 1 : Régression Logistique  
    try:  
        model_lr = LogisticRegression()  
        model_lr.fit(X_lr, y_lr)  
        prediction_lr = model_lr.predict(X_lr)[0]  
        probabilite_lr = model_lr.predict_proba(X_lr)[0][prediction_lr] * 100  
    except ValueError as e:  
        st.error(f"Erreur lors de l'entraînement de la Régression Logistique : {e}")  
        prediction_lr = "Erreur"  
        probabilite_lr = 0  
    except Exception as e:  
        st.error(f"Erreur inattendue lors de la prédiction : {e}")  
        prediction_lr = "Erreur"  
        probabilite_lr = 0  

    # Affichage des résultats de la Régression Logistique  
    st.subheader("📈 Régression Logistique")  
    st.write(f"Prédiction : {'Équipe A' if prediction_lr == 0 else 'Équipe B'}")  
    st.write(f"Probabilité : {probabilite_lr:.2f}%")  

    # Méthode 2 : Random Forest  
    try:  
        model_rf = RandomForestClassifier()  
        model_rf.fit(X_lr, y_lr)  
        prediction_rf = model_rf.predict(X_lr)[0]  
        probabilite_rf = model_rf.predict_proba(X_lr)[0][prediction_rf] * 100  
    except ValueError as e:  
        st.error(f"Erreur lors de l'entraînement de Random Forest : {e}")  
        prediction_rf = "Erreur"  
        probabilite_rf = 0  
    except Exception as e:  
        st.error(f"Erreur inattendue lors de la prédiction : {e}")  
        prediction_rf = "Erreur"  
        probabilite_rf = 0  

    # Affichage des résultats de Random Forest  
    st.subheader("🌳 Random Forest")  
    st.write(f"Prédiction : {'Équipe A' if prediction_rf == 0 else 'Équipe B'}")  
    st.write(f"Probabilité : {probabilite_rf:.2f}%")  

    # Méthode 3 : Modèle de Poisson  
    try:  
        # Calcul des probabilités de victoire, nul et défaite avec Poisson  
        lambda_A = safe_float(st.session_state.data["buts_marques_A"])  
        lambda_B = safe_float(st.session_state.data["buts_marques_B"])  
        proba_victoire_A = poisson.pmf(1, lambda_A) * poisson.pmf(0, lambda_B)  
        proba_nul = poisson.pmf(1, lambda_A) * poisson.pmf(1, lambda_B)  
        proba_victoire_B = poisson.pmf(0, lambda_A) * poisson.pmf(1, lambda_B)  

        # Normalisation des probabilités  
        total = proba_victoire_A + proba_nul + proba_victoire_B  
        proba_victoire_A /= total  
        proba_nul /= total  
        proba_victoire_B /= total  

        # Détermination de la prédiction  
        if proba_victoire_A > proba_victoire_B and proba_victoire_A > proba_nul:  
            prediction_poisson = "Équipe A"  
            probabilite_poisson = proba_victoire_A * 100  
        elif proba_victoire_B > proba_victoire_A and proba_victoire_B > proba_nul:  
            prediction_poisson = "Équipe B"  
            probabilite_poisson = proba_victoire_B * 100  
        else:  
            prediction_poisson = "Match Nul"  
            probabilite_poisson = proba_nul * 100  
    except Exception as e:  
        st.error(f"Erreur lors du calcul du modèle de Poisson : {e}")  
        prediction_poisson = "Erreur"  
        probabilite_poisson = 0  

    # Affichage des résultats du modèle de Poisson  
    st.subheader("📊 Modèle de Poisson")  
    st.write(f"Prédiction : {prediction_poisson}")  
    st.write(f"Probabilité : {probabilite_poisson:.2f}%")  

# Onglet 3 : Cotes et Value Bet  
with tab3:  
    st.header("🎰 Cotes et Value Bet")  
    st.write("Cette section calcule les cotes et identifie les value bets.")  

    # Formulaire pour les cotes  
    with st.form("Cotes et Value Bet"):  
        st.session_state.data["cote_victoire_X"] = st.number_input(  
            "🏆 Cote Victoire Équipe A",  
            value=st.session_state.data["cote_victoire_X"],  
            key="cote_victoire_X_input",  
            step=0.01  
        )  
        st.session_state.data["cote_nul"] = st.number_input(  
            "➖ Cote Match Nul",  
            value=st.session_state.data["cote_nul"],  
            key="cote_nul_input",  
            step=0.01  
        )  
        st.session_state.data["cote_victoire_Z"] = st.number_input(  
            "🏆 Cote Victoire Équipe B",  
            value=st.session_state.data["cote_victoire_Z"],  
            key="cote_victoire_Z_input",  
            step=0.01  
        )  
        st.form_submit_button("💾 Enregistrer les cotes")  

    # Calcul des value bets  
    if prediction_lr != "Erreur":  
        value_bet_A = (probabilite_lr / 100) * st.session_state.data["cote_victoire_X"] - 1  
        value_bet_nul = (proba_nul / 100) * st.session_state.data["cote_nul"] - 1  
        value_bet_B = (probabilite_poisson / 100) * st.session_state.data["cote_victoire_Z"] - 1  

        st.subheader("💎 Value Bets")  
        st.write(f"Value Bet Équipe A : {value_bet_A:.2f}")  
        st.write(f"Value Bet Match Nul : {value_bet_nul:.2f}")  
        st.write(f"Value Bet Équipe B : {value_bet_B:.2f}")  

# Onglet 4 : Système de Mise  
with tab4:  
    st.header("💰 Système de Mise")  
    st.write("Cette section calcule la mise optimale selon la méthode de Kelly.")  

    # Formulaire pour la bankroll  
    with st.form("Système de Mise"):  
        st.session_state.data["bankroll"] = st.number_input(  
            "💵 Bankroll",  
            value=st.session_state.data["bankroll"],  
            key="bankroll_input",  
            step=10.0  
        )  
        st.form_submit_button("💾 Enregistrer la bankroll")  

    # Calcul de la mise Kelly  
    if prediction_lr != "Erreur":  
        mise_A = mise_kelly(probabilite_lr / 100, st.session_state.data["cote_victoire_X"], st.session_state.data["bankroll"])  
        mise_nul = mise_kelly(proba_nul / 100, st.session_state.data["cote_nul"], st.session_state.data["bankroll"])  
        mise_B = mise_kelly(probabilite_poisson / 100, st.session_state.data["cote_victoire_Z"], st.session_state.data["bankroll"])  

        st.subheader("📊 Mise Kelly")  
        st.write(f"Mise optimale pour l'Équipe A : {mise_A:.2f} €")  
        st.write(f"Mise optimale pour le Match Nul : {mise_nul:.2f} €")  
        st.write(f"Mise optimale pour l'Équipe B : {mise_B:.2f} €")
