import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  

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
        "possession_A": 55.0,  
        "possession_B": 48.0,  
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
            st.subheader("Équipe A")  
            st.session_state.data["tirs_cadres_par_match_A"] = st.number_input(  
                "Tirs Cadres par Match (A)",  
                value=st.session_state.data["tirs_cadres_par_match_A"],  
                key="tirs_cadres_A_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_A"] = st.number_input(  
                "Grandes Chances (A)",  
                value=int(st.session_state.data["grandes_chances_A"]),  
                key="grandes_chances_A_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_A"] = st.number_input(  
                "Passes Réussies par Match (A)",  
                value=st.session_state.data["passes_reussies_par_match_A"],  
                key="passes_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_A"] = st.number_input(  
                "Centres Réussies par Match (A)",  
                value=st.session_state.data["centres_reussies_par_match_A"],  
                key="centres_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_A"] = st.number_input(  
                "Dribbles Réussis par Match (A)",  
                value=st.session_state.data["dribbles_reussis_par_match_A"],  
                key="dribbles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_A"] = st.number_input(  
                "Buts Attendus Concédés (A)",  
                value=st.session_state.data["buts_attendus_concedes_A"],  
                key="buts_attendus_concedes_A_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_A"] = st.number_input(  
                "Interceptions (A)",  
                value=st.session_state.data["interceptions_A"],  
                key="interceptions_A_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_A"] = st.number_input(  
                "Tacles Réussis par Match (A)",  
                value=st.session_state.data["tacles_reussis_par_match_A"],  
                key="tacles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_A"] = st.number_input(  
                "Penalties Concédées (A)",  
                value=int(st.session_state.data["penalties_concedees_A"]),  
                key="penalties_concedees_A_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_A"] = st.number_input(  
                "Fautes par Match (A)",  
                value=st.session_state.data["fautes_par_match_A"],  
                key="fautes_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_A"] = st.number_input(  
                "Motivation (A)",  
                value=int(st.session_state.data["motivation_A"]),  
                key="motivation_A_input",  
                step=1  
            )  
            # Nouveaux critères pour Poisson  
            st.session_state.data["buts_marques_A"] = st.number_input(  
                "Buts Marqués par Match (A)",  
                value=st.session_state.data["buts_marques_A"],  
                key="buts_marques_A_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_A"] = st.number_input(  
                "Tirs par Match (A)",  
                value=st.session_state.data["tirs_par_match_A"],  
                key="tirs_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["possession_A"] = st.number_input(  
                "Possession (%) (A)",  
                value=st.session_state.data["possession_A"],  
                key="possession_A_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_A"] = st.number_input(  
                "Corners par Match (A)",  
                value=st.session_state.data["corners_par_match_A"],  
                key="corners_par_match_A_input",  
                step=0.01  
            )  
            # Forme récente de l'Équipe A  
            st.subheader("Forme Récente (Équipe A)")  
            st.session_state.data["forme_recente_A_victoires"] = st.number_input(  
                "Victoires (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_victoires"],  
                key="forme_recente_A_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_nuls"] = st.number_input(  
                "Nuls (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_nuls"],  
                key="forme_recente_A_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_defaites"] = st.number_input(  
                "Défaites (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_defaites"],  
                key="forme_recente_A_defaites_input",  
                step=1  
            )  

        # Équipe B  
        with col2:  
            st.subheader("Équipe B")  
            st.session_state.data["tirs_cadres_par_match_B"] = st.number_input(  
                "Tirs Cadres par Match (B)",  
                value=st.session_state.data["tirs_cadres_par_match_B"],  
                key="tirs_cadres_B_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_B"] = st.number_input(  
                "Grandes Chances (B)",  
                value=int(st.session_state.data["grandes_chances_B"]),  
                key="grandes_chances_B_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_B"] = st.number_input(  
                "Passes Réussies par Match (B)",  
                value=st.session_state.data["passes_reussies_par_match_B"],  
                key="passes_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_B"] = st.number_input(  
                "Centres Réussies par Match (B)",  
                value=st.session_state.data["centres_reussies_par_match_B"],  
                key="centres_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_B"] = st.number_input(  
                "Dribbles Réussis par Match (B)",  
                value=st.session_state.data["dribbles_reussis_par_match_B"],  
                key="dribbles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_B"] = st.number_input(  
                "Buts Attendus Concédés (B)",  
                value=st.session_state.data["buts_attendus_concedes_B"],  
                key="buts_attendus_concedes_B_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_B"] = st.number_input(  
                "Interceptions (B)",  
                value=st.session_state.data["interceptions_B"],  
                key="interceptions_B_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_B"] = st.number_input(  
                "Tacles Réussis par Match (B)",  
                value=st.session_state.data["tacles_reussis_par_match_B"],  
                key="tacles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_B"] = st.number_input(  
                "Penalties Concédées (B)",  
                value=int(st.session_state.data["penalties_concedees_B"]),  
                key="penalties_concedees_B_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_B"] = st.number_input(  
                "Fautes par Match (B)",  
                value=st.session_state.data["fautes_par_match_B"],  
                key="fautes_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_B"] = st.number_input(  
                "Motivation (B)",  
                value=int(st.session_state.data["motivation_B"]),  
                key="motivation_B_input",  
                step=1  
            )  
            # Nouveaux critères pour Poisson  
            st.session_state.data["buts_marques_B"] = st.number_input(  
                "Buts Marqués par Match (B)",  
                value=st.session_state.data["buts_marques_B"],  
                key="buts_marques_B_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_B"] = st.number_input(  
                "Tirs par Match (B)",  
                value=st.session_state.data["tirs_par_match_B"],  
                key="tirs_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["possession_B"] = st.number_input(  
                "Possession (%) (B)",  
                value=st.session_state.data["possession_B"],  
                key="possession_B_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_B"] = st.number_input(  
                "Corners par Match (B)",  
                value=st.session_state.data["corners_par_match_B"],  
                key="corners_par_match_B_input",  
                step=0.01  
            )  
            # Forme récente de l'Équipe B  
            st.subheader("Forme Récente (Équipe B)")  
            st.session_state.data["forme_recente_B_victoires"] = st.number_input(  
                "Victoires (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_victoires"],  
                key="forme_recente_B_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_nuls"] = st.number_input(  
                "Nuls (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_nuls"],  
                key="forme_recente_B_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_defaites"] = st.number_input(  
                "Défaites (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_defaites"],  
                key="forme_recente_B_defaites_input",  
                step=1  
            )  

        # Face-à-face  
        st.subheader("Face-à-Face")  
        st.session_state.data["face_a_face_A"] = st.number_input(  
            "Victoires de l'Équipe A (Face-à-Face)",  
            value=int(st.session_state.data["face_a_face_A"]),  
            key="face_a_face_A_input",  
            step=1  
        )  
        st.session_state.data["face_a_face_B"] = st.number_input(  
            "Victoires de l'Équipe B (Face-à-Face)",  
            value=int(st.session_state.data["face_a_face_B"]),  
            key="face_a_face_B_input",  
            step=1  
        )  

        # Bouton de soumission  
        st.form_submit_button("Enregistrer les données")  

# Onglet 2 : Prédictions  
with tab2:  
    st.header("🔮 Prédictions")  
    st.write("Cette section prédit le gagnant entre l'Équipe A et l'Équipe B en utilisant plusieurs méthodes de prédiction.")  

    # Préparation des données pour Random Forest et Régression Logistique  
    X = pd.DataFrame([st.session_state.data])  
    y = np.random.choice([0, 1, 2], size=1)  # Utilisation de valeurs numériques pour y  

    # Méthode 1 : Random Forest  
    try:  
        model_rf = RandomForestClassifier()  
        model_rf.fit(X, y)  
        prediction_rf = model_rf.predict(X)[0]  
        probabilite_rf = model_rf.predict_proba(X)[0][prediction_rf] * 100  
    except ValueError as e:  
        st.error(f"Erreur lors de l'entraînement de Random Forest : {e}")  
        prediction_rf = "Erreur"  
        probabilite_rf = 0  

    # Méthode 2 : Régression Logistique  
    try:  
        model_lr = LogisticRegression()  
        model_lr.fit(X, y)  
        prediction_lr = model_lr.predict(X)[0]  
        probabilite_lr = model_lr.predict_proba(X)[0][prediction_lr] * 100  
    except ValueError as e:  
        st.error(f"Erreur lors de l'entraînement de la Régression Logistique : {e}")  
        prediction_lr = "Erreur"  
        probabilite_lr = 0  

       # Simulation des buts avec Poisson  
    buts_A = poisson.rvs(buts_attendus_A, size=1000)  
    buts_B = poisson.rvs(buts_attendus_B, size=1000)  
    victoire_A = np.mean(buts_A > buts_B)  
    victoire_B = np.mean(buts_B > buts_A)  
    match_nul = np.mean(buts_A == buts_B)  

    # Affichage des résultats  
    st.write("📊 **Résultats des Différentes Méthodes de Prédiction**")  
    col1, col2, col3 = st.columns(3)  
    with col1:  
        st.metric("Random Forest", f"{prediction_rf} ({probabilite_rf:.2f}%)")  
    with col2:  
        st.metric("Régression Logistique", f"{prediction_lr} ({probabilite_lr:.2f}%)")  
    with col3:  
        st.metric("Probabilité de victoire de l'Équipe A", f"{victoire_A:.2%}")  

    st.write("📊 **Prédiction des Buts avec Poisson**")  
    col4, col5 = st.columns(2)  
    with col4:  
        st.metric("Buts prédits pour l'Équipe A", f"{int(np.mean(buts_A))}", f"{np.mean(buts_A):.2%}")  
    with col5:  
        st.metric("Buts prédits pour l'Équipe B", f"{int(np.mean(buts_B))}", f"{np.mean(buts_B):.2%}")  

# Onglet 3 : Cotes et Value Bet  
with tab3:  
    st.header("🎰 Cotes et Value Bet")  
    st.write("Cette section analyse les cotes et les opportunités de value bet.")  
    st.session_state.data["cote_victoire_X"] = st.number_input(  
        "Cote Victoire Équipe A",  
        value=st.session_state.data["cote_victoire_X"],  
        key="cote_victoire_X_input",  
        step=0.01  
    )  
    st.session_state.data["cote_nul"] = st.number_input(  
        "Cote Match Nul",  
        value=st.session_state.data["cote_nul"],  
        key="cote_nul_input",  
        step=0.01  
    )  
    st.session_state.data["cote_victoire_Z"] = st.number_input(  
        "Cote Victoire Équipe B",  
        value=st.session_state.data["cote_victoire_Z"],  
        key="cote_victoire_Z_input",  
        step=0.01  
    )  

# Onglet 4 : Système de Mise  
with tab4:  
    st.header("💰 Système de Mise")  
    st.write("Cette section permet de configurer et de gérer le système de mise.")  
    st.session_state.data["bankroll"] = st.number_input(  
        "Bankroll",  
        value=st.session_state.data["bankroll"],  
        key="bankroll_input",  
        step=0.01  
    )
