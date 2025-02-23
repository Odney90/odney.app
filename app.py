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
        # Cotes et bankroll  
        "cote_victoire_X": 2.0,  
        "cote_nul": 3.0,  
        "cote_victoire_Z": 4.0,  
        "bankroll": 1000.0,  
    }  

# Création des onglets  
tab1, tab2, tab3, tab4, tab5 = st.tabs(  
    ["📊 Statistiques", "🌦️ Conditions et Motivation", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"]  
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
                key="tirs_cadres_A_input"  
            )  
            st.session_state.data["grandes_chances_A"] = st.number_input(  
                "Grandes Chances (A)",  
                value=int(st.session_state.data["grandes_chances_A"]),  
                key="grandes_chances_A_input"  
            )  
            st.session_state.data["passes_reussies_par_match_A"] = st.number_input(  
                "Passes Réussies par Match (A)",  
                value=st.session_state.data["passes_reussies_par_match_A"],  
                key="passes_reussies_A_input"  
            )  
            st.session_state.data["centres_reussies_par_match_A"] = st.number_input(  
                "Centres Réussies par Match (A)",  
                value=st.session_state.data["centres_reussies_par_match_A"],  
                key="centres_reussies_A_input"  
            )  
            st.session_state.data["dribbles_reussis_par_match_A"] = st.number_input(  
                "Dribbles Réussis par Match (A)",  
                value=st.session_state.data["dribbles_reussis_par_match_A"],  
                key="dribbles_reussis_A_input"  
            )  

        # Équipe B  
        with col2:  
            st.subheader("Équipe B")  
            st.session_state.data["tirs_cadres_par_match_B"] = st.number_input(  
                "Tirs Cadres par Match (B)",  
                value=st.session_state.data["tirs_cadres_par_match_B"],  
                key="tirs_cadres_B_input"  
            )  
            st.session_state.data["grandes_chances_B"] = st.number_input(  
                "Grandes Chances (B)",  
                value=int(st.session_state.data["grandes_chances_B"]),  
                key="grandes_chances_B_input"  
            )  
            st.session_state.data["passes_reussies_par_match_B"] = st.number_input(  
                "Passes Réussies par Match (B)",  
                value=st.session_state.data["passes_reussies_par_match_B"],  
                key="passes_reussies_B_input"  
            )  
            st.session_state.data["centres_reussies_par_match_B"] = st.number_input(  
                "Centres Réussies par Match (B)",  
                value=st.session_state.data["centres_reussies_par_match_B"],  
                key="centres_reussies_B_input"  
            )  
            st.session_state.data["dribbles_reussis_par_match_B"] = st.number_input(  
                "Dribbles Réussis par Match (B)",  
                value=st.session_state.data["dribbles_reussis_par_match_B"],  
                key="dribbles_reussis_B_input"  
            )  

        # Face-à-face  
        st.subheader("Face-à-Face")  
        st.session_state.data["face_a_face_A"] = st.number_input(  
            "Victoires de l'Équipe A (Face-à-Face)",  
            value=st.session_state.data["face_a_face_A"],  
            key="face_a_face_A_input"  
        )  
        st.session_state.data["face_a_face_B"] = st.number_input(  
            "Victoires de l'Équipe B (Face-à-Face)",  
            value=st.session_state.data["face_a_face_B"],  
            key="face_a_face_B_input"  
        )  

        # Bouton de soumission  
        st.form_submit_button("Enregistrer les données")  

# Onglet 2 : Conditions et Motivation  
with tab2:  
    st.header("🌦️ Conditions et Motivation")  
    st.write("Cette section permet d'analyser les conditions du match et la motivation des équipes.")  

# Onglet 3 : Prédictions  
with tab3:  
    st.header("🔮 Prédictions")  
    st.write("Cette section prédit le gagnant entre l'Équipe A et l'Équipe B en utilisant plusieurs méthodes de prédiction.")  

    # Préparation des données pour Random Forest  
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

    # Méthode 2 : Distribution de Poisson  
    buts_attendus_A = st.session_state.data["buts_attendus_concedes_B"]  
    buts_attendus_B = st.session_state.data["buts_attendus_concedes_A"]  
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
        st.metric("Probabilité de victoire de l'Équipe A", f"{victoire_A:.2%}")  
    with col3:  
        st.metric("Probabilité de victoire de l'Équipe B", f"{victoire_B:.2%}")  

    st.write("📊 **Prédiction des Buts avec Poisson**")  
    col4, col5 = st.columns(2)  
    with col4:  
        st.metric("Buts prédits pour l'Équipe A", f"{int(np.mean(buts_A))}", f"{np.mean(buts_A):.2%}")  
    with col5:  
        st.metric("Buts prédits pour l'Équipe B", f"{int(np.mean(buts_B))}", f"{np.mean(buts_B):.2%}")  

# Onglet 4 : Cotes et Value Bet  
with tab4:  
    st.header("🎰 Cotes et Value Bet")  
    st.write("Cette section analyse les cotes et les opportunités de value bet.")  
    st.session_state.data["cote_victoire_X"] = st.number_input(  
        "Cote Victoire Équipe A",  
        value=st.session_state.data["cote_victoire_X"],  
        key="cote_victoire_X_input"  
    )  
    st.session_state.data["cote_nul"] = st.number_input(  
        "Cote Match Nul",  
        value=st.session_state.data["cote_nul"],  
        key="cote_nul_input"  
    )  
    st.session_state.data["cote_victoire_Z"] = st.number_input(  
        "Cote Victoire Équipe B",  
        value=st.session_state.data["cote_victoire_Z"],  
        key="cote_victoire_Z_input"  
    )  

# Onglet 5 : Système de Mise  
with tab5:  
    st.header("💰 Système de Mise")  
    st.write("Cette section permet de configurer et de gérer le système de mise.")  
    st.session_state.data["bankroll"] = st.number_input(  
        "Bankroll",  
        value=st.session_state.data["bankroll"],  
        key="bankroll_input"  
    )  

    # Calcul de la Mise Kelly  
    st.write("📊 **Mise Kelly**")  
    col6, col7, col8 = st.columns(3)  
    with col6:  
        mise_kelly_A = mise_kelly(victoire_A, st.session_state.data["cote_victoire_X"], st.session_state.data["bankroll"])  
        st.metric("Mise Kelly pour l'Équipe A", f"{mise_kelly_A:.2f} €")  
    with col7:  
        mise_kelly_nul = mise_kelly(match_nul, st.session_state.data["cote_nul"], st.session_state.data["bankroll"])  
        st.metric("Mise Kelly pour le Match Nul", f"{mise_kelly_nul:.2f} €")  
    with col8:  
        mise_kelly_B = mise_kelly(victoire_B, st.session_state.data["cote_victoire_Z"], st.session_state.data["bankroll"])  
        st.metric("Mise Kelly pour l'Équipe B", f"{mise_kelly_B:.2f} €")
