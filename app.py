import streamlit as st  
import numpy as np  
from scipy.stats import poisson  

# Création des onglets  
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(  
    ["📊 Statistiques", "🌦️ Conditions et Motivation", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise", "📝 Formulaire et Prédiction"]  
)  

# Initialisation des données dans st.session_state  
if "data" not in st.session_state:  
    st.session_state.data = {  
        # Critères d'attaque  
        "tirs_cadres_par_match_A": 0.45,  
        "grandes_chances_A": 25,  # Assurez-vous que c'est un entier  
        "passes_reussies_par_match_A": 15.0,  
        "centres_reussies_par_match_A": 5.5,  
        "dribbles_reussis_par_match_A": 15.0,  
        "tirs_cadres_par_match_B": 0.35,  
        "grandes_chances_B": 20,  # Assurez-vous que c'est un entier  
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
        "face_a_face_A": 3,  # Nombre de victoires de l'Équipe A  
        "face_a_face_B": 2,  # Nombre de victoires de l'Équipe B  
        # Cotes et bankroll  
        "cote_victoire_X": 2.0,  
        "cote_nul": 3.0,  
        "cote_victoire_Z": 4.0,  
        "bankroll": 1000.0,  
    }  

# Onglet 1 : Statistiques  
with tab1:  
    st.header("📊 Statistiques des Équipes")  
    col_a, col_b = st.columns(2)  

    # Statistiques de l'Équipe A  
    with col_a:  
        st.subheader("Équipe A")  
        st.session_state.data["grandes_chances_A"] = st.slider(  
            "Grandes Chances (A)",  
            min_value=0,  
            max_value=50,  
            value=int(st.session_state.data["grandes_chances_A"]),  # Convertir en entier  
        )
    # Statistiques de l'Équipe A  
    with col_a:  
        st.subheader("Équipe A")  
        st.session_state.data["tirs_cadres_par_match_A"] = st.slider("Tirs Cadres par Match (A)", 0.0, 1.0, st.session_state.data["tirs_cadres_par_match_A"])  
        st.session_state.data["grandes_chances_A"] = st.slider("Grandes Chances (A)", 0, 50, st.session_state.data["grandes_chances_A"])  
        st.session_state.data["passes_reussies_par_match_A"] = st.slider("Passes Réussies par Match (A)", 0.0, 50.0, st.session_state.data["passes_reussies_par_match_A"])  
        st.session_state.data["centres_reussies_par_match_A"] = st.slider("Centres Réussies par Match (A)", 0.0, 10.0, st.session_state.data["centres_reussies_par_match_A"])  
        st.session_state.data["dribbles_reussis_par_match_A"] = st.slider("Dribbles Réussis par Match (A)", 0.0, 20.0, st.session_state.data["dribbles_reussis_par_match_A"])  
        st.session_state.data["buts_attendus_concedes_A"] = st.slider("Buts Attendus Concedés (A)", 0.0, 5.0, st.session_state.data["buts_attendus_concedes_A"])  
        st.session_state.data["interceptions_A"] = st.slider("Interceptions (A)", 0.0, 50.0, st.session_state.data["interceptions_A"])  
        st.session_state.data["tacles_reussis_par_match_A"] = st.slider("Tacles Réussis par Match (A)", 0.0, 50.0, st.session_state.data["tacles_reussis_par_match_A"])  
        st.session_state.data["penalties_concedees_A"] = st.slider("Penalties Concedées (A)", 0, 10, st.session_state.data["penalties_concedees_A"])  
        st.session_state.data["fautes_par_match_A"] = st.slider("Fautes par Match (A)", 0.0, 30.0, st.session_state.data["fautes_par_match_A"])  

    # Statistiques de l'Équipe B  
    with col_b:  
        st.subheader("Équipe B")  
        st.session_state.data["tirs_cadres_par_match_B"] = st.slider("Tirs Cadres par Match (B)", 0.0, 1.0, st.session_state.data["tirs_cadres_par_match_B"])  
        st.session_state.data["grandes_chances_B"] = st.slider("Grandes Chances (B)", 0, 50, st.session_state.data["grandes_chances_B"])  
        st.session_state.data["passes_reussies_par_match_B"] = st.slider("Passes Réussies par Match (B)", 0.0, 50.0, st.session_state.data["passes_reussies_par_match_B"])  
        st.session_state.data["centres_reussies_par_match_B"] = st.slider("Centres Réussies par Match (B)", 0.0, 10.0, st.session_state.data["centres_reussies_par_match_B"])  
        st.session_state.data["dribbles_reussis_par_match_B"] = st.slider("Dribbles Réussis par Match (B)", 0.0, 20.0, st.session_state.data["dribbles_reussis_par_match_B"])  
        st.session_state.data["buts_attendus_concedes_B"] = st.slider("Buts Attendus Concedés (B)", 0.0, 5.0, st.session_state.data["buts_attendus_concedes_B"])  
        st.session_state.data["interceptions_B"] = st.slider("Interceptions (B)", 0.0, 50.0, st.session_state.data["interceptions_B"])  
        st.session_state.data["tacles_reussis_par_match_B"] = st.slider("Tacles Réussis par Match (B)", 0.0, 50.0, st.session_state.data["tacles_reussis_par_match_B"])  
        st.session_state.data["penalties_concedees_B"] = st.slider("Penalties Concedées (B)", 0, 10, st.session_state.data["penalties_concedees_B"])  
        st.session_state.data["fautes_par_match_B"] = st.slider("Fautes par Match (B)", 0.0, 30.0, st.session_state.data["fautes_par_match_B"])  

# Onglet 2 : Conditions et Motivation  
with tab2:  
    st.header("🌦️ Conditions et Motivation")  
    st.write("Cette section permet d'analyser les conditions du match et la motivation des équipes.")  
    st.session_state.data["face_a_face_A"] = st.slider("Victoires de l'Équipe A (Face-à-Face)", 0, 10, st.session_state.data["face_a_face_A"])  
    st.session_state.data["face_a_face_B"] = st.slider("Victoires de l'Équipe B (Face-à-Face)", 0, 10, st.session_state.data["face_a_face_B"])  

# Onglet 3 : Prédictions  
with tab3:  
    st.header("🔮 Prédictions")  
    st.write("Cette section prédit le gagnant entre l'Équipe A et l'Équipe B en utilisant la distribution de Poisson.")  

    # Prédiction des buts avec la distribution de Poisson  
    buts_attendus_A = st.session_state.data["buts_attendus_concedes_B"]  # Buts attendus de l'Équipe A  
    buts_attendus_B = st.session_state.data["buts_attendus_concedes_A"]  # Buts attendus de l'Équipe B  

    # Simulation des buts avec Poisson  
    buts_A = poisson.rvs(buts_attendus_A, size=1000)  
    buts_B = poisson.rvs(buts_attendus_B, size=1000)  

    # Calcul des probabilités de victoire, match nul et défaite  
    victoire_A = np.mean(buts_A > buts_B)  
    victoire_B = np.mean(buts_B > buts_A)  
    match_nul = np.mean(buts_A == buts_B)  

    # Affichage des résultats  
    st.write("📊 **Résultat de la Distribution de Poisson**")  
    col1, col2, col3 = st.columns(3)  
    with col1:  
        st.metric("Probabilité de victoire de l'Équipe A", f"{victoire_A:.2%}")  
    with col2:  
        st.metric("Probabilité de victoire de l'Équipe B", f"{victoire_B:.2%}")  
    with col3:  
        st.metric("Probabilité de match nul", f"{match_nul:.2%}")  

    # Prédiction des buts en nombre entier avec leur pourcentage  
    st.write("📊 **Prédiction des Buts avec Poisson**")  
    col4, col5 = st.columns(2)  
    with col4:  
        st.metric("Buts prédits pour l'Équipe A", f"{int(np.mean(buts_A))}", f"{np.mean(buts_A):.2%}")  
    with col5:  
        st.metric("Buts prédits pour l'Équipe B", f"{int(np.mean(buts_B))}", f"{np.mean(buts_B):.2%}")  

    # Gagnant final  
    st.write("🏆 **Gagnant Prédit**")  
    if victoire_A > victoire_B and victoire_A > match_nul:  
        st.success("Équipe A")  
    elif victoire_B > victoire_A and victoire_B > match_nul:  
        st.success("Équipe B")  
    else:  
        st.success("Match Nul")  

# Onglet 4 : Cotes et Value Bet  
with tab4:  
    st.header("🎰 Cotes et Value Bet")  
    st.write("Cette section analyse les cotes et les opportunités de value bet.")  
    st.session_state.data["cote_victoire_X"] = st.number_input("Cote Victoire X", value=st.session_state.data["cote_victoire_X"])  
    st.session_state.data["cote_nul"] = st.number_input("Cote Nul", value=st.session_state.data["cote_nul"])  
    st.session_state.data["cote_victoire_Z"] = st.number_input("Cote Victoire Z", value=st.session_state.data["cote_victoire_Z"])  

# Onglet 5 : Système de Mise  
with tab5:  
    st.header("💰 Système de Mise")  
    st.write("Cette section permet de configurer et de gérer le système de mise.")  
    st.session_state.data["bankroll"] = st.number_input("Bankroll", value=st.session_state.data["bankroll"])  

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

# Onglet 6 : Formulaire et Prédiction  
with tab6:  
    st.header("📝 Formulaire et Prédiction")  
    st.write("Cette section permet de saisir manuellement les critères pour prédire les buts en utilisant le modèle Poisson.")  

    # Formulaire pour saisir les critères  
    st.subheader("📋 Saisie des Critères")  
    col_a, col_b = st.columns(2)  

    # Critères pour l'Équipe A  
    with col_a:  
        st.write("**Équipe A**")  
        tirs_cadres_par_match_A = st.slider("Tirs Cadres par Match (A)", 0.0, 1.0, st.session_state.data["tirs_cadres_par_match_A"])  
        grandes_chances_A = st.slider("Grandes Chances (A)", 0, 50, st.session_state.data["grandes_chances_A"])  
        passes_reussies_par_match_A = st.slider("Passes Réussies par Match (A)", 0.0, 50.0, st.session_state.data["passes_reussies_par_match_A"])  
        centres_reussies_par_match_A = st.slider("Centres Réussies par Match (A)", 0.0, 10.0, st.session_state.data["centres_reussies_par_match_A"])  
        dribbles_reussis_par_match_A = st.slider("Dribbles Réussis par Match (A)", 0.0, 20.0, st.session_state.data["dribbles_reussis_par_match_A"])  

    # Critères pour l'Équipe B  
    with col_b:  
        st.write("**Équipe B**")  
        tirs_cadres_par_match_B = st.slider("Tirs Cadres par Match (B)", 0.0, 1.0, st.session_state.data["tirs_cadres_par_match_B"])  
        grandes_chances_B = st.slider("Grandes Chances (B)", 0, 50, st.session_state.data["grandes_chances_B"])  
        passes_reussies_par_match_B = st.slider("Passes Réussies par Match (B)", 0.0, 50.0, st.session_state.data["passes_reussies_par_match_B"])  
        centres_reussies_par_match_B = st.slider("Centres Réussies par Match (B)", 0.0, 10.0, st.session_state.data["centres_reussies_par_match_B"])  
        dribbles_reussis_par_match_B = st.slider("Dribbles Réussis par Match (B)", 0.0, 20.0, st.session_state.data["dribbles_reussis_par_match_B"])  

    # Bouton pour lancer la prédiction  
    if st.button("🔮 Lancer la Prédiction"):  
        # Prédiction des buts avec la distribution de Poisson  
        buts_attendus_A = st.session_state.data["buts_attendus_concedes_B"]  # Buts attendus de l'Équipe A  
        buts_attendus_B = st.session_state.data["buts_attendus_concedes_A"]  # Buts attendus de l'Équipe B  

        # Simulation des buts avec Poisson  
        buts_A = poisson.rvs(buts_attendus_A, size=1000)  
        buts_B = poisson.rvs(buts_attendus_B, size=1000)  

        # Affichage des résultats  
        st.write("📊 **Prédiction des Buts avec Poisson**")  
        col1, col2 = st.columns(2)  
        with col1:  
            st.metric("Buts prédits pour l'Équipe A", f"{int(np.mean(buts_A))}", f"{np.mean(buts_A):.2%}")  
        with col2:  
                        st.metric("Buts prédits pour l'Équipe B", f"{int(np.mean(buts_B))}", f"{np.mean(buts_B):.2%}")  

        # Calcul des probabilités de victoire, match nul et défaite  
        victoire_A = np.mean(buts_A > buts_B)  
        victoire_B = np.mean(buts_B > buts_A)  
        match_nul = np.mean(buts_A == buts_B)  

        # Affichage des résultats  
        st.write("📊 **Résultat de la Distribution de Poisson**")  
        col3, col4, col5 = st.columns(3)  
        with col3:  
            st.metric("Probabilité de victoire de l'Équipe A", f"{victoire_A:.2%}")  
        with col4:  
            st.metric("Probabilité de victoire de l'Équipe B", f"{victoire_B:.2%}")  
        with col5:  
            st.metric("Probabilité de match nul", f"{match_nul:.2%}")  

        # Gagnant final  
        st.write("🏆 **Gagnant Prédit**")  
        if victoire_A > victoire_B and victoire_A > match_nul:  
            st.success("Équipe A")  
        elif victoire_B > victoire_A and victoire_B > match_nul:  
            st.success("Équipe B")  
        else:  
            st.success("Match Nul")
