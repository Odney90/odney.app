import streamlit as st  
import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from scipy.stats import poisson  

# Fonction pour convertir les valeurs en float de manière sécurisée  
def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

# Fonction pour calculer la Mise Kelly  
def mise_kelly(probabilite, cote, bankroll):  
    if cote <= 1:  
        return 0  
    return (probabilite * (cote - 1) - (1 - probabilite)) / (cote - 1) * bankroll  

# Initialisation des données dans st.session_state  
if "data" not in st.session_state:  
    st.session_state.data = {  
        "tirs_cadres_par_match_A": 0.45,  # Exemple de données pour l'Équipe A  
        "grandes_chances_A": 25,  
        "grandes_chances_manquées_A": 40,  
        "passes_reussies_A": 400,  
        "passes_reussies_par_match_A": 15.0,  
        "passes_longues_par_match_A": 35.0,  
        "centres_réussies_par_match_A": 5.5,  
        "penelaties_obtenues_A": 4,  
        "balle_toucher_dans_la_surface_adverse_A": 50,  
        "corners_A": 55,  
        "Buts_attendus_concedes_A": 1.5,  
        "interceptions_A": 45.0,  
        "tacles_reussis_par_match_A": 40.0,  
        "dégagements_par_match_A": 10.0,  
        "penalities_concedees_A": 3,  
        "tirs_arretes_par_match_A": 0.75,  
        "fautes_par_match_A": 18.0,  
        "cartons_jaunes_A": 5,  
        "cartons_rouges_A": 1,  
        "dribbles_reussis_par_match_A": 15.0,  
        "nombre_de_joueurs_cles_absents_A": 1,  
        "victoires_exterieur_A": 5,  
        "jours_repos_A": 4.0,  
        "matchs-sur_30_jours_A": 8,  
        "motivation_A": 4,  
        "tirs_cadres_par_match_B": 0.35,  
        "grandes_chances_B": 20,  
        "grandes_chances_manquées_B": 50,  
        "passes_reussies_B": 350,  
        "passes_reussies_par_match_B": 12.0,  
        "passes_longues_par_match_B": 40.9,  
        "centres_réussies_par_match_B": 4.8,  
        "penelaties_obtenues_B": 5,  
        "balle_toucher_dans_la_surface_adverse_B": 48,  
        "corners_B": 50,  
        "Buts_attendus_concedes_B": 1.8,  
        "interceptions_B": 40.0,  
        "tacles_reussis_par_match_B": 35.0,  
        "dégagements_par_match_B": 12.6,  
        "penalities_concedees_B": 5,  
        "tirs_arretes_par_match_B": 0.65,  
        "fautes_par_match_B": 20.0,  
        "cartons_jaunes_B": 6,  
        "cartons_rouges_B": 2,  
        "dribbles_reussis_par_match_B": 13.9,  
        "nombre_de_joueurs_cles_absents_B": 0,  
        "victoires_exterieur_B": 4,  
        "jours_repos_B": 3.0,  
        "matchs-sur_30_jours_B": 9,  
        "motivation_B": 3,  
        "cote_victoire_X": 2.0,  
        "cote_nul": 3.0,  
        "cote_victoire_Z": 4.0,  
        "bankroll": 1000.0,  
        "conditions_match": "",  
        "face_a_face": "",  
        "forme_recente_A": ["V", "V", "V", "V", "V"],  # Données fictives pour l'Équipe A  
        "forme_recente_B": ["D", "N", "V", "D", "V"],  # Données fictives pour l'Équipe B  
    }  

# Création des onglets  
tab1, tab2, tab3, tab4, tab5 = st.tabs(  
    ["📊 Statistiques", "🌦️ Conditions et Motivation", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"]  
)  

# Onglet 1 : Statistiques  
with tab1:  
    st.header("📊 Statistiques des Équipes")  
    col_a, col_b = st.columns(2)  

    # Statistiques de l'Équipe A  
    with col_a:  
        st.subheader("Équipe A")  
        for key in st.session_state.data:  
            if key.endswith("_A"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    value=safe_float(st.session_state.data[key]),  
                    min_value=-1e6,  
                    max_value=1e6,  
                )  

    # Statistiques de l'Équipe B  
    with col_b:  
        st.subheader("Équipe B")  
        for key in st.session_state.data:  
            if key.endswith("_B"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    value=safe_float(st.session_state.data[key]),  
                    min_value=-1e6,  
                    max_value=1e6,  
                )  

# Onglet 2 : Conditions et Motivation  
with tab2:  
    st.header("🌦️ Conditions et Motivation")  
    st.write("Cette section permet d'analyser les conditions du match et la motivation des équipes.")  
    st.session_state.data["conditions_match"] = st.text_input("Conditions du Match", value=st.session_state.data["conditions_match"])  
    st.session_state.data["face_a_face"] = st.text_input("Historique Face-à-Face", value=st.session_state.data["face_a_face"])  

# Onglet 3 : Prédictions  
with tab3:  
    st.header("🔮 Prédictions")  
    st.write("Cette section prédit le gagnant entre l'Équipe A et l'Équipe B en utilisant trois méthodes : Random Forest, régression logistique et distribution de Poisson.")  

    # Caractéristiques pour les modèles  
    features_A = [  
        st.session_state.data["tirs_cadres_par_match_A"],  
        st.session_state.data["grandes_chances_A"],  
        st.session_state.data["grandes_chances_manquées_A"],  
        st.session_state.data["passes_reussies_A"],  
        st.session_state.data["passes_reussies_par_match_A"],  
        st.session_state.data["passes_longues_par_match_A"],  
        st.session_state.data["centres_réussies_par_match_A"],  
        st.session_state.data["penelaties_obtenues_A"],  
        st.session_state.data["balle_toucher_dans_la_surface_adverse_A"],  
        st.session_state.data["corners_A"],  
        st.session_state.data["Buts_attendus_concedes_A"],  
        st.session_state.data["interceptions_A"],  
        st.session_state.data["tacles_reussis_par_match_A"],  
        st.session_state.data["dégagements_par_match_A"],  
        st.session_state.data["penalities_concedees_A"],  
        st.session_state.data["tirs_arretes_par_match_A"],  
        st.session_state.data["fautes_par_match_A"],  
        st.session_state.data["cartons_jaunes_A"],  
        st.session_state.data["cartons_rouges_A"],  
        st.session_state.data["dribbles_reussis_par_match_A"],  
        st.session_state.data["nombre_de_joueurs_cles_absents_A"],  
        st.session_state.data["victoires_exterieur_A"],  
        st.session_state.data["jours_repos_A"],  
        st.session_state.data["matchs-sur_30_jours_A"],  
        st.session_state.data["motivation_A"],  
    ]  

    features_B = [  
        st.session_state.data["tirs_cadres_par_match_B"],  
        st.session_state.data["grandes_chances_B"],  
        st.session_state.data["grandes_chances_manquées_B"],  
        st.session_state.data["passes_reussies_B"],  
        st.session_state.data["passes_reussies_par_match_B"],  
        st.session_state.data["passes_longues_par_match_B"],  
        st.session_state.data["centres_réussies_par_match_B"],  
        st.session_state.data["penelaties_obtenues_B"],  
        st.session_state.data["balle_toucher_dans_la_surface_adverse_B"],  
        st.session_state.data["corners_B"],  
        st.session_state.data["Buts_attendus_concedes_B"],  
        st.session_state.data["interceptions_B"],  
        st.session_state.data["tacles_reussis_par_match_B"],  
        st.session_state.data["dégagements_par_match_B"],  
        st.session_state.data["penalities_concedees_B"],  
        st.session_state.data["tirs_arretes_par_match_B"],  
        st.session_state.data["fautes_par_match_B"],  
        st.session_state.data["cartons_jaunes_B"],  
        st.session_state.data["cartons_rouges_B"],  
        st.session_state.data["dribbles_reussis_par_match_B"],  
        st.session_state.data["nombre_de_joueurs_cles_absents_B"],  
        st.session_state.data["victoires_exterieur_B"],  
        st.session_state.data["jours_repos_B"],  
        st.session_state.data["matchs-sur_30_jours_B"],  
        st.session_state.data["motivation_B"],  
    ]  

    # Données d'entraînement fictives (à remplacer par vos données réelles)  
    X_train = np.array([features_A, features_B])  # Caractéristiques des équipes A et B  
    y_train = np.array([1, 0])  # 1 pour Équipe A, 0 pour Équipe B  

    # Entraînement du modèle Random Forest  
    model_rf = RandomForestClassifier()  
    model_rf.fit(X_train, y_train)  

    # Prédiction avec Random Forest  
    prediction_rf_A = model_rf.predict([features_A])  
    prediction_rf_B = model_rf.predict([features_B])  

    # Probabilités de victoire avec Random Forest  
    prob_victoire_rf_A = model_rf.predict_proba([features_A])[0][1]  
    prob_victoire_rf_B = model_rf.predict_proba([features_B])[0][0]  

    # Entraînement du modèle de régression logistique  
    model_lr = LogisticRegression()  
    model_lr.fit(X_train, y_train)  

    # Prédiction avec la régression logistique  
    prediction_lr_A = model_lr.predict([features_A])  
    prediction_lr_B = model_lr.predict([features_B])  

    # Probabilités de victoire avec régression logistique  
    prob_victoire_lr_A = model_lr.predict_proba([features_A])[0][1]  
    prob_victoire_lr_B = model_lr.predict_proba([features_B])[0][0]  

    # Prédiction des buts avec la distribution de Poisson  
    buts_attendus_A = st.session_state.data["Buts_attendus_concedes_B"]  # Buts attendus de l'Équipe A  
    buts_attendus_B = st.session_state.data["Buts_attendus_concedes_A"]  # Buts attendus de l'Équipe B  

    # Simulation des buts avec Poisson  
    buts_A = poisson.rvs(buts_attendus_A, size=1000)  
    buts_B = poisson.rvs(buts_attendus_B, size=1000)  

    # Calcul des probabilités de victoire, match nul et défaite  
    victoire_A = np.mean(buts_A > buts_B)  
    victoire_B = np.mean(buts_B > buts_A)  
    match_nul = np.mean(buts_A == buts_B)  

    # Affichage des résultats  
    st.write("📊 **Résultat du Random Forest**")  
    col1, col2 = st.columns(2)  
    with col1:  
        st.metric("Probabilité de victoire de l'Équipe A", f"{prob_victoire_rf_A:.2%}")  
    with col2:  
        st.metric("Probabilité de victoire de l'Équipe B", f"{prob_victoire_rf_B:.2%}")  

    st.write("📊 **Résultat de la Régression Logistique**")  
    col3, col4 = st.columns(2)  
    with col3:  
        st.metric("Probabilité de victoire de l'Équipe A", f"{prob_victoire_lr_A:.2%}")  
    with col4:  
        st.metric("Probabilité de victoire de l'Équipe B", f"{prob_victoire_lr_B:.2%}")  

    st.write("📊 **Résultat de la Distribution de Poisson**")  
    col5, col6, col7 = st.columns(3)  
    with col5:  
        st.metric("Probabilité de victoire de l'Équipe A", f"{victoire_A:.2%}")  
    with col6:  
        st.metric("Probabilité de victoire de l'Équipe B", f"{victoire_B:.2%}")  
    with col7:  
        st.metric("Probabilité de match nul", f"{match_nul:.2%}")  

    # Prédiction des buts en nombre entier avec leur pourcentage  
    st.write("📊 **Prédiction des Buts avec Poisson**")  
    col8, col9 = st.columns(2)  
    with col8:  
        st.metric("Buts prédits pour l'Équipe A", f"{int(np.mean(buts_A))}", f"{np.mean(buts_A):.2%}")  
    with col9:  
        st.metric("Buts prédits pour l'Équipe B", f"{int(np.mean(buts_B))}", f"{np.mean(buts_B):.2%}")  

    # Gagnant final (basé sur la combinaison de la régression logistique et de Poisson)  
    st.write("🏆 **Gagnant Prédit (Régression Logistique + Poisson)**")  
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
    col10, col11, col12 = st.columns(3)  
    with col10:  
        mise_kelly_A = mise_kelly(victoire_A, st.session_state.data["cote_victoire_X"], st.session_state.data["bankroll"])  
        st.metric("Mise Kelly pour l'Équipe A", f"{mise_kelly_A:.2f} €")  
    with col11:  
        mise_kelly_nul = mise_kelly(match_nul, st.session_state.data["cote_nul"], st.session_state.data["bankroll"])  
        st.metric("Mise Kelly pour le Match Nul", f"{mise_kelly_nul:.2f} €")  
    with col12:  
        mise_kelly_B = mise_kelly(victoire_B, st.session_state.data["cote_victoire_Z"], st.session_state.data["bankroll"])  
        st.metric("Mise Kelly pour l'Équipe B", f"{mise_kelly_B:.2f} €")

# Onglet 6 : Formulaire et Prédiction  
with tab6:  
    st.header("📝 Formulaire et Prédiction")  
    st.write("Cette section permet de saisir manuellement les critères pour prédire le résultat en utilisant la régression logistique et la distribution de Poisson.")  

    # Formulaire pour saisir les critères  
    st.subheader("📋 Saisie des Critères")  
    col_a, col_b = st.columns(2)  

    # Critères pour l'Équipe A  
    with col_a:  
        st.write("**Équipe A**")  
        tirs_cadres_par_match_A = st.number_input("Tirs Cadres par Match (A)", value=0.45)  
        grandes_chances_A = st.number_input("Grandes Chances (A)", value=25)  
        grandes_chances_manquees_A = st.number_input("Grandes Chances Manquées (A)", value=40)  
        passes_reussies_A = st.number_input("Passes Réussies (A)", value=400)  
        passes_reussies_par_match_A = st.number_input("Passes Réussies par Match (A)", value=15.0)  
        passes_longues_par_match_A = st.number_input("Passes Longues par Match (A)", value=35.0)  
        centres_reussies_par_match_A = st.number_input("Centres Réussies par Match (A)", value=5.5)  
        penalties_obtenues_A = st.number_input("Penalties Obtenues (A)", value=4)  
        balle_toucher_surface_adverse_A = st.number_input("Balle Toucher dans la Surface Adverse (A)", value=50)  
        corners_A = st.number_input("Corners (A)", value=55)  
        buts_attendus_concedes_A = st.number_input("Buts Attendus Concedés (A)", value=1.5)  
        interceptions_A = st.number_input("Interceptions (A)", value=45.0)  
        tacles_reussis_par_match_A = st.number_input("Tacles Réussis par Match (A)", value=40.0)  
        degagements_par_match_A = st.number_input("Dégagements par Match (A)", value=10.0)  
        penalties_concedees_A = st.number_input("Penalties Concedées (A)", value=3)  
        tirs_arretes_par_match_A = st.number_input("Tirs Arrêtés par Match (A)", value=0.75)  
        fautes_par_match_A = st.number_input("Fautes par Match (A)", value=18.0)  
        cartons_jaunes_A = st.number_input("Cartons Jaunes (A)", value=5)  
        cartons_rouges_A = st.number_input("Cartons Rouges (A)", value=1)  
        dribbles_reussis_par_match_A = st.number_input("Dribbles Réussis par Match (A)", value=15.0)  
        nombre_joueurs_cles_absents_A = st.number_input("Nombre de Joueurs Clés Absents (A)", value=1)  
        victoires_exterieur_A = st.number_input("Victoires à l'Extérieur (A)", value=5)  
        jours_repos_A = st.number_input("Jours de Repos (A)", value=4.0)  
        matchs_sur_30_jours_A = st.number_input("Matchs sur 30 Jours (A)", value=8)  
        motivation_A = st.number_input("Motivation (A)", value=4)  

    # Critères pour l'Équipe B  
    with col_b:  
        st.write("**Équipe B**")  
        tirs_cadres_par_match_B = st.number_input("Tirs Cadres par Match (B)", value=0.35)  
        grandes_chances_B = st.number_input("Grandes Chances (B)", value=20)  
        grandes_chances_manquees_B = st.number_input("Grandes Chances Manquées (B)", value=50)  
        passes_reussies_B = st.number_input("Passes Réussies (B)", value=350)  
        passes_reussies_par_match_B = st.number_input("Passes Réussies par Match (B)", value=12.0)  
        passes_longues_par_match_B = st.number_input("Passes Longues par Match (B)", value=40.9)  
        centres_reussies_par_match_B = st.number_input("Centres Réussies par Match (B)", value=4.8)  
        penalties_obtenues_B = st.number_input("Penalties Obtenues (B)", value=5)  
        balle_toucher_surface_adverse_B = st.number_input("Balle Toucher dans la Surface Adverse (B)", value=48)  
        corners_B = st.number_input("Corners (B)", value=50)  
        buts_attendus_concedes_B = st.number_input("Buts Attendus Concedés (B)", value=1.8)  
        interceptions_B = st.number_input("Interceptions (B)", value=40.0)  
        tacles_reussis_par_match_B = st.number_input("Tacles Réussis par Match (B)", value=35.0)  
        degagements_par_match_B = st.number_input("Dégagements par Match (B)", value=12.6)  
        penalties_concedees_B = st.number_input("Penalties Concedées (B)", value=5)  
        tirs_arretes_par_match_B = st.number_input("Tirs Arrêtés par Match (B)", value=0.65)  
        fautes_par_match_B = st.number_input("Fautes par Match (B)", value=20.0)  
        cartons_jaunes_B = st.number_input("Cartons Jaunes (B)", value=6)  
        cartons_rouges_B = st.number_input("Cartons Rouges (B)", value=2)  
        dribbles_reussis_par_match_B = st.number_input("Dribbles Réussis par Match (B)", value=13.9)  
        nombre_joueurs_cles_absents_B = st.number_input("Nombre de Joueurs Clés Absents (B)", value=0)  
        victoires_exterieur_B = st.number_input("Victoires à l'Extérieur (B)", value=4)  
        jours_repos_B = st.number_input("Jours de Repos (B)", value=3.0)  
        matchs_sur_30_jours_B = st.number_input("Matchs sur 30 Jours (B)", value=9)  
        motivation_B = st.number_input("Motivation (B)", value=3)  

    # Bouton pour lancer la prédiction  
    if st.button("🔮 Lancer la Prédiction"):  
        # Caractéristiques pour les modèles  
        features_A = [  
            tirs_cadres_par_match_A,  
            grandes_chances_A,  
            grandes_chances_manquees_A,  
            passes_reussies_A,  
            passes_reussies_par_match_A,  
            passes_longues_par_match_A,  
            centres_reussies_par_match_A,  
            penalties_obtenues_A,  
            balle_toucher_surface_adverse_A,  
            corners_A,  
            buts_attendus_concedes_A,  
            interceptions_A,  
            tacles_reussis_par_match_A,  
            degagements_par_match_A,  
            penalties_concedees_A,  
            tirs_arretes_par_match_A,  
            fautes_par_match_A,  
            cartons_jaunes_A,  
            cartons_rouges_A,  
            dribbles_reussis_par_match_A,  
            nombre_joueurs_cles_absents_A,  
            victoires_exterieur_A,  
            jours_repos_A,  
            matchs_sur_30_jours_A,  
            motivation_A,  
        ]  

        features_B = [  
            tirs_cadres_par_match_B,  
            grandes_chances_B,  
            grandes_chances_manquees_B,  
            passes_reussies_B,  
            passes_reussies_par_match_B,  
            passes_longues_par_match_B,  
            centres_reussies_par_match_B,  
            penalties_obtenues_B,  
            balle_toucher_surface_adverse_B,  
            corners_B,  
            buts_attendus_concedes_B,  
            interceptions_B,  
            tacles_reussis_par_match_B,  
            degagements_par_match_B,  
            penalties_concedees_B,  
            tirs_arretes_par_match_B,  
            fautes_par_match_B,  
            cartons_jaunes_B,  
            cartons_rouges_B,  
            dribbles_reussis_par_match_B,  
            nombre_joueurs_cles_absents_B,  
            victoires_exterieur_B,  
            jours_repos_B,  
            matchs_sur_30_jours_B,  
            motivation_B,  
        ]  

        # Prédiction avec la régression logistique  
        prediction_lr_A = model_lr.predict([features_A])  
        prediction_lr_B = model_lr.predict([features_B])  

        # Probabilités de victoire avec régression logistique  
        prob_victoire_lr_A = model_lr.predict_proba([features_A])[0][1]  
        prob_victoire_lr_B = model_lr.predict_proba([features_B])[0][0]  

        # Prédiction des buts avec la distribution de Poisson  
        buts_A = poisson.rvs(buts_attendus_concedes_B, size=1000)  
        buts_B = poisson.rvs(buts_attendus_concedes_A, size=1000)  

        # Calcul des probabilités de victoire, match nul et défaite  
        victoire_A = np.mean(buts_A > buts_B)  
        victoire_B = np.mean(buts_B > buts_A)  
        match_nul = np.mean(buts_A == buts_B)  

        # Affichage des résultats  
        st.write("📊 **Résultat de la Régression Logistique**")  
        col1, col2 = st.columns(2)  
        with col1:  
            st.metric("Probabilité de victoire de l'Équipe A", f"{prob_victoire_lr_A:.2%}")  
        with col2:  
            st.metric("Probabilité de victoire de l'Équipe B", f"{prob_victoire_lr_B:.2%}")  

        st.write("📊 **Résultat de la Distribution de Poisson**")  
        col3, col4, col5 = st.columns(3)  
        with col3:  
            st.metric("Probabilité de victoire de l'Équipe A", f"{victoire_A:.2%}")  
        with col4:  
            st.metric("Probabilité de victoire de l'Équipe B", f"{victoire_B:.2%}")  
        with col5:  
            st.metric("Probabilité de match nul", f"{match_nul:.2%}")  

        # Prédiction des buts en nombre entier avec leur pourcentage  
        st.write("📊 **Prédiction des Buts avec Poisson**")  
        col6, col7 = st.columns(2)  
        with col6:  
            st.metric("Buts prédits pour l'Équipe A", f"{int(np.mean(buts_A))}", f"{np.mean(buts_A):.2%}")  
        with col7:  
            st.metric("Buts prédits pour l'Équipe B", f"{int(np.mean(buts_B))}", f"{np.mean(buts_B):.2%}")  

        # Gagnant final (basé sur la combinaison de la régression logistique et de Poisson)  
        st.write("🏆 **Gagnant Prédit (Régression Logistique + Poisson)**")  
        if victoire_A > victoire_B and victoire_A > match_nul:  
            st.success("Équipe A")  
        elif victoire_B > victoire_A and victoire_B > match_nul:  
            st.success("Équipe B")  
        else:  
            st.success("Match Nul")
