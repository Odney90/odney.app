import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  

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
    forme_recente_A = st.number_input("Forme récente de l'équipe A (1 à 10)", min_value=1, max_value=10, value=7, key="forme_recente_A")  
    ratio_face_a_face_A = st.number_input("Ratio de victoires en face à face (1 à 10)", min_value=1, max_value=10, value=6, key="ratio_face_a_face_A")  

with col2:  
    equipe_B = st.text_input("Nom de l'équipe B", value="Équipe B")  
    forme_recente_B = st.number_input("Forme récente de l'équipe B (1 à 10)", min_value=1, max_value=10, value=6, key="forme_recente_B")  
    ratio_face_a_face_B = st.number_input("Ratio de victoires en face à face (1 à 10)", min_value=1, max_value=10, value=4, key="ratio_face_a_face_B")
    # Critères détaillés pour les équipes  
st.header("🔍 Critères Détaillés des Équipes")  
st.markdown("Entrez les statistiques détaillées pour chaque équipe.")  

col5, col6 = st.columns(2)  

with col5:  
    st.subheader("Équipe A")  
    buts_produits_A = st.number_input("Buts produits par match", min_value=0.0, value=1.5, step=0.1, key="buts_produits_A")  
    buts_encaisses_A = st.number_input("Buts encaissés par match", min_value=0.0, value=1.0, step=0.1, key="buts_encaisses_A")  
    total_buts_produits_A = st.number_input("Buts total produits", min_value=0, value=50, key="total_buts_produits_A")  
    total_buts_encaisses_A = st.number_input("Buts total encaissés", min_value=0, value=30, key="total_buts_encaisses_A")  
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
    buts_produits_B = st.number_input("Buts produits par match", min_value=0.0, value=1.0, step=0.1, key="buts_produits_B")  
    buts_encaisses_B = st.number_input("Buts encaissés par match", min_value=0.0, value=1.5, step=0.1, key="buts_encaisses_B")  
    total_buts_produits_B = st.number_input("Buts total produits", min_value=0, value=40, key="total_buts_produits_B")  
    total_buts_encaisses_B = st.number_input("Buts total encaissés", min_value=0, value=35, key="total_buts_encaisses_B")  
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
        # Section 4 : Prédiction du Match  
st.header("🔮 Prédiction du Match")  
st.markdown("Cliquez sur le bouton pour lancer la prédiction du match.")  

if st.button("Lancer la Prédiction"):  
    # Préparation des données pour le modèle  
    data = {  
        'forme_recente_A': forme_recente_A,  
        'forme_recente_B': forme_recente_B,  
        'ratio_face_a_face_A': ratio_face_a_face_A,  
        'ratio_face_a_face_B': ratio_face_a_face_B,  
        'buts_produits_A': buts_produits_A,  
        'buts_encaisses_A': buts_encaisses_A,  
        'total_buts_produits_A': total_buts_produits_A,  
        'total_buts_encaisses_A': total_buts_encaisses_A,  
        'poss_moyenne_A': poss_moyenne_A,  
        'xG_A': xG_A,  
        'tirs_cadres_A': tirs_cadres_A,  
        'pourcentage_tirs_convertis_A': pourcentage_tirs_convertis_A,  
        'grosses_occasions_A': grosses_occasions_A,  
        'grosses_occasions_ratees_A': grosses_occasions_ratees_A,  
        'passes_reussies_A': passes_reussies_A,  
        'passes_longues_precises_A': passes_longues_precises_A,  
        'centres_reussis_A': centres_reussis_A,  
        'penalties_obtenus_A': penalties_obtenus_A,  
        'touches_surface_adverse_A': touches_surface_adverse_A,  
        'corners_A': corners_A,  
        'corners_par_match_A': corners_par_match_A,  
        'corners_concedes_A': corners_concedes_A,  
        'xG_concedes_A': xG_concedes_A,  
        'interceptions_A': interceptions_A,  
        'tacles_reussis_A': tacles_reussis_A,  
        'degagements_A': degagements_A,  
        'possessions_recuperees_A': possessions_recuperees_A,  
        'penalties_concedes_A': penalties_concedes_A,  
        'arrets_A': arrets_A,  
        'fautes_A': fautes_A,  
        'cartons_jaunes_A': cartons_jaunes_A,  
        'cartons_rouges_A': cartons_rouges_A,  
        'absents_A': absents_A,  
        'rating_joueur_cle_A_1': ratings_A[0],  
        'rating_joueur_cle_A_2': ratings_A[1],  
        'rating_joueur_cle_A_3': ratings_A[2],  
        'rating_joueur_cle_A_4': ratings_A[3],  
        'rating_joueur_cle_A_5': ratings_A[4],  
        'buts_produits_B': buts_produits_B,  
        'buts_encaisses_B': buts_encaisses_B,  
        'total_buts_produits_B': total_buts_produits_B,  
        'total_buts_encaisses_B': total_buts_encaisses_B,  
        'poss_moyenne_B': poss_moyenne_B,  
        'xG_B': xG_B,  
        'tirs_cadres_B': tirs_cadres_B,  
        'pourcentage_tirs_convertis_B': pourcentage_tirs_convertis_B,  
        'grosses_occasions_B': grosses_occasions_B,  
        'grosses_occasions_ratees_B': grosses_occasions_ratees_B,  
        'passes_reussies_B': passes_reussies_B,  
        'passes_longues_precises_B': passes_longues_precises_B,  
        'centres_reussis_B': centres_reussis_B,  
        'penalties_obtenus_B': penalties_obtenus_B,  
        'touches_surface_adverse_B': touches_surface_adverse_B,  
        'corners_B': corners_B,  
        'corners_par_match_B': corners_par_match_B,  
        'corners_concedes_B': corners_concedes_B,  
        'xG_concedes_B': xG_concedes_B,  
        'interceptions_B': interceptions_B,  
        'tacles_reussis_B': tacles_reussis_B,  
        'degagements_B': degagements_B,  
        'possessions_recuperees_B': possessions_recuperees_B,  
        'penalties_concedes_B': penalties_concedes_B,  
        'arrets_B': arrets_B,  
        'fautes_B': fautes_B,  
        'cartons_jaunes_B': cartons_jaunes_B,  
        'cartons_rouges_B': cartons_rouges_B,  
        'absents_B': absents_B,  
        'rating_joueur_cle_B_1': ratings_B[0],  
        'rating_joueur_cle_B_2': ratings_B[1],  
        'rating_joueur_cle_B_3': ratings_B[2],  
        'rating_joueur_cle_B_4': ratings_B[3],  
        'rating_joueur_cle_B_5': ratings_B[4],  
    }  

    df = pd.DataFrame([data])  

    # Charger le modèle entraîné  
    model = LogisticRegression()  
    model.coef_ = np.load('model_coef.npy')  
    model.intercept_ = np.load('model_intercept.npy')  
    model.classes_ = np.load('model_classes.npy')  

    # Charger le scaler  
    scaler = StandardScaler()  
    scaler.mean_ = np.load('scaler_mean.npy')  
    scaler.scale_ = np.load('scaler_scale.npy')  
    scaler.var_ = np.load('scaler_var.npy')  
    scaler.n_features_in_ = np.load('scaler_n_features_in.npy')  
    scaler.feature_names_in_ = np.load('scaler_feature_names_in.npy')  

    # Standardiser les données  
    df_scaled = scaler.transform(df)  

    # Faire la prédiction  
    proba = model.predict_proba(df_scaled)  
    resultat = pd.DataFrame(proba, columns=model.classes_)  

    # Afficher les résultats  
    st.subheader("Résultats de la Prédiction :")  
    st.write(f"Probabilité de victoire pour {equipe_A}: {resultat['Victoire'].values[0]:.2f}")  
    st.write(f"Probabilité de match nul: {resultat['Nul'].values[0]:.2f}")  
    st.write(f"Probabilité de victoire pour {equipe_B}: {resultat['Défaite'].values[0]:.2f}")  

    # Prédiction du score (exemple simplifié)  
    buts_A = np.random.poisson(buts_produits_A, 1)[0]  
    buts_B = np.random.poisson(buts_produits_B, 1)[0]  

    st.subheader("Prédiction du Score (simplifiée) :")  
    st.write(f"Score probable: {equipe_A} {buts_A} - {equipe_B} {buts_B}")
    # Section 5 : Outils de Paris  
st.header("💰 Outils de Paris")  
st.markdown("Explorez les outils pour optimiser vos paris.")  

# Mise de base  
mise_base = st.number_input("Mise de base (€)", min_value=1, value=10)  

# Cotes des bookmakers  
st.subheader("Cotes des Bookmakers")  
cote_victoire_A = st.number_input(f"Cote victoire {equipe_A}", min_value=1.0, value=2.5, step=0.1)  
cote_nul = st.number_input("Cote match nul", min_value=1.0, value=3.2, step=0.1)  
cote_victoire_B = st.number_input(f"Cote victoire {equipe_B}", min_value=1.0, value=2.8, step=0.1)  

# Calcul de la valeur attendue  
st.subheader("Calcul de la Valeur Attendue")  
valeur_attendue_victoire_A = (cote_victoire_A * resultat['Victoire'].values[0] - 1) * mise_base  
valeur_attendue_nul = (cote_nul * resultat['Nul'].values[0] - 1) * mise_base  
valeur_attendue_victoire_B = (cote_victoire_B * resultat['Défaite'].values[0] - 1) * mise_base  

st.write(f"Valeur attendue victoire {equipe_A}: {valeur_attendue_victoire_A:.2f} €")  
st.write(f"Valeur attendue match nul: {valeur_attendue_nul:.2f} €")  
st.write(f"Valeur attendue victoire {equipe_B}: {valeur_attendue_victoire_B:.2f} €")  

# Analyse de risque  
st.subheader("Analyse de Risque")  
st.write("Cette section pourrait inclure une analyse plus approfondie des risques associés à chaque pari.")  

# Conseils de paris (exemple simple)  
st.subheader("Conseils de Paris")  
if valeur_attendue_victoire_A > 0 and valeur_attendue_victoire_A > valeur_attendue_nul and valeur_attendue_victoire_A > valeur_attendue_victoire_B:  
    st.write(f"Potentiel de pari intéressant sur la victoire de {equipe_A}.")  
elif valeur_attendue_nul > 0 and valeur_attendue_nul > valeur_attendue_victoire_A and valeur_attendue_nul > valeur_attendue_victoire_B:  
    st.write("Potentiel de pari intéressant sur un match nul.")  
elif valeur_attendue_victoire_B > 0 and valeur_attendue_victoire_B > valeur_attendue_victoire_A and valeur_attendue_victoire_B > valeur_attendue_nul:  
    st.write(f"Potentiel de pari intéressant sur la victoire de {equipe_B}.")  
else:  
    st.write("Aucun pari ne semble particulièrement avantageux selon les cotes et les probabilités.")
