import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression, PoissonRegressor  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import KFold, cross_val_score  
import altair as alt  

# Initialisation des données de session si elles n'existent pas  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        'score_rating_A': 85.0,  
        'score_rating_B': 78.0,  
        'buts_par_match_A': 1.8,  
        'buts_par_match_B': 1.5,  
        'buts_concedes_par_match_A': 1.2,  
        'buts_concedes_par_match_B': 1.4,  
        'possession_moyenne_A': 58.0,  
        'possession_moyenne_B': 52.0,  
        'expected_but_A': 1.7,  
        'expected_but_B': 1.4,  
        'tirs_cadres_A': 5,  
        'tirs_cadres_B': 4,  
        'grandes_chances_A': 2,  
        'grandes_chances_B': 2,  
        'victoires_domicile_A': 8,  
        'victoires_domicile_B': 6,  
        'victoires_exterieur_A': 5,  
        'victoires_exterieur_B': 4,  
        'joueurs_absents_A': 1,  
        'joueurs_absents_B': 2,  
        'moyenne_age_joueurs_A': 27.0,  
        'moyenne_age_joueurs_B': 26.5,  
        'experience_entraineur_A': 5,  
        'experience_entraineur_B': 3,  
        'nombre_blessures_A': 2,  
        'nombre_blessures_B': 1,  
        'cartons_jaunes_A': 10,  
        'cartons_jaunes_B': 8,  
        'cartons_rouges_A': 1,  
        'cartons_rouges_B': 0,  
        'taux_reussite_passes_A': 85.0,  
        'taux_reussite_passes_B': 80.0,  
        'taux_possession_A': 55.0,  
        'taux_possession_B': 50.0,  
        'nombre_tirs_totaux_A': 15,  
        'nombre_tirs_totaux_B': 12,  
        'nombre_tirs_cadres_A': 5,  
        'nombre_tirs_cadres_B': 4,  
        'ratio_buts_tirs_A': 0.12,  
        'ratio_buts_tirs_B': 0.10,  
        'ratio_buts_encais_tirs_A': 0.08,  
        'ratio_buts_encais_tirs_B': 0.09,  
        'performance_domicile_A': 2.1,  
        'performance_domicile_B': 1.8,  
        'performance_exterieur_A': 1.5,  
        'performance_exterieur_B': 1.2,  
        'historique_confrontations_A_B': 3,  
        'historique_confrontations_B_A': 1,  
        'moyenne_buts_marques_A': 2.0,  
        'moyenne_buts_marques_B': 1.5,  
        'moyenne_buts_encais_A': 1.0,  
        'moyenne_buts_encais_B': 1.2,  
        'impact_joueurs_cles_A': 0.8,  
        'impact_joueurs_cles_B': 0.7,  
        'taux_reussite_corners_A': 30.0,  
        'taux_reussite_corners_B': 25.0,  
        'nombre_degagements_A': 15,  # Nouveau champ pour le nombre de dégagements  
        'nombre_degagements_B': 12,  
        'tactique_A': 8,  # Nouveau champ pour la tactique sur une échelle de 1 à 10  
        'tactique_B': 7,  
        'moyenne_temps_possession_A': 60.0,  
        'moyenne_temps_possession_B': 55.0,  
        'motivation_A': 8,  
        'motivation_B': 7,  
        'forme_recente_A': 12,  
        'forme_recente_B': 9,  
        'historique_face_a_face_A_B': 2,  
        'historique_face_a_face_B_A': -1,  
    }  

# Formulaire flottant pour la saisie des données  
with st.form("formulaire_saisie"):  
    st.header("📊 Saisie des Données d'Analyse")  
    
    # Champs pour les données de l'équipe A  
    st.subheader("Équipe A 🏆")  
    st.session_state.data['score_rating_A'] = st.number_input("Score de Performance (Équipe A)", value=float(st.session_state.data['score_rating_A']), step=0.1)  
    st.session_state.data['buts_par_match_A'] = st.number_input("Buts par Match (Équipe A)", value=float(st.session_state.data['buts_par_match_A']), step=0.1)  
    st.session_state.data['buts_concedes_par_match_A'] = st.number_input("Buts Concedes par Match (Équipe A)", value=float(st.session_state.data['buts_concedes_par_match_A']), step=0.1)  
    st.session_state.data['possession_moyenne_A'] = st.number_input("Possession Moyenne (%) (Équipe A)", value=float(st.session_state.data['possession_moyenne_A']), step=0.1)  
    st.session_state.data['expected_but_A'] = st.number_input("Expected Goals (Équipe A)", value=float(st.session_state.data['expected_but_A']), step=0.1)  
    st.session_state.data['tirs_cadres_A'] = st.number_input("Tirs Cadres (Équipe A)", value=int(st.session_state.data['tirs_cadres_A']), step=1)  
    st.session_state.data['grandes_chances_A'] = st.number_input("Grandes Chances (Équipe A)", value=int(st.session_state.data['grandes_chances_A']), step=1)  
    st.session_state.data['victoires_domicile_A'] = st.number_input("Victoires à Domicile (Équipe A)", value=int(st.session_state.data['victoires_domicile_A']), step=1)  
    st.session_state.data['victoires_exterieur_A'] = st.number_input("Victoires à l'Extérieur (Équipe A)", value=int(st.session_state.data['victoires_exterieur_A']), step=1)  
    st.session_state.data['joueurs_absents_A'] = st.number_input("Joueurs Absents (Équipe A)", value=int(st.session_state.data['joueurs_absents_A']), step=1)  
    
    # Nouveaux champs pour l'équipe A  
    st.session_state.data['moyenne_age_joueurs_A'] = st.number_input("Moyenne d'Âge des Joueurs (Équipe A)", value=float(st.session_state.data['moyenne_age_joueurs_A']), step=0.1)  
    st.session_state.data['experience_entraineur_A'] = st.number_input("Expérience de l'Entraîneur (Années)", value=int(st.session_state.data['experience_entraineur_A']), step=1)  
    st.session_state.data['nombre_blessures_A'] = st.number_input("Nombre de Blessures (Équipe A)", value=int(st.session_state.data['nombre_blessures_A']), step=1)  
    st.session_state.data['cartons_jaunes_A'] = st.number_input("Cartons Jaunes (Équipe A)", value=int(st.session_state.data['cartons_jaunes_A']), step=1)  
    st.session_state.data['cartons_rouges_A'] = st.number_input("Cartons Rouges (Équipe A)", value=int(st.session_state.data['cartons_rouges_A']), step=1)  
    st.session_state.data['taux_reussite_passes_A'] = st.number_input("Taux de Réussite des Passes (%) (Équipe A)", value=float(st.session_state.data['taux_reussite_passes_A']), step=0.1)  
    st.session_state.data['taux_possession_A'] = st.number_input("Taux de Possession (%) (Équipe A)", value=float(st.session_state.data['taux_possession_A']), step=0.1)  
    st.session_state.data['nombre_tirs_totaux_A'] = st.number_input("Nombre de Tirs Totaux (Équipe A)", value=int(st.session_state.data['nombre_tirs_totaux_A']), step=1)  
    st.session_state.data['nombre_tirs_cadres_A'] = st.number_input("Nombre de Tirs Cadrés (Équipe A)", value=int(st.session_state.data['nombre_tirs_cadres_A']), step=1)  
    
    # Ratios avec descriptions  
    st.session_state.data['ratio_buts_tirs_A'] = st.number_input("Ratio Buts/Tirs (Équipe A)", value=float(st.session_state.data['ratio_buts_tirs_A']), step=0.01, help="Ratio des buts marqués par rapport au nombre total de tirs.")  
    st.session_state.data['ratio_buts_encais_tirs_A'] = st.number_input("Ratio Buts Encaissés/Tirs (Équipe A)", value=float(st.session_state.data['ratio_buts_encais_tirs_A']), step=0.01, help="Ratio des buts encaissés par rapport au nombre total de tirs subis.")  
    
    st.session_state.data['performance_domicile_A'] = st.number_input("Performance à Domicile (Points) (Équipe A)", value=float(st.session_state.data['performance_domicile_A']), step=0.1)  
    st.session_state.data['performance_exterieur_A'] = st.number_input("Performance à l'Extérieur (Points) (Équipe A)", value=float(st.session_state.data['performance_exterieur_A']), step=0.1)  
    st.session_state.data['historique_confrontations_A_B'] = st.number_input("Historique Confrontations (Équipe A contre Équipe B)", value=int(st.session_state.data['historique_confrontations_A_B']), step=1)  
    st.session_state.data['moyenne_buts_marques_A'] = st.number_input("Moyenne de Buts Marqués par Match (Équipe A)", value=float(st.session_state.data['moyenne_buts_marques_A']), step=0.1)  
    st.session_state.data['moyenne_buts_encais_A'] = st.number_input("Moyenne de Buts Encaissés par Match (Équipe A)", value=float(st.session_state.data['moyenne_buts_encais_A']), step=0.1)  
    st.session_state.data['impact_joueurs_cles_A'] = st.number_input("Impact des Joueurs Clés (Équipe A)", value=float(st.session_state.data['impact_joueurs_cles_A']), step=0.1)  
    st.session_state.data['taux_reussite_corners_A'] = st.number_input("Taux de Réussite des Corners (%) (Équipe A)", value=float(st.session_state.data['taux_reussite_corners_A']), step=0.1)  
    
    # Nouveau champ pour le nombre de dégagements  
    st.session_state.data['nombre_degagements_A'] = st.number_input("Nombre de Dégagements (Équipe A)", value=int(st.session_state.data['nombre_degagements_A']), step=1)  
    
    # Nouveau champ pour la tactique  
    st.session_state.data['tactique_A'] = st.number_input("Tactique (Équipe A)", value=int(st.session_state.data['tactique_A']), min_value=1, max_value=10, help="Évaluez l'efficacité de la tactique sur une échelle de 1 à 10.")  
    
    # Nouveaux champs pour la motivation et la forme récente  
    st.session_state.data['motivation_A'] = st.number_input("Motivation (Équipe A)", value=int(st.session_state.data['motivation_A']), min_value=1, max_value=10)  
    st.session_state.data['forme_recente_A'] = st.number_input("Forme Récente (Points sur 5 matchs) (Équipe A)", value=int(st.session_state.data['forme_recente_A']), step=1)  
    
    # Champs pour les données de l'équipe B  
    st.subheader("Équipe B 🥈")  
    st.session_state.data['score_rating_B'] = st.number_input("Score de Performance (Équipe B)", value=float(st.session_state.data['score_rating_B']), step=0.1)  
    st.session_state.data['buts_par_match_B'] = st.number_input("Buts par Match (Équipe B)", value=float(st.session_state.data['buts_par_match_B']), step=0.1)  
    st.session_state.data['buts_concedes_par_match_B'] = st.number_input("Buts Concedes par Match (Équipe B)", value=float(st.session_state.data['buts_concedes_par_match_B']), step=0.1)  
    st.session_state.data['possession_moyenne_B'] = st.number_input("Possession Moyenne (%) (Équipe B)", value=float(st.session_state.data['possession_moyenne_B']), step=0.1)  
    st.session_state.data['expected_but_B'] = st.number_input("Expected Goals (Équipe B)", value=float(st.session_state.data['expected_but_B']), step=0.1)  
    st.session_state.data['tirs_cadres_B'] = st.number_input("Tirs Cadres (Équipe B)", value=int(st.session_state.data['tirs_cadres_B']), step=1)  
    st.session_state.data['grandes_chances_B'] = st.number_input("Grandes Chances (Équipe B)", value=int(st.session_state.data['grandes_chances_B']), step=1)  
    st.session_state.data['victoires_domicile_B'] = st.number_input("Victoires à Domicile (Équipe B)", value=int(st.session_state.data['victoires_domicile_B']), step=1)  
    st.session_state.data['victoires_exterieur_B'] = st.number_input("Victoires à l'Extérieur (Équipe B)", value=int(st.session_state.data['victoires_exterieur_B']), step=1)  
    st.session_state.data['joueurs_absents_B'] = st.number_input("Joueurs Absents (Équipe B)", value=int(st.session_state.data['joueurs_absents_B']), step=1)  

    # Nouveaux champs pour l'équipe B  
    st.session_state.data['moyenne_age_joueurs_B'] = st.number_input("Moyenne d'Âge des Joueurs (Équipe B)", value=float(st.session_state.data['moyenne_age_joueurs_B']), step=0.1)  
    st.session_state.data['experience_entraineur_B'] = st.number_input("Expérience de l'Entraîneur (Années)", value=int(st.session_state.data['experience_entraineur_B']), step=1)  
    st.session_state.data['nombre_blessures_B'] = st.number_input("Nombre de Blessures (Équipe B)", value=int(st.session_state.data['nombre_blessures_B']), step=1)  
    st.session_state.data['cartons_jaunes_B'] = st.number_input("Cartons Jaunes (Équipe B)", value=int(st.session_state.data['cartons_jaunes_B']), step=1)  
    st.session_state.data['cartons_rouges_B'] = st.number_input("Cartons Rouges (Équipe B)", value=int(st.session_state.data['cartons_rouges_B']), step=1)  
    st.session_state.data['taux_reussite_passes_B'] = st.number_input("Taux de Réussite des Passes (%) (Équipe B)", value=float(st.session_state.data['taux_reussite_passes_B']), step=0.1)  
    st.session_state.data['taux_possession_B'] = st.number_input("Taux de Possession (%) (Équipe B)", value=float(st.session_state.data['taux_possession_B']), step=0.1)  
    st.session_state.data['nombre_tirs_totaux_B'] = st.number_input("Nombre de Tirs Totaux (Équipe B)", value=int(st.session_state.data['nombre_tirs_totaux_B']), step=1)  
    st.session_state.data['nombre_tirs_cadres_B'] = st.number_input("Nombre de Tirs Cadrés (Équipe B)", value=int(st.session_state.data['nombre_tirs_cadres_B']), step=1)  
    
    # Ratios avec descriptions  
    st.session_state.data['ratio_buts_tirs_B'] = st.number_input("Ratio Buts/Tirs (Équipe B)", value=float(st.session_state.data['ratio_buts_tirs_B']), step=0.01, help="Ratio des buts marqués par rapport au nombre total de tirs.")  
    st.session_state.data['ratio_buts_encais_tirs_B'] = st.number_input("Ratio Buts Encaissés/Tirs (Équipe B)", value=float(st.session_state.data['ratio_buts_encais_tirs_B']), step=0.01, help="Ratio des buts encaissés par rapport au nombre total de tirs subis.")  
    
    st.session_state.data['performance_domicile_B'] = st.number_input("Performance à Domicile (Points) (Équipe B)", value=float(st.session_state.data['performance_domicile_B']), step=0.1)  
    st.session_state.data['performance_exterieur_B'] = st.number_input("Performance à l'Extérieur (Points) (Équipe B)", value=float(st.session_state.data['performance_exterieur_B']), step=0.1)  
    st.session_state.data['historique_confrontations_B_A'] = st.number_input("Historique Confrontations (Équipe B contre Équipe A)", value=int(st.session_state.data['historique_confrontations_B_A']), step=1)  
    st.session_state.data['moyenne_buts_marques_B'] = st.number_input("Moyenne de Buts Marqués par Match (Équipe B)", value=float(st.session_state.data['moyenne_buts_marques_B']), step=0.1)  
    st.session_state.data['moyenne_buts_encais_B'] = st.number_input("Moyenne de Buts Encaissés par Match (Équipe B)", value=float(st.session_state.data['moyenne_buts_encais_B']), step=0.1)       
    st.session_state.data['moyenne_buts_encais_B'] = st.number_input("Moyenne de Buts Encaissés par Match (Équipe B)", value=float(st.session_state.data['moyenne_buts_encais_B']), step=0.1)  
    st.session_state.data['impact_joueurs_cles_B'] = st.number_input("Impact des Joueurs Clés (Équipe B)", value=float(st.session_state.data['impact_joueurs_cles_B']), step=0.1)  
    st.session_state.data['taux_reussite_corners_B'] = st.number_input("Taux de Réussite des Corners (%) (Équipe B)", value=float(st.session_state.data['taux_reussite_corners_B']), step=0.1)  
    
    # Nouveau champ pour le nombre de dégagements  
    st.session_state.data['nombre_degagements_B'] = st.number_input("Nombre de Dégagements (Équipe B)", value=int(st.session_state.data['nombre_degagements_B']), step=1)  
    
    # Nouveau champ pour la tactique  
    st.session_state.data['tactique_B'] = st.number_input("Tactique (Équipe B)", value=int(st.session_state.data['tactique_B']), min_value=1, max_value=10, help="Évaluez l'efficacité de la tactique sur une échelle de 1 à 10.")  
    
    # Nouveaux champs pour la motivation et la forme récente  
    st.session_state.data['motivation_B'] = st.number_input("Motivation (Équipe B)", value=int(st.session_state.data['motivation_B']), min_value=1, max_value=10)  
    st.session_state.data['forme_recente_B'] = st.number_input("Forme Récente (Points sur 5 matchs) (Équipe B)", value=int(st.session_state.data['forme_recente_B']), step=1)  

    # Bouton pour soumettre le formulaire  
    submitted = st.form_submit_button("✅ Soumettre les Données")  
    if submitted:  
        st.success("Données soumises avec succès! 🎉")  

# Comparateur des équipes  
st.header("🔍 Comparateur des Équipes")  
data_comparaison = {  
    "Statistiques": [  
        "Score de Performance",  
        "Buts par Match",  
        "Buts Concedes par Match",  
        "Possession Moyenne (%)",  
        "Expected Goals",  
        "Tirs Cadres",  
        "Grandes Chances",  
        "Victoires à Domicile",  
        "Victoires à l'Extérieur",  
        "Joueurs Absents",  
        "Moyenne d'Âge des Joueurs",  
        "Expérience de l'Entraîneur (Années)",  
        "Nombre de Blessures",  
        "Cartons Jaunes",  
        "Cartons Rouges",  
        "Taux de Réussite des Passes (%)",  
        "Taux de Possession (%)",  
        "Nombre de Tirs Totaux",  
        "Nombre de Tirs Cadrés",  
        "Ratio Buts/Tirs",  
        "Ratio Buts Encaissés/Tirs",  
        "Performance à Domicile",  
        "Performance à l'Extérieur",  
        "Historique Confrontations",  
        "Moyenne de Buts Marqués par Match",  
        "Moyenne de Buts Encaissés par Match",  
        "Impact des Joueurs Clés",  
        "Taux de Réussite des Corners (%)",  
        "Nombre de Dégagements",  
        "Tactique",  
        "Motivation",  
        "Forme Récente"  
    ],  
    "Équipe A": [  
        st.session_state.data['score_rating_A'],  
        st.session_state.data['buts_par_match_A'],  
        st.session_state.data['buts_concedes_par_match_A'],  
        st.session_state.data['possession_moyenne_A'],  
        st.session_state.data['expected_but_A'],  
        st.session_state.data['tirs_cadres_A'],  
        st.session_state.data['grandes_chances_A'],  
        st.session_state.data['victoires_domicile_A'],  
        st.session_state.data['victoires_exterieur_A'],  
        st.session_state.data['joueurs_absents_A'],  
        st.session_state.data['moyenne_age_joueurs_A'],  
        st.session_state.data['experience_entraineur_A'],  
        st.session_state.data['nombre_blessures_A'],  
        st.session_state.data['cartons_jaunes_A'],  
        st.session_state.data['cartons_rouges_A'],  
        st.session_state.data['taux_reussite_passes_A'],  
        st.session_state.data['taux_possession_A'],  
        st.session_state.data['nombre_tirs_totaux_A'],  
        st.session_state.data['nombre_tirs_cadres_A'],  
        st.session_state.data['ratio_buts_tirs_A'],  
        st.session_state.data['ratio_buts_encais_tirs_A'],  
        st.session_state.data['performance_domicile_A'],  
        st.session_state.data['performance_exterieur_A'],  
        st.session_state.data['historique_confrontations_A_B'],  
        st.session_state.data['moyenne_buts_marques_A'],  
        st.session_state.data['moyenne_buts_encais_A'],  
        st.session_state.data['impact_joueurs_cles_A'],  
        st.session_state.data['taux_reussite_corners_A'],  
        st.session_state.data['nombre_degagements_A'],  
        st.session_state.data['tactique_A'],  
        st.session_state.data['motivation_A'],  
        st.session_state.data['forme_recente_A']  
    ],  
    "Équipe B": [  
        st.session_state.data['score_rating_B'],  
        st.session_state.data['buts_par_match_B'],  
        st.session_state.data['buts_concedes_par_match_B'],  
        st.session_state.data['possession_moyenne_B'],  
        st.session_state.data['expected_but_B'],  
        st.session_state.data['tirs_cadres_B'],  
        st.session_state.data['grandes_chances_B'],  
        st.session_state.data['victoires_domicile_B'],  
        st.session_state.data['victoires_exterieur_B'],  
        st.session_state.data['joueurs_absents_B'],  
        st.session_state.data['moyenne_age_joueurs_B'],  
        st.session_state.data['experience_entraineur_B'],  
        st.session_state.data['nombre_blessures_B'],  
        st.session_state.data['cartons_jaunes_B'],  
        st.session_state.data['cartons_rouges_B'],  
        st.session_state.data['taux_reussite_passes_B'],  
        st.session_state.data['taux_possession_B'],  
        st.session_state.data['nombre_tirs_totaux_B'],  
        st.session_state.data['nombre_tirs_cadres_B'],  
        st.session_state.data['ratio_buts_tirs_B'],  
        st.session_state.data['ratio_buts_encais_tirs_B'],  
        st.session_state.data['performance_domicile_B'],  
        st.session_state.data['performance_exterieur_B'],  
        st.session_state.data['historique_confrontations_B_A'],  
        st.session_state.data['moyenne_buts_marques_B'],  
        st.session_state.data['moyenne_buts_encais_B'],  
        st.session_state.data['impact_joueurs_cles_B'],  
        st.session_state.data['taux_reussite_corners_B'],  
        st.session_state.data['nombre_degagements_B'],  
        st.session_state.data['tactique_B'],  
        st.session_state.data['motivation_B'],  
        st.session_state.data['forme_recente_B']  
    ]  
}  

# Création d'un DataFrame pour afficher les données  
df_comparaison = pd.DataFrame(data_comparaison)  

# Affichage du tableau avec des couleurs  
st.write("### Comparaison des Équipes")  
st.dataframe(df_comparaison.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='salmon'))  

# Graphique de comparaison  
st.write("### Graphique de Comparaison")  
chart_data = pd.DataFrame({  
    'Statistiques': df_comparaison['Statistiques'],  
    'Équipe A': df_comparaison['Équipe A'],  
    'Équipe B': df_comparaison['Équipe B']  
})  

st.bar_chart(chart_data.set_index('Statistiques'))  

# Fin de l'application  
st.write("Merci d'utiliser l'outil d'analyse des équipes! ⚽️")
