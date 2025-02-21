import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  

# Fonction pour prédire le nombre de buts  
def predire_buts(probabilite_victoire, attaque, defense):  
    """  
    Prédit le nombre de buts en fonction de la probabilité de victoire,  
    de la force offensive (attaque) et de la force défensive (defense).  
    """  
    base_buts = probabilite_victoire * 3  # Base : max 3 buts en fonction de la probabilité  
    ajustement = (attaque - defense) * 0.5  # Ajustement basé sur la différence attaque/défense  
    buts = max(0, base_buts + ajustement)  # Les buts ne peuvent pas être négatifs  
    return round(buts, 1)  # Retourne un nombre avec une décimale  

# Fonction pour convertir les cotes en probabilités implicites  
def cotes_en_probabilites(cote):  
    return 1 / cote  

# Titre de l'application  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de match et gestion de bankroll</h1>", unsafe_allow_html=True)  
st.write("Analysez les données des matchs, simulez les résultats, et suivez vos paris pour optimiser votre bankroll.")  

# Partie 1 : Analyse des équipes  
st.markdown("<h2 style='color: #2196F3;'>1. Analyse des équipes</h2>", unsafe_allow_html=True)  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
buts_par_match_A = st.number_input("Buts par match", min_value=0, max_value=10, value=2, key="buts_par_match_A", help="Entrez le nombre moyen de buts marqués par l'équipe A par match.")  
nombre_buts_produits_A = st.number_input("Nombre de buts produits", min_value=0, max_value=100, value=80, key="nombre_buts_produits_A")  
nombre_buts_encaisse_A = st.number_input("Nombre de buts encaissés", min_value=0, max_value=100, value=30, key="nombre_buts_encaisse_A")  
buts_encaisse_par_match_A = st.number_input("Buts encaissés par match", min_value=0, max_value=10, value=1, key="buts_encaisse_par_match_A")  
poss_moyenne_A = st.number_input("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
matchs_sans_encaisser_A = st.number_input("Nombre de matchs sans encaisser de but", min_value=0, max_value=40, value=10, key="matchs_sans_encaisser_A")  
xG_A = st.number_input("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=2.5, key="xG_A")  
tirs_cadres_A = st.number_input("Tirs cadrés par match", min_value=0, max_value=20, value=5, key="tirs_cadres_A")  
pourcentage_tirs_convertis_A = st.number_input("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=25, key="pourcentage_tirs_convertis_A")  
grosses_occasions_A = st.number_input("Grosses occasions", min_value=0, max_value=50, value=20, key="grosses_occasions_A")  
grosses_occasions_ratees_A = st.number_input("Grosses occasions ratées", min_value=0, max_value=50, value=10, key="grosses_occasions_ratees_A")  
passes_reussies_A = st.number_input("Passes réussies par match", min_value=0, max_value=100, value=70, key="passes_reussies_A")  
passes_longues_precises_A = st.number_input("Passes longues précises par match", min_value=0, max_value=50, value=20, key="passes_longues_precises_A")  
centres_reussis_A = st.number_input("Centres réussis par match", min_value=0, max_value=50, value=15, key="centres_reussis_A")  
penalties_obtenus_A = st.number_input("Pénalties obtenus", min_value=0, max_value=10, value=3, key="penalties_obtenus_A")  
touches_surface_adverse_A = st.number_input("Touches dans la surface adverse", min_value=0, max_value=50, value=25, key="touches_surface_adverse_A")  
corners_A = st.number_input("Nombre de corners", min_value=0, max_value=50, value=20, key="corners_A")  
corners_par_match_A = st.number_input("Nombre de corners par match", min_value=0, max_value=10, value=5, key="corners_par_match_A")  
corners_concedes_A = st.number_input("Nombre de corners concédés", min_value=0, max_value=50, value=15, key="corners_concedes_A")  
xG_concedes_A = st.number_input("xG concédés", min_value=0.0, max_value=5.0, value=1.5, key="xG_concedes_A")  
interceptions_A = st.number_input("Interceptions par match", min_value=0, max_value=20, value=5, key="interceptions_A")  
tacles_reussis_A = st.number_input("Tacles réussis par match", min_value=0, max_value=20, value=7, key="tacles_reussis_A")  
degagements_A = st.number_input("Dégagements par match", min_value=0, max_value=20, value=5, key="degagements_A")  
possessions_recuperees_A = st.number_input("Possessions récupérées au milieu de terrain par match", min_value=0, max_value=20, value=5, key="possessions_recuperees_A")  
penalties_concedes_A = st.number_input("Pénalties concédés", min_value=0, max_value=10, value=2, key="penalties_concedes_A")  
arrets_A = st.number_input("Arrêts par match", min_value=0, max_value=20, value=5, key="arrets_A")  
fautes_A = st.number_input("Fautes par match", min_value=0, max_value=20, value=10, key="fautes_A")  
cartons_jaunes_A = st.number_input("Cartons jaunes", min_value=0, max_value=10, value=3, key="cartons_jaunes_A")  
cartons_rouges_A = st.number_input("Cartons rouges", min_value=0, max_value=5, value=1, key="cartons_rouges_A")  
tactique_A = st.text_input("Tactique de l'équipe A", "4-3-3", key="tactique_A")  
joueurs_cles_A = st.text_input("Joueurs clés de l'équipe A", "Joueur 1, Joueur 2", key="joueurs_cles_A")  
score_joueurs_cles_A = st.number_input("Score des joueurs clés (sur 10)", min_value=0, max_value=10, value=7, key="score_joueurs_cles_A")
