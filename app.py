import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt  

# Configuration de l'application  
st.set_page_config(page_title="Prédiction de Match", layout="wide")  

# Titre de la page d'analyse  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Analyse des Équipes</h1>", unsafe_allow_html=True)  

# Historique des équipes  
st.subheader("Historique des équipes")  
st.markdown("Veuillez entrer les résultats des 10 derniers matchs pour chaque équipe (1 pour victoire, 0 pour match nul, -1 pour défaite).")  

# Historique de l'équipe A  
historique_A = []  
for i in range(10):  
    resultat = st.selectbox(f"Match {i + 1} de l'équipe A", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_A_{i}")  
    historique_A.append(1 if resultat == "Victoire (1)" else (0 if resultat == "Match Nul (0)" else -1))  

# Historique de l'équipe B  
historique_B = []  
for i in range(10):  
    resultat = st.selectbox(f"Match {i + 1} de l'équipe B", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_B_{i}")  
    historique_B.append(1 if resultat == "Victoire (1)" else (0 if resultat == "Match Nul (0)" else -1))  

# Calcul de la forme récente sur 10 matchs  
def calculer_forme(historique):  
    return sum(historique) / len(historique)  # Moyenne des résultats  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  

st.write(f"Forme récente de l'équipe A (moyenne sur 10 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 10 matchs) : **{forme_recente_B:.2f}**")  

# Visualisation de la forme des équipes  
st.subheader("Visualisation de la Forme des Équipes")  
fig, ax = plt.subplots()  
ax.plot(range(1, 11), historique_A, marker='o', label='Équipe A', color='blue')  
ax.plot(range(1, 11), historique_B, marker='o', label='Équipe B', color='red')  
ax.set_xticks(range(1, 11))  
ax.set_xticklabels([f"Match {i}" for i in range(1, 11)])  
ax.set_ylim(-1.5, 1.5)  
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')  
ax.set_ylabel('Résultat')  
ax.set_title('Historique des Performances des Équipes')  
ax.legend()  
st.pyplot(fig)  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
buts_produits_A = st.slider("Nombre de buts produits", min_value=0, max_value=100, value=80, key="buts_produits_A")  
buts_encaisse_A = st.slider("Nombre de buts encaissés", min_value=0, max_value=100, value=30, key="buts_encaisse_A")  
buts_par_match_A = st.slider("Buts par match", min_value=0, max_value=10, value=2, key="buts_par_match_A")  
poss_moyenne_A = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
xG_A = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=2.5, key="xG_A")  
tirs_cadres_A = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=10, key="tirs_cadres_A")  
pourcentage_tirs_convertis_A = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=20, key="pourcentage_tirs_convertis_A")  
grosses_occasions_A = st.slider("Grosses occasions", min_value=0, max_value=100, value=10, key="grosses_occasions_A")  
grosses_occasions_ratees_A = st.slider("Grosses occasions ratées", min_value=0, max_value=100, value=5, key="grosses_occasions_ratees_A")  
passes_reussies_A = st.slider("Passes réussies par match", min_value=0, max_value=100, value=30, key="passes_reussies_A")  
passes_longues_A = st.slider("Passes longues précises par match", min_value=0, max_value=50, value=10, key="passes_longues_A")  
centres_reussis_A = st.slider("Centres réussis par match", min_value=0, max_value=50, value=5, key="centres_reussis_A")  
penalties_obtenus_A = st.slider("Pénalties obtenus", min_value=0, max_value=10, value=2, key="penalties_obtenus_A")  
touches_surface_A = st.slider("Touches dans la surface adverse", min_value=0, max_value=50, value=15, key="touches_surface_A")  
corners_A = st.slider("Nombre de corners", min_value=0, max_value=20, value=5, key="corners_A")  
corners_par_match_A = st.slider("Nombre de corners par match", min_value=0, max_value=10, value=3, key="corners_par_match_A")  
corners_concedes_A = st.slider("Nombre de corners concédés par match", min_value=0, max_value=10, value=4, key="corners_concedes_A")  
xG_concedes_A = st.slider("xG concédés", min_value=0.0, max_value=5.0, value=1.5, key="xG_concedes_A")  
interceptions_A = st.slider("Interceptions par match", min_value=0, max_value=20, value=5, key="interceptions_A")  
tacles_reussis_A = st.slider("Tacles réussis par match", min_value=0, max_value=20, value=5, key="tacles_reussis_A")  
degagements_A = st.slider("Dégagements par match", min_value=0, max_value=20, value=5, key="degagements_A")  
possessions_recuperees_A = st.slider("Possessions récupérées au milieu de terrain par match", min_value=0, max_value=20, value=5, key="possessions_recuperees_A")  
penalties_concedes_A = st.slider("Pénalties concédés", min_value=0, max_value=10, value=1, key="penalties_concedes_A")  
arrets_A = st.slider("Arrêts par match", min_value=0, max_value=20, value=3, key="arrets_A")  
fautes_A = st.slider("Fautes par match", min_value=0, max_value=20, value=10, key="fautes_A")  
cartons_jaunes_A = st.slider("Cartons jaunes", min_value=0, max_value=10, value=2, key="cartons_jaunes_A")  
cartons_rouges_A = st.slider("Cartons rouges", min_value=0, max_value=10, value=0, key="cartons_rouges_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_produits_B = st.slider("Nombre de buts produits", min_value=0, max_value=100, value=60, key="buts_produits_B")  
buts_encaisse_B = st.slider("Nombre de buts encaissés", min_value=0, max_value=100, value=40, key="buts_encaisse_B")  
buts_par_match_B = st.slider("Buts par match", min_value=0, max_value=10, value=1, key="buts_par_match_B")  
poss_moyenne_B = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
xG_B = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.5, key="xG_B")  
tirs_cadres_B = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=8, key="tirs_cadres_B")  
pourcentage_tirs_convertis_B = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=15, key="pourcentage_tirs_convertis_B")  
grosses_occasions_B = st.slider("Grosses occasions", min_value=0, max_value=100, value=8, key="grosses_occasions_B")  
grosses_occasions_ratees_B = st.slider("Grosses occasions ratées", min_value=0, max_value=100, value=4, key="grosses_occasions_ratees_B")  
passes_reussies_B = st.slider("Passes réussies par match", min_value=0, max_value=100, value=25, key="passes_reussies_B")  
passes_longues_B = st.slider("Passes longues précises par match", min_value=0, max_value=50, value=8, key="passes_longues_B")  
centres_reussis_B = st.slider("Centres réussis par match", min_value=0, max_value=50, value=4, key="centres_reussis_B")  
penalties_obtenus_B = st.slider("Pénalties obtenus", min_value=0, max_value=10, value=1, key="penalties_obtenus_B")  
touches_surface_B = st.slider("Touches dans la surface adverse", min_value=0, max_value=50, value=12, key="touches_surface_B")  
corners_B = st.slider("Nombre de corners", min_value=0, max_value=20, value=4, key="corners_B")  
corners_par_match_B = st.slider("Nombre de corners par match", min_value=0, max_value=10, value=3, key="corners_par_match_B")  
corners_concedes_B = st.slider("Nombre de corners concédés par match", min_value=0, max_value=10, value=5, key="corners_concedes_B")  
xG_concedes_B = st.slider("xG concédés", min_value=0.0, max_value=5.0, value=1.8, key="xG_concedes_B")  
interceptions_B = st.slider("Interceptions par match", min_value=0, max_value=20, value=6, key="interceptions_B")  
tacles_reussis_B = st.slider("Tacles réussis par match", min_value=0, max_value=20, value=6, key="tacles_reussis_B")  
degagements_B = st.slider("Dégagements par match", min_value=0, max_value=20, value=6, key="degagements_B")  
possessions_recuperees_B = st.slider("Possessions récupérées au milieu de terrain par match", min_value=0, max_value=20, value=6, key="possessions_recuperees_B")  
penalties_concedes_B = st.slider("Pénalties concédés", min_value=0, max_value=10, value=1, key="penalties_concedes_B")  
arrets_B = st.slider("Arrêts par match", min_value=0, max_value=20, value=4, key="arrets_B")  
fautes_B = st.slider("Fautes par match", min_value=0, max_value=20, value=8, key="fautes_B")  
cartons_jaunes_B = st.slider("Cartons jaunes", min_value=0, max_value=10, value=1, key="cartons_jaunes_B")  
cartons_rouges_B = st.slider("Cartons rouges", min_value=0, max_value=10, value=0, key="cartons_rouges_B")  

# Joueurs clés  
st.subheader("Joueurs Clés")  
joueurs_cles_A = st.text_input("Joueurs clés de l'équipe A (séparés par des virgules)", value="Joueur1, Joueur2")  
scores_joueurs_cles_A = st.text_input("Scores des joueurs clés de l'équipe A (séparés par des virgules)", value="8, 7")  

joueurs_cles_B = st.text_input("Joueurs clés de l'équipe B (séparés par des virgules)", value="Joueur3, Joueur4")  
scores_joueurs_cles_B = st.text_input("Scores des joueurs clés de l'équipe B (séparés par des virgules)", value="6, 9")  

# Fonction pour analyser les systèmes tactiques  
def analyser_tactiques(buts_produits, buts_encaisse, possession, xG, tirs_cadres, corners, interceptions):  
    score_tactique = (buts_produits * 0.4) + (possessions_recuperees * 0.2) + (tirs_cadres * 0.2) - (buts_encaisse * 0.4) - (corners * 0.2) - (interceptions * 0.1)  
    return score_tactique  

# Calcul des scores tactiques  
score_tactique_A = analyser_tactiques(buts_produits_A, buts_encaisse_A, poss_moyenne_A, xG_A, tirs_cadres_A, corners_A, interceptions_A)  
score_tactique_B = analyser_tactiques(buts_produits_B, buts_encaisse_B, poss_moyenne_B, xG_B, tirs_cadres_B, corners_B, interceptions_B)  

# Sélection du meilleur système tactique  
meilleur_tactique_A = "4-3-3" if score_tactique_A > score_tactique_B else "4-4-2"  
meilleur_tactique_B = "4-3-3" if score_tactique_B > score_tactique_A else "4-4-2"  

# Affichage des meilleurs systèmes tactiques  
st.write(f"Meilleur système tactique pour l'équipe A : **{meilleur_tactique_A}**")  
st.write(f"Meilleur système tactique pour l'équipe B : **{meilleur_tactique_B}**")  

# Modèle de prédiction des buts  
if st.button("Analyser le Match"):  
    # Données d'entraînement fictives  
    data = {  
        "buts_produits_A": [80, 60, 70, 90, 50],  
        "buts_encaisse_A": [30, 40, 20, 50, 35],  
        "buts_par_match_A": [2, 1, 2, 3, 1],  
        "poss_moyenne_A": [55, 45, 60, 40, 50],  
        "xG_A": [2.5, 1.5, 3.0, 0.5, 2.0],  
        "tirs_cadres_A": [10, 8, 12, 5, 9],  
        "pourcentage_tirs_convertis_A": [20, 25, 15, 30, 10],  
        "grosses_occasions_A": [10, 8, 12, 5, 9],  
        "grosses_occasions_ratees_A": [5, 3, 4, 2, 1],  
        "passes_reussies_A": [30, 25, 35, 20, 28],  
        "passes_longues_A": [10, 12, 15, 8, 11],  
        "centres_reussis_A": [5, 4, 6, 3, 2],  
        "penalties_obtenus_A": [2, 1, 3, 0, 1],  
        "touches_surface_A": [15, 20, 18, 10, 14],  
        "corners_A": [5, 3, 6, 2, 4],  
        "corners_par_match_A": [3, 2, 4, 1, 3],  
        "corners_concedes_A": [4, 5, 3, 2, 4],  
        "xG_concedes_A": [1.5, 2.0, 1.0, 2.5, 1.8],  
        "interceptions_A": [5, 6, 4, 7, 5],  
        "tacles_reussis_A": [5, 4, 6, 3, 5],  
        "degagements_A": [5, 6, 4, 7, 5],  
        "possessions_recuperees_A": [5, 4, 6, 3, 5],  
        "penalties_concedes_A": [1, 0, 2, 1, 1],  
        "arrets_A": [3, 2, 4, 1, 3],  
        "fautes_A": [10, 12, 8, 9, 11],  
        "cartons_jaunes_A": [2, 1, 3
