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
st.markdown("Veuillez entrer les résultats des 5 derniers matchs pour chaque équipe (1 pour victoire, 0 pour match nul, -1 pour défaite).")  

# Historique des résultats  
historique_A = []  
historique_B = []  
for i in range(5):  
    resultat_A = st.selectbox(f"Match {i + 1} de l'équipe A", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_A_{i}")  
    historique_A.append(1 if resultat_A == "Victoire (1)" else (0 if resultat_A == "Match Nul (0)" else -1))  
    
    resultat_B = st.selectbox(f"Match {i + 1} de l'équipe B", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_B_{i}")  
    historique_B.append(1 if resultat_B == "Victoire (1)" else (0 if resultat_B == "Match Nul (0)" else -1))  

# Historique de face-à-face  
st.subheader("Historique de Face-à-Face")  
st.markdown("Veuillez entrer les résultats des 5 derniers matchs entre les deux équipes.")  
historique_face_a_face = []  
for i in range(5):  
    resultat = st.selectbox(f"Match {i + 1} entre A et B", options=["Victoire A (1)", "Match Nul (0)", "Victoire B (-1)"], key=f"match_face_a_face_{i}")  
    historique_face_a_face.append(1 if resultat == "Victoire A (1)" else (0 if resultat == "Match Nul (0)" else -1))  

# Calcul de la forme récente sur 5 matchs  
def calculer_forme(historique):  
    return sum(historique) / len(historique)  # Moyenne des résultats  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  
forme_face_a_face = calculer_forme(historique_face_a_face)  

st.write(f"Forme récente de l'équipe A (moyenne sur 5 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 5 matchs) : **{forme_recente_B:.2f}**")  
st.write(f"Forme récente en face-à-face (moyenne sur 5 matchs) : **{forme_face_a_face:.2f}**")  

# Visualisation de la forme des équipes  
st.subheader("Visualisation de la Forme des Équipes")  
fig, ax = plt.subplots()  
ax.plot(range(1, 6), historique_A, marker='o', label='Équipe A', color='blue')  
ax.plot(range(1, 6), historique_B, marker='o', label='Équipe B', color='red')  
ax.plot(range(1, 6), historique_face_a_face, marker='o', label='Face-à-Face', color='green')  
ax.set_xticks(range(1, 6))  
ax.set_xticklabels([f"Match {i}" for i in range(1, 6)])  
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

# Fonction pour analyser les systèmes tactiques  
def analyser_tactiques(buts_produits, buts_encaisse, possession, xG, tirs_cadres, corners, interceptions):  
    score_tactique = (buts_produits * 0.4) + (poss_moyenne_A * 0.2) + (tirs_cadres * 0.2) - (buts_encaisse * 0.4) - (corners * 0.2) - (interceptions * 0.1)  
    return score_tactique  

# Calcul des scores tactiques  
score_tactique_A = analyser_tactiques(buts_produits_A, buts_encaisse_A, poss_moyenne_A, xG_A, tirs_cadres_A, corners_A, interceptions_A)  
score_tactique_B = analyser_tactiques(buts_produits_B, buts_encaisse_B, poss_moyenne_B, xG_B, tirs_cadres_B, corners_B, interceptions_B)  

# Prédiction des buts  
def prediction_buts(score_A, score_B):  
    total_score = score_A + score_B  
    if total_score == 0:  
        return 0, 0  # Éviter la division par zéro  
    return (score_A / total_score) * 100, (score_B / total_score) * 100  

# Calcul des pourcentages de victoire  
pourcentage_victoire_A, pourcentage_victoire_B = prediction_buts(score_tactique_A, score_tactique_B)  

# Affichage des meilleurs systèmes tactiques  
st.write(f"<h3 style='color: #FF5722;'>Meilleur système tactique pour l'équipe A : {'4-3-3' if score_tactique_A > score_tactique_B else '4-4-2'}</h3>", unsafe_allow_html=True)  
st.write(f"<h3 style='color: #FF5722;'>Meilleur système tactique pour l'équipe B : {'4-3-3' if score_tactique_B > score_tactique_A else '4-4-2'}</h3>", unsafe_allow_html=True)  

# Affichage des résultats de la prédiction  
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
        "cartons_jaunes_A": [2, 1, 3, 0, 1],  
        "cartons_rouges_A": [0, 0, 1, 0, 0],  
        "buts_produits_B": [60, 70, 50, 20, 40],  
        "buts_encaisse_B": [40, 30, 20, 50, 35],  
        "buts_par_match_B": [1, 2, 1, 0, 1],  
        "poss_moyenne_B": [45, 55, 40, 50, 45],  
        "xG_B": [1.5, 2.0, 1.0, 0.5, 1.2],  
        "tirs_cadres_B": [8, 7, 9, 5, 6],  
        "pourcentage_tirs_convertis_B": [15, 20, 10, 25, 5],  
        "grosses_occasions_B": [8, 10, 6, 4, 7],  
        "grosses_occasions_ratees_B": [4, 3, 2, 1, 2],  
        "passes_reussies_B": [25, 30, 20, 15, 28],  
        "passes_longues_B": [8, 10, 12, 6, 9],  
        "centres_reussis_B": [4, 5, 3, 2, 4],  
        "penalties_obtenus_B": [1, 2, 0, 1, 1],  
        "touches_surface_B": [12, 15, 10, 8, 11],  
        "corners_B": [4, 5, 3, 2, 4],  
        "corners_par_match_B": [3, 2, 4, 1, 3],  
        "corners_concedes_B": [5, 4, 6, 3, 5],  
        "xG_concedes_B": [1.8, 2.2, 1.5, 2.0, 1.9],  
        "interceptions_B": [6, 5, 7, 4, 6],  
        "tacles_reussis_B": [6, 5, 7, 4, 6],  
        "degagements_B": [6, 5, 7, 4, 6],  
        "possessions_recuperees_B": [6, 5, 7, 4, 6],  
        "penalties_concedes_B": [1, 2, 1, 0, 1],  
        "arrets_B": [4, 3, 5, 2, 4],  
        "fautes_B": [8, 9, 7, 6, 8],  
        "cartons_jaunes_B": [1, 2, 1, 0, 1],  
        "cartons_rouges_B": [0, 0, 0, 0, 0],  
    }  

    # Conversion des données en DataFrame  
    df = pd.DataFrame(data)  

    # Séparation des caractéristiques et de la cible  
    X = df.drop(columns=["buts_produits_A", "buts_produits_B"])  
    y_A = df["buts_produits_A"]  
    y_B = df["buts_produits_B"]  

    # Division des données en ensembles d'entraînement et de test  
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X, y_A, test_size=0.2, random_state=42)  
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y_B, test_size=0.2, random_state=42)  

    # Modèle de régression linéaire  
    model_A = LinearRegression()  
    model_B = LinearRegression()  

    # Entraînement des modèles  
    model_A.fit(X_train_A, y_train_A)  
    model_B.fit(X_train_B, y_train_B)  

    # Prédictions  
    prediction_A = model_A.predict(X_test_A)  
    prediction_B = model_B.predict(X_test_B)  

    # Calcul des buts moyens  
    buts_moyens_A = prediction_A.mean()  
    buts_moyens_B = prediction_B.mean()  

    # Affichage des résultats de la prédiction  
    st.subheader("Résultats de la Prédiction")  
    st.write(f"Prédiction des buts pour l'équipe A : **{buts_moyens_A:.2f}**")  
    st.write(f"Prédiction des buts pour l'équipe B : **{buts_moyens_B:.2f}**")  

    # Calcul des pourcentages de victoire  
    pourcentage_victoire_A, pourcentage_victoire_B = prediction_buts(buts_moyens_A, buts_moyens_B)  

    # Affichage des pourcentages de victoire  
    st.write(f"Pourcentage de victoire pour l'équipe A : **{pourcentage_victoire_A:.2f}%**")  
    st.write(f"Pourcentage de victoire pour l'équipe B : **{pourcentage_victoire_B:.2f}%**")  

    # Visualisation des prédictions  
    fig, ax = plt.subplots()  
    ax.bar(["Équipe A", "Équipe B"], [buts_moyens_A, buts_moyens_B], color=['blue', 'red'])  
    ax.set_ylabel('Buts Prédits')  
    ax.set_title('Prédictions de Buts pour le Match')  
    st.pyplot(fig)  

# Fin de l'application  
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Merci d'avoir utilisé l'outil d'analyse de match !</h2>", unsafe_allow_html=True)
