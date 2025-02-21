import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  
import csv  
from io import StringIO  


# Fonction pour prédire le nombre de buts  
def predire_buts(probabilite_victoire, attaque, defense):  
    """  
    Prédit le nombre de buts en fonction de la probabilité de victoire,  
    de la force offensive (attaque) et de la force défensive (defense).  
    """  
    base_buts = probabilité_victoire * 3  # Base : max 3 buts en fonction de la probabilité  
    ajustement = (attaque - defense) * 0.5  # Ajustement basé sur la différence attaque/défense  
    buts = max(0, base_buts + ajustement)  # Les buts ne peuvent pas être négatifs  
    return round(buts, 1)  # Retourne un nombre avec une décimale  


# Fonction pour calculer la mise optimale selon Kelly  
def calculer_mise_kelly(probabilite, cotes, facteur_kelly=1):  
    b = cotes - 1  # Gain net pour 1 unité misée  
    q = 1 - probabilite  # Probabilité d'échec  
    fraction_kelly = (b * probabilite - q) / b  
    fraction_kelly = max(0, fraction_kelly)  # La mise ne peut pas être négative  
    return fraction_kelly * facteur_kelly  


# Fonction pour convertir la mise Kelly en unités (1 à 5)  
def convertir_mise_en_unites(mise, bankroll, max_unites=5):  
    # Mise maximale en fonction de la bankroll  
    mise_max = bankroll * 0.05  # Par exemple, 5% de la bankroll  
    # Mise en unités (arrondie à l'entier le plus proche)  
    unites = round((mise / mise_max) * max_unites)  
    return min(max(unites, 1), max_unites)  # Limiter entre 1 et max_unites  


# Titre de l'application  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de match et gestion de bankroll</h1>", unsafe_allow_html=True)  
st.write("Analysez les données des matchs, simulez les résultats, et suivez vos paris pour optimiser votre bankroll.")  

# Partie 1 : Analyse des équipes  
st.markdown("<h2 style='color: #2196F3;'>1. Analyse des équipes</h2>", unsafe_allow_html=True)  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
# Données fictives modifiables pour 40 matchs  
buts_par_match_A = st.slider("Buts par match", 0, 10, 2, key="buts_par_match_A")  
nombre_buts_produits_A = st.slider("Nombre de buts produits", 0, 100, 80, key="nombre_buts_produits_A")  
nombre_buts_encaisse_A = st.slider("Nombre de buts encaissés", 0, 100, 30, key="nombre_buts_encaisse_A")  
buts_encaisse_par_match_A = st.slider("Buts encaissés par match", 0, 10, 1, key="buts_encaisse_par_match_A")  
poss_moyenne_A = st.slider("Possession moyenne (%)", 0, 100, 55, key="poss_moyenne_A")  
matchs_sans_encaisser_A = st.slider("Nombre de matchs sans encaisser de but", 0, 40, 10, key="matchs_sans_encaisser_A")  
xG_A = st.slider("Buts attendus (xG)", 0.0, 5.0, 2.5, key="xG_A")  
tirs_cadres_A = st.slider("Tirs cadrés par match", 0, 20, 5, key="tirs_cadres_A")  
pourcentage_tirs_convertis_A = st.slider("Pourcentage de tirs convertis (%)", 0, 100, 25, key="pourcentage_tirs_convertis_A")  
grosses_occasions_A = st.slider("Grosses occasions", 0, 50, 20, key="grosses_occasions_A")  
grosses_occasions_ratees_A = st.slider("Grosses occasions ratées", 0, 50, 10, key="grosses_occasions_ratees_A")  
passes_reussies_A = st.slider("Passes réussies par match", 0, 100, 70, key="passes_reussies_A")  
passes_longues_precises_A = st.slider("Passes longues précises par match", 0, 50, 20, key="passes_longues_precises_A")  
centres_reussis_A = st.slider("Centres réussis par match", 0, 50, 15, key="centres_reussis_A")  
penalties_obtenus_A = st.slider("Pénalties obtenus", 0, 10, 3, key="penalties_obtenus_A")  
touches_surface_adverse_A = st.slider("Touches dans la surface adverse", 0, 50, 25, key="touches_surface_adverse_A")  
corners_A = st.slider("Nombre de corners", 0, 50, 20, key="corners_A")  
corners_par_match_A = st.slider("Nombre de corners par match", 0, 10, 5, key="corners_par_match_A")  
corners_concedes_A = st.slider("Nombre de corners concédés", 0, 50, 15, key="corners_concedes_A")  
xG_concedes_A = st.slider("xG concédés", 0.0, 5.0, 1.5, key="xG_concedes_A")  
interceptions_A = st.slider("Interceptions par match", 0, 20, 5, key="interceptions_A")  
tacles_reussis_A = st.slider("Tacles réussis par match", 0, 20, 7, key="tacles_reussis_A")  
degagements_A = st.slider("Dégagements par match", 0, 20, 5, key="degagements_A")  
possessions_recuperees_A = st.slider("Possessions récupérées au milieu de terrain par match", 0, 20, 5, key="possessions_recuperees_A")  
penalties_concedes_A = st.slider("Pénalties concédés", 0, 10, 2, key="penalties_concedes_A")  
arrets_A = st.slider("Arrêts par match", 0, 20, 5, key="arrets_A")  
fautes_A = st.slider("Fautes par match", 0, 20, 10, key="fautes_A")  
cartons_jaunes_A = st.slider("Cartons jaunes", 0, 10, 3, key="cartons_jaunes_A")  
cartons_rouges_A = st.slider("Cartons rouges", 0, 5, 1, key="cartons_rouges_A")  
tactique_A = st.text_input("Tactique de l'équipe A", "4-3-3", key="tactique_A")  
joueurs_cles_A = st.text_input("Joueurs clés de l'équipe A", "Joueur 1, Joueur 2", key="joueurs_cles_A")  
score_joueurs_cles_A = st.slider("Score des joueurs clés (sur 10)", 0, 10, 7, key="score_joueurs_cles_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_par_match_B = st.slider("Buts par match", 0, 10, 1, key="buts_par_match_B")  
nombre_buts_produits_B = st.slider("Nombre de buts produits", 0, 100, 60, key="nombre_buts_produits_B")  
nombre_buts_encaisse_B = st.slider("Nombre de buts encaissés", 0, 100, 40, key="nombre_buts_encaisse_B")  
buts_encaisse_par_match_B = st.slider("Buts encaissés par match", 0, 10, 2, key="buts_encaisse_par_match_B")  
poss_moyenne_B = st.slider("Possession moyenne (%)", 0, 100, 45, key="poss_moyenne_B")  
matchs_sans_encaisser_B = st.slider("Nombre de matchs sans encaisser de but", 0, 40, 5, key="matchs_sans_encaisser_B")  
xG_B = st.slider("Buts attendus (xG)", 0.0, 5.0, 1.5, key="xG_B")  
tirs_cadres_B = st.slider("Tirs cadrés par match", 0, 20, 3, key="tirs_cadres_B")  
pourcentage_tirs_convertis_B = st.slider("Pourcentage de tirs convertis (%)", 0, 100, 15, key="pourcentage_tirs_convertis_B")  
grosses_occasions_B = st.slider("Grosses occasions", 0, 50, 15, key="grosses_occasions_B")  
grosses_occasions_ratees_B = st.slider("Grosses occasions ratées", 0, 50, 5, key="grosses_occasions_ratees_B")  
passes_reussies_B = st.slider("Passes réussies par match", 0, 100, 60, key="passes_reussies_B")  
passes_longues_precises_B = st.slider("Passes longues précises par match", 0, 50, 15, key="passes_longues_precises_B")  
centres_reussis_B = st.slider("Centres réussis par match", 0, 50, 10, key="centres_reussis_B")  
penalties_obtenus_B = st.slider("Pénalties obtenus", 0, 10, 2, key="penalties_obtenus_B")  
touches_surface_adverse_B = st.slider("Touches dans la surface adverse", 0, 50, 20, key="touches_surface_adverse_B")  
corners_B = st.slider("Nombre de corners", 0, 50, 15, key="corners_B")  
corners_par_match_B = st.slider("Nombre de corners par match", 0, 10, 4, key="corners_par_match_B")  
corners_concedes_B = st.slider("Nombre de corners concédés", 0, 50, 10, key="corners_concedes_B")  
xG_concedes_B = st.slider("xG concédés", 0.0, 5.0, 2.0, key="xG_concedes_B")  
interceptions_B = st.slider("Interceptions par match", 0, 20, 4, key="interceptions_B")  
tacles_reussis_B = st.slider("Tacles réussis par match", 0, 20, 5, key="tacles_reussis_B")  
degagements_B = st.slider("Dégagements par match", 0, 20, 4, key="degagements_B")  
possessions_recuperees_B = st.slider("Possessions récupérées au milieu de terrain par match", 0, 20, 4, key="possessions_recuperees_B")  
penalties_concedes_B = st.slider("Pénalties concédés", 0, 10, 3, key="penalties_concedes_B")  
arrets_B = st.slider("Arrêts par match", 0, 20, 4, key="arrets_B")  
fautes_B = st.slider("Fautes par match", 0, 20, 8, key="fautes_B")  
cartons_jaunes_B = st.slider("Cartons jaunes", 0, 10, 4, key="cartons_jaunes_B")  
cartons_rouges_B = st.slider("Cartons rouges", 0, 5, 1, key="cartons_rouges_B")  
tactique_B = st.text_input("Tactique de l'équipe B", "4-4-2", key="tactique_B")  
joueurs_cles_B = st.text_input("Joueurs clés de l'équipe B", "Joueur 3, Joueur 4", key="joueurs_cles_B")  
score_joueurs_cles_B = st.slider("Score des joueurs clés (sur 10)", 0, 10, 6, key="score_joueurs_cles_B")  

# Partie 2 : Historique et contexte du match  
st.markdown("<h2 style='color: #FF5722;'>2. Historique et contexte du match</h2>", unsafe_allow_html=True)  
historique = st.radio(  
    "Résultats des 5 dernières confrontations",  
    ["Équipe A a gagné 3 fois", "Équipe B a gagné 3 fois", "Équilibré (2-2-1)"],  
    key="historique"  
)  
domicile = st.radio(  
    "Quelle équipe joue à domicile ?",  
    ["Équipe A", "Équipe B", "Terrain neutre"],  
    key="domicile"  
)  
meteo = st.radio(  
    "Conditions météo pendant le match",  
    ["Ensoleillé", "Pluie", "Vent"],  
    key="meteo"  
)  

# Partie 3 : Simulateur de match (Prédiction des résultats)  
st.markdown("<h2 style='color: #9C27B0;'>3. Simulateur de match</h2>", unsafe_allow_html=True)  

# Exemple de données fictives pour entraîner les modèles  
data = {  
    "attaque": [70, 65, 80, 60, 75],  
    "defense": [60, 70, 50, 80, 65],  
    "forme_recente": [3.5, 3.0, 4.0, 2.5, 3.8],  
    "motivation": [4.0, 3.5, 4.5, 3.0, 4.2],  
    "resultat": [1, 0, 1, 0, 1]  # 1 = victoire équipe A, 0 = victoire équipe B  
}  
df = pd.DataFrame(data)  

# Entraînement des modèles  
features = ["attaque", "defense", "forme_recente", "motivation"]  
X = df[features]  
y = df["resultat"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Modèle 1 : RandomForest  
model_rf = RandomForestClassifier(random_state=42)  
model_rf.fit(X_train, y_train)  
precision_rf = accuracy_score(y_test, model_rf.predict(X_test))  

# Modèle 2 : Logistic Regression  
model_lr = LogisticRegression(random_state=42)  
model_lr.fit(X_train, y_train)  
precision_lr = accuracy_score(y_test, model_lr.predict(X_test))  

# Affichage des précisions  
st.write(f"Précision du modèle RandomForest : **{precision_rf * 100:.2f}%**")  
st.write(f"Précision du modèle Logistic Regression : **{precision_lr * 100:.2f}%**")  

# Prédictions pour les nouvelles données  
nouvelle_donnee = pd.DataFrame({  
    "attaque": [buts_par_match_A, buts_par_match_B],  
    "defense": [nombre_buts_encaisse_A, nombre_buts_encaisse_B],  
    "forme_recente": [forme_recente_A, forme_recente_B],  
    "motivation": [motivation_A, motivation_B]  
})  

prediction_rf = model_rf.predict_proba(nouvelle_donnee)[0][1]  # Probabilité de victoire pour l'équipe A  
prediction_lr = model_lr.predict_proba(nouvelle_donnee)[0][1]  # Probabilité de victoire pour l'équipe A  

# Prédiction du nombre de buts  
buts_rf_A = predire_buts(prediction_rf, buts_par_match_A, nombre_buts_encaisse_B)  
buts_rf_B = predire_buts(1 - prediction_rf, buts_par_match_B, nombre_buts_encaisse_A)  
buts_lr_A = predire_buts(prediction_lr, buts_par_match_A, nombre_buts_encaisse_B)  
buts_lr_B = predire_buts(1 - prediction_lr, buts_par_match_B, nombre_buts_encaisse_A)  

st.write(f"Probabilité de victoire de l'équipe A selon RandomForest : **{prediction_rf * 100:.2f}%**")  
st.write(f"Probabilité de victoire de l'équipe A selon Logistic Regression : **{prediction_lr * 100:.2f}%**")  

# Partie 4 : Simulateur sous forme de schéma  
st.markdown("<h2 style='color: #FFC107;'>4. Simulateur de match</h2>", unsafe_allow_html=True)  

# Création du graphique  
fig, ax = plt.subplots(figsize=(8, 5))  
bar_width = 0.35  
index = np.arange(2)  

# Données pour le graphique  
buts_rf = [buts_rf_A, buts_rf_B]  
buts_lr = [buts_lr_A, buts_lr_B]  

# Barres pour RandomForest  
ax.bar(index, buts_rf, bar_width, label="RandomForest", color="blue")  

# Barres pour Logistic Regression  
ax.bar(index + bar_width, buts_lr, bar_width, label="Logistic Regression", color="orange")  

# Configuration du graphique  
ax.set_xlabel("Équipes")  
ax.set_ylabel("Nombre de buts prédit")  
ax.set_title("Prédiction du nombre de buts par équipe")  
ax.set_xticks(index + bar_width / 2)  
ax.set_xticklabels(["Équipe A", "Équipe B"])  
ax.legend()  

# Affichage du graphique  
st.pyplot(fig)  

# Partie combinée : Calcul des probabilités et mise optimale  
st.markdown("<h2 style='color: #4CAF50;'>5. Simulateur de paris combinés</h2>", unsafe_allow_html=True)  

# Sélection des équipes pour le combiné
