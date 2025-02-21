import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import poisson  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  

# Configuration de l'application  
st.set_page_config(page_title="Prédiction de Match", layout="wide")  

# Titre de la page d'analyse  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Analyse des Équipes</h1>", unsafe_allow_html=True)  

# Historique des équipes  
st.subheader("Historique des équipes")  
st.markdown("Veuillez entrer le nombre de victoires, de défaites et de nuls pour chaque équipe sur la saison.")  

# Saisir les résultats de la saison  
victoires_A = st.number_input("Victoires de l'équipe A", min_value=0, value=10, key="victoires_A")  
nuls_A = st.number_input("Nuls de l'équipe A", min_value=0, value=5, key="nuls_A")  
defaites_A = st.number_input("Défaites de l'équipe A", min_value=0, value=5, key="defaites_A")  

victoires_B = st.number_input("Victoires de l'équipe B", min_value=0, value=8, key="victoires_B")  
nuls_B = st.number_input("Nuls de l'équipe B", min_value=0, value=7, key="nuls_B")  
defaites_B = st.number_input("Défaites de l'équipe B", min_value=0, value=5, key="defaites_B")  

# Historique des 5 derniers matchs  
st.subheader("Historique des 5 Derniers Matchs")  
st.markdown("Veuillez entrer les résultats des 5 derniers matchs pour chaque équipe (1 pour victoire, 0 pour match nul, -1 pour défaite).")  

historique_A = []  
historique_B = []  
for i in range(5):  
    resultat_A = st.selectbox(f"Match {i + 1} de l'équipe A", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_A_{i}")  
    historique_A.append(1 if resultat_A == "Victoire (1)" else (0 if resultat_A == "Match Nul (0)" else -1))  
    
    resultat_B = st.selectbox(f"Match {i + 1} de l'équipe B", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_B_{i}")  
    historique_B.append(1 if resultat_B == "Victoire (1)" else (0 if resultat_B == "Match Nul (0)" else -1))  

# Calcul de la forme récente sur 5 matchs  
def calculer_forme(historique):  
    return sum(historique) / len(historique)  # Moyenne des résultats  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  

st.write(f"Forme récente de l'équipe A (moyenne sur 5 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 5 matchs) : **{forme_recente_B:.2f}**")  

# Visualisation de la forme des équipes  
st.subheader("Visualisation de la Forme des Équipes")  
fig, ax = plt.subplots()  
ax.plot(range(1, 6), historique_A, marker='o', label='Équipe A', color='blue')  
ax.plot(range(1, 6), historique_B, marker='o', label='Équipe B', color='red')  
ax.set_xticks(range(1, 6))  
ax.set_xticklabels([f"Match {i}" for i in range(1, 6)], rotation=45)  
ax.set_ylim(-1.5, 1.5)  
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')  
ax.set_ylabel('Résultat')  
ax.set_title('Historique des Performances des Équipes (5 Derniers Matchs)')  
ax.legend()  
st.pyplot(fig)  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
buts_produits_A = st.slider("Nombre de buts produits (moyenne sur la saison)", min_value=0, max_value=10, value=2, key="buts_produits_A")  
buts_encaisse_A = st.slider("Nombre de buts encaissés (moyenne sur la saison)", min_value=0, max_value=10, value=1, key="buts_encaisse_A")  
poss_moyenne_A = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
xG_A = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.5, key="xG_A")  
tirs_cadres_A = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=10, key="tirs_cadres_A")  
pourcentage_tirs_convertis_A = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=20, key="pourcentage_tirs_convertis_A")  
grosses_occasions_A = st.number_input("Grosses occasions", min_value=0, value=5, key="grosses_occasions_A")  
grosses_occasions_ratees_A = st.number_input("Grosses occasions ratées", min_value=0, value=2, key="grosses_occasions_ratees_A")  
passes_reussies_A = st.slider("Passes réussies par match (sur 1000)", min_value=0, max_value=1000, value=300, key="passes_reussies_A")  
passes_longues_precises_A = st.slider("Passes longues précises par match", min_value=0, max_value=100, value=15, key="passes_longues_precises_A")  
centres_reussis_A = st.slider("Centres réussis par match", min_value=0, max_value=50, value=5, key="centres_reussis_A")  
penalties_obtenus_A = st.number_input("Pénalties obtenus", min_value=0, value=1, key="penalties_obtenus_A")  
touches_surface_adverse_A = st.number_input("Touches dans la surface adverse", min_value=0, value=10, key="touches_surface_adverse_A")  
corners_A = st.number_input("Nombre de corners", min_value=0, value=4, key="corners_A")  
corners_par_match_A = st.number_input("Nombre de corners par match", min_value=0, value=2, key="corners_par_match_A")  
corners_concedes_A = st.number_input("Nombre de corners concédés par match", min_value=0, value=3, key="corners_concedes_A")  
xG_concedes_A = st.slider("xG concédés", min_value=0.0, max_value=5.0, value=1.0, key="xG_concedes_A")  
interceptions_A = st.slider("Interceptions par match", min_value=0, max_value=50, value=10, key="interceptions_A")  
tacles_reussis_A = st.slider("Tacles réussis par match", min_value=0, max_value=50, value=5, key="tacles_reussis_A")  
degagements_A = st.slider("Dégagements par match", min_value=0, max_value=50, value=8, key="degagements_A")  
possessions_recuperees_A = st.slider("Possessions récupérées au milieu de terrain par match", min_value=0, max_value=50, value=5, key="possessions_recuperees_A")  
penalties_concedes_A = st.number_input("Pénalties concédés", min_value=0, value=0, key="penalties_concedes_A")  
arrets_A = st.slider("Arrêts par match", min_value=0, max_value=20, value=3, key="arrets_A")  
fautes_A = st.slider("Fautes par match", min_value=0, max_value=50, value=10, key="fautes_A")  
cartons_jaunes_A = st.number_input("Cartons jaunes", min_value=0, value=2, key="cartons_jaunes_A")  
cartons_rouges_A = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_A")  
tactique_A = st.text_input("Tactique de l'équipe A", value="4-3-3", key="tactique_A")  
joueurs_cles_A = st.text_input("Joueurs clés de l'équipe A", value="Joueur 1, Joueur 2", key="joueurs_cles_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_produits_B = st.slider("Nombre de buts produits (moyenne sur la saison)", min_value=0, max_value=10, value=1, key="buts_produits_B")  
buts_encaisse_B = st.slider("Nombre de buts encaissés (moyenne sur la saison)", min_value=0, max_value=10, value=2, key="buts_encaisse_B")  
poss_moyenne_B = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
xG_B = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.0, key="xG_B")  
tirs_cadres_B = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=8, key="tirs_cadres_B")  
pourcentage_tirs_convertis_B = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=15, key="pourcentage_tirs_convertis_B")  
grosses_occasions_B = st.number_input("Grosses occasions", min_value=0, value=3, key="grosses_occasions_B")  
grosses_occasions_ratees_B = st.number_input("Grosses occasions ratées", min_value=0, value=1, key="grosses_occasions_ratees_B")  
passes_reussies_B = st.slider("Passes réussies par match (sur 1000)", min_value=0, max_value=1000, value=250, key="passes_reussies_B")  
passes_longues_precises_B = st.slider("Passes longues précises par match", min_value=0, max_value=100, value=10, key="passes_longues_precises_B")  
centres_reussis_B = st.slider("Centres réussis par match", min_value=0, max_value=50, value=3, key="centres_reussis_B")  
penalties_obtenus_B = st.number_input("Pénalties obtenus", min_value=0, value=0, key="penalties_obtenus_B")  
touches_surface_adverse_B = st.number_input("Touches dans la surface adverse", min_value=0, value=8, key="touches_surface_adverse_B")  
corners_B = st.number_input("Nombre de corners", min_value=0, value=3, key="corners_B")  
corners_par_match_B = st.number_input("Nombre de corners par match", min_value=0, value=1, key="corners_par_match_B")  
corners_concedes_B = st.number_input("Nombre de corners concédés par match", min_value=0, value=2, key="corners_concedes_B")  
xG_concedes_B = st.slider("xG concédés", min_value=0.0, max_value=5.0, value=1.5, key="xG_concedes_B")  
interceptions_B = st.slider("Interceptions par match", min_value=0, max_value=50, value=8, key="interceptions_B")  
tacles_reussis_B = st.slider("Tacles réussis par match", min_value=0, max_value=50, value=4, key="tacles_reussis_B")  
degagements_B = st.slider("Dégagements par match", min_value=0, max_value=50, value=6, key="degagements_B")  
possessions_recuperees_B = st.slider("Possessions récupérées au milieu de terrain par match", min_value=0, max_value=50, value=4, key="possessions_recuperees_B")  
penalties_concedes_B = st.number_input("Pénalties concédés", min_value=0, value=1, key="penalties_concedes_B")  
arrets_B = st.slider("Arrêts par match", min_value=0, max_value=20, value=2, key="arrets_B")  
fautes_B = st.slider("Fautes par match", min_value=0, max_value=50, value=12, key="fautes_B")  
cartons_jaunes_B = st.number_input("Cartons jaunes", min_value=0, value=1, key="cartons_jaunes_B")  
cartons_rouges_B = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_B")  
tactique_B = st.text_input("Tactique de l'équipe B", value="4-2-3-1", key="tactique_B")  
joueurs_cles_B = st.text_input("Joueurs clés de l'équipe B", value="Joueur 3, Joueur 4", key="joueurs_cles_B")  

# Conditions du match (affichage uniquement)  
st.subheader("Conditions du Match")  
meteo = st.selectbox("Conditions Météorologiques", options=["Ensoleillé", "Pluvieux", "Neigeux", "Nuageux"], index=0, key="meteo")  
avantage_terrain = st.selectbox("Équipe à domicile", options=["Équipe A", "Équipe B"], index=0, key="avantage_terrain")  
motivation = st.slider("Niveau de motivation (1 à 10)", min_value=1, max_value=10, value=5, key="motivation")  

# Affichage des conditions météorologiques  
st.write(f"Conditions Météorologiques : **{meteo}**")  
st.write(f"Équipe à domicile : **{avantage_terrain}**")  
st.write(f"Niveau de motivation : **{motivation}**")  

# Fonction pour prédire les buts avec distribution de Poisson  
def prediction_buts_poisson(xG_A, xG_B):  
    # Calcul des probabilités de marquer 0 à 5 buts  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  
    
    # Calcul des buts attendus  
    buts_attendus_A = sum(i * prob for i, prob in enumerate(buts_A))  
    buts_attendus_B = sum(i * prob for i, prob in enumerate(buts_B))  
    
    return buts_attendus_A, buts_attendus_B  

# Prédiction des buts avec la méthode de Poisson  
buts_moyens_A, buts_moyens_B = prediction_buts_poisson(xG_A, xG_B)  

# Affichage des résultats de la prédiction  
st.subheader("Résultats de la Prédiction (Méthode de Poisson)")  
st.write(f"Prédiction des buts pour l'équipe A : **{buts_moyens_A:.2f}**")  
st.write(f"Prédiction des buts pour l'équipe B : **{buts_moyens_B:.2f}**")  

# Calcul des pourcentages de victoire  
pourcentage_victoire_A = (buts_moyens_A / (buts_moyens_A + buts_moyens_B)) * 100 if (buts_moyens_A + buts_moyens_B) > 0 else 0  
pourcentage_victoire_B = (buts_moyens_B / (buts_moyens_A + buts_moyens_B)) * 100 if (buts_moyens_A + buts_moyens_B) > 0 else 0  

# Affichage des pourcentages de victoire  
st.write(f"Pourcentage de victoire pour l'équipe A : **{pourcentage_victoire_A:.2f}%**")  
st.write(f"Pourcentage de victoire pour l'équipe B : **{pourcentage_victoire_B:.2f}%**")  

# Visualisation des prédictions  
fig, ax = plt.subplots()  
ax.bar(["Équipe A", "Équipe B"], [buts_moyens_A, buts_moyens_B], color=['blue', 'red'])  
ax.set_ylabel('Buts Prédits')  
ax.set_title('Prédictions de Buts pour le Match (Méthode de Poisson)')  
st.pyplot(fig)  

# Section de conseils de paris  
st.subheader("Conseils de Paris (Méthode de Poisson)")  
if pourcentage_victoire_A > pourcentage_victoire_B:  
    st.write("Conseil : Pariez sur la victoire de l'équipe A.")  
else:  
    st.write("Conseil : Pariez sur la victoire de l'équipe B.")  

# Méthode d'analyse multi-variable  
st.subheader("Analyse Multi-Variable")  

# Préparation des données pour la régression logistique  
data = {  
    'xG_A': [xG_A],
    'xG_B': [xG_B],  
    'tirs_cadres_A': [tirs_cadres_A],  
    'tirs_cadres_B': [tirs_cadres_B],  
    'poss_moyenne_A': [poss_moyenne_A],  
    'poss_moyenne_B': [poss_moyenne_B],  
    'motivation': [motivation],  
    'avantage_terrain': [1 if avantage_terrain == "Équipe A" else 0]  # Convertir en binaire  
}  

# Convertir en DataFrame  
df_multi = pd.DataFrame(data)  

# Exemple de données historiques pour l'entraînement (à remplacer par des données réelles)  
historical_data = {  
    'xG_A': [1.5, 2.0, 1.0, 2.5],  # Buts attendus pour l'équipe A  
    'xG_B': [1.0, 1.5, 2.0, 1.0],  # Buts attendus pour l'équipe B  
    'tirs_cadres_A': [5, 7, 3, 8],  # Tirs cadrés par l'équipe A  
    'tirs_cadres_B': [4, 6, 5, 2],  # Tirs cadrés par l'équipe B  
    'poss_moyenne_A': [55, 60, 50, 65],  # Possession moyenne de l'équipe A  
    'poss_moyenne_B': [45, 40, 50, 35],  # Possession moyenne de l'équipe B  
    'motivation': [8, 7, 6, 9],  # Niveau de motivation  
    'result': [1, 1, 0, 1]  # 1 = victoire A, 0 = victoire B  
}  

# Créer le DataFrame à partir des données historiques  
df_historical = pd.DataFrame(historical_data)  

# Séparer les caractéristiques et la cible  
X = df_historical[['xG_A', 'xG_B', 'tirs_cadres_A', 'tirs_cadres_B', 'poss_moyenne_A', 'poss_moyenne_B', 'motivation', 'avantage_terrain']]  
y = df_historical['result']  

# Diviser les données en ensembles d'entraînement et de test  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Créer et entraîner le modèle  
model = LogisticRegression()  
model.fit(X_train, y_train)  

# Prédire les résultats avec les nouvelles données  
prediction_multi = model.predict(df_multi)  

# Affichage des résultats de la prédiction multi-variable  
st.subheader("Résultats de la Prédiction (Méthode Multi-Variable)")  
if prediction_multi[0] == 1:  
    st.write("L'équipe A est prédite pour gagner.")  
else:  
    st.write("L'équipe B est prédite pour gagner.")  

# Évaluer le modèle sur l'ensemble de test  
accuracy = accuracy_score(y_test, model.predict(X_test))  
st.write(f"Précision du modèle multi-variable : **{accuracy:.2f}**")  

# Visualisation des résultats de la méthode multi-variable  
fig, ax = plt.subplots()  
ax.bar(["Équipe A", "Équipe B"], [prediction_multi[0], 1 - prediction_multi[0]], color=['blue', 'red'])  
ax.set_ylabel('Probabilité de Victoire')  
ax.set_title('Prédictions de Victoire pour le Match (Méthode Multi-Variable)')  
st.pyplot(fig)  

# Section de conseils de paris pour la méthode multi-variable  
st.subheader("Conseils de Paris (Méthode Multi-Variable)")  
if prediction_multi[0] == 1:  
    st.write("Conseil : Pariez sur la victoire de l'équipe A.")  
else:  
    st.write("Conseil : Pariez sur la victoire de l'équipe B.")  

# Fin de l'application  
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Merci d'avoir utilisé l'outil d'analyse de match !</h2>", unsafe_allow_html=True)
