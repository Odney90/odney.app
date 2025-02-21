import streamlit as st  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  

# Configuration de l'application  
st.set_page_config(page_title="Prédiction de Match", layout="wide")  

# Ajout d'une image de fond (à remplacer par votre URL)  
st.markdown(  
    """  
    <style>  
    .reportview-container {  
        background: url('https://example.com/your-background-image.jpg'); /* Remplacez par l'URL de votre image */  
        background-size: cover;  
        background-position: center;  
        color: white; /* Changez la couleur du texte si nécessaire */  
    }  
    h1 {  
        text-align: center;  
        color: #4CAF50;  
        font-size: 2.5em;  
    }  
    h2 {  
        color: #FF5733; /* Couleur pour les sous-titres */  
    }  
    h3 {  
        color: #FFC300; /* Couleur pour les titres de sections */  
    }  
    h4 {  
        color: #DAF7A6; /* Couleur pour les titres de sous-sections */  
    }  
    .sidebar .sidebar-content {  
        background-color: rgba(255, 255, 255, 0.8); /* Fond semi-transparent pour la barre latérale */  
        border-radius: 10px;  
        padding: 10px;  
    }  
    .stButton>button {  
        background-color: #4CAF50; /* Couleur des boutons */  
        color: white;  
        border-radius: 5px;  
    }  
    .stButton>button:hover {  
        background-color: #45a049; /* Couleur au survol */  
    }  
    </style>  
    """,  
    unsafe_allow_html=True  
)  

# Titre de la page d'analyse  
st.markdown("<h1>Analyse des Équipes</h1>", unsafe_allow_html=True)
# Historique des équipes  
st.subheader("Historique des équipes")  
st.markdown("Veuillez entrer le nombre de victoires, de défaites et de nuls pour chaque équipe.")  

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

total_buts_produits_A = st.number_input("Buts total produits", min_value=0, value=50, key="total_buts_produits_A", step=1)  
moyenne_buts_produits_A = st.number_input("Moyenne de buts produits par match", min_value=0.0, value=1.5, step=0.1, key="moyenne_buts_produits_A")  
total_buts_encaisses_A = st.number_input("Buts total encaissés", min_value=0, value=30, key="total_buts_encaisses_A", step=1)  
moyenne_buts_encaisses_A = st.number_input("Moyenne de buts encaissés par match", min_value=0.0, value=1.0, step=0.1, key="moyenne_buts_encaisses_A")  
poss_moyenne_A = st.number_input("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A", step=1)  
xG_A = st.number_input("Buts attendus (xG)", min_value=0.0, value=1.5, step=0.1, key="xG_A")  
tirs_cadres_A = st.number_input("Tirs cadrés par match", min_value=0, value=10, key="tirs_cadres_A", step=1)  
pourcentage_tirs_convertis_A = st.number_input("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=20, key="pourcentage_tirs_convertis_A", step=1)  
grosses_occasions_A = st.number_input("Grosses occasions", min_value=0, value=5, key="grosses_occasions_A", step=1)  
grosses_occasions_ratees_A = st.number_input("Grosses occasions ratées", min_value=0, value=2, key="grosses_occasions_ratees_A", step=1)  
passes_reussies_A = st.number_input("Passes réussies/match (sur 1000)", min_value=0, value=300, key="passes_reussies_A", step=1)  
passes_longues_precises_A = st.number_input("Passes longues précises/match", min_value=0, value=15, key="passes_longues_precises_A", step=1)  
centres_reussis_A = st.number_input("Centres réussis/match", min_value=0, value=5, key="centres_reussis_A", step=1)  
penalties_obtenus_A = st.number_input("Pénalties obtenus", min_value=0, value=1, key="penalties_obtenus_A", step=1)  
touches_surface_adverse_A = st.number_input("Touches dans la surface adverse", min_value=0, value=10, key="touches_surface_adverse_A", step=1)  
corners_A = st.number_input("Nombre de corners", min_value=0, value=4, key="corners_A", step=1)  
corners_par_match_A = st.number_input("Corners par match", min_value=0, value=2, key="corners_par_match_A", step=1)  
corners_concedes_A = st.number_input("Corners concédés par match", min_value=0, value=3, key="corners_concedes_A", step=1)  
xG_concedes_A = st.number_input("xG concédés", min_value=0.0, value=1.0, step=0.1, key="xG_concedes_A")  
interceptions_A = st.number_input("Interceptions par match", min_value=0, value=10, key="interceptions_A", step=1)  
tacles_reussis_A = st.number_input("Tacles réussis par match", min_value=0, value=5, key="tacles_reussis_A", step=1)  
degagements_A = st.number_input("Dégagements par match", min_value=0, value=8, key="degagements_A", step=1)  
possessions_recuperees_A = st.number_input("Possessions récupérées au milieu/match", min_value=0, value=5, key="possessions_recuperees_A", step=1)  
penalties_concedes_A = st.number_input("Pénalties concédés", min_value=0, value=0, key="penalties_concedes_A", step=1)  
arrets_A = st.number_input("Arrêts par match", min_value=0, value=3, key="arrets_A", step=1)  
fautes_A = st.number_input("Fautes par match", min_value=0, value=10, key="fautes_A", step=1)  
cartons_jaunes_A = st.number_input("Cartons jaunes", min_value=0, value=2, key="cartons_jaunes_A", step=1)  
cartons_rouges_A = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_A", step=1)  
tactique_A = st.text_input("Tactique de l'équipe A", value="4-3-3", key="tactique_A")  

# Section pour les joueurs clés de l'équipe A  
st.subheader("Joueurs Clés de l'équipe A")  
absents_A = st.number_input("Nombre de joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_A")  
ratings_A = []  
for i in range(5):  
    rating = st.number_input(f"Rating du joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_A_{i}")  
    ratings_A.append(rating)
    # Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  

total_buts_produits_B = st.number_input("Buts total produits", min_value=0, value=40, key="total_buts_produits_B", step=1)  
moyenne_buts_produits_B = st.number_input("Moyenne de buts produits par match", min_value=0.0, value=1.0, step=0.1, key="moyenne_buts_produits_B")  
total_buts_encaisses_B = st.number_input("Buts total encaissés", min_value=0, value=35, key="total_buts_encaisses_B", step=1)  
moyenne_buts_encaisses_B = st.number_input("Moyenne de buts encaissés par match", min_value=0.0, value=1.5, step=0.1, key="moyenne_buts_encaisses_B")  
poss_moyenne_B = st.number_input("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B", step=1)  
xG_B = st.number_input("Buts attendus (xG)", min_value=0.0, value=1.0, step=0.1, key="xG_B")  
tirs_cadres_B = st.number_input("Tirs cadrés par match", min_value=0, value=8, key="tirs_cadres_B", step=1)  
pourcentage_tirs_convertis_B = st.number_input("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=15, key="pourcentage_tirs_convertis_B", step=1)  
grosses_occasions_B = st.number_input("Grosses occasions", min_value=0, value=3, key="grosses_occasions_B", step=1)  
grosses_occasions_ratees_B = st.number_input("Grosses occasions ratées", min_value=0, value=1, key="grosses_occasions_ratees_B", step=1)  
passes_reussies_B = st.number_input("Passes réussies/match (sur 1000)", min_value=0, value=250, key="passes_reussies_B", step=1)  
passes_longues_precises_B = st.number_input("Passes longues précises/match", min_value=0, value=10, key="passes_longues_precises_B", step=1)  
centres_reussis_B = st.number_input("Centres réussis/match", min_value=0, value=3, key="centres_reussis_B", step=1)  
penalties_obtenus_B = st.number_input("Pénalties obtenus", min_value=0, value=0, key="penalties_obtenus_B", step=1)  
touches_surface_adverse_B = st.number_input("Touches dans la surface adverse", min_value=0, value=8, key="touches_surface_adverse_B", step=1)  
corners_B = st.number_input("Nombre de corners", min_value=0, value=3, key="corners_B", step=1)  
corners_par_match_B = st.number_input("Corners par match", min_value=0, value=1, key="corners_par_match_B", step=1)  
corners_concedes_B = st.number_input("Corners concédés par match", min_value=0, value=2, key="corners_concedes_B", step=1)  
xG_concedes_B = st.number_input("xG concédés", min_value=0.0, value=1.5, step=0.1, key="xG_concedes_B")  
interceptions_B = st.number_input("Interceptions par match", min_value=0, value=8, key="interceptions_B", step=1)  
tacles_reussis_B = st.number_input("Tacles réussis par match", min_value=0, value=4, key="tacles_reussis_B", step=1)  
degagements_B = st.number_input("Dégagements par match", min_value=0, value=6, key="degagements_B", step=1)  
possessions_recuperees_B = st.number_input("Possessions récupérées au milieu/match", min_value=0, value=4, key="possessions_recuperees_B", step=1)  
penalties_concedes_B = st.number_input("Pénalties concédés", min_value=0, value=1, key="penalties_concedes_B", step=1)  
arrets_B = st.number_input("Arrêts par match", min_value=0, value=2, key="arrets_B", step=1)  
fautes_B = st.number_input("Fautes par match", min_value=0, value=12, key="fautes_B", step=1)  
cartons_jaunes_B = st.number_input("Cartons jaunes", min_value=0, value=1, key="cartons_jaunes_B", step=1)  
cartons_rouges_B = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_B", step=1)  
tactique_B = st.text_input("Tactique de l'équipe B", value="4-2-3-1", key="tactique_B")  

# Section pour les joueurs clés de l'équipe B  
st.subheader("Joueurs Clés de l'équipe B")  
absents_B = st.number_input("Nombre de joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_B")  
ratings_B = []  
for i in range(5):  
    rating = st.number_input(f"Rating du joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_B_{i}")  
    ratings_B.append(rating)
    # Motivation des équipes  
st.subheader("Motivation des Équipes")  
motivation_A = st.number_input("Niveau de motivation de l'équipe A (1 à 10)", min_value=1, max_value=10, value=5, key="motivation_A", step=1)  
motivation_B = st.number_input("Niveau de motivation de l'équipe B (1 à 10)", min_value=1, max_value=10, value=5, key="motivation_B", step=1)  

# Affichage des niveaux de motivation  
st.write(f"Niveau de motivation de l'équipe A : **{motivation_A}**")  
st.write(f"Niveau de motivation de l'équipe B : **{motivation_B}**")
# Prédiction des buts avec la méthode de Poisson  
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
st.subheader("Prédiction des Buts")  
st.write(f"Buts attendus pour l'équipe A : **{buts_moyens_A:.2f}**")  
st.write(f"Buts attendus pour l'équipe B : **{buts_moyens_B:.2f}**")
# Prédiction multivariable avec régression logistique (version améliorée)  
st.subheader("Prédiction Multivariable avec Régression Logistique (Améliorée)")  

# Préparation des données pour le modèle  
X = np.array([  
    [total_buts_produits_A, moyenne_buts_produits_A, total_buts_encaisses_A, moyenne_buts_encaisses_A, poss_moyenne_A, motivation_A, absents_A, np.mean(ratings_A), forme_recente_A],  
    [total_buts_produits_B, moyenne_buts_produits_B, total_buts_encaisses_B, moyenne_buts_encaisses_B, poss_moyenne_B, motivation_B, absents_B, np.mean(ratings_B), forme_recente_B]  
])  
y = np.array([1, 0])  # 1 pour l'équipe A, 0 pour l'équipe B  

# Normalisation des données (important pour la régression logistique)  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Entraînement du modèle avec régularisation pour éviter le surapprentissage  
model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')  # L2 régularisation  
model.fit(X_scaled, y)  

# Prédiction des résultats  
prediction = model.predict_proba(X_scaled)  

# Affichage des probabilités de victoire  
st.write(f"Probabilité de victoire pour l'équipe A : **{prediction[0][1]:.2%}**")  
st.write(f"Probabilité de victoire pour l'équipe B : **{prediction[1][1]:.2%}**")
# Interprétation des coefficients du modèle  
st.subheader("Interprétation du Modèle")  
feature_names = ['Buts produits totaux', 'Buts produits/match', 'Buts encaissés totaux', 'Buts encaissés/match', 'Possession moyenne', 'Motivation', 'Absents', 'Rating joueurs', 'Forme récente']  
coefficients = model.coef_[0]  

for feature, coef in zip(feature_names, coefficients):  
    st.write(f"{feature}: Coefficient = {coef:.2f}")  

# Calcul des paris alternatifs  
double_chance_A = prediction[0][1] + (1 - prediction[1][1])  # Équipe A gagne ou match nul  
double_chance_B = prediction[1][1] + (1 - prediction[0][1])  # Équipe B gagne ou match nul  

# Affichage des paris alternatifs  
st.subheader("Paris Alternatifs")  
st.write(f"Probabilité de double chance pour l'équipe A (gagner ou match nul) : **{double_chance_A:.2%}**")  
st.write(f"Probabilité de double chance pour l'équipe B (gagner ou match nul) : **{double_chance_B:.2%}**")
# Seuil de probabilité pour le résultat nul  
seuil_nul = st.number_input("Seuil de probabilité pour le résultat nul", min_value=0.0, max_value=1.0, value=0.3, step=0.05)  

# Analyse du résultat nul  
proba_nul = 1 - (prediction[0][1] + prediction[1][1])  
st.write(f"Probabilité estimée d'un match nul : **{proba_nul:.2%}**")  

if proba_nul >= seuil_nul:  
    st.warning("Attention : La probabilité d'un match nul est élevée selon le seuil défini.")  
else:  
    st.success("La probabilité d'un match nul est considérée comme faible.")  

# Conclusion  
st.subheader("Conclusion")  
st.write("Merci d'avoir utilisé l'outil d'analyse de match ! Cette application vous permet d'évaluer les performances des équipes et de prendre des décisions éclairées sur les paris.")  
st.write("N'oubliez pas que les prédictions sont basées sur les données que vous avez fournies et que le football reste imprévisible.")
