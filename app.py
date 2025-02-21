import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import poisson  

# Configuration de l'application  
st.set_page_config(page_title="Prédiction de Match", layout="wide")  

# Titre de la page d'analyse  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Analyse des Équipes</h1>", unsafe_allow_html=True)  

# Historique des équipes  
st.subheader("Historique des équipes")  
st.markdown("Veuillez entrer le nombre de victoires, de défaites et de nuls pour chaque équipe sur la saison.")  

# Saisir les résultats de la saison  
victoires_A = st.number_input("Victoires de l'équipe A", min_value=0, value=10)  
nuls_A = st.number_input("Nuls de l'équipe A", min_value=0, value=5)  
defaites_A = st.number_input("Défaites de l'équipe A", min_value=0, value=5)  

victoires_B = st.number_input("Victoires de l'équipe B", min_value=0, value=8)  
nuls_B = st.number_input("Nuls de l'équipe B", min_value=0, value=7)  
defaites_B = st.number_input("Défaites de l'équipe B", min_value=0, value=5)  

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
passes_reussies_A = st.slider("Passes réussies par match", min_value=0, max_value=100, value=30, key="passes_reussies_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_produits_B = st.slider("Nombre de buts produits (moyenne sur la saison)", min_value=0, max_value=10, value=1, key="buts_produits_B")  
buts_encaisse_B = st.slider("Nombre de buts encaissés (moyenne sur la saison)", min_value=0, max_value=10, value=2, key="buts_encaisse_B")  
poss_moyenne_B = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
xG_B = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.0, key="xG_B")  
tirs_cadres_B = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=8, key="tirs_cadres_B")  
passes_reussies_B = st.slider("Passes réussies par match", min_value=0, max_value=100, value=25, key="passes_reussies_B")  

# Fonction pour prédire les buts avec distribution de Poisson  
def prediction_buts_poisson(xG_A, xG_B):  
    # Calcul des probabilités de marquer 0 à 5 buts  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  
    
    # Calcul des buts attendus  
    buts_attendus_A = sum(i * prob for i, prob in enumerate(buts_A))  
    buts_attendus_B = sum(i * prob for i, prob in enumerate(buts_B))  
    
    return buts_attendus_A, buts_attendus_B  

# Prédiction des buts  
buts_moyens_A, buts_moyens_B = prediction_buts_poisson(xG_A, xG_B)  

# Affichage des résultats de la prédiction  
st.subheader("Résultats de la Prédiction")  
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
ax.set_title('Prédictions de Buts pour le Match')  
st.pyplot(fig)  

# Section de conseils de paris  
st.subheader("Conseils de Paris")  
if pourcentage_victoire_A > pourcentage_victoire_B:  
    st.write("Conseil : Pariez sur la victoire de l'équipe A.")  
else:  
    st.write("Conseil : Pariez sur la victoire de l'équipe B.")  

# Fin de l'application  
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Merci d'avoir utilisé l'outil d'analyse de match !</h2>", unsafe_allow_html=True)
