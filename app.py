import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt  
from scipy.stats import poisson  

# Configuration de l'application  
st.set_page_config(page_title="Prédiction de Match", layout="wide")  

# Titre de la page d'analyse  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Analyse des Équipes</h1>", unsafe_allow_html=True)  

# Historique des équipes  
st.subheader("Historique des équipes")  
st.markdown("Veuillez entrer les résultats des 20 derniers matchs pour chaque équipe (1 pour victoire, 0 pour match nul, -1 pour défaite).")  

# Historique des résultats  
historique_A = []  
historique_B = []  
for i in range(20):  
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

# Calcul de la forme récente sur 20 matchs  
def calculer_forme(historique):  
    return sum(historique) / len(historique)  # Moyenne des résultats  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  
forme_face_a_face = calculer_forme(historique_face_a_face)  

st.write(f"Forme récente de l'équipe A (moyenne sur 20 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 20 matchs) : **{forme_recente_B:.2f}**")  
st.write(f"Forme récente en face-à-face (moyenne sur 5 matchs) : **{forme_face_a_face:.2f}**")  

# Visualisation de la forme des équipes  
st.subheader("Visualisation de la Forme des Équipes")  
fig, ax = plt.subplots()  
ax.plot(range(1, 21), historique_A, marker='o', label='Équipe A', color='blue')  
ax.plot(range(1, 21), historique_B, marker='o', label='Équipe B', color='red')  
ax.plot(range(1, 6), historique_face_a_face, marker='o', label='Face-à-Face', color='green')  
ax.set_xticks(range(1, 21))  
ax.set_xticklabels([f"Match {i}" for i in range(1, 21)], rotation=45)  
ax.set_ylim(-1.5, 1.5)  
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')  
ax.set_ylabel('Résultat')  
ax.set_title('Historique des Performances des Équipes')  
ax.legend()  
st.pyplot(fig)  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
buts_produits_A = st.slider("Nombre de buts produits (moyenne sur 20 matchs)", min_value=0, max_value=10, value=2, key="buts_produits_A")  
buts_encaisse_A = st.slider("Nombre de buts encaissés (moyenne sur 20 matchs)", min_value=0, max_value=10, value=1, key="buts_encaisse_A")  
poss_moyenne_A = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
xG_A = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.5, key="xG_A")  
tirs_cadres_A = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=10, key="tirs_cadres_A")  
passes_reussies_A = st.slider("Passes réussies par match", min_value=0, max_value=100, value=30, key="passes_reussies_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_produits_B = st.slider("Nombre de buts produits (moyenne sur 20 matchs)", min_value=0, max_value=10, value=1, key="buts_produits_B")  
buts_encaisse_B = st.slider("Nombre de buts encaissés (moyenne sur 20 matchs)", min_value=0, max_value=10, value=2, key="buts_encaisse_B")  
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
