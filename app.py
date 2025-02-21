import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données des équipes
@st.cache_data
def load_data():
    return pd.read_excel('equipes_europeennes.xlsx')

data = load_data()

# Fonction pour convertir les cotes en probabilités implicites
def convertir_cote_en_probabilite(cote):
    return 1 / cote

# Fonction pour calculer la mise optimale selon Kelly
def mise_kelly(prob, cote, bankroll, fraction=1):
    avantage = (cote * prob) - 1
    mise_fraction = avantage / (cote - 1)
    return bankroll * mise_fraction * fraction

# Interface utilisateur
st.title("Analyse et Prédiction de Matchs de Football")

# Sélection des équipes
st.header("Sélection des Équipes")
equipe_A = st.selectbox("Sélectionnez l'équipe A", data['Équipe'].unique())
equipe_B = st.selectbox("Sélectionnez l'équipe B", data['Équipe'].unique())

# Affichage des statistiques
def afficher_stats(equipe):
    st.subheader(f"Statistiques de {equipe}")
    st.write(data[data['Équipe'] == equipe])

afficher_stats(equipe_A)
afficher_stats(equipe_B)

# Saisie des cotes
st.header("Saisie des Cotes")
cote_A = st.number_input(f"Cote pour {equipe_A}", min_value=1.01, value=2.0)
cote_B = st.number_input(f"Cote pour {equipe_B}", min_value=1.01, value=3.0)
cote_nul = st.number_input("Cote pour le match nul", min_value=1.01, value=3.5)

# Calcul des probabilités implicites
prob_A = convertir_cote_en_probabilite(cote_A)
prob_B = convertir_cote_en_probabilite(cote_B)
prob_nul = convertir_cote_en_probabilite(cote_nul)
somme_probs = prob_A + prob_B + prob_nul
prob_A /= somme_probs
prob_B /= somme_probs
prob_nul /= somme_probs

st.write(f"Probabilité implicite de victoire de {equipe_A} : {prob_A:.2%}")
st.write(f"Probabilité implicite de victoire de {equipe_B} : {prob_B:.2%}")
st.write(f"Probabilité implicite de match nul : {prob_nul:.2%}")

# Partie pour la mise de Kelly
st.header("Mise de Kelly")
bankroll = st.number_input("Entrez votre bankroll", min_value=10, value=100)
kelly_A = mise_kelly(prob_A, cote_A, bankroll)
kelly_B = mise_kelly(prob_B, cote_B, bankroll)

st.write(f"Mise recommandée sur {equipe_A} : {kelly_A:.2f}")
st.write(f"Mise recommandée sur {equipe_B} : {kelly_B:.2f}")

# Partie pour la gestion de bankroll
st.header("Gestion de Bankroll")
mises = st.number_input("Total des mises en cours", min_value=0.0, value=0.0)
solde_restant = bankroll - mises
st.write(f"Solde restant après mise : {solde_restant:.2f}")

# Simulateur de match basé sur algorithmes
st.header("Simulateur de Match")
if st.button("Simuler le match"):
    pred_prob_A = np.random.rand()
    pred_prob_B = np.random.rand()
    pred_prob_nul = 1 - (pred_prob_A + pred_prob_B)
    
    st.write(f"Probabilité de victoire {equipe_A}: {pred_prob_A:.2%}")
    st.write(f"Probabilité de victoire {equipe_B}: {pred_prob_B:.2%}")
    st.write(f"Probabilité de match nul: {pred_prob_nul:.2%}")
