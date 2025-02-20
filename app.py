import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  


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


# Fonction pour ajuster les probabilités en fonction des tactiques  
def ajuster_probabilite_tactique(probabilite, tactique_A, tactique_B):  
    avantages = {  
        "4-4-2": {"contre": ["4-3-3", "3-4-3"], "bonus": 0.03},  
        "4-2-3-1": {"contre": ["4-4-2", "4-5-1"], "bonus": 0.04},  
        "4-3-3": {"contre": ["4-4-2", "5-3-2"], "bonus": 0.05},  
        "4-5-1": {"contre": ["3-4-3", "4-3-3"], "bonus": 0.03},  
        "3-4-3": {"contre": ["5-3-2", "4-5-1"], "bonus": 0.05},  
        "3-5-2": {"contre": ["4-3-3", "4-4-2"], "bonus": 0.04},  
        "5-3-2": {"contre": ["3-4-3", "4-3-3"], "bonus": 0.03},  
        "4-1-4-1": {"contre": ["4-2-3-1", "4-3-3"], "bonus": 0.03},  
        "4-3-2-1": {"contre": ["4-4-2", "5-3-2"], "bonus": 0.04},  
        "4-4-1-1": {"contre": ["4-3-3", "3-4-3"], "bonus": 0.03},  
        "4-2-2-2": {"contre": ["4-5-1", "5-3-2"], "bonus": 0.04},  
        "3-6-1": {"contre": ["4-4-2", "4-3-3"], "bonus": 0.05},  
        "5-4-1": {"contre": ["3-4-3", "4-3-3"], "bonus": 0.03},  
        "4-1-3-2": {"contre": ["4-4-2", "5-3-2"], "bonus": 0.04},  
        "4-3-1-2": {"contre": ["4-3-3", "3-4-3"], "bonus": 0.04},  
    }  

    if tactique_B in avantages[tactique_A]["contre"]:  
        probabilite += avantages[tactique_A]["bonus"]  
    elif tactique_A in avantages[tactique_B]["contre"]:  
        probabilite -= avantages[tactique_B]["bonus"]  

    return max(0, min(1, probabilite))  


# Titre de l'application  
st.title("Prédiction de match avec RandomForest et gestion des mises")  
st.write("Analysez les données des matchs, calculez les probabilités et optimisez vos mises.")  

# Section : Critères pour l'équipe A  
st.header("Critères pour l'équipe A")  
tirs_cadres_A = st.slider("Tirs cadrés par l'équipe A", 0, 20, 10, key="tirs_cadres_A")  
possession_A = st.slider("Possession de l'équipe A (%)", 0, 100, 55, key="possession_A")  
cartons_jaunes_A = st.slider("Cartons jaunes pour l'équipe A", 0, 10, 2, key="cartons_jaunes_A")  
fautes_A = st.slider("Fautes commises par l'équipe A", 0, 30, 15, key="fautes_A")  
forme_recente_A = st.slider("Forme récente de l'équipe A (sur 5)", 0.0, 5.0, 3.5, key="forme_recente_A")  
absences_A = st.slider("Nombre d'absences dans l'équipe A", 0, 10, 1, key="absences_A")  
arrets_A = st.slider("Arrêts moyens par match pour l'équipe A", 0, 20, 5, key="arrets_A")  
penalites_concedees_A = st.slider("Pénalités concédées par l'équipe A", 0, 10, 1, key="penalites_concedees_A")  
tacles_reussis_A = st.slider("Tacles réussis par match pour l'équipe A", 0, 50, 20, key="tacles_reussis_A")  
degagements_A = st.slider("Dégagements par match pour l'équipe A", 0, 50, 15, key="degagements_A")  
tactique_A = st.selectbox(  
    "Tactique de l'équipe A",  
    [  
        "4-4-2", "4-2-3-1", "4-3-3", "4-5-1", "3-4-3", "3-5-2", "5-3-2",  
        "4-1-4-1", "4-3-2-1", "4-4-1-1", "4-2-2-2", "3-6-1", "5-4-1",  
        "4-1-3-2", "4-3-1-2"  
    ],  
    key="tactique_A"  
)  

# Section : Critères pour l'équipe B  
st.header("Critères pour l'équipe B")  
tirs_cadres_B = st.slider("Tirs cadrés par l'équipe B", 0, 20, 8, key="tirs_cadres_B")  
possession_B = st.slider("Possession de l'équipe B (%)", 0, 100, 45, key="possession_B")  
cartons_jaunes_B = st.slider("Cartons jaunes pour l'équipe B", 0, 10, 3, key="cartons_jaunes_B")  
fautes_B = st.slider("Fautes commises par l'équipe B", 0, 30, 18, key="fautes_B")  
forme_recente_B = st.slider("Forme récente de l'équipe B (sur 5)", 0.0, 5.0, 3.0, key="forme_recente_B")  
absences_B = st.slider("Nombre d'absences dans l'équipe B", 0, 10, 2, key="absences_B")  
arrets_B = st.slider("Arrêts moyens par match pour l'équipe B", 0, 20, 4, key="arrets_B")  
penalites_concedees_B = st.slider("Pénalités concédées par l'équipe B", 0, 10, 2, key="penalites_concedees_B")  
tacles_reussis_B = st.slider("Tacles réussis par match pour l'équipe B", 0, 50, 18, key="tacles_reussis_B")  
degagements_B = st.slider("Dégagements par match pour l'équipe B", 0, 50, 12, key="degagements_B")  
tactique_B = st.selectbox(  
    "Tactique de l'équipe B",  
    [  
        "4-4-2", "4-2-3-1", "4-3-3", "4-5-1", "3-4-3", "3-5-2", "5-3-2",  
        "4-1-4-1", "4-3-2-1", "4-4-1-1", "4-2-2-2", "3-6-1", "5-4-1",  
        "4-1-3-2", "4-3-1-2"  
    ],  
    key="tactique_B"  
)  

# Gestion de la bankroll  
st.header("Gestion de la bankroll")  
bankroll = st.number_input("Entrez votre bankroll totale (€)", min_value=1.0, value=1000.0, step=1.0)  

# Calcul des probabilités ajustées  
probabilite_A = 0.55  # Exemple de probabilité pour l'équipe A  
probabilite_B = 0.45  # Exemple de probabilité pour l'équipe B  

# Calcul des mises Kelly  
mise_A = calculer_mise_kelly(probabilite_A, 2.0) * bankroll  # Exemple : cote de 2.0 pour l'équipe A  
mise_B = calculer_mise_kelly(probabilite_B, 3.0) * bankroll  # Exemple : cote de 3.0 pour l'équipe B  

# Conversion des mises en unités (1 à 5)  
unites_A = convertir_mise_en_unites(mise_A, bankroll)  
unites_B = convertir_mise_en_unites(mise_B, bankroll)  

st.write(f"Mise optimale pour l'équipe A : {unites_A} unités (sur 5)")  
st.write(f"Mise optimale pour l'équipe B : {unites_B} unités (sur 5)")
