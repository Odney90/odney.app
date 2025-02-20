import streamlit as st  
import numpy as np  
import qrcode  
import io  
from PIL import Image  
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

# Simuler un jeu de données pour entraîner le modèle Random Forest  
def generer_donnees_simulees():  
    np.random.seed(42)  
    # Générer des données aléatoires pour les critères  
    X = np.random.rand(1000, 10) * 100  # 10 critères, valeurs entre 0 et 100  
    # Générer des résultats (0 = défaite, 1 = victoire)  
    y = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])  
    return X, y  

# Entraîner le modèle Random Forest  
def entrainer_modele_random_forest(X, y):  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    modele = RandomForestClassifier(n_estimators=100, random_state=42)  
    modele.fit(X_train, y_train)  
    y_pred = modele.predict(X_test)  
    accuracy = accuracy_score(y_test, y_pred)  
    return modele, accuracy  

# Prédire la probabilité de victoire avec le modèle  
def predire_probabilite(modele, criteres):  
    probabilite = modele.predict_proba([criteres])[0][1]  # Probabilité de victoire (classe 1)  
    return probabilite  

# Charger les données simulées et entraîner le modèle  
X, y = generer_donnees_simulees()  
modele, accuracy = entrainer_modele_random_forest(X, y)  

# Afficher la précision du modèle  
st.sidebar.write(f"Précision du modèle Random Forest : {accuracy * 100:.2f}%")  

# Titre de l'application  
st.title("Prédiction de match et gestion des mises avec Random Forest")  
st.write("Analysez les données des matchs et calculez les mises optimales selon la méthode de Kelly.")  

# Évaluation des équipes individuelles  
st.header("Évaluation des équipes individuelles")  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
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

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
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

# Préparer les critères pour la prédiction  
criteres_A = [  
    tirs_cadres_A, possession_A, cartons_jaunes_A, fautes_A, forme_recente_A,  
    absences_A, arrets_A, penalites_concedees_A, tacles_reussis_A, degagements_A  
]  
criteres_B = [  
    tirs_cadres_B, possession_B, cartons_jaunes_B, fautes_B, forme_recente_B,  
    absences_B, arrets_B, penalites_concedees_B, tacles_reussis_B, degagements_B  
]  

# Prédire les probabilités de victoire  
probabilite_A = predire_probabilite(modele, criteres_A)  
probabilite_B = predire_probabilite(modele, criteres_B)  

# Facteur Kelly  
facteur_kelly = st.slider("Facteur Kelly (1 = conservateur, 5 = agressif)", 1, 5, 1, key="facteur_kelly")  

# Calcul des mises optimales  
mise_A = calculer_mise_kelly(probabilite_A, 2.0, facteur_kelly)  # Exemple : cotes = 2.0  
mise_B = calculer_mise_kelly(probabilite_B, 3.0, facteur_kelly)  # Exemple : cotes = 3.0  

# Affichage des résultats  
st.subheader("Résultats des mises")  
st.write(f"**Équipe A** : Probabilité de victoire = {probabilite_A * 100:.2f}%, Mise optimale = {mise_A * 100:.2f}% du capital")  
st.write(f"**Équipe B** : Probabilité de victoire = {probabilite_B * 100:.2f}%, Mise optimale = {mise_B * 100:.2f}% du capital")
