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

# Partie 1 : Analyse des équipes  
st.header("1. Analyse des équipes")  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
tirs_cadres_A = st.slider("Tirs cadrés par l'équipe A", 0, 20, 10, key="tirs_cadres_A")  
possession_A = st.slider("Possession de l'équipe A (%)", 0, 100, 55, key="possession_A")  
cartons_jaunes_A = st.slider("Cartons jaunes pour l'équipe A", 0, 10, 2, key="cartons_jaunes_A")  
fautes_A = st.slider("Fautes commises par l'équipe A", 0, 30, 15, key="fautes_A")  
forme_recente_A = st.slider("Forme récente de l'équipe A (sur 5)", 0.0, 5.0, 3.5, key="forme_recente_A")  
motivation_A = st.slider("Motivation de l'équipe A (sur 5)", 0.0, 5.0, 4.0, key="motivation_A")  
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

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
tirs_cadres_B = st.slider("Tirs cadrés par l'équipe B", 0, 20, 8, key="tirs_cadres_B")  
possession_B = st.slider("Possession de l'équipe B (%)", 0, 100, 45, key="possession_B")  
cartons_jaunes_B = st.slider("Cartons jaunes pour l'équipe B", 0, 10, 3, key="cartons_jaunes_B")  
fautes_B = st.slider("Fautes commises par l'équipe B", 0, 30, 18, key="fautes_B")  
forme_recente_B = st.slider("Forme récente de l'équipe B (sur 5)", 0.0, 5.0, 3.0, key="forme_recente_B")  
motivation_B = st.slider("Motivation de l'équipe B (sur 5)", 0.0, 5.0, 3.5, key="motivation_B")  
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

# Partie 2 : Historique et contexte du match  
st.header("2. Historique et contexte du match")  
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

# Partie 3 : Prédiction avec RandomForest  
st.header("3. Prédiction avec RandomForest")  
st.write("Le modèle RandomForest est utilisé pour prédire les probabilités de victoire.")  

# Exemple de données fictives pour entraîner le modèle  
data = {  
    "tirs_cadres": [10, 8, 12, 6, 9],  
    "possession": [55, 45, 60, 40, 50],  
    "cartons_jaunes": [2, 3, 1, 4, 2],  
    "fautes": [15, 18, 12, 20, 14],  
    "forme_recente": [3.5, 3.0, 4.0, 2.5, 3.8],  
    "absences": [1, 2, 0, 3, 1],  
    "arrets": [5, 4, 6, 3, 5],  
    "penalites_concedees": [1, 2, 0, 3, 1],  
    "tacles_reussis": [20, 18, 25, 15, 22],  
    "degagements": [15, 12, 18, 10, 14],  
    "resultat": [1, 0, 1, 0, 1]  # 1 = victoire équipe A, 0 = victoire équipe B  
}  
df = pd.DataFrame(data)  

# Entraînement du modèle  
features = [  
    "tirs_cadres", "possession", "cartons_jaunes", "fautes", "forme_recente", "absences",  
    "arrets", "penalites_concedees", "tacles_reussis", "degagements"  
]  
X = df[features]  
y = df["resultat"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
model = RandomForestClassifier(random_state=42)  
model.fit(X_train, y_train)  

# Précision du modèle  
precision = accuracy_score(y_test, model.predict(X_test))  
st.write(f"Précision du modèle RandomForest : **{precision * 100:.2f}%**")  

# Partie 4 : Gestion de la bankroll et des mises  
st.header("4. Gestion de la bankroll et des mises")  
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

# Partie 5 : Analyse combinée  
st.header("5. Analyse combinée")  
st.write("Choisissez trois équipes parmi celles analysées pour calculer la probabilité combinée et la mise optimale.")  

# Sélection des équipes pour le combiné  
equipe_1 = st.selectbox("Choisissez la première équipe", ["Équipe A", "Équipe B"], key="equipe_1")  
cote_1 = st.number_input(f"Cote pour {equipe_1}", min_value=1.01, step=0.01, value=2.0, key="cote_1")  

equipe_2 = st.selectbox("Choisissez la deuxième équipe", ["Équipe A", "Équipe B"], key="equipe_2")  
cote_2 = st.number_input(f"Cote pour {equipe_2}", min_value=1.01, step=0.01, value=1.8, key="cote_2")  

equipe_3 = st.selectbox("Choisissez la troisième équipe", ["Équipe A", "Équipe B"], key="equipe_3")  
cote_3 = st.number_input(f"Cote pour {equipe_3}", min_value=1.01, step=0.01, value=1.6, key="cote_3")  

# Calcul des probabilités implicites  
prob_1 = 1 / cote_1  
prob_2 = 1 / cote_2  
prob_3 = 1 / cote_3  

# Probabilité combinée  
prob_combinee = prob_1 * prob_2 * prob_3  

st.write(f"Probabilité combinée pour les trois équipes : {prob_combinee * 100:.2f}%")  

# Allocation de mise combinée  
mise_combinee = calculer_mise_kelly(prob_combinee, cote_1 * cote_2 * cote_3) * bankroll  
unites_combinee = convertir_mise_en_unites(mise_combinee, bankroll)  

st.write(f"Mise optimale pour le combiné des trois équipes : {unites_combinee} unités (sur 5)")
