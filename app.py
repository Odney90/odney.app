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

# Historique des confrontations  
st.subheader("Historique des confrontations")  
historique = st.radio(  
    "Résultats des 5 dernières confrontations",  
    ["Équipe A a gagné 3 fois", "Équipe B a gagné 3 fois", "Équilibré (2-2-1)"],  
    key="historique"  
)  

# Facteur domicile/extérieur  
st.subheader("Lieu du match")  
domicile = st.radio(  
    "Quelle équipe joue à domicile ?",  
    ["Équipe A", "Équipe B", "Terrain neutre"],  
    key="domicile"  
)  

# Facteur météo  
st.subheader("Conditions météorologiques")  
meteo = st.radio(  
    "Conditions météo pendant le match",  
    ["Ensoleillé", "Pluie", "Vent"],  
    key="meteo"  
)  

# Modèle RandomForest  
st.header("Prédiction avec RandomForest")  
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

# Données utilisateur pour prédiction  
input_data = pd.DataFrame({  
    "tirs_cadres": [tirs_cadres_A - tirs_cadres_B],  
    "possession": [possession_A - possession_B],  
    "cartons_jaunes": [cartons_jaunes_A - cartons_jaunes_B],  
    "fautes": [fautes_A - fautes_B],  
    "forme_recente": [forme_recente_A - forme_recente_B],  
    "absences": [absences_A - absences_B],  
    "arrets": [arrets_A - arrets_B],  
    "penalites_concedees": [penalites_concedees_A - penalites_concedees_B],  
    "tacles_reussis": [tacles_reussis_A - tacles_reussis_B],  
    "degagements": [degagements_A - degagements_B]  
})  

# Prédiction  
prediction = model.predict(input_data)[0]  
probabilites = model.predict_proba(input_data)[0]  
probabilite_A = probabilites[1]  
probabilite_B = probabilites[0]  

st.write(f"Probabilité de victoire pour l'équipe A : **{probabilite_A * 100:.2f}%**")  
st.write(f"Probabilité de victoire pour l'équipe B : **{probabilite_B * 100:.2f}%**")  

# Mise Kelly pour les équipes A et B  
capital = st.number_input("Capital total pour le pari (€)", min_value=1.0, value=100.0, step=1.0)  
mise_A = calculer_mise_kelly(probabilite_A, 2.0) * capital  # Exemple : cote de 2.0 pour l'équipe A  
mise_B = calculer_mise_kelly(probabilite_B, 3.0) * capital  # Exemple : cote de 3.0 pour l'équipe B  

st.write(f"Mise optimale pour l'équipe A : {mise_A:.2f} €")  
st.write(f"Mise optimale pour l'équipe B : {mise_B:.2f} €")  

# Analyse combinée  
st.header("Analyse combinée (3 équipes choisies)")  
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
capital_combine = st.number_input("Capital total pour le pari combiné (€)", min_value=1.0, value=100.0, step=1.0)  
mise_combinee = calculer_mise_kelly(prob_combinee, cote_1 * cote_2 * cote_3) * capital_combine  

st.write(f"Mise optimale pour le combiné des trois équipes : {mise_combinee:.2f} €")
