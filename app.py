import streamlit as st  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.ensemble import RandomForestClassifier  

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

# Critères pour l'équipe A  
st.header("Critères pour l'équipe A")  
tirs_cadres_A = st.slider("Tirs cadrés par l'équipe A", 0, 20, 10, key="tirs_cadres_A")  
possession_A = st.slider("Possession de l'équipe A (%)", 0, 100, 55, key="possession_A")  
cartons_jaunes_A = st.slider("Cartons jaunes pour l'équipe A", 0, 10, 2, key="cartons_jaunes_A")  
fautes_A = st.slider("Fautes commises par l'équipe A", 0, 30, 15, key="fautes_A")  
forme_recente_A = st.slider("Forme récente de l'équipe A (sur 5)", 0.0, 5.0, 3.5, key="forme_recente_A")  
absences_A = st.slider("Nombre d'absences dans l'équipe A", 0, 10, 1, key="absences_A")  

# Critères pour l'équipe B  
st.header("Critères pour l'équipe B")  
tirs_cadres_B = st.slider("Tirs cadrés par l'équipe B", 0, 20, 8, key="tirs_cadres_B")  
possession_B = st.slider("Possession de l'équipe B (%)", 0, 100, 45, key="possession_B")  
cartons_jaunes_B = st.slider("Cartons jaunes pour l'équipe B", 0, 10, 3, key="cartons_jaunes_B")  
fautes_B = st.slider("Fautes commises par l'équipe B", 0, 30, 18, key="fautes_B")  
forme_recente_B = st.slider("Forme récente de l'équipe B (sur 5)", 0.0, 5.0, 3.0, key="forme_recente_B")  
absences_B = st.slider("Nombre d'absences dans l'équipe B", 0, 10, 2, key="absences_B")  

# Préparation des données pour RandomForest  
st.subheader("Prédiction avec RandomForest")  
features = [  
    "tirs_cadres", "possession", "cartons_jaunes", "fautes", "forme_recente", "absences"  
]  

# Création d'un DataFrame pour les données d'entraînement (exemple fictif)  
data = {  
    "tirs_cadres": [10, 8, 12, 6, 9],  
    "possession": [55, 45, 60, 40, 50],  
    "cartons_jaunes": [2, 3, 1, 4, 2],  
    "fautes": [15, 18, 12, 20, 14],  
    "forme_recente": [3.5, 3.0, 4.0, 2.5, 3.8],  
    "absences": [1, 2, 0, 3, 1],  
    "resultat": [1, 0, 1, 0, 1]  # 1 = victoire équipe A, 0 = victoire équipe B  
}  
df = pd.DataFrame(data)  

# Modèle RandomForest  
X = df[features]  
y = df["resultat"]  
model = RandomForestClassifier()  
model.fit(X, y)  

# Données de l'utilisateur  
input_data = pd.DataFrame({  
    "tirs_cadres": [tirs_cadres_A - tirs_cadres_B],  
    "possession": [possession_A - possession_B],  
    "cartons_jaunes": [cartons_jaunes_A - cartons_jaunes_B],  
    "fautes": [fautes_A - fautes_B],  
    "forme_recente": [forme_recente_A - forme_recente_B],  
    "absences": [absences_A - absences_B]  
})  

# Prédiction  
prediction = model.predict(input_data)[0]  
probabilites = model.predict_proba(input_data)[0]  
probabilite_A = probabilites[1]  
probabilite_B = probabilites[0]  

# Affichage des résultats  
st.write(f"Probabilité de victoire pour l'équipe A : {probabilite_A * 100:.2f}%")  
st.write(f"Probabilité de victoire pour l'équipe B : {probabilite_B * 100:.2f}%")  

# Visualisation des probabilités  
st.header("Visualisation des probabilités")  
fig, ax = plt.subplots()  
ax.bar(["Équipe A", "Équipe B"], [probabilite_A * 100, probabilite_B * 100], color=["green", "blue"])  
ax.set_ylabel("Probabilité (%)")  
ax.set_title("Probabilités de victoire")  
st.pyplot(fig)  

# Gestion du capital  
st.header("Gestion du capital")  
capital_initial = st.number_input("Capital initial (€)", min_value=1.0, value=100.0, step=1.0)  
mise_A = calculer_mise_kelly(probabilite_A, 2.0)  
mise_B = calculer_mise_kelly(probabilite_B, 3.0)  
mise_A_euros = mise_A * capital_initial  
mise_B_euros = mise_B * capital_initial  

st.write(f"Mise optimale pour l'équipe A : {mise_A_euros:.2f} €")  
st.write(f"Mise optimale pour l'équipe B : {mise_B_euros:.2f} €")  

# Export des résultats  
st.header("Exporter les résultats")  
if st.button("Exporter en CSV"):  
    data = {  
        "Équipe": ["A", "B"],  
        "Probabilité (%)": [probabilite_A * 100, probabilite_B * 100],  
        "Mise optimale (€)": [mise_A_euros, mise_B_euros]  
    }  
    df_export = pd.DataFrame(data)  
    csv = df_export.to_csv(index=False).encode('utf-8')  
    st.download_button(label="Télécharger les résultats", data=csv, file_name="resultats.csv", mime="text/csv")
