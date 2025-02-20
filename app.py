import streamlit as st  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

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
st.title("Prédiction de match et gestion des mises avec analyse avancée")  
st.write("Analysez les données des matchs, calculez les probabilités et optimisez vos mises.")  

# Critères pour l'équipe A  
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

# Critères pour l'équipe B  
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

# Probabilités initiales (exemple arbitraire)  
probabilite_A = 0.5  
probabilite_B = 0.5  

# Ajuster les probabilités en fonction des tactiques  
probabilite_A = ajuster_probabilite_tactique(probabilite_A, tactique_A, tactique_B)  
probabilite_B = ajuster_probabilite_tactique(probabilite_B, tactique_B, tactique_A)  

# Ajuster les probabilités en fonction de l'historique  
if historique == "Équipe A a gagné 3 fois":  
    probabilite_A += 0.05  # Avantage pour l'équipe A  
elif historique == "Équipe B a gagné 3 fois":  
    probabilite_B += 0.05  # Avantage pour l'équipe B  

# Ajuster les probabilités en fonction du lieu  
if domicile == "Équipe A":  
    probabilite_A += 0.10  # Avantage pour l'équipe A  
    probabilite_B -= 0.10  
elif domicile == "Équipe B":  
    probabilite_B += 0.10  # Avantage pour l'équipe B  
    probabilite_A -= 0.10  

# Ajuster les probabilités en fonction de la météo  
if meteo == "Pluie":  
    probabilite_B += 0.03  # Avantage pour une équipe défensive  
elif meteo == "Vent":  
    probabilite_A += 0.02  # Avantage pour une équipe offensive  

# Limiter les probabilités entre 0 et 1  
probabilite_A = max(0, min(1, probabilite_A))  
probabilite_B = max(0, min(1, probabilite_B))  

# Gestion du capital  
st.header("Gestion du capital")  
capital_initial = st.number_input("Capital initial (€)", min_value=1.0, value=100.0, step=1.0)  
mise_A = calculer_mise_kelly(probabilite_A, 2.0)  
mise_B = calculer_mise_kelly(probabilite_B, 3.0)  
mise_A_euros = mise_A * capital_initial  
mise_B_euros = mise_B * capital_initial  

st.write(f"Mise optimale pour l'équipe A : {mise_A_euros:.2f} €")  
st.write(f"Mise optimale pour l'équipe B : {mise_B_euros:.2f} €")  

# Prédiction de score  
st.header("Prédiction de score")  
score_A = int(probabilite_A * 5)  # Exemple : multiplier la probabilité par 5 pour obtenir un score  
score_B = int(probabilite_B * 5)  
st.write(f"Score prédit : Équipe A {score_A} - {score_B} Équipe B")  

# Analyse des performances des joueurs clés  
st.header("Performances des joueurs clés")  
performance_buteur_A = st.slider("Performance du buteur de l'équipe A (sur 10)", 0, 10, 7)  
performance_gardien_A = st.slider("Performance du gardien de l'équipe A (sur 10)", 0, 10, 8)  
performance_buteur_B = st.slider("Performance du buteur de l'équipe B (sur 10)", 0, 10, 6)  
performance_gardien_B = st.slider("Performance du gardien de l'équipe B (sur 10)", 0, 10, 7)  

# Ajuster les probabilités en fonction des performances  
probabilite_A += (performance_buteur_A - performance_gardien_B) * 0.01  
probabilite_B += (performance_buteur_B - performance_gardien_A) * 0.01  

# Analyse des tendances  
st.header("Analyse des tendances")  
resultats_A = st.text_input("Résultats des 5 derniers matchs de l'équipe A (V pour victoire, N pour nul, D pour défaite)", "V,V,N,D,V")  
resultats_B = st.text_input("Résultats des 5 derniers matchs de l'équipe B (V pour victoire, N pour nul, D pour défaite)", "D,N,V,V,D")  

# Calculer le score de forme  
score_forme_A = resultats_A.count("V") * 3 + resultats_A.count("N")  
score_forme_B = resultats_B.count("V") * 3 + resultats_B.count("N")  

# Ajuster les probabilités  
probabilite_A += score_forme_A * 0.01  
probabilite_B += score_forme_B * 0.01  

# Visualisation des probabilités  
st.header("Visualisation des probabilités")  
fig, ax = plt.subplots()  
ax.bar(["Équipe A", "Équipe B"], [probabilite_A * 100, probabilite_B * 100], color=["green", "blue"])  
ax.set_ylabel("Probabilité (%)")  
ax.set_title("Probabilités de victoire")  
st.pyplot(fig)  

# Export des résultats  
st.header("Exporter les résultats")  
if st.button("Exporter en CSV"):  
    data = {  
        "Équipe": ["A", "B"],  
        "Probabilité (%)": [probabilite_A * 100, probabilite_B * 100],  
        "Mise optimale (%)": [mise_A * 100, mise_B * 100]  
    }  
    df = pd.DataFrame(data)  
    csv = df.to_csv(index=False).encode('utf-8')  
    st.download_button(label="Télécharger les résultats", data=csv, file_name="resultats.csv", mime="text/csv")
