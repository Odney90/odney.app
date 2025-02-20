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
tactique_B = st.selectbox(  
    "Tactique de l'équipe B",  
    [  
        "4-4-2", "4-2-3-1", "4-3-3", "4-5-1", "3-4-3", "3-5-2", "5-3-2",  
        "4-1-4-1", "4-3-2-1", "4-4-1-1", "4-2-2-2", "3-6-1", "5-4-1",  
        "4-1-3-2", "4-3-1-2"  
    ],  
    key="tactique_B"  
)  

# Probabilités initiales (exemple arbitraire)  
probabilite_A = 0.5  
probabilite_B = 0.5  

# Ajuster les probabilités en fonction des tactiques  
probabilite_A = ajuster_probabilite_tactique(probabilite_A, tactique_A, tactique_B)  
probabilite_B = ajuster_probabilite_tactique(probabilite_B, tactique_B, tactique_A)  

# Analyse des cotes implicites  
st.header("Analyse des cotes implicites")  
cote_A = st.number_input("Cote pour l'équipe A", min_value=1.01, value=2.0, step=0.01)  
cote_B = st.number_input("Cote pour l'équipe B", min_value=1.01, value=3.0, step=0.01)  

# Calcul des probabilités implicites  
prob_implicite_A = 1 / cote_A  
prob_implicite_B = 1 / cote_B  

# Affichage des probabilités implicites  
st.write(f"Probabilité implicite pour l'équipe A : {prob_implicite_A * 100:.2f}%")  
st.write(f"Probabilité implicite pour l'équipe B : {prob_implicite_B * 100:.2f}%")  

# Comparaison avec les probabilités calculées  
st.write(f"Écart pour l'équipe A : {(probabilite_A - prob_implicite_A) * 100:.2f}%")  
st.write(f"Écart pour l'équipe B : {(probabilite_B - prob_implicite_B) * 100:.2f}%")  

# Gestion du capital  
st.header("Gestion du capital")  
capital_initial = st.number_input("Capital initial (€)", min_value=1.0, value=100.0, step=1.0)  
mise_A = calculer_mise_kelly(probabilite_A, cote_A)  
mise_B = calculer_mise_kelly(probabilite_B, cote_B)  
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
