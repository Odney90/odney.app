import streamlit as st  
import numpy as np  

# Fonction pour calculer la probabilité implicite à partir des cotes  
def calculer_probabilite_implicite(cotes):  
    return 1 / cotes if cotes > 0 else 0  

# Titre de l'application  
st.title("Prédiction de match et gestion des mises avec Kelly")  
st.write("Analysez les données des matchs et calculez les mises optimales selon la méthode de Kelly.")  

# Évaluation des équipes individuelles  
st.header("Évaluation des équipes individuelles")  

# Entrées pour l'équipe A  
cotes_A = st.number_input("Cotes pour l'équipe A", min_value=1.01, value=2.0)  
probabilite_implicite_A = calculer_probabilite_implicite(cotes_A)  
probabilite_A = st.slider(  
    "Probabilité de victoire de l'équipe A (%)",   
    0, 100, int(probabilite_implicite_A * 100)  
) / 100  

# Entrées pour l'équipe B  
cotes_B = st.number_input("Cotes pour l'équipe B", min_value=1.01, value=3.0)  
probabilite_implicite_B = calculer_probabilite_implicite(cotes_B)  
probabilite_B = st.slider(  
    "Probabilité de victoire de l'équipe B (%)",   
    0, 100, int(probabilite_implicite_B * 100)  
) / 100  

# Affichage des probabilités implicites  
st.write(f"**Probabilité implicite pour l'équipe A :** {probabilite_implicite_A * 100:.2f}%")  
st.write(f"**Probabilité implicite pour l'équipe B :** {probabilite_implicite_B * 100:.2f}%")  

# Critères supplémentaires pour les équipes  
tirs_cadres_A = st.slider("Tirs cadrés par l'équipe A", 0, 20, 10)  
tirs_cadres_B = st.slider("Tirs cadrés par l'équipe B", 0, 20, 8)  
possession_A = st.slider("Possession de l'équipe A (%)", 0, 100, 55)  
possession_B = st.slider("Possession de l'équipe B (%)", 0, 100, 45)  
cartons_jaunes_A = st.slider("Cartons jaunes pour l'équipe A", 0, 10, 2)  
cartons_jaunes_B = st.slider("Cartons jaunes pour l'équipe B", 0, 10, 3)  
fautes_A = st.slider("Fautes commises par l'équipe A", 0, 30, 15)  
fautes_B = st.slider("Fautes commises par l'équipe B", 0, 30, 18)  
forme_recente_A = st.slider("Forme récente de l'équipe A (sur 5)", 0.0, 5.0, 3.5)  
forme_recente_B = st.slider("Forme récente de l'équipe B (sur 5)", 0.0, 5.0, 3.0)  
absences_A = st.slider("Nombre d'absences dans l'équipe A", 0, 10, 1)  
absences_B = st.slider("Nombre d'absences dans l'équipe B", 0, 10, 2)  

# Facteur Kelly  
facteur_kelly = st.slider("Facteur Kelly (1 = conservateur, 5 = agressif)", 1, 5, 1)  

# Fonction pour calculer la mise optimale selon Kelly  
def calculer_mise_kelly(probabilite, cotes, facteur_kelly=1):  
    b = cotes - 1  # Gain net pour 1 unité misée  
    q = 1 - probabilite  # Probabilité d'échec  
    fraction_kelly = (b * probabilite - q) / b  
    fraction_kelly = max(0, fraction_kelly)  # La mise ne peut pas être négative  
    return fraction_kelly * facteur_kelly  

# Calcul des mises optimales pour les équipes  
mise_A = calculer_mise_kelly(probabilite_A, cotes_A, facteur_kelly)  
mise_B = calculer_mise_kelly(probabilite_B, cotes_B, facteur_kelly)  

# Affichage des résultats pour les équipes individuelles  
st.subheader("Résultats des mises individuelles")  
st.write(f"- **Mise optimale sur l'équipe A :** {mise_A * 100:.2f}% du capital")  
st.write(f"- **Mise optimale sur l'équipe B :** {mise_B * 100:.2f}% du capital")
