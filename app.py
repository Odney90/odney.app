import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
import qrcode  
from PIL import Image  
import io  # Pour gérer les flux de données en mémoire  

# Fonction pour confronter les styles tactiques  
def confronter_styles(style_A, style_B):  
    if style_A == style_B:  
        return 0  # Styles identiques  
    elif (style_A == 0 and style_B == 2) or (style_A == 1 and style_B == 0) or (style_A == 2 and style_B == 1):  
        return 1  # Avantage équipe A  
    else:  
        return -1  # Avantage équipe B  

# Fonction pour calculer la mise selon Kelly  
def calculer_mise_kelly(probabilite, cote, facteur_kelly=1):  
    b = cote - 1  # Gain net pour 1 unité misée  
    q = 1 - probabilite  # Probabilité d'échec  
    kelly_fraction = (b * probabilite - q) / b  
    kelly_fraction = max(0, kelly_fraction)  # La mise ne peut pas être négative  
    return kelly_fraction * facteur_kelly  

# Fonction pour calculer la probabilité combinée  
def calculer_probabilite_combinee(probabilites):  
    prob_combinee = 1  
    for prob in probabilites:  
        prob_combinee *= prob  
    return prob_combinee  

# Interface utilisateur avec Streamlit  
st.title("Prédiction de Match et Gestion des Mises avec Kelly")  
st.write("Analysez les données des matchs et calculez les mises optimales selon la méthode de Kelly.")  

# Saisie des données pour les équipes individuelles  
st.header("Évaluation des équipes individuelles")  
probabilite_A = st.slider("Probabilité de victoire de l'équipe A (%)", 0, 100, 60) / 100  
cote_A = st.number_input("Cote de l'équipe A", min_value=1.01, value=2.0)  
probabilite_B = st.slider("Probabilité de victoire de l'équipe B (%)", 0, 100, 50) / 100  
cote_B = st.number_input("Cote de l'équipe B", min_value=1.01, value=3.0)  

# Critères supplémentaires pour les équipes  
tirs_cadres_A = st.slider("Tirs cadrés de l'équipe A", 0, 20, 10)  
tirs_cadres_B = st.slider("Tirs cadrés de l'équipe B", 0, 20, 8)  
possession_A = st.slider("Possession de l'équipe A (%)", 0, 100, 55)  
possession_B = st.slider("Possession de l'équipe B (%)", 0, 100, 45)  
cartons_jaunes_A = st.slider("Cartons jaunes de l'équipe A", 0, 10, 2)  
cartons_jaunes_B = st.slider("Cartons jaunes de l'équipe B", 0, 10, 3)  
fautes_A = st.slider("Fautes commises par l'équipe A", 0, 30, 15)  
fautes_B = st.slider("Fautes commises par l'équipe B", 0, 30, 18)  
forme_recente_A = st.slider("Forme récente de l'équipe A (sur 5)", 0.0, 5.0, 3.5)  
forme_recente_B = st.slider("Forme récente de l'équipe B (sur 5)", 0.0, 5.0, 3.0)  
absences_A = st.slider("Nombre d'absences dans l'équipe A", 0, 10, 1)  
absences_B = st.slider("Nombre d'absences dans l'équipe B", 0, 10, 2)  

# Saisie des styles tactiques et importance du match  
style_tactique_A = st.selectbox("Style tactique de l'équipe A", ["Possession", "Pressing", "Contre-attaque"])  
style_tactique_B = st.selectbox("Style tactique de l'équipe B", ["Possession", "Pressing", "Contre-attaque"])  
importance_match = st.selectbox("Importance du match", ["Championnat", "Finale", "Amical"])  

# Calcul des mises pour les équipes individuelles  
facteur_kelly = st.slider("Facteur de Kelly (1 = conservateur, 5 = agressif)", 1, 5, 1)  
mise_A = calculer_mise_kelly(probabilite_A, cote_A, facteur_kelly)  
mise_B = calculer_mise_kelly(probabilite_B, cote_B, facteur_kelly)  

# Affichage des résultats pour les équipes individuelles  
st.subheader("Résultats des Mises Individuelles")  
st.write(f"- **Mise optimale sur l'équipe A :** {mise_A * 100:.2f}% du capital")  
st.write(f"- **Mise optimale sur l'équipe B :** {mise_B * 100:.2f}% du capital")  

# Partie combinée pour analyser jusqu'à 3 équipes  
st.header("Combinaison de plusieurs équipes")  
st.write("Renseignez les probabilités et les cotes pour combiner jusqu'à 3 équipes.")  

# Saisie des données pour les équipes combinées  
col1, col2, col3 = st.columns(3)  

with col1:  
    prob_equipe_1 = st.slider("Probabilité de l'équipe 1 (%)", 0, 100, 60) / 100  
    cote_equipe_1 = st.number_input("Cote de l'équipe 1", min_value=1.01, value=2.0)  

with col2:  
    prob_equipe_2 = st.slider("Probabilité de l'équipe 2 (%)", 0, 100, 50) / 100  
    cote_equipe_2 = st.number_input("Cote de l'équipe 2", min_value=1.01, value=3.0)  

with col3:  
    prob_equipe_3 = st.slider("Probabilité de l'équipe 3 (%)", 0, 100, 40) / 100  
    cote_equipe_3 = st.number_input("Cote de l'équipe 3", min_value=1.01, value=4.0)  

# Calcul de la probabilité combinée et de la cote combinée  
probabilites = [prob_equipe_1, prob_equipe_2, prob_equipe_3]  
cotes = [cote_equipe_1, cote_equipe_2, cote_equipe_3]  

# Filtrer les équipes avec des probabilités > 0  
probabilites = [p for p in probabilites if p > 0]  
cotes = [c for c in cotes if c > 1.01]  

if len(probabilites) > 0:  
    prob_combinee = calculer_probabilite_combinee(probabilites)  
    cote_combinee = np.prod(cotes)  

    # Calcul de la mise combinée  
    mise_combinee = calculer_mise_kelly(prob_combinee, cote_combinee, facteur_kelly)  

    # Affichage des résultats pour la combinaison  
    st.subheader("Résultats de la Combinaison")  
    st.write(f"- **Probabilité combinée :** {prob_combinee * 100:.2f}%")  
    st.write(f"- **Cote combinée :** {cote_combinee:.2f}")  
    st.write(f"- **Mise optimale sur la combinaison :** {mise_combinee * 100:.2f}% du capital")  
else:  
    st.write("Veuillez renseigner au moins une équipe avec une probabilité valide.")  

# Génération du QR code  
st.subheader("Accédez à cette application sur votre téléphone")  
url = "http://127.0.0.1:8501"  # Remplacez par votre IP locale si nécessaire  
qr = qrcode.make(url)  
buffer = io.BytesIO()  
qr.save(buffer, format="PNG")  
buffer.seek(0)  
st.image(buffer, caption="Scannez ce QR code pour accéder à l'application sur votre téléphone")