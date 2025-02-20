import streamlit as st  
import numpy as np  
import qrcode  
import io  # Pour gérer les flux de données en mémoire  
from PIL import Image  

# Fonction pour calculer la probabilité implicite à partir des cotes  
def calculer_probabilite_implicite(cotes):  
    return 1 / cotes if cotes > 0 else 0  

# Fonction pour calculer la mise optimale selon Kelly  
def calculer_mise_kelly(probabilite, cotes, facteur_kelly=1):  
    b = cotes - 1  # Gain net pour 1 unité misée  
    q = 1 - probabilite  # Probabilité d'échec  
    fraction_kelly = (b * probabilite - q) / b  
    fraction_kelly = max(0, fraction_kelly)  # La mise ne peut pas être négative  
    return fraction_kelly * facteur_kelly  

# Fonction pour calculer la probabilité combinée  
def calculer_probabilite_combinee(probabilites):  
    prob_combinee = 1  
    for prob in probabilites:  
        prob_combinee *= prob  
    return prob_combinee  

# Titre de l'application  
st.title("Prédiction de match et gestion des mises avec Kelly")  
st.write("Analysez les données des matchs et calculez les mises optimales selon la méthode de Kelly.")  

# Évaluation des équipes individuelles  
st.header("Évaluation des équipes individuelles")  

# Entrées pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
cotes_A = st.number_input("Cotes pour l'équipe A", min_value=1.01, value=2.0)  
probabilite_implicite_A = calculer_probabilite_implicite(cotes_A)  
probabilite_A = st.slider(  
    "Probabilité de victoire de l'équipe A (%)",   
    0, 100, int(probabilite_implicite_A * 100)  
) / 100  

# Affichage des probabilités implicites pour l'équipe A  
st.write(f"**Probabilité implicite pour l'équipe A :** {probabilite_implicite_A * 100:.2f}%")  

# Critères pour l'équipe A  
tirs_cadres_A = st.slider("Tirs cadrés par l'équipe A", 0, 20, 10)  
possession_A = st.slider("Possession de l'équipe A (%)", 0, 100, 55)  
cartons_jaunes_A = st.slider("Cartons jaunes pour l'équipe A", 0, 10, 2)  
fautes_A = st.slider("Fautes commises par l'équipe A", 0, 30, 15)  
forme_recente_A = st.slider("Forme récente de l'équipe A (sur 5)", 0.0, 5.0, 3.5)  
absences_A = st.slider("Nombre d'absences dans l'équipe A", 0, 10, 1)  
arrets_A = st.slider("Arrêts moyens par match pour l'équipe A", 0, 20, 5)  
penalites_concedees_A = st.slider("Pénalités concédées par l'équipe A", 0, 10, 1)  
tacles_reussis_A = st.slider("Tacles réussis par match pour l'équipe A", 0, 50, 20)  
degagements_A = st.slider("Dégagements par match pour l'équipe A", 0, 50, 15)  
interceptions_A = st.slider("Interceptions par match pour l'équipe A", 0, 50, 10)  
buts_concedes_A = st.slider("Buts concédés sur 100 pour l'équipe A", 0, 100, 30)  
corners_A = st.slider("Corners sur 300 pour l'équipe A", 0, 300, 150)  
penalites_obtenues_A = st.slider("Pénalités obtenues par l'équipe A", 0, 10, 3)  
centres_reussis_A = st.slider("Centres réussis par match pour l'équipe A", 0, 50, 25)  
passes_longues_A = st.slider("Passes longues précises par match pour l'équipe A", 0, 50, 15)  
grandes_chances_A = st.slider("Grandes chances de marquer pour l'équipe A", 0, 20, 8)  
cartons_rouges_A = st.slider("Cartons rouges pour l'équipe A", 0, 5, 1)  

# Entrées pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
cotes_B = st.number_input("Cotes pour l'équipe B", min_value=1.01, value=3.0)  
probabilite_implicite_B = calculer_probabilite_implicite(cotes_B)  
probabilite_B = st.slider(  
    "Probabilité de victoire de l'équipe B (%)",   
    0, 100, int(probabilite_implicite_B * 100)  
) / 100  

# Affichage des probabilités implicites pour l'équipe B  
st.write(f"**Probabilité implicite pour l'équipe B :** {probabilite_implicite_B * 100:.2f}%")  

# Critères pour l'équipe B  
tirs_cadres_B = st.slider("Tirs cadrés par l'équipe B", 0, 20, 8)  
possession_B = st.slider("Possession de l'équipe B (%)", 0, 100, 45)  
cartons_jaunes_B = st.slider("Cartons jaunes pour l'équipe B", 0, 10, 3)  
fautes_B = st.slider("Fautes commises par l'équipe B", 0, 30, 18)  
forme_recente_B = st.slider("Forme récente de l'équipe B (sur 5)", 0.0, 5.0, 3.0)  
absences_B = st.slider("Nombre d'absences dans l'équipe B", 0, 10, 2)  
arrets_B = st.slider("Arrêts moyens par match pour l'équipe B", 0, 20, 4)  
penalites_concedees_B = st.slider("Pénalités concédées par l'équipe B", 0, 10, 2)  
tacles_reussis_B = st.slider("Tacles réussis par match pour l'équipe B", 0, 50, 18)  
degagements_B = st.slider("Dégagements par match pour l'équipe B", 0, 50, 12)  
interceptions_B = st.slider("Interceptions par match pour l'équipe B", 0, 50, 8)  
buts_concedes_B = st.slider("Buts concédés sur 100 pour l'équipe B", 0, 100, 40)  
corners_B = st.slider("Corners sur 300 pour l'équipe B", 0, 300, 140)  
penalites_obtenues_B = st.slider("Pénalités obtenues par l'équipe B", 0, 10, 2)  
centres_reussis_B = st.slider("Centres réussis par match pour l'équipe B", 0, 50, 20)  
passes_longues_B = st.slider("Passes longues précises par match pour l'équipe B", 0, 50, 12)  
grandes_chances_B = st.slider("Grandes chances de marquer pour l'équipe B", 0, 20, 6)  
cartons_rouges_B = st.slider("Cartons rouges pour l'équipe B", 0, 5, 0)  

# Facteur Kelly  
facteur_kelly = st.slider("Facteur Kelly (1 = conservateur, 5 = agressif)", 1, 5, 1)  

# Calcul des mises optimales pour les équipes  
mise_A = calculer_mise_kelly(probabilite_A, cotes_A, facteur_kelly)  
mise_B = calculer_mise_kelly(probabilite_B, cotes_B, facteur_kelly)  

# Affichage des résultats pour les équipes individuelles  
st.subheader("Résultats des mises individuelles")  
st.write(f"- **Mise optimale sur l'équipe A :** {mise_A * 100:.2f}% du capital")  
st.write(f"- **Mise optimale sur l'équipe B :** {mise_B * 100:.2f}% du capital")  

# Combinaison de plusieurs équipes  
st.header("Combinaison de plusieurs équipes")  
st.write("Entrez les probabilités et les cotes pour combiner jusqu'à 3 équipes.")  

# Entrées pour les équipes combinées  
col1, col2, col3 = st.columns(3)  

with col1:  
    prob_equipe_1 = st.slider("Probabilité de l'équipe 1 (%)", 0, 100, 60) / 100  
    cotes_equipe_1 = st.number_input("Cotes pour l'équipe 1", min_value=1.01, value=2.0)  

with col2:  
    prob_equipe_2 = st.slider("Probabilité de l'équipe 2 (%)", 0, 100, 50) / 100  
    cotes_equipe_2 = st.number_input("Cotes pour l'équipe 2", min_value=1.01, value=3.0)  

with col3:  
    prob_equipe_3 = st.slider("Probabilité de l'équipe 3 (%)", 0, 100, 40) / 100  
    cotes_equipe_3 = st.number_input("Cotes pour l'équipe 3", min_value=1.01, value=4.0)  

# Calcul des probabilités et cotes combinées  
probabilites = [prob_equipe_1, prob_equipe_2, prob_equipe_3]  
cotes = [cotes_equipe_1, cotes_equipe_2, cotes_equipe_3]  

# Filtrer les équipes avec des probabilités valides  
probabilites = [p for p in probabilites if p > 0]  
cotes = [c for c in cotes if c > 1.01]  

if len(probabilites) > 0:  
    prob_combinee = calculer_probabilite_combinee(probabilites)  
    cotes_combinees = np.prod(cotes)  

    # Calcul de la mise combinée  
    mise_combinee = calculer_mise_kelly(prob_combinee, cotes_combinees, facteur_kelly)  

    # Affichage des résultats pour la combinaison  
    st.subheader("Résultats de la combinaison")  
    st.write(f"- **Probabilité combinée :** {prob_combinee * 100:.2f}%")  
    st.write(f"- **Cotes combinées :** {cotes_combinees:.2f}")  
    st.write(f"- **Mise optimale sur la combinaison :** {mise_combinee * 100:.2f}% du capital")  
else:  
    st.write("Veuillez fournir au moins une équipe avec une probabilité valide.")  

# Génération de QR code pour accéder à l'application  
st.subheader("Accédez à cette application sur votre téléphone")  
url = "http://127.0.0.1:8501"  # Remplacez par votre IP locale si nécessaire  
qr = qrcode.make(url)  
buffer = io.BytesIO()  
qr.save(buffer, format="PNG")  
buffer.seek(0)  
st.image(buffer, caption="Scannez ce QR code pour accéder à l'application sur votre téléphone")
