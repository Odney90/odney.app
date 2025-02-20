import streamlit as st  
import numpy as np  
import pandas as pd  
import qrcode  
import io  
from PIL import Image  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  

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

# Fonction pour ajuster les probabilités en fonction des tactiques  
def ajuster_probabilite_tactique(probabilite, tactique_A, tactique_B):  
    # Logique d'ajustement des probabilités en fonction des tactiques  
    if tactique_A == "3-4-3" and tactique_B == "5-3-2":  
        probabilite += 0.05  # Avantage pour 3-4-3 contre 5-3-2  
    elif tactique_A == "5-3-2" and tactique_B == "3-4-3":  
        probabilite -= 0.05  # Désavantage pour 5-3-2 contre 3-4-3  
    elif tactique_A == "4-4-2" and tactique_B == "3-4-3":  
        probabilite -= 0.03  # Légère faiblesse pour 4-4-2 contre 3-4-3  
    elif tactique_A == "3-4-3" and tactique_B == "4-4-2":  
        probabilite += 0.03  # Avantage pour 3-4-3 contre 4-4-2  
    # Limiter les probabilités entre 0 et 1  
    return max(0, min(1, probabilite))  

# Charger les données simulées et entraîner le modèle  
X, y = generer_donnees_simulees()  
modele, accuracy = entrainer_modele_random_forest(X, y)  

# Titre de l'application  
st.title("Prédiction de match et gestion des mises avec Random Forest")  
st.write("Analysez les données des matchs et calculez les mises optimales selon la méthode de Kelly.")  

# Affichage de la précision du modèle sous forme de badge  
st.markdown(  
    f"""  
    <div style="background-color:#4CAF50;padding:10px;border-radius:10px;text-align:center;width:50%;margin:auto;">  
        <span style="color:white;font-size:16px;font-weight:bold;">Précision du modèle : {accuracy * 100:.2f}%</span>  
    </div>  
    """,  
    unsafe_allow_html=True,  
)  

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
tactique_A = st.selectbox("Tactique de l'équipe A", ["3-4-3", "4-4-2", "5-3-2"], key="tactique_A")  

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
tactique_B = st.selectbox("Tactique de l'équipe B", ["3-4-3", "4-4-2", "5-3-2"], key="tactique_B")  

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

# Facteur Kelly  
facteur_kelly = st.slider("Facteur Kelly (1 = conservateur, 5 = agressif)", 1, 5, 1, key="facteur_kelly")  

# Calcul des mises optimales  
mise_A = calculer_mise_kelly(probabilite_A, 2.0, facteur_kelly)  # Exemple : cotes = 2.0  
mise_B = calculer_mise_kelly(probabilite_B, 3.0, facteur_kelly)  # Exemple : cotes = 3.0  

# Affichage des résultats pour les équipes individuelles  
st.subheader("Résultats des mises individuelles")  
col1, col2 = st.columns(2)  

with col1:  
    st.markdown('<div style="background-color:#4CAF50;padding:10px;border-radius:10px;">', unsafe_allow_html=True)  
    st.write(f"**Équipe A** :")  
    st.write(f"Probabilité de victoire : {probabilite_A * 100:.2f}%")  
    st.write(f"Mise optimale : {mise_A * 100:.2f}% du capital")  
    st.markdown('</div>', unsafe_allow_html=True)  

with col2:  
    st.markdown('<div style="background-color:#2196F3;padding:10px;border-radius:10px;">', unsafe_allow_html=True)  
    st.write(f"**Équipe B** :")  
    st.write(f"Probabilité de victoire : {probabilite_B * 100:.2f}%")  
    st.write(f"Mise optimale : {mise_B * 100:.2f}% du capital")  
    st.markdown('</div>', unsafe_allow_html=True)  

# Simulation de match  
st.header("Simulation de match")  
if st.button("Simuler le match"):  
    resultat = np.random.choice(["Équipe A gagne", "Équipe B gagne", "Match nul"], p=[probabilite_A, probabilite_B, 1 - probabilite_A - probabilite_B])  
    st.write(f"**Résultat simulé : {resultat}**")  

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
