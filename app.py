import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  
import csv  
from io import StringIO  


# Fonction pour prédire le nombre de buts  
def predire_buts(probabilite_victoire, attaque, defense):  
    """  
    Prédit le nombre de buts en fonction de la probabilité de victoire,  
    de la force offensive (attaque) et de la force défensive (defense).  
    """  
    base_buts = probabilite_victoire * 3  # Base : max 3 buts en fonction de la probabilité  
    ajustement = (attaque - defense) * 0.5  # Ajustement basé sur la différence attaque/défense  
    buts = max(0, base_buts + ajustement)  # Les buts ne peuvent pas être négatifs  
    return round(buts, 1)  # Retourne un nombre avec une décimale  


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


# Titre de l'application  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de match et gestion de bankroll</h1>", unsafe_allow_html=True)  
st.write("Analysez les données des matchs, simulez les résultats, et suivez vos paris pour optimiser votre bankroll.")  

# Partie 1 : Analyse des équipes  
st.markdown("<h2 style='color: #2196F3;'>1. Analyse des équipes</h2>", unsafe_allow_html=True)  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
attaque_A = st.slider("Force offensive de l'équipe A (attaque)", 0, 100, 70, key="attaque_A")  
defense_A = st.slider("Force défensive de l'équipe A (défense)", 0, 100, 60, key="defense_A")  
forme_recente_A = st.slider("Forme récente de l'équipe A (sur 5)", 0.0, 5.0, 3.5, key="forme_recente_A")  
motivation_A = st.slider("Motivation de l'équipe A (sur 5)", 0.0, 5.0, 4.0, key="motivation_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
attaque_B = st.slider("Force offensive de l'équipe B (attaque)", 0, 100, 65, key="attaque_B")  
defense_B = st.slider("Force défensive de l'équipe B (défense)", 0, 100, 70, key="defense_B")  
forme_recente_B = st.slider("Forme récente de l'équipe B (sur 5)", 0.0, 5.0, 3.0, key="forme_recente_B")  
motivation_B = st.slider("Motivation de l'équipe B (sur 5)", 0.0, 5.0, 3.5, key="motivation_B")  

# Partie 2 : Historique et contexte du match  
st.markdown("<h2 style='color: #FF5722;'>2. Historique et contexte du match</h2>", unsafe_allow_html=True)  
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

# Partie 3 : Simulateur de match (Prédiction des résultats)  
st.markdown("<h2 style='color: #9C27B0;'>3. Simulateur de match</h2>", unsafe_allow_html=True)  

# Exemple de données fictives pour entraîner les modèles  
data = {  
    "attaque": [70, 65, 80, 60, 75],  
    "defense": [60, 70, 50, 80, 65],  
    "forme_recente": [3.5, 3.0, 4.0, 2.5, 3.8],  
    "motivation": [4.0, 3.5, 4.5, 3.0, 4.2],  
    "resultat": [1, 0, 1, 0, 1]  # 1 = victoire équipe A, 0 = victoire équipe B  
}  
df = pd.DataFrame(data)  

# Entraînement des modèles  
features = ["attaque", "defense", "forme_recente", "motivation"]  
X = df[features]  
y = df["resultat"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Modèle 1 : RandomForest  
model_rf = RandomForestClassifier(random_state=42)  
model_rf.fit(X_train, y_train)  
precision_rf = accuracy_score(y_test, model_rf.predict(X_test))  

# Modèle 2 : Logistic Regression  
model_lr = LogisticRegression(random_state=42)  
model_lr.fit(X_train, y_train)  
precision_lr = accuracy_score(y_test, model_lr.predict(X_test))  

# Affichage des précisions  
st.write(f"Précision du modèle RandomForest : **{precision_rf * 100:.2f}%**")  
st.write(f"Précision du modèle Logistic Regression : **{precision_lr * 100:.2f}%**")  

# Prédictions pour les nouvelles données  
nouvelle_donnee = pd.DataFrame({  
    "attaque": [attaque_A, attaque_B],  
    "defense": [defense_A, defense_B],  
    "forme_recente": [forme_recente_A, forme_recente_B],  
    "motivation": [motivation_A, motivation_B]  
})  

prediction_rf = model_rf.predict_proba(nouvelle_donnee)[0][1]  # Probabilité de victoire pour l'équipe A  
prediction_lr = model_lr.predict_proba(nouvelle_donnee)[0][1]  # Probabilité de victoire pour l'équipe A  

# Prédiction du nombre de buts  
buts_rf_A = predire_buts(prediction_rf, attaque_A, defense_B)  
buts_rf_B = predire_buts(1 - prediction_rf, attaque_B, defense_A)  
buts_lr_A = predire_buts(prediction_lr, attaque_A, defense_B)  
buts_lr_B = predire_buts(1 - prediction_lr, attaque_B, defense_A)  

st.write(f"Probabilité de victoire de l'équipe A selon RandomForest : **{prediction_rf * 100:.2f}%**")  
st.write(f"Probabilité de victoire de l'équipe A selon Logistic Regression : **{prediction_lr * 100:.2f}%**")  

# Partie 4 : Simulateur sous forme de schéma  
st.markdown("<h2 style='color: #FFC107;'>4. Simulateur de match</h2>", unsafe_allow_html=True)  

# Création du graphique  
fig, ax = plt.subplots(figsize=(8, 5))  
bar_width = 0.35  
index = np.arange(2)  

# Données pour le graphique  
buts_rf = [buts_rf_A, buts_rf_B]  
buts_lr = [buts_lr_A, buts_lr_B]  

# Barres pour RandomForest  
ax.bar(index, buts_rf, bar_width, label="RandomForest", color="blue")  

# Barres pour Logistic Regression  
ax.bar(index + bar_width, buts_lr, bar_width, label="Logistic Regression", color="orange")  

# Configuration du graphique  
ax.set_xlabel("Équipes")  
ax.set_ylabel("Nombre de buts prédit")  
ax.set_title("Prédiction du nombre de buts par équipe")  
ax.set_xticks(index + bar_width / 2)  
ax.set_xticklabels(["Équipe A", "Équipe B"])  
ax.legend()  

# Affichage du graphique  
st.pyplot(fig)  

# Partie combinée : Calcul des probabilités et mise optimale  
st.markdown("<h2 style='color: #4CAF50;'>5. Simulateur de paris combinés</h2>", unsafe_allow_html=True)  

# Sélection des équipes pour le combiné  
equipe_1 = st.selectbox("Choisissez la première équipe", ["Équipe A", "Équipe B"], key="equipe_1")  
cote_1 = st.number_input(f"Cote pour {equipe_1}", min_value=1.01, step=0.01, value=2.0, key="cote_1")  

equipe_2 = st.selectbox("Choisissez la deuxième équipe", ["Équipe A", "Équipe B"], key="equipe_2")  
cote_2 = st.number_input(f"Cote pour {equipe_2}", min_value=1.01, step=0.01, value=1.8, key="cote_2")  

equipe_3 = st.selectbox("Choisissez la troisième équipe", ["Équipe A", "Équipe B"], key="equipe_3")  
cote_3 = st.number_input(f"Cote pour {equipe_3}", min_value=1.01, step=0.01, value=1.6, key="cote_3")  

# Vérification des cotes pour éviter les erreurs  
if cote_1 > 1 and cote_2 > 1 and cote_3 > 1:  
    # Calcul des probabilités implicites  
    prob_1 = 1 / cote_1  
    prob_2 = 1 / cote_2  
    prob_3 = 1 / cote_3  

    # Probabilité combinée  
    prob_combinee = prob_1 * prob_2 * prob_3  

    # Calcul de la mise combinée  
    cotes_combinees = cote_1 * cote_2 * cote_3  
    bankroll_initiale = 1000  # Exemple de bankroll initiale  
    mise_combinee = calculer_mise_kelly(prob_combinee, cotes_combinees) * bankroll_initiale  
    unites_combinee = convertir_mise_en_unites(mise_combinee, bankroll_initiale)  

    # Affichage des résultats  
    st.write(f"Probabilité combinée pour les trois équipes : **{prob_combinee * 100:.2f}%**")  
    st.write(f"Cotes combinées : **{cotes_combinees:.2f}**")  
    st.write(f"Mise optimale pour le combiné des trois équipes : **{unites_combinee} unités (sur 5)**")  
else:  
    st.error("Les cotes doivent être supérieures à 1 pour calculer les probabilités combinées.")
