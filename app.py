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


# Fonction pour calculer le retour sur investissement (ROI)  
def calculer_roi(paris_gagnants, paris_perdants, bankroll_initiale):  
    gains = sum([pari['gain'] for pari in paris_gagnants])  
    pertes = sum([pari['mise'] for pari in paris_perdants])  
    profit = gains - pertes  
    roi = (profit / bankroll_initiale) * 100  
    return profit, roi  


# Fonction pour prédire le nombre de buts  
def predire_buts(probabilite_victoire):  
    # On suppose que la probabilité de victoire est proportionnelle au nombre de buts  
    return round(probabilite_victoire * 5)  # Exemple : max 5 buts  


# Titre de l'application  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de match et gestion de bankroll</h1>", unsafe_allow_html=True)  
st.write("Analysez les données des matchs, simulez les résultats, et suivez vos paris pour optimiser votre bankroll.")  

# Partie 1 : Analyse des équipes  
st.markdown("<h2 style='color: #2196F3;'>1. Analyse des équipes</h2>", unsafe_allow_html=True)  

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

# Entraînement des modèles  
features = [  
    "tirs_cadres", "possession", "cartons_jaunes", "fautes", "forme_recente", "absences",  
    "arrets", "penalites_concedees", "tacles_reussis", "degagements"  
]  
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
    "tirs_cadres": [tirs_cadres_A, tirs_cadres_B],  
    "possession": [possession_A, possession_B],  
    "cartons_jaunes": [cartons_jaunes_A, cartons_jaunes_B],  
    "fautes": [fautes_A, fautes_B],  
    "forme_recente": [forme_recente_A, forme_recente_B],  
    "absences": [absences_A, absences_B],  
    "arrets": [arrets_A, arrets_B],  
    "penalites_concedees": [penalites_concedees_A, penalites_concedees_B],  
    "tacles_reussis": [tacles_reussis_A, tacles_reussis_B],  
    "degagements": [degagements_A, degagements_B]  
})  

prediction_rf = model_rf.predict_proba(nouvelle_donnee)[0][1]  # Probabilité de victoire pour l'équipe A  
prediction_lr = model_lr.predict_proba(nouvelle_donnee)[0][1]  # Probabilité de victoire pour l'équipe A  

# Prédiction du nombre de buts  
buts_rf_A = predire_buts(prediction_rf)  
buts_rf_B = predire_buts(1 - prediction_rf)  
buts_lr_A = predire_buts(prediction_lr)  
buts_lr_B = predire_buts(1 - prediction_lr)  

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

# Partie 5 : Simulateur de paris combinés  
st.markdown("<h2 style='color: #4CAF50;'>5. Simulateur de paris combinés</h2>", unsafe_allow_html=True)  

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
mise_combinee = calculer_mise_kelly(prob_combinee, cote_1 * cote_2 * cote_3) * bankroll_initiale  
unites_combinee = convertir_mise_en_unites(mise_combinee, bankroll_initiale)  

st.write(f"Mise optimale pour le combiné des trois équipes : {unites_combinee} unités (sur 5)")  

# Partie 6 : Exportation des données  
st.markdown("<h2 style='color: #FFC107;'>6. Exportation des données</h2>", unsafe_allow_html=True)  

# Préparation des données pour exportation  
export_data = {  
    "Bankroll initiale": bankroll_initiale,  
    "Probabilité combinée": prob_combinee,  
    "Mise combinée": mise_combinee,  
    "Unité combinée": unites_combinee  
}  

# Exportation en CSV  
if st.button("Exporter les données en CSV"):  
    output = StringIO()  
    writer = csv.writer(output)  
    writer.writerow(["Bankroll initiale", "Probabilité combinée", "Mise combinée", "Unité combinée"])  
    writer.writerow([bankroll_initiale, prob_combinee, mise_combinee, unites_combinee])  
    st.download_button(  
        label="Télécharger les données",  
        data=output.getvalue(),  
        file_name="paris_combine.csv",  
        mime="text/csv"  
    )
