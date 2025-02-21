import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import RandomForestRegressor  

# Configuration de l'application  
st.set_page_config(page_title="Prédiction de Match et Gestion de Bankroll", layout="wide")  

# Titre de la page d'analyse  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Analyse des Équipes</h1>", unsafe_allow_html=True)  
st.write("Analysez les données des matchs et simulez les résultats.")  

# Historique des équipes  
st.subheader("Historique des équipes")  
historique_A = st.text_area("Historique des performances de l'équipe A (10 derniers matchs)", height=100)  
historique_B = st.text_area("Historique des performances de l'équipe B (10 derniers matchs)", height=100)  

# Calcul de la forme récente sur 10 matchs  
def calculer_forme(historique):  
    if historique:  
        matchs = historique.split(',')  
        if len(matchs) > 10:  
            matchs = matchs[-10:]  # Prendre les 10 derniers matchs  
        return sum(int(resultat) for resultat in matchs) / len(matchs)  # Moyenne des résultats  
    return 0.0  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  

st.write(f"Forme récente de l'équipe A (moyenne sur 10 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 10 matchs) : **{forme_recente_B:.2f}**")  

# Séparateur  
st.markdown("---")  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
buts_par_match_A = st.slider("Buts par match", min_value=0, max_value=10, value=2, key="buts_par_match_A")  
nombre_buts_produits_A = st.slider("Nombre de buts produits", min_value=0, max_value=100, value=80, key="nombre_buts_produits_A")  
nombre_buts_encaisse_A = st.slider("Nombre de buts encaissés", min_value=0, max_value=100, value=30, key="nombre_buts_encaisse_A")  
buts_encaisse_par_match_A = st.slider("Buts encaissés par match", min_value=0, max_value=10, value=1, key="buts_encaisse_par_match_A")  
poss_moyenne_A = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
matchs_sans_encaisser_A = st.slider("Nombre de matchs sans encaisser de but", min_value=0, max_value=40, value=10, key="matchs_sans_encaisser_A")  
xG_A = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=2.5, key="xG_A")  
tirs_cadres_A = st.slider("Tirs cadrés par match", min_value=0, max_value=20, value=5, key="tirs_cadres_A")  
pourcentage_tirs_convertis_A = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=25, key="pourcentage_tirs_convertis_A")  
grosses_occasions_A = st.slider("Grosses occasions créées", min_value=0, max_value=100, value=10, key="grosses_occasions_A")  
grosses_occasions_ratees_A = st.slider("Grosses occasions ratées", min_value=0, max_value=100, value=5, key="grosses_occasions_ratees_A")  

# Nouveaux critères pour l'équipe A  
passes_longues_A = st.slider("Passes longues précises par match", min_value=0, max_value=50, value=10, key="passes_longues_A")  
touches_surface_A = st.slider("Touches dans la surface adverse", min_value=0, max_value=50, value=15, key="touches_surface_A")  
corners_A = st.slider("Nombre de corners", min_value=0, max_value=20, value=5, key="corners_A")  
corners_concedes_A = st.slider("Nombre de corners concédés", min_value=0, max_value=20, value=3, key="corners_concedes_A")  
interceptions_A = st.slider("Interceptions par match", min_value=0, max_value=20, value=5, key="interceptions_A")  
arrets_A = st.slider("Arrêts par match", min_value=0, max_value=20, value=3, key="arrets_A")  
fautes_A = st.slider("Fautes par match", min_value=0, max_value=20, value=10, key="fautes_A")  
tactique_A = st.text_input("Tactique de l'équipe A", value="Offensive")  
joueurs_absents_A = st.text_input("Joueurs clés absents de l'équipe A", value="Aucun")  

# Séparateur  
st.markdown("---")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_par_match_B = st.slider("Buts par match", min_value=0, max_value=10, value=1, key="buts_par_match_B")  
nombre_buts_produits_B = st.slider("Nombre de buts produits", min_value=0, max_value=100, value=60, key="nombre_buts_produits_B")  
nombre_buts_encaisse_B = st.slider("Nombre de buts encaissés", min_value=0, max_value=100, value=40, key="nombre_buts_encaisse_B")  
buts_encaisse_par_match_B = st.slider("Buts encaissés par match", min_value=0, max_value=10, value=2, key="buts_encaisse_par_match_B")  
poss_moyenne_B = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
matchs_sans_encaisser_B = st.slider("Nombre de matchs sans encaisser de but", min_value=0, max_value=40, value=5, key="matchs_sans_encaisser_B")  
xG_B = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.5, key="xG_B")  
tirs_cadres_B = st.slider("Tirs cadrés par match", min_value=0, max_value=20, value=3, key="tirs_cadres_B")  
pourcentage_tirs_convertis_B = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=15, key="pourcentage_tirs_convertis_B")  
grosses_occasions_B = st.slider("Grosses occasions créées", min_value=0, max_value=100, value=8, key="grosses_occasions_B")  
grosses_occasions_ratees_B = st.slider("Grosses occasions ratées", min_value=0, max_value=100, value=4, key="grosses_occasions_ratees_B")  

# Nouveaux critères pour l'équipe B  
passes_longues_B = st.slider("Passes longues précises par match", min_value=0, max_value=50, value=8, key="passes_longues_B")  
touches_surface_B = st.slider("Touches dans la surface adverse", min_value=0, max_value=50, value=12, key="touches_surface_B")  
corners_B = st.slider("Nombre de corners", min_value=0, max_value=20, value=4, key="corners_B")  
corners_concedes_B = st.slider("Nombre de corners concédés", min_value=0, max_value=20, value=5, key="corners_concedes_B")  
interceptions_B = st.slider("Interceptions par match", min_value=0, max_value=20, value=6, key="interceptions_B")  
arrets_B = st.slider("Arrêts par match", min_value=0, max_value=20, value=4, key="arrets_B")  
fautes_B = st.slider("Fautes par match", min_value=0, max_value=20, value=8, key="fautes_B")  
tactique_B = st.text_input("Tactique de l'équipe B", value="Défensive")  
joueurs_absents_B = st.text_input("Joueurs clés absents de l'équipe B", value="Aucun")  

# Séparateur  
st.markdown("---")  

# Critères météorologiques  
st.subheader("Conditions Météorologiques")  
temperature = st.slider("Température (°C)", min_value=-30, max_value=50, value=20)  
humidité = st.slider("Humidité (%)", min_value=0, max_value=100, value=50)  
conditions_meteo = st.selectbox("Conditions Météorologiques", ["Ensoleillé", "Pluvieux", "Neigeux", "Nuageux"])  

# Partie 2 : Modèles de Prédiction  
st.markdown("<h2 style='color: #FF5722;'>Prédiction des résultats</h2>", unsafe_allow_html=True)  

# Exemple de données fictives pour entraîner les modèles  
data = {  
    "buts_par_match_A": [2, 1, 3, 0, 2],  
    "nombre_buts_produits_A": [80, 60, 90, 30, 70],  
    "nombre_buts_encaisse_A": [30, 40, 20, 50, 25],  
    "buts_encaisse_par_match_A": [1, 2, 0, 3, 1],  
    "poss_moyenne_A": [55, 45, 60, 40, 50],  
    "xG_A": [2.5, 1.5, 3.0, 0.5, 2.0],  
    "grosses_occasions_A": [10, 8, 12, 5, 9],  
    "passes_longues_A": [10, 12, 15, 8, 11],  
    "touches_surface_A": [15, 20, 18, 10, 14],  
    "corners_A": [5, 3, 6, 2, 4],  
    "corners_concedes_A": [3, 4, 2, 5, 3],  
    "interceptions_A": [5, 6, 4, 7, 5],  
    "arrets_A": [3, 2, 4, 1, 3],  
    "fautes_A": [10, 12, 8, 9, 11],  
    "buts_par_match_B": [1, 2, 1, 0, 1],  
    "nombre_buts_produits_B": [60, 70, 50, 20, 40],  
    "nombre_buts_encaisse_B": [40, 30, 20, 50, 35],  
    "buts_encaisse_par_match_B": [2, 1, 1, 3, 2],  
    "poss_moyenne_B": [45, 55, 40, 50, 45],  
    "xG_B": [1.5, 2.0, 1.0, 0.5, 1.2],  
    "grosses_occasions_B": [8, 10, 6, 4, 7],  
    "passes_longues_B": [8, 10, 12, 6, 9],  
    "touches_surface_B": [12, 15, 10, 8, 11],  
    "corners_B": [4, 5, 3, 2, 4],  
    "corners_concedes_B": [5, 6, 4, 3, 5],  
    "interceptions_B": [6, 5, 7, 4, 5],  
    "arrets_B": [4, 3, 5, 2, 4],  
    "fautes_B": [8, 9, 7, 10, 8],  
    "temperature": [20, 15, 25, 10, 30],  
    "humidité": [50, 60, 40, 70, 30],  
}  

df = pd.DataFrame(data)  

# Entraînement des modèles  
features_A = [  
    "nombre_buts_produits_A", "nombre_buts_encaisse_A", "buts_encaisse_par_match_A",  
    "poss_moyenne_A", "xG_A", "grosses_occasions_A", "passes_longues_A",  
    "touches_surface_A", "corners_A", "corners_concedes_A", "interceptions_A",  
    "arrets_A", "fautes_A", "temperature", "humidité"  
]  
features_B = [  
    "nombre_buts_produits_B", "nombre_buts_encaisse_B", "buts_encaisse_par_match_B",  
    "poss_moyenne_B", "xG_B", "grosses_occasions_B", "passes_longues_B",  
    "touches_surface_B", "corners_B", "corners_concedes_B", "interceptions_B",  
    "arrets_B", "fautes_B", "temperature", "humidité"  
]  

X_A = df[features_A]  
y_A = df["buts_par_match_A"]  

X_B = df[features_B]  
y_B = df["buts_par_match_B"]  

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)  
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)  

# Modèle 1 : Régression Linéaire pour l'équipe A  
model_lr_A = LinearRegression()  
model_lr_A.fit(X_train_A, y_train_A)  

# Modèle 2 : Régression Linéaire pour l'équipe B  
model_lr_B = LinearRegression()  
model_lr_B.fit(X_train_B, y_train_B)  

# Modèle de prédiction des probabilités de victoire  
model_rf = RandomForestRegressor(random_state=42)  

# Utilisation de pd.concat() pour combiner les DataFrames  
X_combined = pd.concat([X_A, X_B], ignore_index=True)  
y_combined = pd.concat([y_A, y_B], ignore_index=True)  

model_rf.fit(X_combined, y_combined)  

# Prédictions pour les nouvelles données  
nouvelle_donnee_A = pd.DataFrame({  
    "nombre_buts_produits_A": [nombre_buts_produits_A],  
    "nombre_buts_encaisse_A": [nombre_buts_encaisse_A],  
    "buts_encaisse_par_match_A": [buts_encaisse_par_match_A],  
    "poss_moyenne_A": [poss_moyenne_A],  
    "xG_A": [xG_A],  
    "grosses_occasions_A": [grosses_occasions_A],  
    "passes_longues_A": [passes_longues_A],  
    "touches_surface_A": [touches_surface_A],  
    "corners_A": [corners_A],  
    "corners_concedes_A": [corners_concedes_A],  
    "interceptions_A": [interceptions_A],  
    "arrets_A": [arrets_A],  
    "fautes_A": [fautes_A],  
    "temperature": [temperature],  
    "humidité": [humidité],  
})  

nouvelle_donnee_B = pd.DataFrame({  
    "nombre_buts_produits_B": [nombre_buts_produits_B],  
    "nombre_buts_encaisse_B": [nombre_buts_encaisse_B],  
    "buts_encaisse_par_match_B": [buts_encaisse_par_match_B],  
    "poss_moyenne_B": [poss_moyenne_B],  
    "xG_B": [xG_B],  
    "grosses_occasions_B": [grosses_occasions_B],  
    "passes_longues_B": [passes_longues_B],  
    "touches_surface_B": [touches_surface_B],  
    "corners_B": [corners_B],  
    "corners_concedes_B": [corners_concedes_B],  
    "interceptions_B": [interceptions_B],  
    "arrets_B": [arrets_B],  
    "fautes_B": [fautes_B],  
    "temperature": [temperature],  
    "humidité": [humidité],  
})  

# Vérification de la longueur des données  
if st.button("Prédire le nombre de buts et les probabilités de victoire"):  
    # Prédictions pour l'équipe A  
    prediction_A = model_lr_A.predict(nouvelle_donnee_A)[0]  
    # Prédictions pour l'équipe B  
    prediction_B = model_lr_B.predict(nouvelle_donnee_B)[0]  

    # Calcul des probabilités de victoire  
    prob_victoire_A = model_rf.predict(nouvelle_donnee_A)[0] / (model_rf.predict(nouvelle_donnee_A)[0] + model_rf.predict(nouvelle_donnee_B)[0])  
    prob_victoire_B = model_rf.predict(nouvelle_donnee_B)[0] / (model_rf.predict(nouvelle_donnee_A)[0] + model_rf.predict(nouvelle_donnee_B)[0])  

    st.write(f"Prédiction des buts pour l'équipe A : **{prediction_A:.2f}**")  
    st.write(f"Prédiction des buts pour l'équipe B : **{prediction_B:.2f}**")  
    st.write(f"Probabilité de victoire pour l'équipe A : **{prob_victoire_A:.2%}**")  
    st.write(f
