import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor  

# Titre de l'application  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de match et gestion de bankroll</h1>", unsafe_allow_html=True)  
st.write("Analysez les données des matchs et simulez les résultats pour optimiser votre bankroll.")  

# Partie 1 : Analyse des équipes  
st.markdown("<h2 style='color: #2196F3;'>1. Analyse des équipes</h2>", unsafe_allow_html=True)  

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

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
buts_par_match_A = st.number_input("Buts par match", min_value=0, max_value=10, value=2, key="buts_par_match_A")  
nombre_buts_produits_A = st.number_input("Nombre de buts produits", min_value=0, max_value=100, value=80, key="nombre_buts_produits_A")  
nombre_buts_encaisse_A = st.number_input("Nombre de buts encaissés", min_value=0, max_value=100, value=30, key="nombre_buts_encaisse_A")  
buts_encaisse_par_match_A = st.number_input("Buts encaissés par match", min_value=0, max_value=10, value=1, key="buts_encaisse_par_match_A")  
poss_moyenne_A = st.number_input("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
matchs_sans_encaisser_A = st.number_input("Nombre de matchs sans encaisser de but", min_value=0, max_value=40, value=10, key="matchs_sans_encaisser_A")  
xG_A = st.number_input("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=2.5, key="xG_A")  
tirs_cadres_A = st.number_input("Tirs cadrés par match", min_value=0, max_value=20, value=5, key="tirs_cadres_A")  
pourcentage_tirs_convertis_A = st.number_input("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=25, key="pourcentage_tirs_convertis_A")  
grosses_occasions_A = st.number_input("Grosses occasions créées", min_value=0, max_value=100, value=10, key="grosses_occasions_A")  
grosses_occasions_ratees_A = st.number_input("Grosses occasions ratées", min_value=0, max_value=100, value=5, key="grosses_occasions_ratees_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_par_match_B = st.number_input("Buts par match", min_value=0, max_value=10, value=1, key="buts_par_match_B")  
nombre_buts_produits_B = st.number_input("Nombre de buts produits", min_value=0, max_value=100, value=60, key="nombre_buts_produits_B")  
nombre_buts_encaisse_B = st.number_input("Nombre de buts encaissés", min_value=0, max_value=100, value=40, key="nombre_buts_encaisse_B")  
buts_encaisse_par_match_B = st.number_input("Buts encaissés par match", min_value=0, max_value=10, value=2, key="buts_encaisse_par_match_B")  
poss_moyenne_B = st.number_input("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
matchs_sans_encaisser_B = st.number_input("Nombre de matchs sans encaisser de but", min_value=0, max_value=40, value=5, key="matchs_sans_encaisser_B")  
xG_B = st.number_input("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.5, key="xG_B")  
tirs_cadres_B = st.number_input("Tirs cadrés par match", min_value=0, max_value=20, value=3, key="tirs_cadres_B")  
pourcentage_tirs_convertis_B = st.number_input("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=15, key="pourcentage_tirs_convertis_B")  
grosses_occasions_B = st.number_input("Grosses occasions créées", min_value=0, max_value=100, value=8, key="grosses_occasions_B")  
grosses_occasions_ratees_B = st.number_input("Grosses occasions ratées", min_value=0, max_value=100, value=4, key="grosses_occasions_ratees_B")  

# Critères météorologiques  
st.subheader("Conditions Météorologiques")  
temperature = st.number_input("Température (°C)", min_value=-30, max_value=50, value=20)  
humidité = st.number_input("Humidité (%)", min_value=0, max_value=100, value=50)  
conditions_meteo = st.selectbox("Conditions Météorologiques", ["Ensoleillé", "Pluvieux", "Neigeux", "Nuageux"])  

# Partie 2 : Modèles de Prédiction  
st.markdown("<h2 style='color: #FF5722;'>2. Prédiction des résultats</h2>", unsafe_allow_html=True)  

# Exemple de données fictives pour entraîner les modèles  
data = {  
    "buts_par_match_A": [2, 1, 3, 0, 2],  
    "nombre_buts_produits_A": [80, 60, 90, 30, 70],  
    "nombre_buts_encaisse_A": [30, 40, 20, 50, 25],  
    "buts_encaisse_par_match_A": [1, 2, 0, 3, 1],  
    "poss_moyenne_A": [55, 45, 60, 40, 50],  
    "xG_A": [2.5, 1.5, 3.0, 0.5, 2.0],  
    "grosses_occasions_A": [10, 8, 12, 5, 9],  
    "buts_par_match_B": [1, 2, 1, 0, 1],  
    "nombre_buts_produits_B": [60, 70, 50, 20, 40],  
    "nombre_buts_encaisse_B": [40, 30, 20, 50, 35],  
    "buts_encaisse_par_match_B": [2, 1, 1, 3, 2],  
    "poss_moyenne_B": [45, 55, 40, 50, 45],  
    "xG_B": [1.5, 2.0, 1.0, 0.5, 1.2],  
    "grosses_occasions_B": [8, 10, 6, 4, 7],  
    "temperature": [20, 15, 25, 10, 30],  
    "humidité": [50, 60, 40, 70, 30],  
}  

df = pd.DataFrame(data)  

# Entraînement des modèles  
features_A = ["nombre_buts_produits_A", "nombre_buts_encaisse_A", "buts_encaisse_par_match_A", "poss_moyenne_A", "xG_A", "grosses_occasions_A", "temperature", "humidité"]  
features_B = ["nombre_buts_produits_B", "nombre_buts_encaisse_B", "buts_encaisse_par_match_B", "poss_moyenne_B", "xG_B", "grosses_occasions_B", "temperature", "humidité"]  

X_A = df[features_A]  
y_A = df["buts_par_match_A"]  

X_B = df[features_B]  
y_B = df["buts_par_match_B"]  

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)  
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)  

# Modèle 1 : Decision Tree pour l'équipe A  
model_dt_A = DecisionTreeRegressor(random_state=42)  
model_dt_A.fit(X_train_A, y_train_A)  

# Modèle 2 : Decision Tree pour l'équipe B  
model_dt_B = DecisionTreeRegressor(random_state=42)  
model_dt_B.fit(X_train_B, y_train_B)  

# Prédictions pour les nouvelles données  
nouvelle_donnee_A = pd.DataFrame({  
    "nombre_buts_produits_A": [nombre_buts_produits_A],  
    "nombre_buts_encaisse_A": [nombre_buts_encaisse_A],  
    "buts_encaisse_par_match_A": [buts_encaisse_par_match_A],  
    "poss_moyenne_A": [poss_moyenne_A],  
    "xG_A": [xG_A],  
    "grosses_occasions_A": [grosses_occasions_A],  
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
    "temperature": [temperature],  
    "humidité": [humidité],  
})  

# Vérification de la longueur des données  
if st.button("Prédire le nombre de buts"):  
    prediction_A = model_dt_A.predict(nouvelle_donnee_A)[0]  # Prédiction pour l'équipe A  
    prediction_B = model_dt_B.predict(nouvelle_donnee_B)[0]  # Prédiction pour l'équipe B  

    # Affichage des résultats de prédiction  
    st.write(f"Prédiction du nombre de buts pour l'équipe A : **{prediction_A:.2f}**")  
    st.write(f"Prédiction du nombre de buts pour l'équipe B : **{prediction_B:.2f}**")  

# Partie 3 : Gestion de la bankroll  
st.markdown("<h2 style='color: #FF5722;'>3. Gestion de la bankroll</h2>", unsafe_allow_html=True)  

# Entrée de la bankroll initiale  
bankroll_initiale = st.number_input("Montant de la bankroll initiale (€)", min_value=0.0, value=1000.0, step=10.0)  

# Mise par pari  
mise = st.number_input("Mise par pari (€)", min_value=0.0, value=10.0, step=1.0)  

# Cote de victoire  
cote_A = st.number_input("Cote de victoire pour l'équipe A", min_value=1.0, value=2.0, step=0.1)  
cote_B = st.number_input("Cote de victoire pour l'équipe B", min_value=1.0, value=1.5, step=0.1)  

# Résultat du pari  
resultat_pari = st.selectbox("Résultat du pari", ["A gagner", "B gagner", "Match nul"])  

# Calcul de la bankroll après le pari  
if st.button("Calculer la bankroll après le pari"):  
    if resultat_pari == "A gagner":  
        bankroll_finale = bankroll_initiale + (mise * (cote_A - 1))  
    elif resultat_pari == "B gagner":  
        bankroll_finale = bankroll_initiale + (mise * (cote_B - 1))  
    else:  
        bankroll_finale = bankroll_initiale - mise  # En cas de match nul, on perd la mise  

    st.write(f"Votre bankroll après le pari est de : **{bankroll_finale:.2f} €**")
