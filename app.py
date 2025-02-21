import streamlit as st  
import pandas as pd  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  

# Titre de l'application  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de match et gestion de bankroll</h1>", unsafe_allow_html=True)  
st.write("Analysez les données des matchs et simulez les résultats pour optimiser votre bankroll.")  

# Partie 1 : Analyse des équipes  
st.markdown("<h2 style='color: #2196F3;'>1. Analyse des équipes</h2>", unsafe_allow_html=True)  

# Historique des équipes  
st.subheader("Historique des équipes")  
historique_A = st.text_area("Historique des performances de l'équipe A", height=100)  
historique_B = st.text_area("Historique des performances de l'équipe B", height=100)  

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

# Partie 2 : Modèles de Prédiction  
st.markdown("<h2 style='color: #FF5722;'>2. Prédiction des résultats</h2>", unsafe_allow_html=True)  

# Exemple de données fictives pour entraîner les modèles  
data = {  
    "buts_par_match": [2, 1, 3, 0, 2],  
    "nombre_buts_produits": [80, 60, 90, 30, 70],  
    "nombre_buts_encaisse": [30, 40, 20, 50, 25],  
    "buts_encaisse_par_match": [1, 2, 0, 3, 1],  
    "poss_moyenne": [55, 45, 60, 40, 50],  
    "xG": [2.5, 1.5, 3.0, 0.5, 2.0],  
    "grosses_occasions": [10, 8, 12, 5, 9],  
    "resultat_A": [1, 0, 1, 0, 1],  # 1 = victoire équipe A, 0 = victoire équipe B  
    "resultat_B": [0, 1, 0, 1, 0]   # 1 = victoire équipe B, 0 = victoire équipe A  
}  
df = pd.DataFrame(data)  

# Entraînement des modèles  
features = ["buts_par_match", "nombre_buts_produits", "nombre_buts_encaisse", "buts_encaisse_par_match", "poss_moyenne", "xG", "grosses_occasions"]  
X = df[features]  
y_A = df["resultat_A"]  
y_B = df["resultat_B"]  

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X, y_A, test_size=0.2, random_state=42)  
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y_B, test_size=0.2, random_state=42)  

# Modèle 1 : RandomForest pour l'équipe A  
model_rf_A = RandomForestRegressor(random_state=42)  
model_rf_A.fit(X_train_A, y_train_A)  

# Modèle 2 : Linear Regression pour l'équipe B  
model_lr_B = LinearRegression()  
model_lr_B.fit(X_train_B, y_train_B)  

# Prédictions pour les nouvelles données  
nouvelle_donnee = pd.DataFrame({  
    "buts_par_match": [buts_par_match_A],  
    "nombre_buts_produits": [nombre_buts_produits_A],  
    "nombre_buts_encaisse": [nombre_buts_encaisse_A],  
    "buts_encaisse_par_match": [buts_encaisse_par_match_A],  
    "poss_moyenne": [poss_moyenne_A],  
    "xG": [xG_A],  
    "grosses_occasions": [grosses_occasions_A],  
})  

nouvelle_donnee_B = pd.DataFrame({  
    "buts_par_match": [buts_par_match_B],  
    "nombre_buts_produits": [nombre_buts_produits_B],  
    "nombre_buts_encaisse": [nombre_buts_encaisse_B],  
    "buts_encaisse_par_match": [buts_encaisse_par_match_B],  
    "poss_moyenne": [poss_moyenne_B],  
    "xG": [xG_B],  
    "grosses_occasions": [grosses_occasions_B],  
})  

# Vérification de la longueur des données  
if st.button("Prédire les résultats"):  
    prediction_A = model_rf_A.predict(nouvelle_donnee)[0]  # Prédiction pour l'équipe A  
    prediction_B = model_lr_B.predict(nouvelle_donnee_B)[0]  # Prédiction pour l'équipe B  

    # Conversion des prédictions en résultats  
    if prediction_A > prediction_B:  
        resultat = "Victoire de l'équipe A"  
    elif prediction_A < prediction_B:  
        resultat = "Victoire de l'équipe B"  
    else:  
        resultat = "Match nul"  

    # Affichage des résultats de prédiction  
    st.write(f"Prédiction de victoire pour l'équipe A : **{prediction_A:.2f}**")  
    st.write(f"Prédiction de victoire pour l'équipe B : **{prediction_B:.2f}**")  
    st.write(f"Résultat prédit : **{resultat}**")  

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
