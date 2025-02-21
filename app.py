import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
import seaborn as sns  
from scipy.stats import poisson  

# Création de données fictives  
def create_fake_data(num_samples=100):  
    np.random.seed(42)  
    data = {  
        'buts_produits': np.random.randint(0, 5, num_samples),  
        'buts_encaisses': np.random.randint(0, 5, num_samples),  
        'possession': np.random.uniform(40, 70, num_samples),  
        'motivation': np.random.uniform(1, 10, num_samples),  
        'absents': np.random.randint(0, 5, num_samples),  
        'rating_joueurs': np.random.uniform(0, 10, num_samples),  
        'forme_recente': np.random.uniform(0, 10, num_samples),  
        'tirs_cadres': np.random.randint(0, 10, num_samples),  
        'corners': np.random.randint(0, 10, num_samples),  
        'xG': np.random.uniform(0, 3, num_samples),  
        'grosses_occasions': np.random.randint(0, 5, num_samples),  
        'grosses_occasions_ratees': np.random.randint(0, 5, num_samples),  
        'passes_reussies': np.random.randint(0, 1000, num_samples),  
        'passes_longues_precises': np.random.randint(0, 20, num_samples),  
        'centres_reussis': np.random.randint(0, 10, num_samples),  
        'penalties_obtenus': np.random.randint(0, 5, num_samples),  
        'touches_surface_adverse': np.random.randint(0, 10, num_samples),  
        'corners_concedes': np.random.randint(0, 10, num_samples),  
        'xG_concedes': np.random.uniform(0, 3, num_samples),  
        'interceptions': np.random.randint(0, 10, num_samples),  
        'tacles_reussis': np.random.randint(0, 10, num_samples),  
        'degagements': np.random.randint(0, 10, num_samples),  
        'possessions_recuperees': np.random.randint(0, 10, num_samples),  
        'penalties_concedes': np.random.randint(0, 5, num_samples),  
        'arrets': np.random.randint(0, 10, num_samples),  
        'fautes': np.random.randint(0, 20, num_samples),  
        'cartons_jaunes': np.random.randint(0, 5, num_samples),  
        'cartons_rouges': np.random.randint(0, 2, num_samples),  
        'resultat': np.random.choice([0, 1], num_samples)  # 0 pour victoire B, 1 pour victoire A  
    }  
    return pd.DataFrame(data)  

# Préparation des données  
def prepare_data(data):  
    features = data[['buts_produits', 'buts_encaisses', 'possession', 'motivation',   
                     'absents', 'rating_joueurs', 'forme_recente', 'tirs_cadres',   
                     'corners', 'xG', 'grosses_occasions', 'grosses_occasions_ratees',   
                     'passes_reussies', 'passes_longues_precises', 'centres_reussis',   
                     'penalties_obtenus', 'touches_surface_adverse', 'corners_concedes',   
                     'xG_concedes', 'interceptions', 'tacles_reussis', 'degagements',   
                     'possessions_recuperees', 'penalties_concedes', 'arrets',   
                     'fautes', 'cartons_jaunes', 'cartons_rouges']]  
    target = data['resultat']  # 1 pour victoire A, 0 pour victoire B  
    return features, target  

# Chargement des données fictives  
data = create_fake_data(100)  
features, target = prepare_data(data)  

# Séparation des données  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)  

# Normalisation des données  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Entraînement du modèle Random Forest  
rf_model = RandomForestClassifier(random_state=42)  
rf_model.fit(X_train_scaled, y_train)  

# Entraînement du modèle de régression logistique  
log_model = LogisticRegression(max_iter=1000)  
log_model.fit(X_train_scaled, y_train)  

# Interface utilisateur Streamlit  
st.title("Prédiction de Matchs de Football")  
st.write("Entrez les statistiques des deux équipes pour prédire le résultat du match.")  

# Saisie des données pour l'équipe A  
st.header("Équipe A")  
buts_produits_A = st.number_input("Buts produits", min_value=0, max_value=5, value=1)  
buts_encaisses_A = st.number_input("Buts encaissés", min_value=0, max_value=5, value=1)  
possession_A = st.number_input("Possession (%)", min_value=0, max_value=100, value=50)  
motivation_A = st.number_input("Motivation (1-10)", min_value=1, max_value=10, value=5)  
absents_A = st.number_input("Joueurs absents", min_value=0, max_value=5, value=0)  
rating_joueurs_A = st.number_input("Rating des joueurs (0-10)", min_value=0.0, max_value=10.0, value=5.0)  
forme_recente_A = st.number_input("Forme récente (0-10)", min_value=0.0, max_value=10.0, value=5.0)  
tirs_cadres_A = st.number_input("Tirs cadrés", min_value=0, max_value=10, value=5)  
corners_A = st.number_input("Corners", min_value=0, max_value=10, value=5)  
xG_A = st.number_input("xG (expected goals)", min_value=0.0, max_value=3.0, value=1.0)  

# Saisie des données pour l'équipe B  
st.header("Équipe B")  
buts_produits_B = st.number_input("Buts produits", min_value=0, max_value=5, value=1, key="B")  
buts_encaisses_B = st.number_input("Buts encaissés", min_value=0, max_value=5, value=1, key="B2")  
possession_B = st.number_input("Possession (%)", min_value=0, max_value=100, value=50, key="B3")  
motivation_B = st.number_input("Motivation (1-10)", min_value=1, max_value=10, value=5, key="B4")  
absents_B = st.number_input("Joueurs absents", min_value=0, max_value=5, value=0, key="B5")  
rating_joueurs_B = st.number_input("Rating des joueurs (0-10)", min_value=0.0, max_value=10.0, value=5.0, key="B6")  
forme_recente_B = st.number_input("Forme récente (0-10)", min_value=0.0, max_value=10.0, value=5.0, key="B7")  
tirs_cadres_B = st.number_input("Tirs cadrés", min_value=0, max_value=10, value=5, key="B8")  
corners_B = st.number_input("Corners", min_value=0, max_value=10, value=5, key="B9")  
xG_B = st.number_input("xG (expected goals)", min_value=0.0, max_value=3.0, value=1.0, key="B10")  

# Prédiction avec Random Forest  
if st.button("Prédire le résultat avec Random Forest"):  
    # Préparation des données d'entrée  
    input_data_rf = np.array([[buts_produits_A, buts_encaisses_B, possession_A, motivation_A,   
                                absents_A, rating_joueurs_A, forme_recente_A, tirs_cadres_A,   
                                corners_A, xG_A,   
                                buts_produits_B, buts_encaisses_A, possession_B, motivation_B,   
                                absents_B, rating_joueurs_B, forme_recente_B, tirs_cadres_B,   
                                corners_B, xG_B]])  
    
    # Normalisation des données d'entrée  
    input_data_scaled_rf = scaler.transform(input_data_rf)  

    # Prédiction avec le modèle Random Forest  
    prediction_rf = rf_model.predict(input_data_scaled_rf)  

    # Affichage du résultat  
    if prediction_rf[0] == 1:  
        st.success("L'équipe A est prédite pour gagner !")  
    else:  
        st.success("L'équipe B est prédite pour gagner !")  

# Prédiction avec régression logistique  
if st.button("Prédire le résultat avec Régression Logistique"):  
    # Préparation des données d'entrée  
    input_data_log = np.array([[buts_produits_A, buts_encaisses_B, possession_A, motivation_A,   
                                 absents_A, rating_joueurs_A, forme_recente_A, tirs_cadres_A,   
                                 corners_A, xG_A,   
                                 buts_produits_B, buts_encaisses_A, possession_B, motivation_B,   
                                 absents_B, rating_joueurs_B, forme_recente_B, tirs_cadres_B,   
                                 corners_B, xG_B]])  
    
    # Normalisation des données d'entrée  
    input_data_scaled_log = scaler.transform(input_data_log)  

    # Prédiction avec le modèle de régression logistique  
    prediction_log = log_model.predict(input_data_scaled_log)  

    # Affichage du résultat  
    if prediction_log[0] == 1:  
        st.success("L'équipe A est prédite pour gagner !")  
    else:  
        st.success("L'équipe B est prédite pour gagner !")  

# Prédiction des buts avec la méthode de Poisson  
def prediction_buts_poisson(xG_A, xG_B):  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  
    buts_attendus_A = sum(i * prob for i, prob in enumerate(buts_A))  
    buts_attendus_B = sum(i * prob for i, prob in enumerate(buts_B))  
    return buts_attendus_A, buts_attendus_B  

# Affichage des résultats de la prédiction des buts  
if st.button("Prédire les buts avec la méthode de Poisson"):  
    buts_moyens_A, buts_moyens_B = prediction_buts_poisson(xG_A, xG_B)  
    st.header("⚽ Prédiction des Buts (Méthode de Poisson)")  
    st.write(f"Buts attendus pour l'équipe A : **{buts_moyens_A:.2f}**")  
    st.write(f"Buts attendus pour l'équipe B : **{buts_moyens_B:.2f}**")
