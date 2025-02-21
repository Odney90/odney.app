import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
import pickle  
import os  
from scipy.stats import poisson  
import matplotlib.pyplot as plt  

# Fonction pour sauvegarder les données  
def save_data(data):  
    with open('data.pkl', 'wb') as f:  
        pickle.dump(data, f)  

# Fonction pour charger les données  
def load_data():  
    if os.path.exists('data.pkl'):  
        with open('data.pkl', 'rb') as f:  
            return pickle.load(f)  
    return None
    
    # Chargement des données précédentes  
previous_data = load_data()  

# Interface utilisateur Streamlit  
st.title("⚽ Prédiction de Matchs de Football")  
st.write("Entrez les statistiques des deux équipes pour prédire le résultat du match.")  

# Couleurs pour les équipes  
color_A = "#1E90FF"  # Bleu pour l'équipe A  
color_B = "#FF4500"  # Rouge pour l'équipe B
# Saisie des données pour l'équipe A  
st.header("Équipe A 🏆")  
col1, col2 = st.columns(2)  

with col1:  
    historique_A = st.text_area("Historique des performances (ex: W, D, L)", height=100)  

with col2:  
    equipe_recevante = st.selectbox("Équipe qui reçoit", ["Équipe A", "Équipe B"])  

# Statistiques de l'équipe A  
st.subheader("Statistiques de l'Équipe A")  
buts_totaux_A = st.number_input("Buts totaux", min_value=0.0, value=0.0)  
buts_par_match_A = st.number_input("Buts par match", min_value=0.0, value=0.0)  
buts_concedes_par_match_A = st.number_input("Buts concédés par match", min_value=0.0, value=0.0)  
buts_concedes_totaux_A = st.number_input("Buts concédés au total", min_value=0.0, value=0.0)  
possession_moyenne_A = st.number_input("Possession moyenne (%)", min_value=0.0, max_value=100.0, value=50.0)  
aucun_but_encaisse_A = st.number_input("Aucun but encaissé (oui=1, non=0)", min_value=0, max_value=1, value=0)
# Critères d'Attaque  
st.subheader("Critères d'Attaque 🔥")  
expected_buts_A = st.number_input("Expected Goals (xG)", min_value=0.0, value=0.0)  
tirs_cadres_A = st.number_input("Tirs cadrés par match", min_value=0.0, value=0.0)  
grosses_chances_A = st.number_input("Grandes chances", min_value=0.0, value=0.0)  
grosses_chances_ratees_A = st.number_input("Grandes chances manquées", min_value=0.0, value=0.0)  
passes_reussies_A = st.number_input("Passes réussies par match", min_value=0.0, value=0.0)  
passes_longues_precises_A = st.number_input("Passes longues précises par match", min_value=0.0, value=0.0)  
centres_reussis_A = st.number_input("Centres réussis par match", min_value=0.0, value=0.0)  
penalites_obtenues_A = st.number_input("Pénalités obtenues", min_value=0.0, value=0.0)  
touches_surface_adverse_A = st.number_input("Balles touchées dans la surface adverse", min_value=0.0, value=0.0)  
corners_A = st.number_input("Nombres de corners", min_value=0.0, value=0.0)  

# Critères de Défense  
st.subheader("Critères de Défense 🛡️")  
expected_concedes_A = st.number_input("Expected Goals concédés (xG)", min_value=0.0, value=0.0)  
interceptions_A = st.number_input("Interceptions par match", min_value=0.0, value=0.0)  
tacles_reussis_A = st.number_input("Tacles réussis par match", min_value=0.0, value=0.0)  
degegements_A = st.number_input("Dégagements par match", min_value=0.0, value=0.0)  
penalites_concedes_A = st.number_input("Pénalités concédées", min_value=0.0, value=0.0)  
possessions_remporte_A = st.number_input("Possessions remportées", min_value=0.0, value=0.0)  
arrets_A = st.number_input("Arrêts par match", min_value=0.0, value=0.0)
# Critères de Discipline  
st.subheader("Critères de Discipline ⚖️")  
fautes_A = st.number_input("Fautes par match", min_value=0.0, value=0.0)  
cartons_jaunes_A = st.number_input("Cartons jaunes", min_value=0.0, value=0.0)  
cartons_rouges_A = st.number_input("Cartons rouges", min_value=0.0, value=0.0)  

# Saisie des données pour l'équipe B  
st.header("Équipe B 🥈")  
col3, col4 = st.columns(2)  

with col3:  
    historique_B = st.text_area("Historique des performances (ex: W, D, L)", height=100, key="B_historique")  

with col4:  
    # Statistiques de l'équipe B  
    st.subheader("Statistiques de l'Équipe B")  
    buts_totaux_B = st.number_input("Buts totaux", min_value=0.0, value=0.0, key="B")  
    buts_par_match_B = st.number_input("Buts par match", min_value=0.0, value=0.0, key="B2")  
    buts_concedes_par_match_B = st.number_input("Buts concédés par match", min_value=0.0, value=0.0, key="B3")  
    buts_concedes_totaux_B = st.number_input("Buts concédés au total", min_value=0.0, value=0.0, key="B4")  
    possession_moyenne_B = st.number_input("Possession moyenne (%)", min_value=0.0, max_value=100.0, value=50.0, key="B5")  
    aucun_but_encaisse_B = st.number_input("Aucun but encaissé (oui=1, non=0)", min_value=0, max_value=1, value=0, key="B6")
    # Critères d'Attaque pour l'équipe B  
st.subheader("Critères d'Attaque 🔥")  
expected_buts_B = st.number_input("Expected Goals (xG)", min_value=0.0, value=0.0, key="B7")  
tirs_cadres_B = st.number_input("Tirs cadrés par match", min_value=0.0, value=0.0, key="B8")  
grosses_chances_B = st.number_input("Grandes chances", min_value=0.0, value=0.0, key="B9")  
grosses_chances_ratees_B = st.number_input("Grandes chances manquées", min_value=0.0, value=0.0, key="B10")  
passes_reussies_B = st.number_input("Passes réussies par match", min_value=0.0, value=0.0, key="B11")  
passes_longues_precises_B = st.number_input("Passes longues précises par match", min_value=0.0, value=0.0, key="B12")  
centres_reussis_B = st.number_input("Centres réussis par match", min_value=0.0, value=0.0, key="B13")  
penalites_obtenues_B = st.number_input("Pénalités obtenues", min_value=0.0, value=0.0, key="B14")  
touches_surface_adverse_B = st.number_input("Balles touchées dans la surface adverse", min_value=0.0, value=0.0, key="B15")  
corners_B = st.number_input("Nombres de corners", min_value=0.0, value=0.0, key="B16")  

# Critères de Défense pour l'équipe B  
st.subheader("Critères de Défense 🛡️")  
expected_concedes_B = st.number_input("Expected Goals concédés (xG)", min_value=0.0, value=0.0, key="B17")  
interceptions_B = st.number_input("Interceptions par match", min_value=0.0, value=0.0, key="B18")  
tacles_reussis_B = st.number_input("Tacles réussis par match", min_value=0.0, value=0.0, key="B19")  
degegements_B = st.number_input("Dégagements par match", min_value=0.0, value=0.0, key="B20")  
penalites_concedes_B = st.number_input("Pénalités concédées", min_value=0.0, value=0.0, key="B21")  
possessions_remporte_B = st.number_input("Possessions remportées", min_value=0.0, value=0.0, key="B22")  
arrets_B = st.number_input("Arrêts par match", min_value=0.0, value=0.0, key="B23")
# Critères de Discipline pour l'équipe B  
st.subheader("Critères de Discipline ⚖️")  
fautes_B = st.number_input("Fautes par match", min_value=0.0, value=0.0, key="B24")  
cartons_jaunes_B = st.number_input("Cartons jaunes", min_value=0.0, value=0.0, key="B25")  
cartons_rouges_B = st.number_input("Cartons rouges", min_value=0.0, value=0.0, key="B26")  

# Sauvegarde automatique des données  
data_to_save = {  
    'Équipe A': {  
        'Historique': historique_A,  
        'Buts totaux': buts_totaux_A,  
        'Buts par match': buts_par_match_A,  
        'Buts concédés par match': buts_concedes_par_match_A,  
        'Buts concédés au total': buts_concedes_totaux_A,  
        'Possession moyenne': possession_moyenne_A,  
        'Aucun but encaissé': aucun_but_encaisse_A,  
        'Expected Goals': expected_buts_A,  
        'Tirs cadrés': tirs_cadres_A,  
        'Grandes chances': grosses_chances_A,  
        'Grandes chances manquées': grosses_chances_ratees_A,  
        'Passes réussies': passes_reussies_A,  
        'Passes longues précises': passes_longues_precises_A,  
        'Centres réussis': centres_reussis_A,  
        'Pénalités obtenues': penalites_obtenues_A,  
        'Touches surface adverse': touches_surface_adverse_A,  
        'Corners': corners_A,  
        'Expected Goals concédés': expected_concedes_A,  
        'Interceptions': interceptions_A,  
        'Tacles réussis': tacles_reussis_A,  
        'Dégagements': degegements_A,  
        'Pénalités concédées': penalites_concedes_A,  
        'Possessions remportées': possessions_remporte_A,  
        'Arrêts': arrets_A,  
        'Fautes': fautes_A,  
        'Cartons jaunes': cartons_jaunes_A,  
        'Cartons rouges': cartons_rouges_A  
    },  
    'Équipe B': {  
        'Historique': historique_B,  
        'Buts totaux': buts_totaux_B,  
        'Buts par match': buts_par_match_B,  
        'Buts concédés par match': buts_concedes_par_match_B,  
        'Buts concédés au total': buts_concedes_totaux_B,  
        'Possession moyenne': possession_moyenne_B,  
        'Aucun but encaissé': aucun_but_encaisse_B,  
        'Expected Goals': expected_buts_B,  
        'Tirs cadrés': tirs_cadres_B,  
        'Grandes chances': grosses_chances_B,  
        'Grandes chances manquées': grosses_chances_ratees_B,  
        'Passes réussies': passes_reussies_B,  
        'Passes longues précises': passes_longues_precises_B,  
        'Centres réussis': centres_reussis_B,  
        'Pénalités obtenues': penalites_obtenues_B,  
        'Touches surface adverse': touches_surface_adverse_B,  
        'Corners': corners_B,  
        'Expected Goals concédés': expected_concedes_B,  
        'Interceptions': interceptions_B,  
        'Tacles réussis': tacles_reussis_B,  
        'Dégagements': degegements_B,  
        'Pénalités concédées': penalites_concedes_B,  
        'Possessions remportées': possessions_remporte_B,  
        'Arrêts': arrets_B,  
        'Fautes': fautes_B,  
        'Cartons jaunes': cartons_jaunes_B,  
        'Cartons rouges': cartons_rouges_B  
    }  
}  
save_data(data_to_save)

# Prédiction avec Random Forest  
if st.button("🔮 Prédire le résultat avec Random Forest"):  
    # Préparation des données d'entrée  
    input_data_rf = np.array([[buts_totaux_A, buts_concedes_totaux_B, possession_moyenne_A,   
                                expected_buts_A, tirs_cadres_A,   
                                buts_totaux_B, buts_concedes_totaux_A, possession_moyenne_B,   
                                expected_buts_B, tirs_cadres_B]])  
    
    # Normalisation des données d'entrée  
    scaler = StandardScaler()  
    input_data_scaled_rf = scaler.fit_transform(input_data_rf)  

    # Prédiction avec le modèle Random Forest  
    rf_model = RandomForestClassifier(random_state=42)  
    rf_model.fit(input_data_scaled_rf, [1])  # Dummy fit for demonstration  
    prediction_rf = rf_model.predict(input_data_scaled_rf)  

    # Affichage du résultat  
    if prediction_rf[0] == 1:  
        st.success("🏆 L'équipe A est prédite pour gagner !")  
    else:  
        st.success("🥈 L'équipe B est prédite pour gagner !")  

# Prédiction avec régression logistique  
if st.button("🔮 Prédire le résultat avec Régression Logistique"):  
    # Préparation des données d'entrée  
    input_data_log = np.array([[buts_totaux_A, buts_concedes_totaux_B, possession_moyenne_A,   
                                 expected_buts_A, tirs_cadres_A,   
                                 buts_totaux_B, buts_concedes_totaux_A, possession_moyenne_B,   
                                 expected_buts_B, tirs_cadres_B]])  
    
    # Normalisation des données d'entrée  
    scaler = StandardScaler()  
    input_data_scaled_log = scaler.fit_transform(input_data_log)  

    # Prédiction avec le modèle de régression logistique  
    log_model = LogisticRegression(max_iter=1000)  
    
    # Dummy fit pour démonstration, remplacez par des données réelles  
    # Vous devez avoir un ensemble de données d'entraînement pour le modèle  
    # Exemple : log_model.fit(X_train, y_train)  
    # Ici, nous allons simuler un entraînement avec des données fictives  
    X_train = np.random.rand(100, 10)  # Remplacez par vos données réelles  
    y_train = np.random.randint(0, 2, size=100)  # Remplacez par vos étiquettes réelles  
    log_model.fit(X_train, y_train)  

    prediction_log = log_model.predict(input_data_scaled_log)  
    prediction_proba = log_model.predict_proba(input_data_scaled_log)  

    # Affichage du résultat  
    if prediction_log[0] == 1:  
        st.success("🏆 L'équipe A est prédite pour gagner !")  
    else:  
        st.success("🥈 L'équipe B est prédite pour gagner !")  

    # Affichage des probabilités  
    st.write(f"Probabilité que l'équipe A gagne : **{prediction_proba[0][1] * 100:.2f}%**")  
    st.write(f"Probabilité que l'équipe B gagne : **{prediction_proba[0][0] * 100:.2f}%**")
# Prédiction des buts avec la méthode de Poisson  
def prediction_buts_poisson(xG_A, xG_B):  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  
    buts_attendus_A = sum(i * prob for i, prob in enumerate(buts_A))  
    buts_attendus_B = sum(i * prob for i, prob in enumerate(buts_B))  
    return buts_attendus_A, buts_attendus_B  

# Affichage des résultats de la prédiction des buts  
if st.button("⚽ Prédire les buts avec la méthode de Poisson"):  
    buts_moyens_A
