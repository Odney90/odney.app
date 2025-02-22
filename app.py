import streamlit as st  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
import pickle  
import os  
from scipy.stats import poisson  
import json  
from datetime import datetime  

# Fonction pour sauvegarder les données  
def save_data(data, filename='data.pkl'):  
    with open(filename, 'wb') as f:  
        pickle.dump(data, f)  
    return True  # Return success status  

# Fonction pour charger les données sauvegardées  
def load_saved_data(filename='data.pkl'):  
    if os.path.exists(filename):  
        with open(filename, 'rb') as f:  
            return pickle.load(f)  
    return None  

# Fonction pour charger les données historiques  
def load_historical_data():  
    return {  
        'Équipe A': {  
            'derniers_matchs': [1, 0, 1, 1, 0],  
            'face_a_face': [1, 1, 0, 1, 0]  
        },  
        'Équipe B': {  
            'derniers_matchs': [0, 1, 1, 0, 1],  
            'face_a_face': [0, 0, 1, 0, 1]  
        }  
    }  

# Fonction pour calculer le score de forme  
def calculate_form_score(last_matches):  
    return np.mean(last_matches)  

# Fonction pour prédire les buts avec Poisson  
def predict_goals_poisson(xG_A, xG_B):  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  
    return buts_A, buts_B  

# Fonction pour la prédiction avec Random Forest  
def predict_random_forest(data):  
    try:  
        scaler = StandardScaler()  
        scaled_data = scaler.fit_transform(data)  
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
        rf_model.fit(scaled_data, [1])  
        prediction = rf_model.predict(scaled_data)  
        
        if prediction[0] == 1:  
            return "Équipe A"  
        else:  
            return "Équipe B"  
    except Exception as e:  
        return f"Error: {str(e)}"  

# Fonction pour la prédiction avec Régression Logistique  
def predict_logistic_regression(data):  
    try:  
        scaler = StandardScaler()  
        scaled_data = scaler.fit_transform(data)  
        
        log_model = LogisticRegression(max_iter=1000)  
        log_model.fit(scaled_data, [1])  
        prediction = log_model.predict(scaled_data)  
        
        if prediction[0] == 1:  
            return "Équipe A"  
        else:  
            return "Équipe B"  
    except Exception as e:  
        return f"Error: {str(e)}"  

# Interface utilisateur Streamlit  
st.title("⚽ Prédiction de Matchs de Football")  
st.write("Entrez les statistiques des deux équipes pour prédire le résultat du match.")  

# Chargement des données historiques  
historical_data = load_historical_data()  

# Section pour les 5 derniers matchs  
with st.expander("Forme récente (5 derniers matchs)"):  
    st.subheader("Équipe A")  
    derniers_matchs_A = []  
    for i in range(5):  
        match_result = st.select_slider(f"Match {i+1} (Équipe A)",   
                                      options=['Victoire', 'Défaite', 'Match nul'],   
                                      key=f"match_A_{i}")  
        if match_result == 'Victoire':  
            derniers_matchs_A.append(1)  
        elif match_result == 'Défaite':  
            derniers_matchs_A.append(0)  
        else:  
            derniers_matchs_A.append(0.5)  
    
    st.subheader("Équipe B")  
    derniers_matchs_B = []  
    for i in range(5):  
        match_result = st.select_slider(f"Match {i+1} (Équipe B)",   
                                      options=['Victoire', 'Défaite', 'Match nul'],   
                                      key=f"match_B_{i}")  
        if match_result == 'Victoire':  
            derniers_matchs_B.append(1)  
        elif match_result == 'Défaite':  
            derniers_matchs_B.append(0)  
        else:  
            derniers_matchs_B.append(0.5)  

# Calcul du score de forme  
score_forme_A = calculate_form_score(derniers_matchs_A)  
score_forme_B = calculate_form_score(derniers_matchs_B)  

# Section pour les statistiques générales  
with st.expander("Statistiques générales"):  
    st.subheader("Équipe A")  
    buts_totaux_A = st.number_input("Buts totaux", min_value=0.0, value=0.0, key="buts_totaux_A")  
    possession_moyenne_A = st.number_input("Possession moyenne (%)", min_value=0.0, max_value=100.0, value=50.0, key="possession_moyenne_A")  
    expected_buts_A = st.number_input("Expected Goals (xG)", min_value=0.0, value=0.0, key="expected_buts_A")  
    tirs_cadres_A = st.number_input("Tirs cadrés par match", min_value=0.0, value=0.0, key="tirs_cadres_A")  
    
    st.subheader("Équipe B")  
    buts_totaux_B = st.number_input("Buts totaux", min_value=0.0, value=0.0, key="buts_totaux_B")  
    possession_moyenne_B = st.number_input("Possession moyenne (%)", min_value=0.0, max_value=100.0, value=50.0, key="possession_moyenne_B")  
    expected_buts_B = st.number_input("Expected Goals (xG)", min_value=0.0, value=0.0, key="expected_buts_B")  
    tirs_cadres_B = st.number_input("Tirs cadrés par match", min_value=0.0, value=0.0, key="tirs_cadres_B")  

# Section pour les critères avancés  
with st.expander("Critères avancés"):  
    st.subheader("Équipe A")  
    passes_reussies_A = st.number_input("Passes réussies par match", min_value=0.0, value=0.0, key="passes_reussies_A")  
    tacles_reussis_A = st.number_input("Tacles réussis par match", min_value=0.0, value=0.0, key="tacles_reussis_A")  
    fautes_A = st.number_input("Fautes par match", min_value=0.0, value=0.0, key="fautes_A")  
    
    st.subheader("Équipe B")  
    passes_reussies_B = st.number_input("Passes réussies par match", min_value=0.0, value=0.0, key="passes_reussies_B")  
    tacles_reussis_B = st.number_input("Tacles réussis par match", min_value=0.0, value=0.0, key="tacles_reussis_B")  
    fautes_B = st.number_input("Fautes par match", min_value=0.0, value=0.0, key="fautes_B")  

# Section pour les critères de discipline  
with st.expander("Critères de discipline"):  
    st.subheader("Équipe A")  
    cartons_jaunes_A = st.number_input("Cartons jaunes", min_value=0.0, value=0.0, key="cartons_jaunes_A")  
    cartons_rouges_A = st.number_input("Cartons rouges", min_value=0.0, value=0.0, key="cartons_rouges_A")  
    
    st.subheader("Équipe B")  
    cartons_jaunes_B = st.number_input("Cartons jaunes", min_value=0.0, value=0.0, key="cartons_jaunes_B")  
    cartons_rouges_B = st.number_input("Cartons rouges", min_value=0.0, value=0.0, key="cartons_rouges_B")  

# Section pour les critères de défense  
with st.expander("Critères de défense"):  
    st.subheader("Équipe A")  
    expected_concedes_A = st.number_input("Expected Goals concédés (xG)", min_value=0.0, value=0.0, key="expected_concedes_A")  
    interceptions_A = st.number_input("Interceptions par match", min_value=0.0, value=0.0, key="interceptions_A")  
    degagements_A = st.number_input("Dégagements par match", min_value=0.0, value=0.0, key="degagements_A")  
    arrets_A = st.number_input("Arrêts par match", min_value=0.0, value=0.0, key="arrets_A")  
    
    st.subheader("Équipe B")  
    expected_concedes_B = st.number_input("Expected Goals concédés (xG)", min_value=0.0, value=0.0, key="expected_concedes_B")  
    interceptions_B = st.number_input("Interceptions par match", min_value=0.0, value=0.0, key="interceptions_B")  
    degagements_B = st.number_input("Dégagements par match", min_value=0.0, value=0.0, key="degagements_B")  
    arrets_B = st.number_input("Arrêts par match", min_value=0.0, value=0.0, key="arrets_B")  

# Section pour les critères d'attaque  
with st.expander("Critères d'attaque"):  
    st.subheader("Équipe A")  
    corners_A = st.number_input("Corners par match", min_value=0.0, value=0.0, key="corners_A")  
    touches_surface_adverse_A = st.number_input("Touches dans la surface adverse", min_value=0.0, value=0.0, key="touches_surface_adverse_A")  
    penalites_obtenues_A = st.number_input("Pénalités obtenues", min_value=0.0, value=0.0, key="penalites_obtenues_A")  
    
    st.subheader("Équipe B")  
    corners_B = st.number_input("Corners par match", min_value=0.0, value=0.0, key="corners_B")  
    touches_surface_adverse_B = st.number_input("Touches dans la surface adverse", min_value=0.0, value=0.0, key="touches_surface_adverse_B")  
    penalites_obtenues_B = st.number_input("Pénalités obtenues", min_value=0.0, value=0.0, key="penalites_obtenues_B")  

# Section pour les statistiques complètes  
with st.expander("Statistiques complètes"):  
    st.subheader("Équipe A")  
    buts_par_match_A = st.number_input("Buts par match", min_value=0.0, value=0.0, key="buts_par_match_A")  
    buts_concedes_par_match_A = st.number_input("Buts concédés par match", min_value=0.0, value=0.0, key="buts_concedes_par_match_A")  
    buts_concedes_totaux_A = st.number_input("Buts concédés au total", min_value=0.0, value=0.0, key="buts_concedes_totaux_A")  
    aucun_but_encaisse_A = st.number_input("Aucun but encaissé (oui=1, non=0)", min_value=0, max_value=1, value=0, key="aucun_but_encaisse_A")  
    
    st.subheader("Équipe B")  
    buts_par_match_B = st.number_input("Buts par match", min_value=0.0, value=0.0, key="buts_par_match_B")  
    buts_concedes_par_match_B = st.number_input("Buts concédés par match", min_value=0.0, value=0.0, key="buts_concedes_par_match_B")  
    buts_concedes_totaux_B = st.number_input("Buts concédés au total", min_value=0.0, value=0.0, key="buts_concedes_totaux_B")  
    aucun_but_encaisse_B = st.number_input("Aucun but encaissé (oui=1, non=0)", min_value=0, max_value=1, value=0, key="aucun_but_encaisse_B")  

# Section pour la visualisation des données  
with st.expander("Visualisation des données"):  
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  
    ax[0].bar(["Équipe A", "Équipe B"], [score_forme_A, score_forme_B])  
    ax[0].set_title("Score de forme (5 derniers matchs)")  
    ax[1].plot([score_forme_A, score_forme_B], marker='o')  
    ax[1].set_title("Évolution de la forme")  
    st.pyplot(fig)  

# Section pour la prédiction  
with st.expander("Prédiction"):  
    col1, col2 = st.columns(2)  
    with col1:  
        if st.button("Prédire avec Random Forest", key="predict_rf"):  
            try:  
                data = np.array([  
                    score_forme_A, score_forme_B, buts_totaux_A, buts_totaux_B,  
                    possession_moyenne_A, possession_moyenne_B, expected_buts_A,  
                    expected_buts_B, tirs_cadres_A, tirs_cadres_B, passes_reussies_A,  
                    passes_reussies_B, tacles_reussis_A, tacles_reussis_B, fautes_A,  
                    fautes_B, cartons_jaunes_A, cartons_jaunes_B, cartons_rouges_A,  
                    cartons_rouges_B, expected_concedes_A, expected_concedes_B,  
                    interceptions_A, interceptions_B, degagements_A, degagements_B,  
                    arrets_A, arrets_B, corners_A, corners_B,  
                    touches_surface_adverse_A, touches_surface_adverse_B,  
                    penalites_obtenues_A, penalites_obtenues_B, buts_par_match_A,  
                    buts_concedes_par_match_A, buts_concedes_totaux_A,  
                    aucun_but_encaisse_A, buts_par_match_B, buts_concedes_par_match_B,  
                    buts_concedes_totaux_B, aucun_but_encaisse_B  
                ]).reshape(1, -1)  
                
                prediction = predict_random_forest(data)  
                if isinstance(prediction, str):  
                    st.success(f"Prédiction : {prediction}", key="rf_prediction")  
                else:  
                    st.error("Erreur lors de la prédiction", key="rf_error")  
            except Exception as e:  
                st.error(f"Erreur lors de la prédiction : {str(e)}", key="rf_error")  
    
    with col2:  
        if st.button("Prédire avec Régression Logistique", key="predict_lr"):  
            try:  
                data = np.array([  
                    score_forme_A, score_forme_B, buts_totaux_A, buts_totaux_B,  
                    possession_moyenne_A, possession_moyenne_B, expected_buts_A,  
                    expected_buts_B, tirs_cadres_A, tirs_cadres_B, passes_reussies_A,  
                    passes_reussies_B, tacles_reussis_A, tacles_reussis_B, fautes_A,  
                    fautes_B, cartons_jaunes_A, cartons_jaunes_B, cartons_rouges_A,  
                    cartons_rouges_B, expected_concedes_A, expected_concedes_B,  
                    interceptions_A, interceptions_B, degagements_A, degagements_B,  
                    arrets_A, arrets_B, corners_A, corners_B,  
                    touches_surface_adverse_A, touches_surface_adverse_B,  
                    penalites_obtenues_A, penalites_obtenues_B, buts_par_match_A,  
                    buts_concedes_par_match_A, buts_concedes_totaux_A,  
                    aucun_but_encaisse_A, buts_par_match_B, buts_concedes_par_match_B,  
                    buts_concedes_totaux_B, aucun_but_encaisse_B  
                ]).reshape(1, -1)  
                
                prediction = predict_logistic_regression(data)  
                if isinstance(prediction, str):  
                    st.success(f"Prédiction : {prediction}", key="lr_prediction")  
                else:  
                    st.error("Erreur lors de la prédiction", key="lr_error")  
            except Exception as e:  
                st.error(f"Erreur lors de la prédiction : {str(e)}", key="lr_error")  

# Section pour la prédiction des buts  
with st.expander("Prédiction des buts"):  
    if st.button("Prédire les buts", key="predict_goals"):  
        try:  
            buts_A, buts_B = predict_goals_poisson(expected_buts_A, expected_buts_B)  
            st.write(f"Buts attendus pour l'équipe A : {buts_A}", key="buts_A")  
            st.write(f"Buts attendus pour l'équipe B : {buts_B}", key="buts_B")  
        except Exception as e:  
            st.error(f"Erreur lors de la prédiction des buts : {str(e)}", key="goals_error")  

# Sauvegarde automatique des données  
def save_all_data():  
    data = {  
        'derniers_matchs_A': derniers_matchs_A,  
        'derniers_matchs_B': derniers_matchs_B,  
        'buts_totaux_A': buts_totaux_A,  
        'buts_totaux_B': buts_totaux_B,  
        'possession_moyenne_A': possession_moyenne_A,  
        'possession_moyenne_B': possession_moyenne_B,  
        'expected_buts_A': expected_buts_A,  
        'expected_buts_B': expected_buts_B,  
        'tirs_cadres_A': tirs_cadres_A,  
        'tirs_cadres_B': tirs_cadres_B,  
        'passes_reussies_A': passes_reussies_A,  
        'passes_reussies_B': passes_reussies_B,  
        'tacles_reussis_A': tacles_reussis_A,  
        'tacles_reussis_B': tacles_reussis_B,  
        'fautes_A': fautes_A,  
        'fautes_B': fautes_B,  
        'cartons_jaunes_A': cartons_jaunes_A,  
        'cartons_jaunes_B': cartons_jaunes_B,  
        'cartons_rouges_A': cartons_rouges_A,  
        'cartons_rouges_B': cartons_rouges_B,  
        'expected_concedes_A': expected_concedes_A,  
        'expected_concedes_B': expected_concedes_B,  
        'interceptions_A': interceptions_A,  
        'interceptions_B': interceptions_B,  
        'degagements_A': degagements_A,  
        'degagements_B': degagements_B,  
        'arrets_A': arrets_A,  
        'arrets_B': arrets_B,  
        'corners_A': corners_A,  
        'corners_B': corners_B,  
        'touches_surface_adverse_A': touches_surface_adverse_A,  
        'touches_surface_adverse_B': touches_surface_adverse_B,  
        'penalites_obtenues_A': penalites_obtenues_A,  
        'penalites_obtenues_B': penalites_obtenues_B,  
        'buts_par_match_A': buts_par_match_A,  
        'buts_concedes_par_match_A': buts_concedes_par_match_A,  
        'buts_concedes_totaux_A': buts_concedes_totaux_A,  
        'aucun_but_encaisse_A': aucun_but_encaisse_A,  
        'buts_par_match_B': buts_par_match_B,  
        'buts_concedes_par_match_B': buts_concedes_par_match_B,  
        'buts_concedes_totaux_B': buts_concedes_totaux_B,  
        'aucun_but_encaisse_B': aucun_but_encaisse_B,  
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
    }  
    
    success = save_data(data)  
    if success:  
        st.success("Données sauvegardées avec succès !")  
    return success  

# Appel à la fonction de sauvegarde  
if __name__ == "__main__":  
    try:  
        save_all_data()  
    except Exception as e:  
        st.error(f"Erreur lors de la sauvegarde : {e}")
