import streamlit as st  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
from imblearn.over_sampling import SMOTE  
import pickle  
import os  

# Configuration de base  
RANDOM_STATE = 42  
MODEL_PATH = "trained_models"  
DATA_PATH = "data"  

# Création des répertoires si ils n'existent pas  
os.makedirs(MODEL_PATH, exist_ok=True)  
os.makedirs(DATA_PATH, exist_ok=True)  

# Fonction pour charger les données  
def load_data(file_name="data.csv"):  
    """Charge les données depuis un fichier CSV"""  
    file_path = os.path.join(DATA_PATH, file_name)  
    if os.path.exists(file_path):  
        return pd.read_csv(file_path)  
    return None  

# Fonction pour préparer les données  
def prepare_data(df):  
    """Prépare les données pour l'entraînement"""  
    try:  
        # Séparation des variables indépendantes et dépendantes  
        X = df.drop(["target"], axis=1)  
        y = df["target"]  
        
        # Équilibrement des données avec SMOTE  
        smote = SMOTE(random_state=RANDOM_STATE)  
        X_res, y_res = smote.fit_resample(X, y)  
        
        # Scalage des données  
        scaler = StandardScaler()  
        X_scaled = scaler.fit_transform(X_res)  
        
        return X_scaled, y_res, scaler  
    except Exception as e:  
        print(f"Error preparing data: {str(e)}")  
        return None, None, None  

# Fonction pour entraîner un modèle  
def train_model(model_type="random_forest"):  
    """Entraîne un modèle selon le type spécifié"""  
    try:  
        # Chargement des données  
        df = load_data()  
        if df is None:  
            raise ValueError("No data found")  
            
        # Préparation des données  
        X, y, scaler = prepare_data(df)  
        if X is None or y is None:  
            raise ValueError("Data preparation failed")  
            
        # Séparation en entraînement et test  
        X_train, X_test, y_train, y_test = train_test_split(  
            X, y,   
            test_size=0.2,   
            random_state=RANDOM_STATE  
        )  
        
        # Initialisation du modèle  
        if model_type == "random_forest":  
            model = RandomForestClassifier(  
                n_estimators=100,  
                random_state=RANDOM_STATE,  
                class_weight="balanced"  
            )  
        elif model_type == "logistic_regression":  
            model = LogisticRegression(  
                max_iter=1000,  
                random_state=RANDOM_STATE,  
                class_weight="balanced"  
            )  
        else:  
            raise ValueError("Invalid model type")  
            
        # Entraînement du modèle  
        model.fit(X_train, y_train)  
        
        # Sauvegarde du modèle et du scaler  
        model_path = os.path.join(MODEL_PATH, f"{model_type}_model.pkl")  
        scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")  
        with open(model_path, "wb") as f:  
            pickle.dump(model, f)  
        with open(scaler_path, "wb") as f:  
            pickle.dump(scaler, f)  
            
        return True  
    except Exception as e:  
        print(f"Error training model: {str(e)}")  
        return False  

# Fonction pour faire une prédiction  
def predict(model_type, data):  
    """Fait une prédiction avec le modèle spécifié"""  
    try:  
        # Chargement du modèle et du scaler  
        model_path = os.path.join(MODEL_PATH, f"{model_type}_model.pkl")  
        scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")  
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):  
            raise ValueError("Model or scaler not found")  
            
        with open(model_path, "rb") as f:  
            model = pickle.load(f)  
        with open(scaler_path, "rb") as f:  
            scaler = pickle.load(f)  
            
        # Préparation des données  
        scaled_data = scaler.transform(data)  
        
        # Prédiction  
        prediction = model.predict(scaled_data)  
        probability = model.predict_proba(scaled_data)  
        
        return prediction, probability  
    except Exception as e:  
        print(f"Error making prediction: {str(e)}")  
        return None, None  

# Interface utilisateur Streamlit  
st.title("⚽ Prédiction de Matchs de Football")  
st.write("Entrez les statistiques des deux équipes pour prédire le résultat du match.")  

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
score_forme_A = np.mean(derniers_matchs_A)  
score_forme_B = np.mean(derniers_matchs_B)  

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

# Section pour l'entraînement des modèles  
with st.expander("Entraînement des modèles"):  
    st.write("Entraînez les modèles avec vos données historiques.")  
    if st.button("Entraîner les modèles"):  
        if train_model("random_forest"):  
            st.success("Modèle Random Forest entraîné avec succès !")  
        else:  
            st.error("Erreur lors de l'entraînement du modèle Random Forest")  
        
        if train_model("logistic_regression"):  
            st.success("Modèle Logistic Regression entraîné avec succès !")  
        else:  
            st.error("Erreur lors de l'entraînement du modèle Logistic Regression")  

# Section pour la prédiction  

with st.expander("Prédiction"):  
    col1, col2 = st.columns(2)  
    
    with col1:  
        if st.button("Prédire avec Random Forest", key="predict_rf"):  
            try:  
                # Préparation des données  
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
                
                # Prédiction  
                prediction, probability = predict("random_forest", data)  
                
                if prediction is not None:  
                    st.write("Prédiction du Random Forest :")  
                    st.write(f"Résultat prédit : {prediction[0]}")  
                    st.write(f"Probabilité : {np.max(probability[0]) * 100:.2f}%")  
                else:  
                    st.error("Erreur lors de la prédiction")  
                
            except Exception as e:  
                st.error(f"Erreur lors de la prédiction avec Random Forest : {str(e)}")  
    
    with col2:  
        if st.button("Prédire avec Régression Logistique", key="predict_lr"):  
            try:  
                # Préparation des données  
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
                
                # Prédiction  
                prediction, probability = predict("logistic_regression", data)  
                
                if prediction is not None:  
                    st.write("Prédiction de la Régression Logistique :")  
                    st.write(f"Résultat prédit : {prediction[0]}")  
                    st.write(f"Probabilité : {np.max(probability[0]) * 100:.2f}%")  
                else:  
                    st.error("Erreur lors de la prédiction")  
                
            except Exception as e:  
                st.error(f"Erreur lors de la prédiction avec Régression Logistique : {str(e)}")
