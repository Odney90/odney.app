import streamlit as st  
import numpy as np  
import pandas as pd  
import traceback  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
from scipy.stats import poisson  

# Fonctions utilitaires  
def safe_float(value, default=0.0):  
    try:  
        return float(value) if value is not None else default  
    except (ValueError, TypeError):  
        return default  

def safe_int(value, default=0):  
    try:  
        return int(value) if value is not None else default  
    except (ValueError, TypeError):  
        return default  

# Fonctions de prédiction et de calcul (gardez les fonctions précédentes)  
# ...  

# Modification des fonctions de préparation de données pour gérer les ensembles vides  
def preparer_donnees_regression_logistique(data):  
    # Vérification et valeurs par défaut  
    if not data:  
        # Données par défaut si l'ensemble est vide  
        data = {  
            'score_rating_A': 1500, 'classement_championnat_A': 10,  
            'score_rating_B': 1500, 'classement_championnat_B': 10,  
            'buts_marques_A': 1.5, 'buts_marques_B': 1.5,  
            'performance_domicile_A': 50, 'performance_domicile_B': 50,  
            'tirs_cadres_par_match_A': 4, 'tirs_cadres_par_match_B': 4,  
            'joueurs_blesses_A': 2, 'joueurs_blesses_B': 2  
        }  
    
    X_lr = np.array([  
        [  
            safe_float(data.get("score_rating_A", 1500)),  
            safe_float(data.get("classement_championnat_A", 10)),  
            safe_float(data.get("score_rating_B", 1500)),  
            safe_float(data.get("classement_championnat_B", 10)),  
            
            safe_float(data.get("buts_marques_A", 1.5)),  
            safe_float(data.get("tirs_cadres_par_match_A", 4)),  
            safe_float(data.get("buts_marques_B", 1.5)),  
            safe_float(data.get("tirs_cadres_par_match_B", 4)),  
            
            safe_float(data.get("performance_domicile_A", 50)),  
            safe_float(data.get("performance_domicile_B", 50)),  
            
            safe_float(data.get("joueurs_blesses_A", 2)),  
            safe_float(data.get("joueurs_blesses_B", 2))  
        ]  
    ])  
    return X_lr  

def preparer_donnees_random_forest(data):  
    # Vérification et valeurs par défaut  
    if not data:  
        # Données par défaut si l'ensemble est vide  
        data = {  
            'classement_championnat_A': 10, 'classement_championnat_B': 10,  
            'interceptions_A': 5, 'interceptions_B': 5,  
            'passes_reussies_par_match_A': 70, 'passes_reussies_par_match_B': 70,  
            'joueurs_blesses_A': 2, 'joueurs_blesses_B': 2  
        }  
    
    X_rf = np.array([  
        [  
            # Données avec valeurs par défaut sécurisées  
            safe_float(data.get("classement_championnat_A", 10)),  
            safe_float(data.get("classement_championnat_B", 10)),  
            
            safe_float(data.get("interceptions_A", 5)),  
            safe_float(data.get("interceptions_B", 5)),  
            
            safe_float(data.get("passes_reussies_par_match_A", 70)),  
            safe_float(data.get("passes_reussies_par_match_B", 70)),  
            
            safe_float(data.get("joueurs_blesses_A", 2)),  
            safe_float(data.get("joueurs_blesses_B", 2))  
        ]  
    ])  
    return X_rf  

# Données historiques par défaut  
def generer_donnees_historiques_defaut(nombre_matchs=50):  
    donnees = []  
    for _ in range(nombre_matchs):  
        match = {  
            'score_rating_A': np.random.normal(1500, 200),  
            'classement_championnat_A': np.random.randint(1, 20),  
            'buts_marques_A': np.random.normal(1.5, 0.5),  
            'tirs_cadres_par_match_A': np.random.normal(5, 1),  
            'performance_domicile_A': np.random.normal(70, 10),  
            'forme_recente_A_victoires': np.random.randint(0, 5),  
            'forme_recente_A_nuls': np.random.randint(0, 3),  
            'forme_recente_A_defaites': np.random.randint(0, 3),  
            
            'score_rating_B': np.random.normal(1500, 200),  
            'classement_championnat_B': np.random.randint(1, 20),  
            'buts_marques_B': np.random.normal(1.5, 0.5),  
            'tirs_cadres_par_match_B': np.random.normal(5, 1),  
            'performance_domicile_B': np.random.normal(70, 10),  
            'forme_recente_B_victoires': np.random.randint(0, 5),  
            'forme_recente_B_nuls': np.random.randint(0, 3),  
            'forme_recente_B_defaites': np.random.randint(0, 3),  
            
            'resultat': np.random.choice([0, 0.5, 1])  # 0: défaite, 0.5: nul, 1: victoire  
        }  
        donnees.append(match)  
    return donnees  

# Configuration Streamlit  
st.set_page_config(page_title="Prédictions de Matchs", page_icon="⚽")  
st.title("🏆 Système de Prédiction de Matchs de Football")  

# Initialisation du state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Formulaire de saisie  
with st.form("Données du Match"):  
    st.subheader("📊 Saisie des Données du Match")  
    
    col1, col2 = st.columns(2)  
    
    with col1:  
        st.markdown("**Équipe A**")  
        st.session_state.data.update({  
            "score_rating_A": st.number_input("⭐ Score Rating", value=1500, key="rating_A"),  
            "classement_championnat_A": st.number_input("🏆 Classement", value=10, key="classement_A"),  
            "buts_marques_A": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_A"),  
            "tirs_cadres_par_match_A": st.number_input("🎯 Tirs Cadrés", value=5, key="tirs_A"),  
            "joueurs_blesses_A": st.number_input("🩺 Joueurs Blessés", value=2, key="blesses_A")  
        })  

    with col2:  
        st.markdown("**Équipe B**")  
        st.session_state.data.update({  
            "score_rating_B": st.number_input("⭐ Score Rating", value=1500, key="rating_B"),  
            "classement_championnat_B": st.number_input("🏆 Classement", value=10, key="classement_B"),  
            "buts_marques_B": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_B"),  
            "tirs_cadres_par_match_B": st.number_input("🎯 Tirs Cadrés", value=5, key="tirs_B"),  
            "joueurs_blesses_B": st.number_input("🩺 Joueurs Blessés", value=2, key="blesses_B")  
        })  

    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Bouton de prédiction  
if st.button("🏆 Prédire le Résultat"):  
    try:  
        # Génération de données historiques si nécessaire  
        donnees_historiques = generer_donnees_historiques_defaut()  
        
        # Préparation des données  
        X_train, y_train = preparer_donnees_entrainement(donnees_historiques)  
        
        # Préparation des données du match actuel  
        X_lr = preparer_donnees_regression_logistique(st.session_state.data)  
        X_rf = preparer_donnees_random_forest(st.session_state.data)  
        
        # Modèles  
        modeles = {  
            "Régression Logistique": LogisticRegression(max_iter=1000),  
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  
        }  
        
        # Résultats  
        st.subheader("🔮 Résultats de Prédiction")  
        
        for nom, modele in modeles.items():  
            # Entraînement et prédiction  
            modele.fit(X_train, y_train)  
            
            # Probabilités de prédiction  
            proba = modele.predict_proba(X_lr if nom == "Régression Logistique" else X_rf)[0]  
            
            # Affichage des probabilités  
            st.markdown(f"**{nom}**")  
            st.write(f"Probabilité Victoire A: {proba[1]:.2%}")  
            st.write(f"Probabilité Victoire B: {proba[0]:.2%}")  
        
    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(traceback.format_exc())
