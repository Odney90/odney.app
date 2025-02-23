import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
from scipy.stats import poisson  

# Fonctions utilitaires  
def safe_int(value):  
    try:  
        return int(value)  
    except (ValueError, TypeError):  
        return 0  

def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

def cote_to_probabilite_implicite(cote):  
    try:  
        return (1 / cote) * 100  
    except (ValueError, ZeroDivisionError):  
        return 0.0  

def calculer_score_forme_avance(victoires, nuls, defaites,   
                                 poids_victoire=3,   
                                 poids_nul=1,   
                                 poids_defaite=-2):  
    return (victoires * poids_victoire +   
            nuls * poids_nul +   
            defaites * poids_defaite)
    # Configuration de la page Streamlit  
st.set_page_config(page_title="Prédictions de Matchs", page_icon="⚽")  
st.title("📊 Prédictions de Matchs de Football")  

# Formulaire pour les données des équipes  
with st.form("Données des Équipes"):  
    # Initialisation du dictionnaire de données  
    if 'data' not in st.session_state:  
        st.session_state.data = {}  

    # Données existantes de l'Équipe A  
    st.subheader("⚽ Données de l'Équipe A")  
    st.session_state.data.update({  
        # Données existantes...  
        "buts_marques_A": st.number_input("⚽ Buts Marqués par Match (A)", value=1.5, key="buts_marques_A_input"),  
        # ... autres champs existants  
    })  

    # Nouvelles colonnes à ajouter  
    st.session_state.data.update({  
        # Données de Classement et Performance Globale  
        "classement_championnat_A": st.number_input("🏆 Classement Championnat (A)", value=5, key="classement_championnat_A_input"),  
        "classement_championnat_B": st.number_input("🏆 Classement Championnat (B)", value=10, key="classement_championnat_B_input"),  
        
        # Performance en Déplacement/Domicile  
        "performance_domicile_A": st.number_input("🏠 Performance Domicile (A)", value=75.0, key="performance_domicile_A_input"),  
        "performance_exterieur_A": st.number_input("✈️ Performance Extérieur (A)", value=60.0, key="performance_exterieur_A_input"),  
        "performance_domicile_B": st.number_input("🏠 Performance Domicile (B)", value=65.0, key="performance_domicile_B_input"),  
        "performance_exterieur_B": st.number_input("✈️ Performance Extérieur (B)", value=55.0, key="performance_exterieur_B_input"),  
        
        # Données de Blessures et Disponibilité  
        "joueurs_blesses_A": st.number_input("🩺 Nombre de Joueurs Blessés (A)", value=2, key="joueurs_blesses_A_input"),  
        "joueurs_blesses_B": st.number_input("🩺 Nombre de Joueurs Blessés (B)", value=3, key="joueurs_blesses_B_input"),  
        
        # Historique des Confrontations Directes  
        "victoires_confrontations_directes_A": st.number_input("🤝 Victoires Confrontations Directes (A)", value=3, key="victoires_confrontations_directes_A_input"),  
        "victoires_confrontations_directes_B": st.number_input("🤝 Victoires Confrontations Directes (B)", value=2, key="victoires_confrontations_directes_B_input"),  
    })  

    # Bouton de soumission du formulaire  
    submitted = st.form_submit_button("💾 Enregistrer les Données")
    
   def validation_croisee_personnalisee(X, y, modele, n_splits=5):  
    from sklearn.model_selection import StratifiedKFold  
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  
    
    resultats = {  
        'accuracy': [],  
        'precision': [],  
        'recall': [],  
        'f1_score': []  
    }  
    
    for train_index, test_index in kf.split(X, y):  
        X_train, X_test = X[train_index], X[test_index]  
        y_train, y_test = y[train_index], y[test_index]  
        
        modele.fit(X_train, y_train)  
        y_pred = modele.predict(X_test)  
        
        resultats['accuracy'].append(accuracy_score(y_test, y_pred))  
        resultats['precision'].append(precision_score(y_test, y_pred, average='weighted'))  
        resultats['recall'].append(recall_score(y_test, y_pred, average='weighted'))  
        resultats['f1_score'].append(f1_score(y_test, y_pred, average='weighted'))  
    
    return {k: np.mean(v) for k, v in resultats.items()}
def preparer_donnees_regression_logistique(data):  
    score_forme_A = calculer_score_forme_avance(  
        data["forme_recente_A_victoires"],  
        data["forme_recente_A_nuls"],  
        data["forme_recente_A_defaites"]  
    )  
    score_forme_B = calculer_score_forme_avance(  
        data["forme_recente_B_victoires"],  
        data["forme_recente_B_nuls"],  
        data["forme_recente_B_defaites"]  
    )  

    X_lr = np.array([  
        [  
            # Performance Globale et Classement  
            safe_float(data["score_rating_A"]),  
            safe_float(data["classement_championnat_A"]),  
            safe_float(data["score_rating_B"]),  
            safe_float(data["classement_championnat_B"]),  
            
            # Performance Offensive  
            safe_float(data["buts_marques_A"]),  
            safe_float(data["tirs_cadres_par_match_A"]),  
            safe_float(data["buts_marques_B"]),  
            safe_float(data["tirs_cadres_par_match_B"]),  
            
            # Performance Domicile/Extérieur  
            safe_float(data["performance_domicile_A"]),  
            safe_float(data["performance_exterieur_A"]),  
            safe_float(data["performance_domicile_B"]),  
            safe_float(data["performance_exterieur_B"]),  
            
            # Forme et Motivation  
            score_forme_A,  
            safe_float(data["motivation_A"]),  
            score_forme_B,  
            safe_float(data["motivation_B"]),  
            
            # Confrontations Directes  
            safe_float(data["victoires_confrontations_directes_A"]),  
            safe_float(data["victoires_confrontations_directes_B"]),  
            
            # Impact des Blessures  
            safe_float(data["joueurs_blesses_A"]),  
            safe_float(data["joueurs_blesses_B"])  
        ]  
    ])  
    return X_lr  

def preparer_donnees_random_forest(data):  
    X_rf = np.array([  
        [  
            # Performance Défensive  
            safe_float(data["buts_attendus_concedes_A"]),  
            safe_float(data["interceptions_A"]),  
            safe_float(data["tacles_reussis_par_match_A"]),  
            safe_float(data["buts_attendus_concedes_B"]),  
            safe_float(data["interceptions_B"]),  
            safe_float(data["tacles_reussis_par_match_B"]),  
            
            # Statistiques de Jeu Avancées  
            safe_float(data["centres_reussies_par_match_A"]),  
            safe_float(data["corners_par_match_A"]),  
            safe_float(data["centres_reussies_par_match_B"]),  
            safe_float(data["corners_par_match_B"]),  
            
            # Discipline  
            safe_float(data["penalties_concedees_A"]),  
            safe_float(data["fautes_par_match_A"]),  
            safe_float(data["penalties_concedees_B"]),  
            safe_float(data["fautes_par_match_B"]),  
            
            # Performance Globale  
            safe_float(data["classement_championnat_A"]),  
            safe_float(data["classement_championnat_B"]),  
            
            # Créativité et Dynamisme  
            safe_float(data["passes_reussies_par_match_A"]),  
            safe_float(data["dribbles_reussis_par_match_A"]),  
            safe_float(data["passes_reussies_par_match_B"]),  
            safe_float(data["dribbles_reussis_par_match_B"]),  
            
            # Données de Blessures et Confrontations  
            safe_float(data["joueurs_blesses_A"]),  
            safe_float(data["joueurs_blesses_B"]),  
            safe_float(data["victoires_confrontations_directes_A"]),  
            safe_float(data["victoires_confrontations_directes_B"])  
        ]  
    ])  
    return X_rf
    # Bouton pour lancer les prédictions  
if st.button("🔮 Lancer les Prédictions"):  
    try:  
        # Préparation des données  
        X_lr = preparer_donnees_regression_logistique(st.session_state.data)  
        X_rf = preparer_donnees_random_forest(st.session_state.data)  
        
        # Données d'entraînement fictives (à remplacer par vos données réelles)  
        X_train_lr = np.random.rand(100, X_lr.shape[1])  
        y_train_lr = np.random.randint(0, 2, 100)  
        
        X_train_rf = np.random.rand(100, X_rf.shape[1])  
        y_train_rf = np.random.randint(0, 2, 100)  
        
        # Modèle de Régression Logistique avec Validation Croisée  
        model_lr = LogisticRegression(max_iter=1000)  
        resultats_lr = validation_croisee_personnalisee(X_train_lr, y_train_lr, model_lr)  
        
        # Modèle Random Forest avec Validation Croisée  
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)  
        resultats_rf = validation_croisee_personnalisee(X_train_rf, y_train_rf, model_rf)  
        
        # Affichage des résultats de validation  
        st.subheader("📊 Résultats de Validation des Modèles")  
        
        col1, col2 = st.columns(2)  
        
        with col1:  
            st.markdown("**Régression Logistique**")  
            for metrique, valeur in resultats_lr.items():  
                st.write(f"{metrique.capitalize()} : {valeur:.2%}")  
        
        with col2:  
            st.markdown("**Random Forest**")  
            for metrique, valeur in resultats_rf.items():  
                st.write(f"{metrique.capitalize()} : {valeur:.2%}")  
        
        # Reste de votre code de prédiction...  

    except Exception as e:  
        st.error(f"Erreur lors de la préparation des modèles : {e}")
        
