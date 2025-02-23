import streamlit as st  
import numpy as np  
import pandas as pd  
import traceback  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
from scipy.stats import poisson  
from sklearn.base import clone  
import math  

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

def calculer_lambda_attendus(data, equipe):  
    facteurs = [  
        data.get(f"buts_marques_{equipe}", 1.5),  
        data.get(f"performance_domicile_{equipe}", 75.0) / 100,  
        data.get(f"tirs_cadres_par_match_{equipe}", 5),  
        data.get(f"classement_championnat_{equipe}", 5) / 20,  
        1 - (data.get(f"joueurs_blesses_{equipe}", 2) / 11)  
    ]  
    
    lambda_base = np.prod(facteurs)  
    return max(0.5, min(lambda_base, 3.0))  

def calculer_probabilite_score_poisson(lambda_moyenne, k_buts):  
    return (lambda_moyenne ** k_buts * np.exp(-lambda_moyenne)) / math.factorial(k_buts)  

def predire_resultat_match_poisson(lambda_A, lambda_B, max_buts=5):  
    probabilites = np.zeros((max_buts + 1, max_buts + 1))  
    
    for buts_A in range(max_buts + 1):  
        for buts_B in range(max_buts + 1):  
            prob_A = calculer_probabilite_score_poisson(lambda_A, buts_A)  
            prob_B = calculer_probabilite_score_poisson(lambda_B, buts_B)  
            probabilites[buts_A, buts_B] = prob_A * prob_B  
    
    victoire_A = np.sum(probabilites[np.triu_indices(max_buts + 1, 1)])  
    victoire_B = np.sum(probabilites[np.tril_indices(max_buts + 1, -1)])  
    nul = probabilites.trace()  
    
    return {  
        'Victoire Équipe A': victoire_A,  
        'Victoire Équipe B': victoire_B,  
        'Match Nul': nul  
    }  

def validation_croisee_stratifiee(X, y, modele, n_splits=5):  
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  
    
    accuracies = []  
    precisions = []  
    recalls = []  
    f1_scores_list = []  
    
    for train_index, test_index in skf.split(X, y):  
        X_train, X_test = X[train_index], X[test_index]  
        y_train, y_test = y[train_index], y[test_index]  
        
        modele_clone = clone(modele)  
        modele_clone.fit(X_train, y_train)  
        
        y_pred = modele_clone.predict(X_test)  
        
        accuracies.append(accuracy_score(y_test, y_pred))  
        precisions.append(precision_score(y_test, y_pred, average='weighted'))  
        recalls.append(recall_score(y_test, y_pred, average='weighted'))  
        f1_scores_list.append(f1_score(y_test, y_pred, average='weighted'))  
    
    return {  
        'accuracy': np.mean(accuracies),  
        'precision': np.mean(precisions),  
        'recall': np.mean(recalls),  
        'f1_score': np.mean(f1_scores_list)  
    }  

def generer_donnees_historiques_defaut(nombre_matchs=50):  
    donnees = []  
    for _ in range(nombre_matchs):  
        match = {  
            'score_rating_A': np.random.normal(1500, 200),  
            'classement_championnat_A': np.random.randint(1, 20),  
            'buts_marques_A': np.random.normal(1.5, 0.5),  
            'tirs_cadres_par_match_A': np.random.normal(5, 1),  
            'performance_domicile_A': np.random.normal(70, 10),  
            
            'score_rating_B': np.random.normal(1500, 200),  
            'classement_championnat_B': np.random.randint(1, 20),  
            'buts_marques_B': np.random.normal(1.5, 0.5),  
            'tirs_cadres_par_match_B': np.random.normal(5, 1),  
            'performance_domicile_B': np.random.normal(70, 10),  
            
            'joueurs_blesses_A': np.random.randint(0, 5),  
            'joueurs_blesses_B': np.random.randint(0, 5),  
            
            'resultat': np.random.choice([0, 1, 2])  # 0: défaite, 1: victoire, 2: nul  
        }  
        donnees.append(match)  
    return donnees  

def preparer_donnees_entrainement(donnees_historiques):  
    if not donnees_historiques:  
        donnees_historiques = generer_donnees_historiques_defaut()  
    
    X = []  
    y = []  
    
    for donnee in donnees_historiques:  
        features = [  
            safe_float(donnee.get('score_rating_A', 1500)),  
            safe_float(donnee.get('classement_championnat_A', 10)),  
            safe_float(donnee.get('buts_marques_A', 1.5)),  
            safe_float(donnee.get('tirs_cadres_par_match_A', 5)),  
            safe_float(donnee.get('performance_domicile_A', 70)),  
            
            safe_float(donnee.get('score_rating_B', 1500)),  
            safe_float(donnee.get('classement_championnat_B', 10)),  
            safe_float(donnee.get('buts_marques_B', 1.5)),  
            safe_float(donnee.get('tirs_cadres_par_match_B', 5)),  
            safe_float(donnee.get('performance_domicile_B', 70)),  
            
            safe_float(donnee.get('joueurs_blesses_A', 2)),  
            safe_float(donnee.get('joueurs_blesses_B', 2))  
        ]  
        
        X.append(features)  
        
        resultat = donnee.get('resultat', 2)  
        y.append(resultat)  # 0: défaite, 1: victoire, 2: nul  
    
    return np.array(X), np.array(y)  

def preparer_donnees_regression_logistique(data):  
    if not data:  
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
    if not data:  
        data = {  
            'classement_championnat_A': 10, 'classement_championnat_B': 10,  
            'score_rating_A': 1500, 'score_rating_B': 1500,  
            'buts_marques_A': 1.5, 'buts_marques_B': 1.5,  
            'tirs_cadres_par_match_A': 5, 'tirs_cadres_par_match_B': 5,  
            'performance_domicile_A': 50, 'performance_domicile_B': 50,  
            'joueurs_blesses_A': 2, 'joueurs_blesses_B': 2  
        }  
    
    X_rf = np.array([  
        [  
            safe_float(data.get("classement_championnat_A", 10)),  
            safe_float(data.get("classement_championnat_B", 10)),  
            
            safe_float(data.get("score_rating_A", 1500)),  
            safe_float(data.get("score_rating_B", 1500)),  
            
            safe_float(data.get("buts_marques_A", 1.5)),  
            safe_float(data.get("buts_marques_B", 1.5)),  
            
            safe_float(data.get("tirs_cadres_par_match_A", 5)),  
            safe_float(data.get("tirs_cadres_par_match_B", 5)),  
            
            safe_float(data.get("performance_domicile_A", 50)),  
            safe_float(data.get("performance_domicile_B", 50)),  
            
            safe_float(data.get("joueurs_blesses_A", 2)),  
            safe_float(data.get("joueurs_blesses_B", 2))  
        ]  
    ])  
    return X_rf  

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
            "performance_domicile_A": st.number_input("🏠 Performance Domicile", value=70, key="domicile_A"),  
            "joueurs_blesses_A": st.number_input("🩺 Joueurs Blessés", value=2, key="blesses_A")  
        })  

    with col2:  
        st.markdown("**Équipe B**")  
        st.session_state.data.update({  
            "score_rating_B": st.number_input("⭐ Score Rating", value=1500, key="rating_B"),  
            "classement_championnat_B": st.number_input("🏆 Classement", value=10, key="classement_B"),  
            "buts_marques_B": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_B"),  
            "tirs_cadres_par_match_B": st.number_input("🎯 Tirs Cadrés", value=5, key="tirs_B"),  
            "performance_domicile_B": st.number_input("🏠 Performance Domicile", value=70, key="domicile_B"),  
            "joueurs_blesses_B": st.number_input("🩺 Joueurs Blessés", value=2, key="blesses_B")  
        })  

    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Section d'analyse et de prédiction  
if st.button("🏆 Prédire le Résultat"):  
    try:  
        # Génération de données historiques  
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
        
        # Calcul des lambda pour Poisson  
        lambda_A = calculer_lambda_attendus(st.session_state.data, 'A')  
        lambda_B = calculer_lambda_attendus(st.session_state.data, 'B')  
        
        # Résultats Poisson  
        resultats_poisson = predire_resultat_match_poisson(lambda_A, lambda_B)  
        
        # Résultats  
        st.subheader("🔮 Résultats de Prédiction")  
        
        col_poisson, col_modeles = st.columns(2)  
        
        with col_poisson:  
            st.markdown("**📊 Probabilités de Poisson**")  
            for resultat, proba in resultats_poisson.items():  
                st.metric(resultat, f"{proba:.2%}")  
        
        with col_modeles:  
            st.markdown("**🤖 Performance des Modèles**")  
            
            resultats_modeles = {}  
            for nom, modele in modeles.items():  
                # Validation croisée stratifiée  
                resultats_cv = validation_croisee_stratifiee(  
                    X_train,   
                    y_train,   
                    modele  
                )  
                
                # Stockage des résultats  
                resultats_modeles[nom] = resultats_cv  
                
                # Affichage des métriques de validation croisée  
                st.markdown(f"**{nom}**")  
                for metrique, valeur in resultats_cv.items():  
                    st.write(f"{metrique.capitalize()} : {valeur:.2%}")  
                
                # Entraînement final et prédiction  
                modele.fit(X_train, y_train)  
                proba = modele.predict_proba(X_lr if nom == "Régression Logistique" else X_rf)[0]  
                
                st.write(f"Probabilité Victoire A: {proba[1]:.2%}")  
                st.write(f"Probabilité Victoire B: {proba[0]:.2%}")  
                st.write(f"Probabilité Match Nul: {proba[2]:.2%}")  
        
        # Analyse finale  
        probabilite_victoire_A = (  
            resultats_poisson['Victoire Équipe A'] +   
            (modeles["Régression Logistique"].predict_proba(X_lr)[0][1] * 0.5) +   
            (modeles["Random Forest"].predict_proba(X_rf)[0][1] * 0.5)  
        ) / 2  
        
        st.subheader("🏆 Résultat Final")  
        st.metric("Probabilité de Victoire de l'Équipe A", f"{probabilite_victoire_A:.2%}")  
        
        # Visualisation des performances des modèles  
        st.subheader("📈 Comparaison des Performances des Modèles")  
        
        # Préparation des données pour le graphique  
        metriques = ['accuracy', 'precision', 'recall', 'f1_score']  
        
        # Création du DataFrame  
        df_performances = pd.DataFrame({  
            nom: [resultats_modeles[nom][metrique] for metrique in metriques]  
            for nom in resultats_modeles.keys()  
        }, index=metriques)  
        
        # Affichage du DataFrame  
        st.dataframe(df_performances)  
        
        # Graphique de comparaison  
        fig, ax = plt.subplots(figsize=(10, 6))  
        df_performances.T.plot(kind='bar', ax=ax)  
        plt.title("Comparaison des Performances des Modèles")  
        plt.xlabel("Modèles")  
        plt.ylabel("Score")  
        plt.legend(title="Métriques", bbox_to_anchor=(1.05, 1), loc='upper left')  
        plt.tight_layout()  
        st.pyplot(fig)  
        
    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(traceback.format_exc())  

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  

- **Probabilités de Poisson** : Basées sur les statistiques historiques et les caractéristiques des équipes  
- **Modèles de Machine Learning** :   
  - Régression Logistique : Modèle linéaire simple  
  - Random Forest : Modèle plus complexe, moins sensible au bruit  
- **Résultat Final** : Moyenne pondérée des différentes méthodes de prédiction  

⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""")
