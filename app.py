import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split, StratifiedKFold  
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

# Méthode de Poisson  
def calculer_probabilite_score_poisson(lambda_moyenne, k_buts):  
    return (lambda_moyenne ** k_buts * np.exp(-lambda_moyenne)) / np.math.factorial(k_buts)  

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

def validation_croisee_personnalisee(X, y, modele, n_splits=5):  
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
        data.get("forme_recente_A_victoires", 0),  
        data.get("forme_recente_A_nuls", 0),  
        data.get("forme_recente_A_defaites", 0)  
    )  
    score_forme_B = calculer_score_forme_avance(  
        data.get("forme_recente_B_victoires", 0),  
        data.get("forme_recente_B_nuls", 0),  
        data.get("forme_recente_B_defaites", 0)  
    )  

    X_lr = np.array([  
        [  
            # Performance Globale et Classement  
            safe_float(data.get("score_rating_A", 0)),  
            safe_float(data.get("classement_championnat_A", 0)),  
            safe_float(data.get("score_rating_B", 0)),  
            safe_float(data.get("classement_championnat_B", 0)),  
            
            # Performance Offensive  
            safe_float(data.get("buts_marques_A", 0)),  
            safe_float(data.get("tirs_cadres_par_match_A", 0)),  
            safe_float(data.get("buts_marques_B", 0)),  
            safe_float(data.get("tirs_cadres_par_match_B", 0)),  
            
            # Performance Domicile/Extérieur  
            safe_float(data.get("performance_domicile_A", 0)),  
            safe_float(data.get("performance_exterieur_A", 0)),  
            safe_float(data.get("performance_domicile_B", 0)),  
            safe_float(data.get("performance_exterieur_B", 0)),  
            
            # Forme et Motivation  
            score_forme_A,  
            safe_float(data.get("motivation_A", 0)),  
            score_forme_B,  
            safe_float(data.get("motivation_B", 0)),  
            
            # Confrontations Directes  
            safe_float(data.get("victoires_confrontations_directes_A", 0)),  
            safe_float(data.get("victoires_confrontations_directes_B", 0)),  
            
            # Impact des Blessures  
            safe_float(data.get("joueurs_blesses_A", 0)),  
            safe_float(data.get("joueurs_blesses_B", 0))  
        ]  
    ])  
    return X_lr  

def preparer_donnees_random_forest(data):  
    X_rf = np.array([  
        [  
            # Performance Défensive  
            safe_float(data.get("buts_attendus_concedes_A", 0)),  
            safe_float(data.get("interceptions_A", 0)),  
            safe_float(data.get("tacles_reussis_par_match_A", 0)),  
            safe_float(data.get("buts_attendus_concedes_B", 0)),  
            safe_float(data.get("interceptions_B", 0)),  
            safe_float(data.get("tacles_reussis_par_match_B", 0)),  
            
            # Statistiques de Jeu Avancées  
            safe_float(data.get("centres_reussies_par_match_A", 0)),  
            safe_float(data.get("corners_par_match_A", 0)),  
            safe_float(data.get("centres_reussies_par_match_B", 0)),  
            safe_float(data.get("corners_par_match_B", 0)),  
            
            # Discipline  
            safe_float(data.get("penalties_concedees_A", 0)),  
            safe_float(data.get("fautes_par_match_A", 0)),  
            safe_float(data.get("penalties_concedees_B", 0)),  
            safe_float(data.get("fautes_par_match_B", 0)),  
            
            # Performance Globale  
            safe_float(data.get("classement_championnat_A", 0)),  
            safe_float(data.get("classement_championnat_B", 0)),  
            
            # Créativité et Dynamisme  
            safe_float(data.get("passes_reussies_par_match_A", 0)),  
            safe_float(data.get("dribbles_reussis_par_match_A", 0)),  
            safe_float(data.get("passes_reussies_par_match_B", 0)),  
            safe_float(data.get("dribbles_reussis_par_match_B", 0)),  
            
            # Données de Blessures et Confrontations  
            safe_float(data.get("joueurs_blesses_A", 0)),  
            safe_float(data.get("joueurs_blesses_B", 0)),  
            safe_float(data.get("victoires_confrontations_directes_A", 0)),  
            safe_float(data.get("victoires_confrontations_directes_B", 0))  
        ]  
    ])  
    return X_rf  

def preparer_donnees_entrainement(donnees_historiques):  
    """  
    Prépare les données d'entraînement à partir de données historiques complètes  
    """  
    X = np.array([  
        [  
            # Caractéristiques de l'Équipe A  
            donnee.get('score_rating_A', 0),  
            donnee.get('classement_championnat_A', 0),  
            donnee.get('buts_marques_A', 0),  
            donnee.get('tirs_cadres_par_match_A', 0),  
            donnee.get('performance_domicile_A', 0),  
            donnee.get('performance_exterieur_A', 0),  
            donnee.get('forme_recente_A_victoires', 0),  
            donnee.get('forme_recente_A_nuls', 0),  
            donnee.get('forme_recente_A_defaites', 0),  
            
            # Caractéristiques de l'Équipe B  
            donnee.get('score_rating_B', 0),  
            donnee.get('classement_championnat_B', 0),  
            donnee.get('buts_marques_B', 0),  
            donnee.get('tirs_cadres_par_match_B', 0),  
            donnee.get('performance_domicile_B', 0),  
            donnee.get('performance_exterieur_B', 0),  
            donnee.get('forme_recente_B_victoires', 0),  
            donnee.get('forme_recente_B_nuls', 0),  
            donnee.get('forme_recente_B_defaites', 0),  
            
            # Caractéristiques du Match  
            donnee.get('victoires_confrontations_directes_A', 0),  
            donnee.get('victoires_confrontations_directes_B', 0),  
            donnee.get('joueurs_blesses_A', 0),  
            donnee.get('joueurs_blesses_B', 0)  
        ] for donnee in donnees_historiques  
    ])  
    
    y = np.array([donnee.get('resultat', 0) for donnee in donnees_historiques])  
    
    return X, y  

# Exemple de données historiques (à remplacer par vos vraies données)  
donnees_historiques_exemple = [  
    {  
        'score_rating_A': 1800, 'classement_championnat_A': 5,   
        'buts_marques_A': 2, 'tirs_cadres_par_match_A': 6,  
        'performance_domicile_A': 75, 'performance_exterieur_A': 60,  
        'forme_recente_A_victoires': 3, 'forme_recente_A_nuls': 1, 'forme_recente_A_defaites': 1,  
        
        'score_rating_B': 1700, 'classement_championnat_B': 10,   
        'buts_marques_B': 1.5, 'tirs_cadres_par_match_B': 5,  
        'performance_domicile_B': 65, 'performance_exterieur_B': 55,  
        'forme_recente_B_victoires': 2, 'forme_recente_B_nuls': 2, 'forme_recente_B_defaites': 2,  
        
        'victoires_confrontations_directes_A': 3,  
        'victoires_confrontations_directes_B': 2,  
        'joueurs_blesses_A': 2,  
        'joueurs_blesses_B': 3,  
        
        'resultat': 1  # 1 pour victoire A, 0 pour victoire B, 0.5 pour nul  
    },  
    # Ajoutez plus de matchs historiques  
]  

# Configuration Streamlit  
st.set_page_config(page_title="Prédictions de Matchs Avancées", page_icon="⚽")  
st.title("🏆 Système de Prédiction de Matchs de Football")  

# Initialisation du state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Formulaire de saisie complet  
with st.form("Données Complètes du Match"):  
    st.subheader("📊 Données Détaillées des Équipes")  
    
    # Colonnes pour une meilleure organisation  
    col1, col2 = st.columns(2)  
    
    with col1:  
        st.markdown("**Équipe A**")  
        st.session_state.data.update({  
            # Données de base  
            "buts_marques_A": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_A"),  
            "classement_championnat_A": st.number_input("🏆 Classement", value=5, key="classement_A"),  
            "score_rating_A": st.number_input("⭐ Score Rating", value=1800, key="rating_A"),  
            
            # Performance  
            "performance_domicile_A": st.number_input("🏠 Performance Domicile", value=75.0, key="domicile_A"),  
            "performance_exterieur_A": st.number_input("✈️ Performance Extérieur", value=60.0, key="exterieur_A"),  
            
            # Statistiques avancées  
            "tirs_cadres_par_match_A": st.number_input("🎯 Tirs Cadrés", value=5, key="tirs_A"),  
            "interceptions_A": st.number_input("🛡️ Interceptions", value=10, key="interceptions_A"),  
            "passes_reussies_par_match_A": st.number_input("🤝 Passes Réussies", value=80, key="passes_A"),  
            
            # Forme et blessures  
            "forme_recente_A_victoires": st.number_input("🏆 Victoires Récentes", value=3, key="victoires_A"),  
            "forme_recente_A_nuls": st.number_input("➖ Matchs Nuls", value=1, key="nuls_A"),  
            "forme_recente_A_defaites": st.number_input("❌ Défaites Récentes", value=1, key="defaites_A"),  
            "joueurs_blesses_A": st.number_input("🩺 Joueurs Blessés", value=2, key="blesses_A"),  
        })  

    with col2:  
        st.markdown("**Équipe B**")  
        st.session_state.data.update({  
            # Données de base  
            "buts_marques_B": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_B"),  
            "classement_championnat_B": st.number_input("🏆 Classement", value=10, key="classement_B"),  
            "score_rating_B": st.number_input("⭐ Score Rating", value=1700, key="rating_B"),  
            
            # Performance  
            "performance_domicile_B": st.number_input("🏠 Performance Domicile", value=65.0, key="domicile_B"),  
            "performance_exterieur_B": st.number_input("✈️ Performance Extérieur", value=55.0, key="exterieur_B"),  
            
            # Statistiques avancées  
            "tirs_cadres_par_match_B": st.number_input("🎯 Tirs Cadrés", value=5, key="tirs_B"),  
            "interceptions_B": st.number_input("🛡️ Interceptions", value=8, key="interceptions_B"),  
            "passes_reussies_par_match_B": st.number_input("🤝 Passes Réussies", value=75, key="passes_B"),  
            
            # Forme et blessures  
            "forme_recente_B_victoires": st.number_input("🏆 Victoires Récentes", value=2, key="victoires_B"),  
            "forme_recente_B_nuls": st.number_input("➖ Matchs Nuls", value=2, key="nuls_B"),  
            "forme_recente_B_defaites": st.number_input("❌ Défaites Récentes", value=2, key="defaites_B"),  
            "joueurs_blesses_B": st.number_input("🩺 Joueurs Blessés", value=3, key="blesses_B"),  
            
            # Confrontations directes  
            "victoires_confrontations_directes_A": st.number_input("🤝 Victoires Confrontations (A)", value=3, key="conf_A"),  
            "victoires_confrontations_directes_B": st.number_input("🤝 Victoires Confrontations (B)", value=2, key="conf_B"),  
        })  

    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Bouton de prédiction principal  
if st.button("🏆 Prédire le Résultat du Match"):  
    try:  
        # Préparation des données pour les modèles  
        X_lr = preparer_donnees_regression_logistique(st.session_state.data)  
        X_rf = preparer_donnees_random_forest(st.session_state.data)  
        
        # Préparation des données d'entraînement  
        X_train, y_train = preparer_donnees_entrainement(donnees_historiques_exemple)  
        
        # Division des données  
        X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_train, y_train, test_size=0.2, random_state=42)  
        
        # Modèles de prédiction  
        modeles = {  
            "Régression Logistique": LogisticRegression(max_iter=1000),  
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  
        }  
        
        # Conteneur pour les résultats  
        st.subheader("🔮 Résultats de Prédiction")  
        
        # Calcul des lambda pour Poisson  
        lambda_A = calculer_lambda_attendus(st.session_state.data, 'A')  
        lambda_B = calculer_lambda_attendus(st.session_state.data, 'B')  
        
        # Prédiction Poisson  
        resultats_poisson = predire_resultat_match_poisson(lambda_A, lambda_B)  
        
        # Colonne pour les probabilités Poisson  
        col_poisson, col_modeles = st.columns(2)  
        
        with col_poisson:  
            st.markdown("**📊 Probabilités de Poisson**")  
            for resultat, proba in resultats_poisson.items():  
                st.metric(resultat, f"{proba:.2%}")  
        
        # Conteneur pour les résultats des modèles  
        with col_modeles:  
            st.markdown("**🤖 Performance des Modèles**")  
            
            # Validation croisée et prédiction pour chaque modèle  
            for nom, modele in modeles.items():  
                if nom == "Régression Logistique":  
                    resultats_cv = validation_croisee_personnalisee(X_train_lr, y_train_lr, modele)  
                    modele.fit(X_train_lr, y_train_lr)  
                    prediction = modele.predict(X_lr)  
                else:  
                    resultats_cv = validation_croisee_personnalisee(X_train_rf, y_train_rf, modele)  
                    modele.fit(X_train_rf, y_train_rf)  
                    prediction = modele.predict(X_rf)  
                
                # Affichage des métriques de validation croisée  
                st.markdown(f"**{nom}**")  
                for metrique, valeur in resultats_cv.items():  
                    st.write(f"{metrique.capitalize()} : {valeur:.2%}")  
        
        # Analyse des probabilités combinées  
        st.subheader("🔬 Analyse Finale")  
        probabilite_victoire_A = (  
            resultats_poisson['Victoire Équipe A'] +   
            (modeles["Régression Logistique"].predict_proba(X_lr)[0][1] * 0.5) +   
            (modeles["Random Forest"].predict_proba(X_rf)[0][1] * 0.5)  
        ) / 2  
        
        st.metric("🏆 Probabilité de Victoire Finale (Équipe A)", f"{probabilite_victoire_A:.2%}")  
        
    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(f"Détails de l'erreur : {traceback.format_exc()}")  

# Pied de page informatif  
st.markdown("""  
---  
📌 **Note Importante** :  
- Les prédictions sont basées sur des modèles statistiques et des données historiques  
- Aucune prédiction n'est garantie à 100%  
- Utilisez ces informations de manière responsable  
""")
