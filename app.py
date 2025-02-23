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
        data.get(f"buts_par_match_{equipe}", 1.5),  
        data.get(f"buts_concedes_par_match_{equipe}", 1.0),  
        data.get(f"possession_moyenne_{equipe}", 50.0) / 100,  
        data.get(f"expected_but_{equipe}", 1.5),  
        data.get(f"expected_concedes_{equipe}", 1.0),  
        data.get(f"tirs_cadres_{equipe}", 5) / 100,  
        data.get(f"grandes_chances_{equipe}", 5) / 100,  
        data.get(f"passes_reussies_{equipe}", 300) / 1000,  
        data.get(f"corners_{equipe}", 5) / 100,  
        data.get(f"interceptions_{equipe}", 5) / 100,  
        data.get(f"tacles_reussis_{equipe}", 5) / 100,  
        data.get(f"fautes_{equipe}", 5) / 100,  
        data.get(f"cartons_jaunes_{equipe}", 1) / 10,  
        data.get(f"cartons_rouges_{equipe}", 0) / 10,  
        data.get(f"joueurs_cles_absents_{equipe}", 0) / 11,  
        data.get(f"motivation_{equipe}", 3) / 5,  
        data.get(f"clean_sheets_gardien_{equipe}", 1) / 10,  
        data.get(f"ratio_tirs_arretes_{equipe}", 0.75),  
        data.get(f"victoires_domicile_{equipe}", 50) / 100,  
        data.get(f"passes_longues_{equipe}", 50) / 100,  
        data.get(f"dribbles_reussis_{equipe}", 10) / 100,  
        data.get(f"ratio_tirs_cadres_{equipe}", 0.4),  
        data.get(f"grandes_chances_manquees_{equipe}", 5) / 100,  
        data.get(f"fautes_zones_dangereuses_{equipe}", 3) / 100,  
        data.get(f"buts_corners_{equipe}", 2) / 100,  
        data.get(f"jours_repos_{equipe}", 4) / 10,  
        data.get(f"matchs_30_jours_{equipe}", 8) / 10  
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
            # Équipe A  
            "score_rating_A": np.random.normal(70, 10),  
            "buts_par_match_A": np.random.normal(1.5, 0.5),  
            "buts_concedes_par_match_A": np.random.normal(1.0, 0.3),  
            "possession_moyenne_A": np.random.normal(55, 5),  
            "expected_but_A": np.random.normal(1.8, 0.5),  
            "expected_concedes_A": np.random.normal(1.2, 0.3),  
            "tirs_cadres_A": np.random.normal(120, 20),  
            "grandes_chances_A": np.random.normal(25, 5),  
            "passes_reussies_A": np.random.normal(400, 50),  
            "corners_A": np.random.normal(60, 10),  
            "interceptions_A": np.random.normal(50, 10),  
            "tacles_reussis_A": np.random.normal(40, 10),  
            "fautes_A": np.random.normal(15, 5),  
            "cartons_jaunes_A": np.random.normal(5, 2),  
            "cartons_rouges_A": np.random.normal(1, 0.5),  
            "joueurs_cles_absents_A": np.random.randint(0, 5),  
            "motivation_A": np.random.randint(1, 5),  
            "clean_sheets_gardien_A": np.random.normal(2, 1),  
            "ratio_tirs_arretes_A": np.random.normal(0.75, 0.1),  
            "victoires_domicile_A": np.random.normal(60, 10),  
            "passes_longues_A": np.random.normal(50, 10),  
            "dribbles_reussis_A": np.random.normal(10, 3),  
            "ratio_tirs_cadres_A": np.random.normal(0.4, 0.1),  
            "grandes_chances_manquees_A": np.random.normal(5, 2),  
            "fautes_zones_dangereuses_A": np.random.normal(3, 1),  
            "buts_corners_A": np.random.normal(2, 1),  
            "jours_repos_A": np.random.normal(4, 1),  
            "matchs_30_jours_A": np.random.normal(8, 2),  
            
            # Équipe B  
            "score_rating_B": np.random.normal(65, 10),  
            "buts_par_match_B": np.random.normal(1.0, 0.5),  
            "buts_concedes_par_match_B": np.random.normal(1.5, 0.3),  
            "possession_moyenne_B": np.random.normal(45, 5),  
            "expected_but_B": np.random.normal(1.2, 0.5),  
            "expected_concedes_B": np.random.normal(1.8, 0.3),  
            "tirs_cadres_B": np.random.normal(100, 20),  
            "grandes_chances_B": np.random.normal(20, 5),  
            "passes_reussies_B": np.random.normal(350, 50),  
            "corners_B": np.random.normal(50, 10),  
            "interceptions_B": np.random.normal(40, 10),  
            "tacles_reussis_B": np.random.normal(35, 10),  
            "fautes_B": np.random.normal(20, 5),  
            "cartons_jaunes_B": np.random.normal(6, 2),  
            "cartons_rouges_B": np.random.normal(2, 0.5),  
            "joueurs_cles_absents_B": np.random.randint(0, 5),  
            "motivation_B": np.random.randint(1, 5),  
            "clean_sheets_gardien_B": np.random.normal(1, 1),  
            "ratio_tirs_arretes_B": np.random.normal(0.65, 0.1),  
            "victoires_exterieur_B": np.random.normal(40, 10),  
            "passes_longues_B": np.random.normal(40, 10),  
            "dribbles_reussis_B": np.random.normal(8, 3),  
            "ratio_tirs_cadres_B": np.random.normal(0.35, 0.1),  
            "grandes_chances_manquees_B": np.random.normal(6, 2),  
            "fautes_zones_dangereuses_B": np.random.normal(4, 1),  
            "buts_corners_B": np.random.normal(1, 1),  
            "jours_repos_B": np.random.normal(3, 1),  
            "matchs_30_jours_B": np.random.normal(9, 2),  
            
            # Résultat  
            "resultat": np.random.choice([0, 1, 2])  # 0: défaite, 1: victoire, 2: nul  
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
            # Équipe A  
            safe_float(donnee.get("score_rating_A", 70)),  
            safe_float(donnee.get("buts_par_match_A", 1.5)),  
            safe_float(donnee.get("buts_concedes_par_match_A", 1.0)),  
            safe_float(donnee.get("possession_moyenne_A", 55)),  
            safe_float(donnee.get("expected_but_A", 1.8)),  
            safe_float(donnee.get("expected_concedes_A", 1.2)),  
            safe_float(donnee.get("tirs_cadres_A", 120)),  
            safe_float(donnee.get("grandes_chances_A", 25)),  
            safe_float(donnee.get("passes_reussies_A", 400)),  
            safe_float(donnee.get("corners_A", 60)),  
            safe_float(donnee.get("interceptions_A", 50)),  
            safe_float(donnee.get("tacles_reussis_A", 40)),  
            safe_float(donnee.get("fautes_A", 15)),  
            safe_float(donnee.get("cartons_jaunes_A", 5)),  
            safe_float(donnee.get("cartons_rouges_A", 1)),  
            safe_float(donnee.get("joueurs_cles_absents_A", 0)),  
            safe_float(donnee.get("motivation_A", 3)),  
            safe_float(donnee.get("clean_sheets_gardien_A", 2)),  
            safe_float(donnee.get("ratio_tirs_arretes_A", 0.75)),  
            safe_float(donnee.get("victoires_domicile_A", 60)),  
            safe_float(donnee.get("passes_longues_A", 50)),  
            safe_float(donnee.get("dribbles_reussis_A", 10)),  
            safe_float(donnee.get("ratio_tirs_cadres_A", 0.4)),  
            safe_float(donnee.get("grandes_chances_manquees_A", 5)),  
            safe_float(donnee.get("fautes_zones_dangereuses_A", 3)),  
            safe_float(donnee.get("buts_corners_A", 2)),  
            safe_float(donnee.get("jours_repos_A", 4)),  
            safe_float(donnee.get("matchs_30_jours_A", 8)),  
            
            # Équipe B  
            safe_float(donnee.get("score_rating_B", 65)),  
            safe_float(donnee.get("buts_par_match_B", 1.0)),  
            safe_float(donnee.get("buts_concedes_par_match_B", 1.5)),  
            safe_float(donnee.get("possession_moyenne_B", 45)),  
            safe_float(donnee.get("expected_but_B", 1.2)),  
            safe_float(donnee.get("expected_concedes_B", 1.8)),  
            safe_float(donnee.get("tirs_cadres_B", 100)),  
            safe_float(donnee.get("grandes_chances_B", 20)),  
            safe_float(donnee.get("passes_reussies_B", 350)),  
            safe_float(donnee.get("corners_B", 50)),  
            safe_float(donnee.get("interceptions_B", 40)),  
            safe_float(donnee.get("tacles_reussis_B", 35)),  
            safe_float(donnee.get("fautes_B", 20)),  
            safe_float(donnee.get("cartons_jaunes_B", 6)),  
            safe_float(donnee.get("cartons_rouges_B", 2)),  
            safe_float(donnee.get("joueurs_cles_absents_B", 0)),  
            safe_float(donnee.get("motivation_B", 3)),  
            safe_float(donnee.get("clean_sheets_gardien_B", 1)),  
            safe_float(donnee.get("ratio_tirs_arretes_B", 0.65)),  
            safe_float(donnee.get("victoires_exterieur_B", 40)),  
            safe_float(donnee.get("passes_longues_B", 40)),  
            safe_float(donnee.get("dribbles_reussis_B", 8)),  
            safe_float(donnee.get("ratio_tirs_cadres_B", 0.35)),  
            safe_float(donnee.get("grandes_chances_manquees_B", 6)),  
            safe_float(donnee.get("fautes_zones_dangereuses_B", 4)),  
            safe_float(donnee.get("buts_corners_B", 1)),  
            safe_float(donnee.get("jours_repos_B", 3)),  
            safe_float(donnee.get("matchs_30_jours_B", 9))  
        ]  
        
        X.append(features)  
        
        resultat = donnee.get("resultat", 2)  
        y.append(resultat)  # 0: défaite, 1: victoire, 2: nul  
    
    return np.array(X), np.array(y)  

def preparer_donnees_regression_logistique(data):  
    if not data:  
        data = {  
            # Équipe A  
            "score_rating_A": 70, "buts_par_match_A": 1.5, "buts_concedes_par_match_A": 1.0,  
            "possession_moyenne_A": 55, "expected_but_A": 1.8, "expected_concedes_A": 1.2,  
            "tirs_cadres_A": 120, "grandes_chances_A": 25, "passes_reussies_A": 400,  
            "corners_A": 60, "interceptions_A": 50, "tacles_reussis_A": 40, "fautes_A": 15,  
            "cartons_jaunes_A": 5, "cartons_rouges_A": 1, "joueurs_cles_absents_A": 0,  
            "motivation_A": 3, "clean_sheets_gardien_A": 2,"ratio_tirs_arretes_A": 0.75, 
            "victoires_domicile_A": 60, "passes_longues_A": 50,"dribbles_reussis_A": 10,
            "ratio_tirs_cadres_A": 0.4, "grandes_chances_manquees_A": 5,"fautes_zones_dangereuses_A": 3, 
            "buts_corners_A": 2, "jours_repos_A": 4, "matchs_30_jours_A": 8,  
            
            # Équipe B  
            "score_rating_B": 65, "buts_par_match_B": 1.0, "buts_concedes_par_match_B": 1.5,  
            "possession_moyenne_B": 45, "expected_but_B": 1.2, "expected_concedes_B": 1.8,  
            "tirs_cadres_B": 100, "grandes_chances_B": 20, "passes_reussies_B": 350,  
            "corners_B": 50, "interceptions_B": 40, "tacles_reussis_B": 35, "fautes_B": 20,  
            "cartons_jaunes_B": 6, "cartons_rouges_B": 2, "joueurs_cles_absents_B": 0,  
            "motivation_B": 3, "clean_sheets_gardien_B": 1, "ratio_tirs_arretes_B": 0.65,  
            "victoires_exterieur_B": 40, "passes_longues_B": 40, "dribbles_reussis_B": 8,  
            "ratio_tirs_cadres_B": 0.35, "grandes_chances_manquees_B": 6,"fautes_zones_dangereuses_B": 4,
            "buts_corners_B": 1, "jours_repos_B": 3,"matchs_30_jours_B": 9,  
            
        }  
    
    X_lr = np.array([  
        [  
            # Équipe A  
            safe_float(data.get("score_rating_A", 70)),  
            safe_float(data.get("buts_par_match_A", 1.5)),  
            safe_float(data.get("buts_concedes_par_match_A", 1.0)),  
            safe_float(data.get("possession_moyenne_A", 55)),  
            safe_float(data.get("expected_but_A", 1.8)),  
            safe_float(data.get("expected_concedes_A", 1.2)),  
            safe_float(data.get("tirs_cadres_A", 120)),  
            safe_float(data.get("grandes_chances_A", 25)),  
            safe_float(data.get("passes_reussies_A", 400)),  
            safe_float(data.get("corners_A", 60)),  
            safe_float(data.get("interceptions_A", 50)),  
            safe_float(data.get("tacles_reussis_A", 40)),  
            safe_float(data.get("fautes_A", 15)),  
            safe_float(data.get("cartons_jaunes_A", 5)),  
            safe_float(data.get("cartons_rouges_A", 1)),  
            safe_float(data.get("joueurs_cles_absents_A", 0)),  
            safe_float(data.get("motivation_A", 3)),  
            safe_float(data.get("clean_sheets_gardien_A", 2)),  
            safe_float(data.get("ratio_tirs_arretes_A", 0.75)),  
            safe_float(data.get("victoires_domicile_A", 60)),  
            safe_float(data.get("passes_longues_A", 50)),  
            safe_float(data.get("dribbles_reussis_A", 10)),  
            safe_float(data.get("ratio_tirs_cadres_A", 0.4)),  
            safe_float(data.get("grandes_chances_manquees_A", 5)),  
            safe_float(data.get("fautes_zones_dangereuses_A", 3)),  
            safe_float(data.get("buts_corners_A", 2)),  
            safe_float(data.get("jours_repos_A", 4)),  
            safe_float(data.get("matchs_30_jours_A", 8)),  
            
            # Équipe B  
            safe_float(data.get("score_rating_B", 65)),  
            safe_float(data.get("buts_par_match_B", 1.0)),  
            safe_float(data.get("buts_concedes_par_match_B", 1.5)),  
            safe_float(data.get("possession_moyenne_B", 45)),  
            safe_float(data.get("expected_but_B", 1.2)),  
            safe_float(data.get("expected_concedes_B", 1.8)),  
            safe_float(data.get("tirs_cadres_B", 100)),  
            safe_float(data.get("grandes_chances_B", 20)),  
            safe_float(data.get("passes_reussies_B", 350)),  
            safe_float(data.get("corners_B", 50)),  
            safe_float(data.get("interceptions_B", 40)),  
            safe_float(data.get("tacles_reussis_B", 35)),  
            safe_float(data.get("fautes_B", 20)),  
            safe_float(data.get("cartons_jaunes_B", 6)),  
            safe_float(data.get("cartons_rouges_B", 2)),  
            safe_float(data.get("joueurs_cles_absents_B", 0)),  
            safe_float(data.get("motivation_B", 3)),  
            safe_float(data.get("clean_sheets_gardien_B", 1)),  
            safe_float(data.get("ratio_tirs_arretes_B", 0.65)),  
            safe_float(data.get("victoires_exterieur_B", 40)),  
            safe_float(data.get("passes_longues_B", 40)),  
            safe_float(data.get("dribbles_reussis_B", 8)),  
            safe_float(data.get("ratio_tirs_cadres_B", 0.35)),  
            safe_float(data.get("grandes_chances_manquees_B", 6)),  
            safe_float(data.get("fautes_zones_dangereuses_B", 4)),  
            safe_float(data.get("buts_corners_B", 1)),  
            safe_float(data.get("jours_repos_B", 3)),  
            safe_float(data.get("matchs_30_jours_B", 9))  
        ]  
    ])  
    return X_lr  

def preparer_donnees_random_forest(data):  
    if not data:  
        data = {  
            # Équipe A  
            "score_rating_A": 70, "buts_par_match_A": 1.5, "buts_concedes_par_match_A": 1.0,  
            "possession_moyenne_A": 55, "expected_but_A": 1.8, "expected_concedes_A": 1.2,  
            "tirs_cadres_A": 120, "grandes_chances_A": 25, "passes_reussies_A": 400,  
            "corners_A": 60, "interceptions_A": 50, "tacles_reussis_A": 40, "fautes_A": 15,  
            "cartons_jaunes_A": 5, "cartons_rouges_A": 1, "joueurs_cles_absents_A": 0,  
            "motivation_A": 3, "clean_sheets_gardien_A": 2, "ratio_tirs_arretes_A": 0.75,  
            "victoires_domicile_A": 60, "passes_longues_A": 50, "dribbles_reussis_A": 10,  
            "ratio_tirs_cadres_A": 0.4, "grandes_chances_manquees_A": 5,  
            "fautes_zones_dangereuses_A": 3, "buts_corners_A": 2, "jours_repos_A": 4,  
            "matchs_30_jours_A": 8,  
            
            # Équipe B  
            "score_rating_B": 65, "buts_par_match_B": 1.0, "buts_concedes_par_match_B": 1.5,  
            "possession_moyenne_B": 45, "expected_but_B": 1.2, "expected_concedes_B": 1.8,  
            "tirs_cadres_B": 100, "grandes_chances_B": 20, "passes_reussies_B": 350,  
            "corners_B": 50, "interceptions_B": 40, "tacles_reussis_B": 35, "fautes_B": 20,  
            "cartons_jaunes_B": 6, "cartons_rouges_B": 2, "joueurs_cles_absents_B": 0,  
            "motivation_B": 3, "clean_sheets_gardien_B": 1, "ratio_tirs_arretes_B": 0.65,  
            "victoires_exterieur_B": 40, "passes_longues_B": 40, "dribbles_reussis_B": 8,  
            "ratio_tirs_cadres_B": 0.35, "grandes_chances_manquees_B": 6,  
            "fautes_zones_dangereuses_B": 4, "buts_corners_B": 1, "jours_repos_B": 3,  
            "matchs_30_jours_B": 9  
        }  
    
    X_rf = np.array([  
        [  
            # Équipe A  
            safe_float(data.get("score_rating_A", 70)),  
            safe_float(data.get("buts_par_match_A", 1.5)),  
            safe_float(data.get("buts_concedes_par_match_A", 1.0)),  
            safe_float(data.get("possession_moyenne_A", 55)),  
            safe_float(data.get("expected_but_A", 1.8)),  
            safe_float(data.get("expected_concedes_A", 1.2)),  
            safe_float(data.get("tirs_cadres_A", 120)),  
            safe_float(data.get("grandes_chances_A", 25)),  
            safe_float(data.get("passes_reussies_A", 400)),  
            safe_float(data.get("corners_A", 60)),  
            safe_float(data.get("interceptions_A", 50)),  
            safe_float(data.get("tacles_reussis_A", 40)),  
            safe_float(data.get("fautes_A", 15)),  
            safe_float(data.get("cartons_jaunes_A", 5)),  
            safe_float(data.get("cartons_rouges_A", 1)),  
            safe_float(data.get("joueurs_cles_absents_A", 0)),  
            safe_float(data.get("motivation_A", 3)),  
            safe_float(data.get("clean_sheets_gardien_A", 2)),  
            safe_float(data.get("ratio_tirs_arretes_A", 0.75)),  
            safe_float(data.get("victoires_domicile_A", 60)),  
            safe_float(data.get("passes_longues_A", 50)),  
            safe_float(data.get("dribbles_reussis_A", 10)),  
            safe_float(data.get("ratio_tirs_cadres_A", 0.4)),  
            safe_float(data.get("grandes_chances_manquees_A", 5)),  
            safe_float(data.get("fautes_zones_dangereuses_A", 3)),  
            safe_float(data.get("buts_corners_A", 2)),  
            safe_float(data.get("jours_repos_A", 4)),  
            safe_float(data.get("matchs_30_jours_A", 8)),  
            
            # Équipe B  
            safe_float(data.get("score_rating_B", 65)),  
            safe_float(data.get("buts_par_match_B", 1.0)),  
            safe_float(data.get("buts_concedes_par_match_B", 1.5)),  
            safe_float(data.get("possession_moyenne_B", 45)),  
            safe_float(data.get("expected_but_B", 1.2)),  
            safe_float(data.get("expected_concedes_B", 1.8)),  
            safe_float(data.get("tirs_cadres_B", 100)),  
            safe_float(data.get("grandes_chances_B", 20)),  
            safe_float(data.get("passes_reussies_B", 350)),  
            safe_float(data.get("corners_B", 50)),  
            safe_float(data.get("interceptions_B", 40)),  
            safe_float(data.get("tacles_reussis_B", 35)),  
            safe_float(data.get("fautes_B", 20)),  
            safe_float(data.get("cartons_jaunes_B", 6)),  
            safe_float(data.get("cartons_rouges_B", 2)),  
            safe_float(data.get("joueurs_cles_absents_B", 0)),  
            safe_float(data.get("motivation_B", 3)),  
            safe_float(data.get("clean_sheets_gardien_B", 1)),  
            safe_float(data.get("ratio_tirs_arretes_B", 0.65)),  
            safe_float(data.get("victoires_exterieur_B", 40)),  
            safe_float(data.get("passes_longues_B", 40)),  
            safe_float(data.get("dribbles_reussis_B", 8)),  
            safe_float(data.get("ratio_tirs_cadres_B", 0.35)),  
            safe_float(data.get("grandes_chances_manquees_B", 6)),  
            safe_float(data.get("fautes_zones_dangereuses_B", 4)),  
            safe_float(data.get("buts_corners_B", 1)),  
            safe_float(data.get("jours_repos_B", 3)),  
            safe_float(data.get("matchs_30_jours_B", 9))  
        ]  
    ])  
    return X_rf
    
# Configuration Streamlit  
st.set_page_config(page_title="Prédictions de Matchs", page_icon="⚽")  
st.title("🏆 Système de Prédiction de Matchs de Football")  

# Initialisation du state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Fonctions utilitaires  
def calculer_lambda_attendus(data, equipe):  
    """Calcule les buts attendus (lambda) pour une équipe."""  
    return data[f"expected_but_{equipe}"]  

def predire_resultat_match_poisson(lambda_A, lambda_B):  
    """Prédit les probabilités de résultat avec le modèle de Poisson."""  
    proba_A = poisson.pmf(np.arange(0, 10), lambda_A)  
    proba_B = poisson.pmf(np.arange(0, 10), lambda_B)  
    
    # Calcul des probabilités de victoire, défaite et match nul  
    victoire_A = np.sum(np.tril(proba_A.reshape(-1, 1) * proba_B, -1))  
    victoire_B = np.sum(np.triu(proba_A.reshape(-1, 1) * proba_B, 1))  
    match_nul = np.sum(np.diag(proba_A.reshape(-1, 1) * proba_B))  
    
    return {  
        "Victoire Équipe A": victoire_A,  
        "Victoire Équipe B": victoire_B,  
        "Match Nul": match_nul  
    }  

def calculer_cotes_implicites(proba):  
    """Calcule les cotes implicites à partir des probabilités."""  
    return {k: round(1 / v, 2) for k, v in proba.items()}  

def detecter_value(cote_implicite, cote_predite):  
    """Détecte si une value existe."""  
    return cote_predite > cote_implicite  

def validation_croisee_stratifiee(X, y, modele):  
    """Effectue une validation croisée stratifiée."""  
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
    scores = cross_val_score(modele, X, y, cv=cv, scoring='accuracy')  
    return {  
        "accuracy": np.mean(scores),  
        "precision": np.mean(cross_val_score(modele, X, y, cv=cv, scoring='precision_weighted')),  
        "recall": np.mean(cross_val_score(modele, X, y, cv=cv, scoring='recall_weighted')),  
        "f1_score": np.mean(cross_val_score(modele, X, y, cv=cv, scoring='f1_weighted'))  
    }  

# Formulaire de saisie  
with st.form("Données du Match"):  
    st.subheader("📊 Saisie des Données du Match")  
    
    col1, col2 = st.columns(2)  
    
    with col1:  
        st.markdown("**Équipe A**")  
        st.session_state.data.update({  
            "score_rating_A": st.number_input("⭐ Score Rating", value=70, key="rating_A"),  
            "buts_par_match_A": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_A"),  
            "buts_concedes_par_match_A": st.number_input("🥅 Buts Concédés", value=1.0, key="concedes_A"),  
            "possession_moyenne_A": st.number_input("🎯 Possession Moyenne", value=55, key="possession_A"),  
            "expected_but_A": st.number_input("📊 Expected Goals (xG)", value=1.8, key="xG_A"),  
            "expected_concedes_A": st.number_input("📉 Expected Goals Against (xGA)", value=1.2, key="xGA_A"),  
            "tirs_cadres_A": st.number_input("🎯 Tirs Cadrés", value=120, key="tirs_A"),  
            "grandes_chances_A": st.number_input("🔥 Grandes Chances", value=25, key="chances_A"),  
            "passes_reussies_A": st.number_input("🔄 Passes Réussies", value=400, key="passes_A"),  
            "corners_A": st.number_input("🔄 Corners", value=60, key="corners_A"),  
            "interceptions_A": st.number_input("🛡️ Interceptions", value=50, key="interceptions_A"),  
            "tacles_reussis_A": st.number_input("🛡️ Tacles Réussis", value=40, key="tacles_A"),  
            "fautes_A": st.number_input("⚠️ Fautes", value=15, key="fautes_A"),  
            "cartons_jaunes_A": st.number_input("🟨 Cartons Jaunes", value=5, key="jaunes_A"),  
            "cartons_rouges_A": st.number_input("🟥 Cartons Rouges", value=1, key="rouges_A"),  
            "joueurs_cles_absents_A": st.number_input("🚑 Joueurs Clés Absents", value=0, key="absents_A"),  
            "motivation_A": st.number_input("💪 Motivation", value=3, key="motivation_A"),  
            "clean_sheets_gardien_A": st.number_input("🧤 Clean Sheets Gardien", value=2, key="clean_sheets_A"),  
            "ratio_tirs_arretes_A": st.number_input("🧤 Ratio Tirs Arrêtés", value=0.75, key="ratio_tirs_A"),  
            "victoires_domicile_A": st.number_input("🏠 Victoires Domicile", value=60, key="victoires_A"),  
            "passes_longues_A": st.number_input("🎯 Passes Longes", value=50, key="passes_longues_A"),  
            "dribbles_reussis_A": st.number_input("🔥 Dribbles Réussis", value=10, key="dribbles_A"),  
            "ratio_tirs_cadres_A": st.number_input("🎯 Ratio Tirs Cadrés", value=0.4, key="ratio_tirs_cadres_A"),  
            "grandes_chances_manquees_A": st.number_input("🔥 Grandes Chances Manquées", value=5, key="chances_manquees_A"),  
            "fautes_zones_dangereuses_A": st.number_input("⚠️ Fautes Zones Dangereuses", value=3, key="fautes_zones_A"),  
            "buts_corners_A": st.number_input("⚽ Buts Corners", value=2, key="buts_corners_A"),  
            "jours_repos_A": st.number_input("⏳ Jours de Repos", value=4, key="repos_A"),  
            "matchs_30_jours_A": st.number_input("📅 Matchs (30 jours)", value=8, key="matchs_A")  
        })  

    with col2:  
        st.markdown("**Équipe B**")  
        st.session_state.data.update({  
            "score_rating_B": st.number_input("⭐ Score Rating", value=65, key="rating_B"),  
            "buts_par_match_B": st.number_input("⚽ Buts Marqués", value=1.0, key="buts_B"),  
            "buts_concedes_par_match_B": st.number_input("🥅 Buts Concédés", value=1.5, key="concedes_B"),  
            "possession_moyenne_B": st.number_input("🎯 Possession Moyenne", value=45, key="possession_B"),  
            "expected_but_B": st.number_input("📊 Expected Goals (xG)", value=1.2, key="xG_B"),  
            "expected_concedes_B": st.number_input("📉 Expected Goals Against (xGA)", value=1.8, key="xGA_B"),  
            "tirs_cadres_B": st.number_input("🎯 Tirs Cadrés", value=100, key="tirs_B"),  
            "grandes_chances_B": st.number_input("🔥 Grandes Chances", value=20, key="chances_B"),  
            "passes_reussies_B": st.number_input("🔄 Passes Réussies", value=350, key="passes_B"),  
            "corners_B": st.number_input("🔄 Corners", value=50, key="corners_B"),  
            "interceptions_B": st.number_input("🛡️ Interceptions", value=40, key="interceptions_B"),  
            "tacles_reussis_B": st.number_input("🛡️ Tacles Réussis", value=35, key="tacles_B"),  
            "fautes_B": st.number_input("⚠️ Fautes", value=20, key="fautes_B"),  
            "cartons_jaunes_B": st.number_input("🟨 Cartons Jaunes", value=6, key="jaunes_B"),  
            "cartons_rouges_B": st.number_input("🟥 Cartons Rouges", value=2, key="rouges_B"),  
            "joueurs_cles_absents_B": st.number_input("🚑 Joueurs Clés Absents", value=0, key="absents_B"),  
            "motivation_B": st.number_input("💪 Motivation", value=3, key="motivation_B"),  
            "clean_sheets_gardien_B": st.number_input("🧤 Clean Sheets Gardien", value=1, key="clean_sheets_B"),  
            "ratio_tirs_arretes_B": st.number_input("🧤 Ratio Tirs Arrêtés", value=0.65, key="ratio_tirs_B"),  
            "victoires_exterieur_B": st.number_input("🏠 Victoires Extérieur", value=40, key="victoires_B"),  
            "passes_longues_B": st.number_input("🎯 Passes Longes", value=40, key="passes_longues_B"),  
            "dribbles_reussis_B": st.number_input("🔥 Dribbles Réussis", value=8, key="dribbles_B"),  
            "ratio_tirs_cadres_B": st.number_input("🎯 Ratio Tirs Cadrés", value=0.35, key="ratio_tirs_cadres_B"),  
            "grandes_chances_manqueuees_B": st.number_input("🔥 Grandes Chances Manquées", value=6, key="chances_manquees_B"),  
            "fautes_zones_dangereuses_B": st.number_input("⚠️ Fautes Zones Dangereuses", value=4, key="fautes_zones_B"),  
            "buts_corners_B": st.number_input("⚽ Buts Corners", value=1, key="buts_corners_B"),  
            "jours_repos_B": st.number_input("⏳ Jours de Repos", value=3, key="repos_B"),  
            "matchs_30_jours_B": st.number_input("📅 Matchs (30 jours)", value=9, key="matchs_B")  
        })  

    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Section d'analyse et de prédiction  
if submitted:  
    try:  
        # Calcul des lambda pour Poisson  
        lambda_A = calculer_lambda_attendus(st.session_state.data, 'A')  
        lambda_B = calculer_lambda_attendus(st.session_state.data, 'B')  
        
        # Résultats Poisson  
        resultats_poisson = predire_resultat_match_poisson(lambda_A, lambda_B)  
        
        # Affichage des résultats  
        st.subheader("🔮 Résultats de Prédiction")  
        
        # Prédiction des buts par équipe avec Poisson  
        st.markdown("### 📊 Prédiction des Buts par Équipe (Modèle de Poisson)")  
        col_buts_A, col_buts_B = st.columns(2)  
        with col_buts_A:  
            st.metric("⚽ Buts Attendus (Équipe A)", f"{lambda_A:.2f}")  
        with col_buts_B:  
            st.metric("⚽ Buts Attendus (Équipe B)", f"{lambda_B:.2f}")  
        
        # Probabilités de Poisson  
        st.markdown("### 📊 Probabilités de Poisson")  
        col_poisson_A, col_poisson_B, col_poisson_nul = st.columns(3)  
        with col_poisson_A:  
            st.metric("🏆 Victoire Équipe A", f"{resultats_poisson['Victoire Équipe A']:.2%}")  
        with col_poisson_B:  
            st.metric("🏆 Victoire Équipe B", f"{resultats_poisson['Victoire Équipe B']:.2%}")  
        with col_poisson_nul:  
            st.metric("🤝 Match Nul", f"{resultats_poisson['Match Nul']:.2%}")  
        
        # Comparaison des cotes  
        st.markdown("### 📈 Comparaison des Cotes Implicites vs Cotes Prédites")  
        cotes_implicites = {"Victoire Équipe A": 2.20, "Victoire Équipe B": 3.33, "Match Nul": 4.00}  
        cotes_predites = calculer_cotes_implicites(resultats_poisson)  
        
        col_cote_A, col_cote_B, col_cote_nul = st.columns(3)  
        with col_cote_A:  
            st.metric("🏆 Cote Implicite (Victoire A)", cotes_implicites["Victoire Équipe A"])  
            st.metric("🏆 Cote Prédite (Victoire A)", cotes_predites["Victoire Équipe A"])  
            st.write("**Value ?**", "✅ Oui" if detecter_value(cotes_implicites["Victoire Équipe A"], cotes_predites["Victoire Équipe A"]) else "❌ Non")  
        with col_cote_B:  
            st.metric("🏆 Cote Implicite (Victoire B)", cotes_implicites["Victoire Équipe B"])  
            st.metric("🏆 Cote Prédite (Victoire B)", cotes_predites["Victoire Équipe B"])  
            st.write("**Value ?**", "✅ Oui" if detecter_value(cotes_implicites["Victoire Équipe B"], cotes_predites["Victoire Équipe B"]) else "❌ Non")  
        with col_cote_nul:  
            st.metric("🤝 Cote Implicite (Match Nul)", cotes_implicites["Match Nul"])  
            st.metric("🤝 Cote Prédite (Match Nul)", cotes_predites["Match Nul"])  
            st.write("**Value ?**", "✅ Oui" if detecter_value(cotes_implicites["Match Nul"], cotes_predites["Match Nul"]) else "❌ Non")  
        
        # Modèles  
        modeles = {  
            "Régression Logistique": LogisticRegression(max_iter=1000),  
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  
        }  
        
        # Validation croisée et prédictions  
        st.markdown("### 🤖 Performance des Modèles")  
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
            st.markdown(f"#### {nom}")  
            col_accuracy, col_precision, col_recall, col_f1 = st.columns(4)  
            with col_accuracy:  
                st.metric("🎯 Précision Globale", f"{resultats_cv['accuracy']:.2%}",   
                         help="Pourcentage de prédictions correctes.")  
            with col_precision:  
                st.metric("🎯 Précision", f"{resultats_cv['precision']:.2%}",   
                         help="Pourcentage de victoires prédites qui étaient correctes.")  
            with col_recall:  
                st.metric("🎯 Rappel", f"{resultats_cv['recall']:.2%}",   
                         help="Pourcentage de victoires réelles correctement prédites.")  
            with col_f1:  
                st.metric("🎯 Score F1", f"{resultats_cv['f1_score']:.2%}",   
                         help="Moyenne équilibrée entre la précision et le rappel.")  
            
            # Prédiction finale  
            modele.fit(X_train, y_train)  
            proba = modele.predict_proba(X_lr if nom == "Régression Logistique" else X_rf)[0]  
            
            st.markdown("**📊 Prédictions**")  
            col_victoire_A, col_victoire_B, col_nul = st.columns(3)  
            with col_victoire_A:  
                st.metric("🏆 Victoire A", f"{proba[1]:.2%}")  
            with col_victoire_B:  
                st.metric("🏆 Victoire B", f"{proba[0]:.2%}")  
            with col_nul:  
                st.metric("🤝 Match Nul", f"{proba[2]:.2%}")  
        
    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
