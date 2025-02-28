# Importation des bibliothèques nécessaires  
import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split, cross_val_score  
from scipy.stats import poisson  
import matplotlib.pyplot as plt  

# Fonction pour entraîner les modèles  
def train_models():  
    # Exemple de données fictives pour l'entraînement  
    X = pd.DataFrame({  
        'home_goals': np.random.randint(0, 3, size=500),  
        'away_goals': np.random.randint(0, 3, size=500),  
        'home_xG': np.random.uniform(0, 2, size=500),  
        'away_xG': np.random.uniform(0, 2, size=500),  
        'home_encais': np.random.uniform(0, 2, size=500),  
        'away_encais': np.random.uniform(0, 2, size=500),  
        'home_victories': np.random.randint(0, 20, size=500),  
        'away_victories': np.random.randint(0, 20, size=500),  
        'home_goals_scored': np.random.randint(0, 50, size=500),  
        'away_goals_scored': np.random.randint(0, 50, size=500),  
        'home_xGA': np.random.uniform(0, 2, size=500),  
        'away_xGA': np.random.uniform(0, 2, size=500),  
        'home_tirs_par_match': np.random.randint(0, 30, size=500),  
        'away_tirs_par_match': np.random.randint(0, 30, size=500),  
        'home_passes_cles_par_match': np.random.randint(0, 50, size=500),  
        'away_passes_cles_par_match': np.random.randint(0, 50, size=500),  
        'home_tirs_cadres': np.random.randint(0, 15, size=500),  
        'away_tirs_cadres': np.random.randint(0, 15, size=500),  
        'home_tirs_concedes': np.random.randint(0, 30, size=500),  
        'away_tirs_concedes': np.random.randint(0, 30, size=500),  
        'home_duels_defensifs': np.random.randint(0, 100, size=500),  
        'away_duels_defensifs': np.random.randint(0, 100, size=500),  
        'home_possession': np.random.uniform(0, 100, size=500),  
        'away_possession': np.random.uniform(0, 100, size=500),  
        'home_passes_reussies': np.random.uniform(0, 100, size=500),  
        'away_passes_reussies': np.random.uniform(0, 100, size=500),  
        'home_touches_surface': np.random.randint(0, 300, size=500),  
        'away_touches_surface': np.random.randint(0, 300, size=500),  
        'home_forme_recente': np.random.randint(0, 15, size=500),  
        'away_forme_recente': np.random.randint(0, 15, size=500)  
    })  
    
    y = np.random.choice([0, 1, 2], size=500)  # 0: Domicile, 1: Nul, 2: Extérieur  

    # Division des données en ensembles d'entraînement et de test  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    # Entraînement des modèles  
    log_reg_model = LogisticRegression()  
    log_reg_model.fit(X_train, y_train)  

    rf_model = RandomForestClassifier()  
    rf_model.fit(X_train, y_train)  

    xgb_model = XGBClassifier()  
    xgb_model.fit(X_train, y_train)  

    svm_model = SVC(probability=True)  
    svm_model.fit(X_train, y_train)  

    return log_reg_model, rf_model, xgb_model, svm_model  

# Fonction pour calculer les probabilités implicites à partir des cotes  
def calculate_implied_prob(odds):  
    return 1 / odds  

# Fonction pour prédire les résultats avec le modèle de Poisson  
def poisson_prediction(goals_pred):  
    return [poisson.pmf(i, goals_pred) for i in range(6)]  # Prédire jusqu'à 5 buts  

# Fonction pour évaluer les modèles avec validation croisée  
def evaluate_models(X, y):  
    models = {  
        "Logistic Regression": LogisticRegression(),  
        "Random Forest": RandomForestClassifier(),  
        "XGBoost": XGBClassifier(),  
        "SVM": SVC(probability=True)  
    }  
    results = {}  
    for name, model in models.items():  
        scores = cross_val_score(model, X, y, cv=3)  
        results[name] = scores.mean()  
    return results  

# Configuration de l'application Streamlit  
st.set_page_config(page_title="Prédiction de Matchs de Football", layout="wide")  
# Interface utilisateur  
st.title("🏆 Analyse de Matchs de Football et Prédictions de Paris Sportifs")  

# Saisie des données des équipes  
st.header("📋 Saisie des données des équipes")  

# Création de deux colonnes pour les équipes  
col1, col2 = st.columns(2)  

# Équipe à domicile  
with col1:  
    st.subheader("Équipe à Domicile")  
    home_team = st.text_input("🏠 Nom de l'équipe à domicile", value="Équipe A")  
    home_goals = st.number_input("⚽ Moyenne de buts marqués par match (domicile)", min_value=0.0, max_value=5.0, value=2.5)  
    home_xG = st.number_input("📈 xG (Expected Goals) (domicile)", min_value=0.0, max_value=5.0, value=2.0)  
    home_encais = st.number_input("🚫 Moyenne de buts encaissés par match (domicile)", min_value=0.0, max_value=5.0, value=1.0)  
    home_victories = st.number_input("🏆 Nombre de victoires à domicile", min_value=0, value=5)  
    home_goals_scored = st.number_input("⚽ Nombre de buts marqués à domicile", min_value=0, value=15)  
    home_xGA = st.number_input("📉 xGA (Expected Goals Against) (domicile)", min_value=0.0, max_value=5.0, value=1.5)  
    home_tirs_par_match = st.number_input("🔫 Nombres de tirs par match (domicile)", min_value=0.0, max_value=30.0, value=15.0)  
    home_passes_cles_par_match = st.number_input("📊 Nombres de passes clés par match (domicile)", min_value=0.0, max_value=50.0, value=10.0)  
    home_tirs_cadres = st.number_input("🎯 Tirs cadrés par match (domicile)", min_value=0.0, max_value=15.0, value=5.0)  
    home_tirs_concedes = st.number_input("🚫 Nombres de tirs concédés par match (domicile)", min_value=0.0, max_value=30.0, value=8.0)  
    home_duels_defensifs = st.number_input("🤼 Duels défensifs gagnés (domicile)", min_value=0.0, max_value=100.0, value=60.0)  
    home_possession = st.number_input("📊 Possession moyenne (%) (domicile)", min_value=0.0, max_value=100.0, value=55.0)  
    home_passes_reussies = st.number_input("✅ Passes réussies (%) par match (domicile)", min_value=0.0, max_value=100.0, value=80.0)  
    home_touches_surface = st.number_input("⚽ Balles touchées dans la surface adverse par match (domicile)", min_value=0.0, max_value=300.0, value=20.0)  
    home_forme_recente = st.number_input("📈 Forme récente (points sur les 5 derniers matchs) (domicile)", min_value=0, max_value=15, value=10)  

# Équipe à l'extérieur  
with col2:  
    st.subheader("Équipe à Extérieur")  
    away_team = st.text_input("🏟️ Nom de l'équipe à l'extérieur", value="Équipe B")  
    away_goals = st.number_input("⚽ Moyenne de buts marqués par match (extérieur)", min_value=0.0, max_value=5.0, value=1.5)  
    away_xG = st.number_input("📈 xG (Expected Goals) (extérieur)", min_value=0.0, max_value=5.0, value=1.8)  
    away_encais = st.number_input("🚫 Moyenne de buts encaissés par match (extérieur)", min_value=0.0, max_value=5.0, value=2.0)  
    away_victories = st.number_input("🏆 Nombre de victoires à l'extérieur", min_value=0, value=3)  
    away_goals_scored = st.number_input("⚽ Nombre de buts marqués à l'extérieur", min_value=0, value=10)  
    away_xGA = st.number_input("📉 xGA (Expected Goals Against) (extérieur)", min_value=0.0, max_value=5.0, value=1.5)  
    away_tirs_par_match = st.number_input("🔫 Nombres de tirs par match (extérieur)", min_value=0.0, max_value=30.0, value=12.0)  
    away_passes_cles_par_match = st.number_input("📊 Nombres de passes clés par match (extérieur)", min_value=0.0, max_value=50.0, value=8.0)  
    away_tirs_cadres = st.number_input("🎯 Tirs cadrés par match (extérieur)", min_value=0.0, max_value=15.0, value=4.0)  
    away_tirs_concedes = st.number_input("🚫 Nombres de tirs concédés par match (extérieur)", min_value=0.0, max_value=30.0, value=10.0)  
    away_duels_defensifs = st.number_input("🤼 Duels défensifs gagnés (extérieur)", min_value=0.0, max_value=100.0, value=55.0)  
    away_possession = st.number_input("📊 Possession moyenne (%) (extérieur)", min_value=0.0, max_value=100.0, value=50.0)  
    away_passes_reussies = st.number_input("✅ Passes réussies (%) (extérieur)", min_value=0.0, max_value=100.0, value=75.0)  
    away_touches_surface = st.number_input("⚽ Balles touchées dans la surface adverse par match (extérieur)", min_value=0.0, max_value=300.0, value=15.0)  
    away_forme_recente = st.number_input("📈 Forme récente (points sur les 5 derniers matchs) (extérieur)", min_value=0, max_value=15, value=8)  

# Saisie des cotes des bookmakers (non utilisées par les modèles)  
st.header("💰 Cotes des Équipes")  
odds_home = st.number_input("🏠 Cote pour l'équipe à domicile", min_value=1.0, value=1.8)  
odds_away = st.number_input("🏟️ Cote pour l'équipe à l'extérieur", min_value=1.0, value=2.2)  

# Bouton pour déclencher les calculs  
if st.button("🔍 Prédire les résultats"):  
    with st.spinner('Calcul des résultats...'):  
        try:  
            # Validation des entrées utilisateur  
            if home_goals < 0 or away_goals < 0:  
                st.error("⚠️ Les moyennes de buts ne peuvent pas être négatives.")  
            else:  
                # Évaluation des modèles avec validation croisée K-Fold  
                X = pd.DataFrame({  
                    'home_goals': np.random.randint(0, 3, size=500),  
                    'away_goals': np.random.randint(0, 3, size=500),  
                    'home_xG': np.random.uniform(0, 2, size=500),  
                    'away_xG': np.random.uniform(0, 2, size=500),  
                    'home_encais': np.random.uniform(0, 2, size=500),  
                    'away_encais': np.random.uniform(0, 2, size=500),  
                    'home_victories': np.random.randint(0, 20, size=500),  
                    'away_victories': np.random.randint(0, 20, size=500),  
                    'home_goals_scored': np.random.randint(0, 50, size=500),  
                    'away_goals_scored': np.random.randint(0, 50, size=500),  
                    'home_xGA': np.random.uniform(0, 2, size=500),  
                    'away_xGA': np.random.uniform(0, 2, size=500),  
                    'home_tirs_par_match': np.random.randint(0, 30, size=500),  
                    'away_tirs_par_match': np.random.randint(0, 30, size=500),  
                    'home_passes_cles_par_match': np.random.randint(0, 50, size=500),  
                    'away_passes_cles_par_match': np.random.randint(0, 50, size=500),  
                    'home_tirs_cadres': np.random.randint(0, 15, size=500),  
                    'away_tirs_cadres': np.random.randint(0, 15, size=500),  
                    'home_tirs_concedes': np.random.randint(0, 30, size=500),  
                    'away_tirs_concedes': np.random.randint(0, 30, size=500),  
                    'home_duels_defensifs': np.random.randint(0, 100, size=500),  
                    'away_duels_defensifs': np.random.randint(0, 100, size=500),  
                    'home_possession': np.random.uniform(0, 100, size=500),  
                    'away_possession': np.random.uniform(0, 100, size=500),  
                    'home_passes_reussies': np.random.uniform(0, 100, size=500),  
                    'away_passes_reussies': np.random.uniform(0, 100, size=500),  
                    'home_touches_surface': np.random.randint(0, 300, size=500),  
                    'away_touches_surface': np.random.randint(0, 300, size=500),  
                    'home_forme_recente': np.random.randint(0, 15, size=500),  
                    'away_forme_recente': np.random.randint(0, 15, size=500)  
                })  
                y = np.random.choice([0, 1, 2], size=500)  # 0: Domicile, 1: Nul, 2: Extérieur  
                
                # Évaluation des modèles avec validation croisée K=3  
                results = evaluate_models(X, y)  
                st.write("📊 Résultats de la validation croisée K-Fold :", results)  

                # Calcul des buts prédit  
                home_goals_pred = home_goals + home_xG - away_encais  
                away_goals_pred = away_goals + away_xG - home_encais  

                # Calcul des probabilités avec le modèle de Poisson  
                home_probabilities = poisson_prediction(home_goals_pred)  
                away_probabilities = poisson_prediction(away_goals_pred)  

                # Formatage des résultats pour l'affichage  
                home_results = {i: home_probabilities[i] * 100 for i in range(len(home_probabilities))}  
                away_results = {i: away_probabilities[i] * 100 for i in range(len(away_probabilities))}  

                # Création d'un DataFrame pour les résultats de Poisson  
                poisson_results = pd.DataFrame({  
                    "Nombre de Buts": range(len(home_probabilities)),  
                    f"Probabilités {home_team} (%)": [f"{home_results[i]:.2f}" for i in range(len(home_results))],  
                    f"Probabilités {away_team} (%)": [f"{away_results[i]:.2f}" for i in range(len(away_results))]  
                })  

                # Affichage des résultats du modèle de Poisson  
                st.markdown("### Résultats du Modèle de Poisson")  
                st.dataframe(poisson_results, use_container_width=True)  

                # Détails sur chaque prédiction des modèles  
                st.markdown("### Détails des Prédictions des Modèles")  
                model_details = {  
                    "Modèle": ["Régression Logistique", "Random Forest", "XGBoost", "SVM"],  
                    "Probabilité Domicile (%)": [  
                        log_reg_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                       home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                       home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                       home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                       away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                       home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                       away_possession, home_passes_reussies, away_passes_reussies,  
                                                       home_touches_surface, away_touches_surface, home_forme_recente,  
                                                       away_forme_recente]])[0][0] * 100,  
                        rf_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                  home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                  home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                  home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                  away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                  home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                  away_possession, home_passes_reussies, away_passes_reussies,  
                                                  home_touches_surface, away_touches_surface, home_forme_recente,  
                                                  away_forme_recente]])[0][0] * 100,  
                        xgb_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                   home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                   home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                   home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                   away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                   home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                   away_possession, home_passes_reussies, away_passes_reussies,  
                                                   home_touches_surface, away_touches_surface, home_forme_recente,  
                                                   away_forme_recente]])[0][0] * 100,  
                        svm_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                   home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                   home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                   home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                   away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                   home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                   away_possession, home_passes_reussies, away_passes_reussies,  
                                                   home_touches_surface, away_touches_surface, home_forme_recente,  
                                                   away_forme_recente]])[0][0] * 100  
                    ],  
                    "Probabilité Nul (%)": [  
                        log_reg_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                       home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                       home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                       home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                       away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                       home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                       away_possession, home_passes_reussies, away_passes_reussies,  
                                                       home_touches_surface, away_touches_surface, home_forme_recente,  
                                                       away_forme_recente]])[0][1] * 100,  
                        rf_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                  home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                  home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                  home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                  away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                  home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                  away_passes_reussies, home_touches_surface, away_touches_surface, home_forme_recente, away_forme_recente]])[0][1] * 100,  
                        xgb_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                   home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                   home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                   home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                   away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                   home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                   away_possession, home_passes_reussies, away_passes_reussies,  
                                                   home_touches_surface, away_touches_surface, home_forme_recente,  
                                                   away_forme_recente]])[0][1] * 100,  
                        svm_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                   home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                   home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                   home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                   away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                   home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                   away_possession, home_passes_reussies, away_passes_reussies,  
                                                   home_touches_surface, away_touches_surface, home_forme_recente,  
                                                   away_forme_recente]])[0][1] * 100  
                    ],  
                    "Probabilité Extérieure (%)": [  
                        log_reg_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                       home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                       home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                       home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                       away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                       home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                       away_possession, home_passes_reussies, away_passes_reussies,  
                                                       home_touches_surface, away_touches_surface, home_forme_recente,  
                                                       away_forme_recente]])[0][2] * 100,  
                        rf_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                  home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                  home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                  home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                  away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                  home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                  away_possession, home_passes_reussies, away_passes_reussies,  
                                                  home_touches_surface, away_touches_surface, home_forme_recente,  
                                                  away_forme_recente]])[0][2] * 100,  
                        xgb_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                   home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                   home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                   home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                   away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                   home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                   away_possession, home_passes_reussies, away_passes_reussies,  
                                                   home_touches_surface, away_touches_surface, home_forme_recente,  
                                                   away_forme_recente]])[0][2] * 100,  
                        svm_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                   home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                   home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                   home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                   away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                   home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                   away_possession, home_passes_reussies, away_passes_reussies,  
                                                   home_touches_surface, away_touches_surface, home_forme_recente,  
                                                   away_forme_recente]])[0][2] * 100  
                    ]  
                }  
                model_details_df = pd.DataFrame(model_details)  
                st.dataframe(model_details_df, use_container_width=True)  

                # Comparaison des probabilités implicites et prédites  
                st.subheader("📊 Comparaison des Probabilités Implicites et Prédites")  
                implied_home_prob = calculate_implied_prob(odds_home)  
                implied_away_prob = calculate_implied_prob(odds_away)  
                implied_draw_prob = 1 - (implied_home_prob + implied_away_prob)  

                comparison_data = {  
                    "Type": ["Implicite Domicile", "Implicite Nul", "Implicite Extérieure",   
                             "Prédite Domicile", "Prédite Nul", "Prédite Extérieure"],  
                    "Probabilité (%)": [  
                        implied_home_prob * 100,  
                        implied_draw_prob * 100,  
                        implied_away_prob * 100,  
                        log_reg_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                       home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                       home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                       home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                       away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                       home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                       away_possession, home_passes_reussies, away_passes_reussies,  
                                                       home_touches_surface, away_touches_surface, home_forme_recente,  
                                                       away_forme_recente]])[0][0] * 100,  
                        log_reg_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                       home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                       home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                       home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                       away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                       home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                       away_possession, home_passes_reussies, away_passes_reussies,  
                                                       home_touches_surface, away_touches_surface, home_forme_recente,  
                                                       away_forme_recente]])[0][1] * 100,  
                        log_reg_model.predict_proba([[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais,  
                                                       home_victories, away_victories, home_goals_scored, away_goals_scored,  
                                                       home_xGA, away_xGA, home_tirs_par_match, away_tirs_par_match,  
                                                       home_passes_cles_par_match, away_passes_cles_par_match, home_tirs_cadres,  
                                                       away_tirs_cadres, home_tirs_concedes, away_tirs_concedes,  
                                                       home_duels_defensifs, away_duels_defensifs, home_possession,  
                                                       away_possession, home_passes_reussies, away_passes_reussies,  
                                                       home_touches_surface, away_touches_surface, home_forme_recente,  
                                                       away_forme_recente]])[0][2] * 100  
                    ]  
                }  
                comparison_df = pd.DataFrame(comparison_data)  
                st.dataframe(comparison_df, use_container_width=True)  

                # Affichage des graphiques des performances des équipes  
                st.subheader("📈 Graphiques des Performances des Équipes")  
                plot_team_performance({  
                    'home_goals_scored': home_goals_scored,  
                    'home_xG': home_xG,  
                    'home_encais': home_encais,  
                    'home_tirs_par_match': home_tirs_par_match,  
                    'home_passes_cles_par_match': home_passes_cles_par_match,  
                    'home_tirs_cadres': home_tirs_cadres,  
                    'home_possession': home_possession  
                }, {  
                    'away_goals_scored': away_goals_scored,  
                    'away_xG': away_xG,  
                    'away_encais': away_encais,  
                    'away_tirs_par_match': away_tirs_par_match,  
                    'away_passes_cles_par_match': away_passes_cles_par_match,  
                    'away_tirs_cadres': away_tirs_cadres,  
                    'away_possession': away_possession  
                })  

        except Exception as e:  
            st.error(f"Une erreur s'est produite : {e}")  

# Fin de l'application  
if __name__ == "__main__":  
    st.write("Merci d'utiliser notre application de prédiction de matchs de football !")  
                                                  
