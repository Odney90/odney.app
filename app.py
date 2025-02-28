import streamlit as st  
import pandas as pd  
import numpy as np  
import altair as alt  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.svm import SVC  
from sklearn.model_selection import cross_val_score, train_test_split  
from sklearn.metrics import accuracy_score  
from scipy.stats import poisson  

# Initialisation de session_state si non existant  
if 'trained_models' not in st.session_state:  
    st.session_state.trained_models = None  
if 'model_scores' not in st.session_state:  
    st.session_state.model_scores = None  
if 'predictions' not in st.session_state:  
    st.session_state.predictions = None  
if 'poisson_results' not in st.session_state:  
    st.session_state.poisson_results = None  
if 'value_bets' not in st.session_state:  
    st.session_state.value_bets = None  
if 'history' not in st.session_state:  
    st.session_state.history = []  

@st.cache_resource  
def train_models(X_train, y_train):  
    models = {  
        "Logistic Regression": LogisticRegression(max_iter=1000),  
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),  
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),  
        "SVM": SVC(probability=True)  
    }  
    return {name: model.fit(X_train, y_train) for name, model in models.items()}  

def poisson_prediction(goals_pred):  
    return np.array([poisson.pmf(i, goals_pred) for i in range(6)])  

def evaluate_models(X, y):  
    # Vérifiez la taille de l'échantillon  
    if len(X) < 3:  # Nombre minimal d'échantillons pour cv=3  
        st.warning("Pas assez d'échantillons pour effectuer une validation croisée. Utilisation d'une validation simple.")  
        return evaluate_models_simple(X, y)  

    # Vérifiez la distribution des classes  
    unique_classes, counts = np.unique(y, return_counts=True)  
    class_distribution = dict(zip(unique_classes, counts))  
    st.write("Distribution des classes :", class_distribution)  

    if len(unique_classes) < 2:  
        st.error("Les données doivent contenir au moins deux classes différentes.")  
        return None  

    # Vérifiez les valeurs manquantes  
    if X.isnull().values.any():  
        st.error("Les données contiennent des valeurs manquantes.")  
        return None  

    # Vérifiez que toutes les colonnes sont de type numérique  
    if not np.issubdtype(X.dtypes, np.number):  
        st.error("Toutes les colonnes de X doivent être numériques.")  
        return None  

    models = {  
        "Logistic Regression": LogisticRegression(max_iter=1000),  
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),  
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),  
        "SVM": SVC(probability=True)  
    }  
    
    return {name: cross_val_score(model, X, y, cv=3).mean() for name, model in models.items()}  

def evaluate_models_simple(X, y):  
    # Vérifiez que X et y ne sont pas vides  
    if len(X) == 0 or len(y) == 0:  
        st.error("Les données d'entrée sont vides.")  
        return None  

    # Vérifiez que X est un DataFrame ou un tableau numérique  
    if not isinstance(X, (pd.DataFrame, np.ndarray)):  
        st.error("Les données d'entrée doivent être un DataFrame ou un tableau numpy.")  
        return None  

    # Vérifiez que y est un tableau numérique  
    if not isinstance(y, (pd.Series, np.ndarray)):  
        st.error("Les étiquettes doivent être un tableau numpy ou une série pandas.")  
        return None  

    # Vérifiez les valeurs manquantes dans X et y  
    if isinstance(X, pd.DataFrame) and X.isnull().values.any():  
        st.error("Les données d'entrée contiennent des valeurs manquantes.")  
        return None  
    if isinstance(y, pd.Series) and y.isnull().values.any():  
        st.error("Les étiquettes contiennent des valeurs manquantes.")  
        return None  

    # Vérifiez que y contient au moins deux classes  
    unique_classes = np.unique(y)  
    if len(unique_classes) < 2:  
        st.error("Les étiquettes doivent contenir au moins deux classes.")  
        return None  

    # Divisez les données en ensembles d'entraînement et de test  
    try:  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    except Exception as e:  
        st.error(f"Erreur lors de la division des données : {e}")  
        return None  

    models = {  
        "Logistic Regression": LogisticRegression(max_iter=1000),  
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),  
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),  
        "SVM": SVC(probability=True)  
    }  

    scores = {}  
    for name, model in models.items():  
        try:  
            model.fit(X_train, y_train)  
            y_pred = model.predict(X_test)  
            scores[name] = accuracy_score(y_test, y_pred)  
        except Exception as e:  
            st.error(f"Erreur lors de l'entraînement du modèle {name}: {e}")  
            scores[name] = None  

    return scores  

def calculate_implied_prob(odds):  
    return 1 / odds  

def detect_value_bet(predicted_prob, implied_prob, threshold=0.05):  
    return predicted_prob > (implied_prob + threshold)  

def predict_match_result(poisson_home, poisson_away):  
    home_goals = np.arange(len(poisson_home))  
    away_goals = np.arange(len(poisson_away))  
    results = []  

    for h in home_goals:  
        for a in away_goals:  
            results.append((h, a, poisson_home[h] * poisson_away[a]))  

    results_df = pd.DataFrame(results, columns=['Home Goals', 'Away Goals', 'Probability'])  
    return results_df  

def validate_input(data):  
    for key, value in data.items():  
        if isinstance(value, (int, float)) and value < 0:  
            st.error(f"La valeur de {key} ne peut pas être négative.")  
            return False  
    return True  

st.set_page_config(page_title="Prédiction de Matchs", layout="wide")  
st.title("🏆 Analyse et Prédictions Football")  

st.header("📋 Saisie des données des équipes")  
col1, col2 = st.columns(2)  

with col1:  
    st.subheader("Équipe à Domicile")  
    home_team = st.text_input("🏠 Nom de l'équipe à domicile", value="Équipe A")  
    home_goals = st.number_input("⚽ Moyenne de buts marqués par match", min_value=0.0, max_value=5.0, value=1.5)  
    home_xG = st.number_input("📈 xG (Expected Goals)", min_value=0.0, max_value=5.0, value=1.8)  
    home_encais = st.number_input("🚫 Moyenne de buts encaissés", min_value=0.0, max_value=5.0, value=1.2)  
    home_possession = st.number_input("📊 Possession moyenne (%)", min_value=0.0, max_value=100.0, value=55.0)  
    home_tirs_par_match = st.number_input("🔫 Nombre de tirs par match", min_value=0, max_value=30, value=15)  
    home_passes_cles_par_match = st.number_input("📊 Nombre de passes clés par match", min_value=0, max_value=50, value=10)  
    home_tirs_cadres = st.number_input("🎯 Tirs cadrés par match", min_value=0, max_value=15, value=5)  
    home_touches_surface = st.number_input("⚽ Balles touchées dans la surface adverse", min_value=0, max_value=300, value=20)  
    home_duels_defensifs = st.number_input("🤼 Duels défensifs gagnés", min_value=0, max_value=100, value=60)  
    home_passes_reussies = st.number_input("✅ Passes réussies (%)", min_value=0.0, max_value=100.0, value=80.0)  
    home_forme_recente = st.number_input("📈 Forme récente (points sur les 5 derniers matchs)", min_value=0, max_value=15, value=10)  
    home_victories = st.number_input("🏆 Nombre de victoires à domicile", min_value=0, max_value=20, value=5)  
    home_fautes_commises = st.number_input("⚠️ Fautes commises par match", min_value=0, max_value=30, value=12)  
    home_interceptions = st.number_input("🛑 Interceptions par match", min_value=0, max_value=30, value=8)  

with col2:  
    st.subheader("Équipe à Extérieur")  
    away_team = st.text_input("🏟️ Nom de l'équipe à l'extérieur", value="Équipe B")  
    away_goals = st.number_input("⚽ Moyenne de buts marqués par match", min_value=0.0, max_value=5.0, value=1.3)  
    away_xG = st.number_input("📈 xG (Expected Goals)", min_value=0.0, max_value=5.0, value=1.5)  
    away_encais = st.number_input("🚫 Moyenne de buts encaissés", min_value=0.0, max_value=5.0, value=1.7)  
    away_possession = st.number_input("📊 Possession moyenne (%)", min_value=0.0, max_value=100.0, value=50.0)  
    away_tirs_par_match = st.number_input("🔫 Nombre de tirs par match", min_value=0, max_value=30, value=12)  
    away_passes_cles_par_match = st.number_input("📊 Nombre de passes clés par match", min_value=0, max_value=50, value=8)  
    away_tirs_cadres = st.number_input("🎯 Tirs cadrés par match", min_value=0, max_value=15, value=4)  
    away_touches_surface = st.number_input("⚽ Balles touchées dans la surface adverse", min_value=0, max_value=300, value=15)  
    away_duels_defensifs = st.number_input("🤼 Duels défensifs gagnés", min_value=0, max_value=100, value=55)  
    away_passes_reussies = st.number_input("✅ Passes réussies (%)", min_value=0.0, max_value=100.0, value=75.0)  
    away_forme_recente = st.number_input("📈 Forme récente (points sur les 5 derniers matchs)", min_value=0, max_value=15, value=8)  
    away_victories = st.number_input("🏆 Nombre de victoires à l'extérieur", min_value=0, max_value=20, value=3)  
    away_fautes_commises = st.number_input("⚠️ Fautes commises par match", min_value=0, max_value=30, value=14)  
    away_interceptions = st.number_input("🛑 Interceptions par match", min_value=0, max_value=30, value=10)  

# Bouton pour prédire les résultats  
if st.button("🔍 Prédire les résultats"):  
    # Rassemblez les données dans un DataFrame  
    home_data = {  
        'goals': home_goals,  
        'xG': home_xG,  
        'encais': home_encais,  
        'possession': home_possession,  
        'tirs_par_match': home_tirs_par_match,  
        'passes_cles_par_match': home_passes_cles_par_match,  
        'tirs_cadres': home_tirs_cadres,  
        'touches_surface': home_touches_surface,  
        'duels_defensifs': home_duels_defensifs,  
        'passes_reussies': home_passes_reussies,  
        'forme_recente': home_forme_recente,  
        'victories': home_victories,  
        'fautes_commises': home_fautes_commises,  
        'interceptions': home_interceptions  
    }  
    
    away_data = {  
        'goals': away_goals,  
        'xG': away_xG,  
        'encais': away_encais,  
        'possession': away_possession,  
        'tirs_par_match': away_tirs_par_match,  
        'passes_cles_par_match': away_passes_cles_par_match,  
        'tirs_cadres': away_tirs_cadres,  
        'touches_surface': away_touches_surface,  
        'duels_defensifs': away_duels_defensifs,  
        'passes_reussies': away_passes_reussies,  
        'forme_recente': away_forme_recente,  
        'victories': away_victories,  
        'fautes_commises': away_fautes_commises,  
        'interceptions': away_interceptions  
    }  

    # Convertir les données en DataFrame pour l'entraînement  
    X = pd.DataFrame([home_data, away_data])  

    # Vérifiez les valeurs manquantes  
    if X.isnull().values.any():  
        st.error("Les données contiennent des valeurs manquantes.")  
    else:  
        # Générer les étiquettes y avec une probabilité d'obtenir deux classes  
        y = np.random.choice([0, 1], size=X.shape[0], p=[0.5, 0.5])  # Assurez-vous que les classes 0 et 1 sont présentes  

        # Vérifiez la forme des données  
        unique_classes = np.unique(y)  
        expected_classes = [0, 1]  # Classes attendues  
        if not all(cls in expected_classes for cls in unique_classes):  
            st.error(f"Classes inattendues dans y: {unique_classes}. Attendu: {expected_classes}.")  
            st.stop()  

        # Entraînez les modèles  
        try:  
            trained_models = train_models(X, y)  
        except Exception as e:  
            st.error(f"Erreur lors de l'entraînement des modèles: {e}")  
            st.stop()  

        # Évaluez les modèles  
        model_scores = evaluate_models(X, y)  

        # Affichez les résultats  
        if model_scores is not None:  
            st.write("Scores des modèles :", model_scores)  

            # Prédictions avec le modèle de Poisson  
            poisson_results_home = poisson_prediction(home_goals)  # Exemple pour l'équipe à domicile  
            poisson_results_away = poisson_prediction(away_goals)  # Exemple pour l'équipe à l'extérieur  
            st.write("Résultats de Poisson pour l'équipe à domicile :", poisson_results_home)  
            st.write("Résultats de Poisson pour l'équipe à l'extérieur :", poisson_results_away)  

            # Détection des paris de valeur (exemple)  
            implied_prob_home = calculate_implied_prob(1.8)  # Remplacez par la cote réelle  
            implied_prob_away = calculate_implied_prob(2.2)  # Remplacez par la cote réelle  
            value_bet_home = detect_value_bet(poisson_results_home[1], implied_prob_home)  # Exemple pour 1 but  
            value_bet_away = detect_value_bet(poisson_results_away[1], implied_prob_away)  # Exemple pour 1 but  

            st.write("Paris de valeur pour l'équipe à domicile :", value_bet_home)  
            st.write("Paris de valeur pour l'équipe à l'extérieur :", value_bet_away)  

            # Prédiction des résultats du match  
            match_results = predict_match_result(poisson_results_home, poisson_results_away)  
            st.subheader("📊 Prédictions des Résultats du Match")  
            st.write(match_results)  

            # Visualisation des résultats de Poisson  
            st.subheader("📊 Visualisation des Résultats de Poisson")  
            poisson_df_home = pd.DataFrame({  
                'Buts': range(6),  
                'Probabilité': poisson_results_home  
            })  
            poisson_df_away = pd.DataFrame({  
                'Buts': range(6),  
                'Probabilité': poisson_results_away  
            })  

            # Graphique pour l'équipe à domicile  
            st.write(f"Distribution des buts pour {home_team}")  
            chart_home = alt.Chart(poisson_df_home).mark_bar().encode(  
                x='Buts:O',  
                y='Probabilité:Q',  
                tooltip=['Buts', 'Probabilité']  
            ).properties(title=f"Distribution des buts pour {home_team}")  
            st.altair
            # Graphique pour l'équipe à l'extérieur  
            st.write(f"Distribution des buts pour {away_team}")  
            chart_away = alt.Chart(poisson_df_away).mark_bar().encode(  
                x='Buts:O',  
                y='Probabilité:Q',  
                tooltip=['Buts', 'Probabilité']  
            ).properties(title=f"Distribution des buts pour {away_team}")  
            st.altair_chart(chart_away, use_container_width=True)  

            # Ajouter l'historique des prédictions  
            st.session_state.history.append({  
                'Home Team': home_team,  
                'Away Team': away_team,  
                'Predictions': match_results  
            })  

            # Afficher l'historique des prédictions  
            st.subheader("📝 Historique des Prédictions")  
            history_df = pd.DataFrame(st.session_state.history)  
            st.write(history_df)  

            # Option pour télécharger les résultats  
            csv = history_df.to_csv(index=False)  
            st.download_button("📥 Télécharger l'historique des prédictions", csv, "predictions_history.csv", "text/csv")  
