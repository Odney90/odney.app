import streamlit as st  
import pandas as pd  
import numpy as np  
from scipy.special import factorial  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import cross_val_score  
from io import BytesIO  
from docx import Document  

# Fonction pour prédire avec le modèle Poisson  
def poisson_prediction(goals):  
    probabilities = []  
    for k in range(6):  # Calculer pour 0 à 5 buts  
        prob = np.exp(-goals) * (goals ** k) / factorial(k)  
        probabilities.append(prob)  
    return probabilities  

# Fonction pour créer un document Word avec les résultats  
def create_doc(results):  
    doc = Document()  
    doc.add_heading('Analyse de Matchs de Football et Prédictions de Paris Sportifs', level=1)  

    # Ajout des données des équipes  
    doc.add_heading('Données des Équipes', level=2)  
    doc.add_paragraph(f"Équipe Domicile: {results['Équipe Domicile']}")  
    doc.add_paragraph(f"Équipe Extérieure: {results['Équipe Extérieure']}")  
    doc.add_paragraph(f"Buts Prédit Domicile: {results['Buts Prédit Domicile']:.2f}")  
    doc.add_paragraph(f"Buts Prédit Extérieur: {results['Buts Prédit Extérieur']:.2f}")  

    # Ajout des probabilités des modèles  
    doc.add_heading('Probabilités des Modèles', level=2)  
    doc.add_paragraph(f"Probabilité Domicile ou Nul: {results['Probabilité Domicile']:.2f}")  
    doc.add_paragraph(f"Probabilité Nul ou Victoire Extérieure: {results['Probabilité Nul']:.2f}")  
    doc.add_paragraph(f"Probabilité Domicile ou Victoire Extérieure: {results['Probabilité Extérieure']:.2f}")  

    # Enregistrement du document  
    buffer = BytesIO()  
    doc.save(buffer)  
    buffer.seek(0)  
    return buffer  

# Fonction pour entraîner et prédire avec les modèles  
@st.cache_resource  
def train_models():  
    np.random.seed(42)  
    data = pd.DataFrame({  
        'home_goals': np.random.randint(0, 3, size=1000),  
        'away_goals': np.random.randint(0, 3, size=1000),  
        'home_xG': np.random.uniform(0, 2, size=1000),  
        'away_xG': np.random.uniform(0, 2, size=1000),  
        'home_encais': np.random.uniform(0, 2, size=1000),  
        'away_encais': np.random.uniform(0, 2, size=1000),  
        'result': np.random.choice([0, 1, 2], size=1000)  # 0: D, 1: N, 2: E  
    })  

    # Création des cibles pour le paris double chance  
    data['double_chance'] = np.where(data['result'] == 0, 0,  
                                      np.where(data['result'] == 1, 1,  
                                               2))  

    X = data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_encais', 'away_encais']]  
    y = data['double_chance']  

    # Modèle de régression logistique  
    log_reg = LogisticRegression(max_iter=100, C=1.0, solver='lbfgs')  
    log_reg.fit(X, y)  

    # Modèle Random Forest  
    rf = RandomForestClassifier(n_estimators=20, max_depth=3, min_samples_split=2)  
    rf.fit(X, y)  

    # Modèle XGBoost  
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=20, max_depth=3, learning_rate=0.1)  
    xgb.fit(X, y)  

    return log_reg, rf, xgb  

# Vérifier si les modèles sont déjà chargés dans l'état de session  
if 'models' not in st.session_state:  
    st.session_state.models = train_models()  

# Assurez-vous que les modèles sont bien chargés  
if st.session_state.models is not None:  
    log_reg_model, rf_model, xgb_model = st.session_state.models  
else:  
    st.error("Les modèles n'ont pas pu être chargés.")  

# Fonction pour évaluer les modèles avec validation croisée K-Fold  
def evaluate_models(X, y):  
    models = {  
        "Régression Logistique": LogisticRegression(max_iter=100, C=1.0, solver='lbfgs'),  
        "Random Forest": RandomForestClassifier(n_estimators=20, max_depth=3, min_samples_split=2),  
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=20, max_depth=3, learning_rate=0.1)  
    }  
    
    results = {}  
    for name, model in models.items():  
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')  # K=3 pour réduire le temps  
        results[name] = scores.mean()  # Moyenne des scores de validation croisée  
    
    return results  

# Évaluation des modèles après l'entraînement  
if st.session_state.models is not None:  
    X = pd.DataFrame({  
        'home_goals': np.random.randint(0, 3, size=1000),  
        'away_goals': np.random.randint(0, 3, size=1000),  
        'home_xG': np.random.uniform(0, 2, size=1000),  
        'away_xG': np.random.uniform(0, 2, size=1000),  
        'home_encais': np.random.uniform(0, 2, size=1000),  
        'away_encais': np.random.uniform(0, 2, size=1000)  
    })  
    y = np.random.choice([0, 1, 2], size=1000)  
    results = evaluate_models(X, y)  
    st.write("Résultats de la validation croisée K-Fold :", results)  

# Interface utilisateur  
st.title("🏆 Analyse de Matchs de Football et Prédictions de Paris Sportifs")  

# Saisie des données des équipes  
st.header("Saisie des données des équipes")  

# Utilisation d'accordéons pour les statistiques des équipes  
with st.expander("Statistiques de l'équipe à domicile", expanded=True):  
    home_team = st.text_input("Nom de l'équipe à domicile", value="Équipe A")  
    home_goals = st.slider("Moyenne de buts marqués par match (domicile)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)  
    home_xG = st.slider("xG (Expected Goals) (domicile)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)  
    home_encais = st.slider("Moyenne de buts encaissés par match (domicile)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)  
    home_xGA = st.slider("xGA (Expected Goals Against) (domicile)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)  
    home_tirs_par_match = st.slider("Nombres de tirs par match (domicile)", min_value=0, max_value=30, value=15)  
    home_passes_menant_a_tir = st.slider("Nombres de passes menant à un tir (domicile)", min_value=0, max_value=30, value=10)  
    home_tirs_cadres = st.slider("Tirs cadrés par match (domicile)", min_value=0, max_value=15, value=5)  
    home_tirs_concedes = st.slider("Nombres de tirs concédés par match (domicile)", min_value=0, max_value=30, value=8)  
    home_duels_defensifs = st.slider("Duels défensifs gagnés (%) (domicile)", min_value=0.0, max_value=100.0, value=60.0)  
    home_possession = st.slider("Possession moyenne (%) (domicile)", min_value=0.0, max_value=100.0, value=55.0)  
    home_passes_reussies = st.slider("Passes réussies (%) (domicile)", min_value=0.0, max_value=100.0, value=80.0)  
    home_touches_surface = st.slider("Balles touchées dans la surface adverse (domicile)", min_value=0, max_value=50, value=20)  
    home_forme_recente = st.slider("Forme récente (points sur les 5 derniers matchs) (domicile)", min_value=0, max_value=15, value=10)  

with st.expander("Statistiques de l'équipe à l'extérieur", expanded=True):  
    away_team = st.text_input("Nom de l'équipe à l'extérieur", value="Équipe B")  
    away_goals = st.slider("Moyenne de buts marqués par match (extérieur)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)  
    away_xG = st.slider("xG (Expected Goals) (extérieur)", min_value=0.0, max_value=5.0, value=1.8, step=0.1)  
    away_encais = st.slider("Moyenne de buts encaissés par match (extérieur)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)  
    away_xGA = st.slider("xGA (Expected Goals Against) (extérieur)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)  
    away_tirs_par_match = st.slider("Nombres de tirs par match (extérieur)", min_value=0, max_value=30, value=12)  
    away_passes_menant_a_tir = st.slider("Nombres de passes menant à un tir (extérieur)", min_value=0, max_value=30, value=8)  
    away_tirs_cadres = st.slider("Tirs cadrés par match (extérieur)", min_value=0, max_value=15, value=4)  
    away_tirs_concedes = st.slider("Nombres de tirs concédés par match (extérieur)", min_value=0, max_value=30, value=10)  
    away_duels_defensifs = st.slider("Duels défensifs gagnés (%) (extérieur)", min_value=0.0, max_value=100.0, value=55.0)  
    away_possession = st.slider("Possession moyenne (%) (extérieur)", min_value=0.0, max_value=100.0, value=50.0)  
    away_passes_reussies = st.slider("Passes réussies (%) (extérieur)", min_value=0.0, max_value=100.0, value=75.0)  
    away_touches_surface = st.slider("Balles touchées dans la surface adverse (extérieur)", min_value=0, max_value=50, value=15)  
    away_forme_recente = st.slider("Forme récente (points sur les 5 derniers matchs) (extérieur)", min_value=0, max_value=15, value=8)  

# Ajout des nouvelles variables de comparaison  
st.header("Variables de Comparaison")  
with st.expander("Statistiques de Victoire", expanded=True):  
    home_wins = st.number_input("Victoires à domicile", min_value=0, value=5)  
    away_wins = st.number_input("Victoires à l'extérieur", min_value=0, value=3)  

with st.expander("Confrontations Directes", expanded=True):  
    head_to_head = st.number_input("Résultats des confrontations directes (Domicile - Extérieur)", min_value=-10, max_value=10, value=0)  

with st.expander("Buts", expanded=True):  
    home_goals_total = st.number_input("Nombre total de buts à domicile", min_value=0, value=15)  
    away_goals_total = st.number_input("Nombre total de buts à l'extérieur", min_value=0, value=10)  

# Variable pour indiquer quelle équipe reçoit  
receiving_team = st.selectbox("Équipe qui reçoit", options=["Domicile", "Extérieur"], index=0)  
is_home = 1 if receiving_team == "Domicile" else 0  

# Saisie des cotes des bookmakers  
st.header("Cotes des Équipes")  
odds_home = st.number_input("Cote pour l'équipe à domicile", min_value=1.0, value=1.8)  
odds_away = st.number_input("Cote pour l'équipe à l'extérieur", min_value=1.0, value=2.2)  

# Calcul des probabilités implicites  
def calculate_implied_prob(odds):  
    return 1 / odds  

# Prédictions  
if st.button("🔍 Prédire les résultats"):  
    with st.spinner('Calcul des résultats...'):  
        # Calcul des buts prédit  
        home_goals_pred = home_goals + home_xG - away_encais  
        away_goals_pred = away_goals + away_xG - home_encais  
    
        # Calcul des probabilités avec le modèle de Poisson  
        home_probabilities = poisson_prediction(home_goals_pred)  
        away_probabilities = poisson_prediction(away_goals_pred)  

        # Formatage des résultats pour l'affichage  
        home_results = ", ".join([f"{i} but {home_probabilities[i] * 100:.1f}%" for i in range(len(home_probabilities))])  
        away_results = ", ".join([f"{i} but {away_probabilities[i] * 100:.1f}%" for i in range(len(away_probabilities))])  

        # Calcul des probabilités implicites  
        implied_home_prob = calculate_implied_prob(odds_home)  
        implied_away_prob = calculate_implied_prob(odds_away)  
        implied_draw_prob = 1 - (implied_home_prob + implied_away_prob)  

        # Prédictions avec les modèles  
        input_data = [[home_goals_pred, away_goals_pred, home_xG, away_xG, home_encais, away_encais, home_wins, away_wins, head_to_head, home_goals_total, away_goals_total, is_home]]  
    
        try:  
            log_reg_prob = log_reg_model.predict_proba(input_data)[0]  
            rf_prob = rf_model.predict_proba(input_data)[0]  
            xgb_prob = xgb_model.predict_proba(input_data)[0]  
        except Exception as e:  
            st.error(f"Erreur lors de la prédiction : {e}")  
            log_reg_prob, rf_prob, xgb_prob = None, None, None  

        # Affichage des résultats  
        st.subheader("📊 Résultats des Prédictions")  

        # Tableau pour le modèle de Poisson  
        poisson_results = pd.DataFrame({  
            "Équipe": [home_team, away_team],  
            "Buts Prédit": [home_results, away_results]  
        })  

        st.markdown("### Résultats du Modèle de Poisson")  
        st.dataframe(poisson_results, use_container_width=True)  

        # Détails sur chaque prédiction des modèles  
        st.markdown("### Détails des Prédictions des Modèles")  
        model_details = {  
            "Modèle": ["Régression Logistique", "Random Forest", "XGBoost"],  
            "Probabilité Domicile ou Nul (%)": [  
                log_reg_prob[0] * 100 if log_reg_prob is not None else 0,  
                rf_prob[0] * 100 if rf_prob is not None else 0,  
                xgb_prob[0] * 100 if xgb_prob is not None else 0  
            ],  
            "Probabilité Nul ou Victoire Extérieure (%)": [  
                log_reg_prob[1] * 100 if log_reg_prob is not None else 0,  
                rf_prob[1] * 100 if rf_prob is not None else 0,  
                xgb_prob[1] * 100 if xgb_prob is not None else 0  
            ],  
            "Probabilité Domicile ou Victoire Extérieure (%)": [  
                log_reg_prob[2] * 100 if log_reg_prob is not None else 0,  
                rf_prob[2] * 100 if rf_prob is not None else 0,  
                xgb_prob[2] * 100 if xgb_prob is not None else 0  
            ]  
        }  
        model_details_df = pd.DataFrame(model_details)  
        st.dataframe(model_details_df, use_container_width=True)  

        # Comparaison des probabilités implicites et prédites  
        st.subheader("📊 Comparaison des Probabilités Implicites et Prédites")  
        comparison_data = {  
            "Type": ["Implicite Domicile ou Nul", "Implicite Nul ou Extérieure", "Implicite Domicile ou Extérieure",   
                     "Prédite Domicile ou Nul", "Prédite Nul ou Victoire Extérieure", "Prédite Domicile ou Victoire Extérieure"],  
            "Probabilité (%)": [  
                implied_home_prob * 100,  
                implied_draw_prob * 100,  
                implied_away_prob * 100,  
                log_reg_prob[0] * 100 if log_reg_prob is not None else None,  
                log_reg_prob[1] * 100 if log_reg_prob is not None else None,  
                                log_reg_prob[2] * 100 if log_reg_prob is not None else None  
            ]  
        }  
        comparison_df = pd.DataFrame(comparison_data)  
        st.dataframe(comparison_df, use_container_width=True)  

        # Graphique des performances des modèles  
        st.subheader("📈 Comparaison des Modèles")  
        model_comparison_data = {  
            "Modèle": ["Régression Logistique", "Random Forest", "XGBoost"],  
            "Probabilité Domicile ou Nul (%)": [log_reg_prob[0] * 100 if log_reg_prob is not None else 0,  
                                                 rf_prob[0] * 100 if rf_prob is not None else 0,  
                                                 xgb_prob[0] * 100 if xgb_prob is not None else 0],  
            "Probabilité Nul ou Victoire Extérieure (%)": [log_reg_prob[1] * 100 if log_reg_prob is not None else 0,  
                                                             rf_prob[1] * 100 if rf_prob is not None else 0,  
                                                             xgb_prob[1] * 100 if xgb_prob is not None else 0],  
            "Probabilité Domicile ou Victoire Extérieure (%)": [log_reg_prob[2] * 100 if log_reg_prob is not None else 0,  
                                                                  rf_prob[2] * 100 if rf_prob is not None else 0,  
                                                                  xgb_prob[2] * 100 if xgb_prob is not None else 0]  
        }  
        model_comparison_df = pd.DataFrame(model_comparison_data)  
        st.dataframe(model_comparison_df, use_container_width=True)  

        # Option pour télécharger le document Word avec les résultats  
        results = {  
            'Équipe Domicile': home_team,  
            'Équipe Extérieure': away_team,  
            'Buts Prédit Domicile': home_results,  
            'Buts Prédit Extérieur': away_results,  
            'Probabilité Domicile ou Nul': log_reg_prob[0] * 100 if log_reg_prob is not None else 0,  
            'Probabilité Nul ou Victoire Extérieure': log_reg_prob[1] * 100 if log_reg_prob is not None else 0,  
            'Probabilité Domicile ou Victoire Extérieure': log_reg_prob[2] * 100 if log_reg_prob is not None else 0,  
        }  
        doc_buffer = create_doc(results)  
        st.download_button("Télécharger le document", doc_buffer, "resultats_match.docx")  

# Fin de l'application
