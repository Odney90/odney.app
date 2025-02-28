import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  
from sklearn.linear_model import LogisticRegression  
from xgboost import XGBClassifier  
import streamlit as st  

# Fixer la graine pour la reproductibilité  
np.random.seed(42)  

# Générer des données fictives pour l'entraînement  
n_samples = 1000  
data = {  
    'home_goals': np.random.poisson(lam=1.5, size=n_samples),  
    'away_goals': np.random.poisson(lam=1.2, size=n_samples),  
    'home_xG': np.random.normal(loc=1.5, scale=0.5, size=n_samples),  
    'away_xG': np.random.normal(loc=1.2, scale=0.5, size=n_samples),  
    'home_possession': np.random.uniform(low=45, high=65, size=n_samples),  
    'away_possession': np.random.uniform(low=35, high=55, size=n_samples),  
    'home_shots': np.random.poisson(lam=10, size=n_samples),  
    'away_shots': np.random.poisson(lam=8, size=n_samples),  
    'home_passes': np.random.poisson(lam=300, size=n_samples),  
    'away_passes': np.random.poisson(lam=250, size=n_samples),  
    'home_fouls': np.random.poisson(lam=12, size=n_samples),  
    'away_fouls': np.random.poisson(lam=10, size=n_samples),  
    'home_corners': np.random.poisson(lam=5, size=n_samples),  
    'away_corners': np.random.poisson(lam=4, size=n_samples),  
    'home_yellow_cards': np.random.poisson(lam=2, size=n_samples),  
    'away_yellow_cards': np.random.poisson(lam=1.5, size=n_samples),  
    'home_red_cards': np.random.poisson(lam=0.2, size=n_samples),  
    'away_red_cards': np.random.poisson(lam=0.1, size=n_samples),  
    'home_defense': np.random.uniform(low=0.5, high=1.0, size=n_samples),  
    'away_defense': np.random.uniform(low=0.5, high=1.0, size=n_samples),  
    'home_attack': np.random.uniform(low=0.5, high=1.0, size=n_samples),  
    'away_attack': np.random.uniform(low=0.5, high=1.0, size=n_samples),  
    'home_injuries': np.random.randint(0, 5, size=n_samples),  
    'away_injuries': np.random.randint(0, 5, size=n_samples),  
    'home_recent_form': np.random.uniform(low=0, high=1, size=n_samples),  
    'away_recent_form': np.random.uniform(low=0, high=1, size=n_samples),  
    'home_fans': np.random.randint(1000, 50000, size=n_samples),  
    'away_fans': np.random.randint(1000, 50000, size=n_samples),  
}  

# Créer le DataFrame  
df = pd.DataFrame(data)  

# Créer une étiquette pour l'issue du match  
df['result'] = (df['home_goals'] > df['away_goals']).astype(int)  

# Séparer les caractéristiques et les étiquettes  
X = df.drop(['result', 'home_goals', 'away_goals'], axis=1)  
y_classification = df['result']  

# Diviser les données en ensembles d'entraînement et de test  
X_train, X_test, y_train_class, y_test_class = train_test_split(  
    X, y_classification, test_size=0.2, random_state=42  
)  

# Modèles de base  
base_models = [  
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),  
    ('svm', SVC(probability=True, random_state=42)),  
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))  
]  

# Modèle de niveau supérieur  
meta_model = LogisticRegression()  

# Créer le modèle de stacking  
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)  

# Entraîner le modèle de stacking  
stacking_model.fit(X_train, y_train_class)  

# Interface utilisateur Streamlit  
st.title("Prédiction de Matchs de Football")  

# Saisie des données des équipes  
col1, col2 = st.columns(2)  

with col1:  
    st.header("Équipe à Domicile")  
    home_xG = st.number_input("xG (Expected Goals)", min_value=0.0, value=1.5)  
    home_possession = st.number_input("Possession moyenne (%)", min_value=0.0, value=55.0)  
    home_shots = st.number_input("Nombre de tirs", min_value=0, value=10)  
    home_passes = st.number_input("Nombre de passes réussies", min_value=0, value=300)  
    home_fouls = st.number_input("Nombre de fautes", min_value=0, value=12)  
    home_corners = st.number_input("Nombre de corners", min_value=0, value=5)  
    home_yellow_cards = st.number_input("Cartons jaunes", min_value=0, value=2)  
    home_red_cards = st.number_input("Cartons rouges", min_value=0, value=0)  
    home_defense = st.number_input("Qualité de la défense (0.0 à 1.0)", min_value=0.0, max_value=1.0, value=0.8)  
    home_attack = st.number_input("Qualité de l'attaque (0.0 à 1.0)", min_value=0.0, max_value=1.0, value=0.8)  
    home_injuries = st.number_input("Nombre de blessures", min_value=0, value=0)  
    home_recent_form = st.number_input("Forme récente (0.0 à 1.0)", min_value=0.0, max_value=1.0, value=0.5)  
    home_fans = st.number_input("Nombre de fans", min_value=1000, value=10000)  

with col2:  
    st.header("Équipe à l'Extérieur")  
    away_xG = st.number_input("xG (Expected Goals)", min_value=0.0, value=1.2)  
    away_possession = st.number_input("Possession moyenne (%)", min_value=0.0, value=45.0)  
    away_shots = st.number_input("Nombre de tirs", min_value=0, value=8)  
    away_passes = st.number_input("Nombre de passes réussies", min_value=0, value=250)  
    away_fouls = st.number_input("Nombre de fautes", min_value=0, value=10)  
    away_corners = st.number_input("Nombre de corners", min_value=0, value=4)  
    away_yellow_cards = st.number_input("Cartons jaunes", min_value=0, value=1)  
    away_red_cards = st.number_input("Cartons rouges", min_value=0, value=0)  
    away_defense = st.number_input("Qualité de la défense (0.0 à 1.0)", min_value=0.0, max_value=1.0, value=0.7)  
    away_attack = st.number_input("Qualité de l'attaque (0.0 à 1.0)", min_value=0.0, max_value=1.0, value=0.7)  
    away_injuries = st.number_input("Nombre de blessures", min_value=0, value=0)  
    away_recent_form = st.number_input("Forme récente (0.0 à 1.0)", min_value=0.0, max_value=1.0, value=0.5)  
    away_fans = st.number_input("Nombre de fans", min_value=1000, value=10000)  

# Saisie des cotes des bookmakers  
st.header("Cotes des Bookmakers")  
home_odds = st.number_input("Cote pour l'équipe à domicile", min_value=1.0, value=2.0)  
away_odds = st.number_input("Cote pour l'équipe à l'extérieur", min_value=1.0, value=2.0)  

# Initialiser les résultats dans le session state  
if 'results' not in st.session_state:  
    st.session_state.results = None  

# Bouton pour prédire  
if st.button("Prédire l'issue du match"):  
    # Préparer les données pour la prédiction  
    new_data = np.array([[home_xG, away_xG, home_possession, away_possession,  
                          home_shots, away_shots, home_passes, away_passes,  
                          home_fouls, away_fouls, home_corners, away_corners,  
                          home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards,  
                          home_defense, away_defense, home_attack, away_attack,  
                          home_injuries, away_injuries, home_recent_form, away_recent_form,  
                          home_fans, away_fans]])  

    # Prédiction de l'issue du match  
    prediction_class = stacking_model.predict(new_data)  
    prediction_prob = stacking_model.predict_proba(new_data)[0]  

    # Calculer les probabilités  
    home_win_prob = prediction_prob[1]  # Probabilité de victoire de l'équipe à domicile  
    away_win_prob = prediction_prob[0]  # Probabilité de victoire de l'équipe à l'extérieur  

    # Affichage des résultats  
    result = "Victoire de l'équipe à domicile" if prediction_class[0] == 1 else "Victoire de l'équipe à l'extérieur"  
    st.session_state.results = {  
        'result': result,  
        'home_win_prob': home_win_prob,  
        'away_win_prob': away_win_prob,  
        'home_odds': home_odds,  
        'away_odds': away_odds  
    }  

# Afficher les résultats si disponibles  
if st.session_state.results:  
    st.write("Résultat prédit :", st.session_state.results['result'])  
    st.write(f"Probabilité de victoire de l'équipe à domicile : {st.session_state.results['home_win_prob']:.2%}")  
    st.write(f"Probabilité de victoire de l'équipe à l'extérieur : {st.session_state.results['away_win_prob']:.2%}")  

    # Détection de value bet  
    st.header("Détection de Value Bet")  
    if st.session_state.results['home_win_prob'] > (1 / st.session_state.results['home_odds']):  
        st.write("Paris de valeur détecté pour l'équipe à domicile.")  
    else:  
        st.write("Pas de paris de valeur pour l'équipe à domicile.")  

    if st.session_state.results['away_win_prob'] > (1 / st.session_state.results['away_odds']):  
        st.write("Paris de valeur détecté pour l'équipe à l'extérieur.")  
    else:  
        st.write("Pas de paris de valeur pour l'équipe à l'extérieur.")  
