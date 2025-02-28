import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.svm import SVC  
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  
from sklearn.linear_model import LogisticRegression  
from xgboost import XGBClassifier  
from sklearn.metrics import accuracy_score, classification_report  
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
home_xG = st.number_input("xG (Expected Goals) de l'équipe à domicile", min_value=0.0, value=1.5)  
away_xG = st.number_input("xG (Expected Goals) de l'équipe à l'extérieur", min_value=0.0, value=1.2)  
home_possession = st.number_input("Possession moyenne (%) de l'équipe à domicile", min_value=0.0, value=55.0)  
away_possession = st.number_input("Possession moyenne (%) de l'équipe à l'extérieur", min_value=0.0, value=45.0)  
home_shots = st.number_input("Nombre de tirs de l'équipe à domicile", min_value=0, value=10)  
away_shots = st.number_input("Nombre de tirs de l'équipe à l'extérieur", min_value=0, value=8)  
home_passes = st.number_input("Nombre de passes réussies de l'équipe à domicile", min_value=0, value=300)  
away_passes = st.number_input("Nombre de passes réussies de l'équipe à l'extérieur", min_value=0, value=250)  
home_fouls = st.number_input("Nombre de fautes de l'équipe à domicile", min_value=0, value=12)  
away_fouls = st.number_input("Nombre de fautes de l'équipe à l'extérieur", min_value=0, value=10)  
home_corners = st.number_input("Nombre de corners de l'équipe à domicile", min_value=0, value=5)  
away_corners = st.number_input("Nombre de corners de l'équipe à l'extérieur", min_value=0, value=4)  
home_yellow_cards = st.number_input("Cartons jaunes de l'équipe à domicile", min_value=0, value=2)  
away_yellow_cards = st.number_input("Cartons jaunes de l'équipe à l'extérieur", min_value=0, value=1)  

# Bouton pour prédire  
if st.button("Prédire l'issue du match"):  
    # Préparer les données pour la prédiction  
    new_data = np.array([[home_xG, away_xG, home_possession, away_possession,  
                          home_shots, away_shots, home_passes, away_passes,  
                          home_fouls, away_fouls, home_corners, away_corners,  
                          home_yellow_cards, away_yellow_cards]])  
    
    # Prédiction de l'issue du match  
    prediction_class = stacking_model.predict(new_data)  
    result = "Victoire de l'équipe à domicile" if prediction_class[0] == 1 else "Victoire de l'équipe à l'extérieur"  
    
    st.write("Résultat prédit :", result)  

    # Prédiction du nombre total de buts avec la méthode de Poisson  
    poisson_model = LogisticRegression()  # Utiliser un modèle de régression pour prédire les buts  
    poisson_model.fit(X_train, y_train_class)  # Entraîner le modèle sur les données d'entraînement  
    predicted_goals = poisson_model.predict(new_data)  # Prédire le nombre de buts  
    st.write("Nombre total de buts prédit :", predicted_goals[0])  
