import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import math  
import altair as alt  

# Exemple de données d'entraînement (remplacez ceci par vos données réelles)  
data = {  
    'xG_home': [1.5, 2.0, 1.2, 1.8, 2.5],  
    'shots_on_target_home': [5, 6, 4, 7, 8],  
    'touches_in_box_home': [15, 20, 10, 18, 25],  
    'xGA_home': [1.0, 1.5, 1.3, 1.2, 2.0],  
    'interceptions_home': [10, 12, 8, 9, 15],  
    'defensive_duels_home': [20, 25, 15, 18, 30],  
    'possession_home': [55, 60, 50, 58, 62],  
    'key_passes_home': [3, 4, 2, 5, 6],  
    'recent_form_home': [10, 12, 8, 9, 15],  
    'home_goals': [2, 3, 1, 2, 4],  
    'home_goals_against': [1, 2, 1, 0, 3],  
    'injuries_home': [1, 0, 2, 1, 0],  
    
    'xG_away': [1.0, 1.5, 1.3, 1.2, 2.0],  
    'shots_on_target_away': [3, 4, 5, 2, 6],  
    'touches_in_box_away': [10, 15, 12, 8, 20],  
    'xGA_away': [1.2, 1.0, 1.5, 1.3, 2.1],  
    'interceptions_away': [8, 10, 9, 7, 12],  
    'defensive_duels_away': [15, 20, 18, 12, 25],  
    'possession_away': [45, 40, 50, 42, 38],  
    'key_passes_away': [2, 3, 4, 1, 5],  
    'recent_form_away': [8, 7, 9, 6, 10],  
    'away_goals': [1, 2, 1, 0, 3],  
    'away_goals_against': [2, 1, 3, 1, 2],  
    'injuries_away': [0, 1, 1, 0, 2],  
    'result': [1, 1, 0, 1, 1]  # 1 = Victoire Domicile, 0 = Victoire Extérieure  
}  

df = pd.DataFrame(data)  

# Fonction pour entraîner le modèle et prédire le résultat  
def train_and_predict(X_train, y_train, X_test):  
    # Modèle Random Forest  
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)  
    model_rf.fit(X_train, y_train)  
    prediction_rf = model_rf.predict(X_test)[0]  
    
    # Modèle XGBoost  
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  
    model_xgb.fit(X_train, y_train)  
    prediction_xgb = model_xgb.predict(X_test)[0]  
    
    return prediction_rf, prediction_xgb  

# Interface Streamlit  
st.title("⚽ Prédiction de Match de Football")  

# Entrée des données de l'utilisateur pour l'équipe domicile  
st.header("🏠 Données de l'Équipe Domicile")  
xG_home = st.number_input("xG (Domicile) 🏆", value=1.5)  
shots_on_target_home = st.number_input("Tirs Cadrés (Domicile) 🎯", value=5)  
touches_in_box_home = st.number_input("Balles Touchées dans la Surface Adverse (Domicile) ⚽", value=15)  
xGA_home = st.number_input("xGA (Domicile) 🚫", value=1.0)  
interceptions_home = st.number_input("Interceptions (Domicile) 🛡️", value=10)  
defensive_duels_home = st.number_input("Duels Défensifs Gagnés (Domicile) 💪", value=20)  
possession_home = st.number_input("Possession (%) (Domicile) 📈", value=55)  
key_passes_home = st.number_input("Passes Clés (Domicile) 🔑", value=3)  
recent_form_home = st.number_input("Forme Récente (Domicile) 📅", value=10)  
home_goals = st.number_input("Buts Marqués à Domicile ⚽", value=2)  
home_goals_against = st.number_input("Buts Encaissés à Domicile 🚫", value=1)  
injuries_home = st.number_input("Blessures (Domicile) 🚑", value=1)  

# Entrée des données de l'utilisateur pour l'équipe extérieure  
st.header("🏟️ Données de l'Équipe Extérieure")  
xG_away = st.number_input("xG (Extérieur) 🏆", value=1.0)  
shots_on_target_away = st.number_input("Tirs Cadrés (Extérieur) 🎯", value=3)  
touches_in_box_away = st.number_input("Balles Touchées dans la Surface Adverse (Extérieur) ⚽", value=10)  
xGA_away = st.number_input("xGA (Extérieur) 🚫", value=1.5)  
interceptions_away = st.number_input("Interceptions (Extérieur) 🛡️", value=8)  
defensive_duels_away = st.number_input("Duels Défensifs Gagnés (Extérieur) 💪", value=15)  
possession_away = st.number_input("Possession (%) (Extérieur) 📈", value=45)  
key_passes_away = st.number_input("Passes Clés (Extérieur) 🔑", value=2)  
recent_form_away = st.number_input("Forme Récente (Extérieur) 📅", value=8)  
away_goals = st.number_input("Buts Marqués à l'Extérieur ⚽", value=1)  
away_goals_against = st.number_input("Buts Encaissés à l'Extérieur 🚫", value=1)  
injuries_away = st.number_input("Blessures (Extérieur) �🚑", value=0)  

# Bouton de prédiction  
if st.button("🔮 Prédire le Résultat"):  
    # Préparation des données pour la prédiction  
    input_data = np.array([[xG_home, shots_on_target_home, touches_in_box_home, xGA_home,  
                            interceptions_home, defensive_duels_home, possession_home,  
                            key_passes_home, recent_form_home, home_goals, home_goals_against,  
                            injuries_home, xG_away, shots_on_target_away, touches_in_box_away,  
                            xGA_away, interceptions_away, defensive_duels_away, possession_away,  
                            key_passes_away, recent_form_away, away_goals, away_goals_against,  
                            injuries_away]])  

    # Séparation des données d'entraînement  
    X = df.drop('result', axis=1)  
    y = df['result']  

    # Prédiction  
    prediction_rf, prediction_xgb = train_and_predict(X, y, input_data)  

    # Affichage des résultats  
    st.write(f"🔮 Prédiction du résultat (Random Forest) : {'Victoire Domicile' if prediction_rf == 1 else 'Victoire Extérieure'}")  
    st.write(f"🔮 Prédiction du résultat (XGBoost) : {'Victoire Domicile' if prediction_xgb == 1 else 'Victoire Extérieure'}")  
