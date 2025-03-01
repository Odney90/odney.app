import streamlit as st  
import numpy as np  
import pandas as pd  
import altair as alt  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  
from xgboost import XGBClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import math  
import random  

# Fonction pour calculer les probabilités de buts avec la méthode de Poisson  
def poisson_prob(lam, k):  
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)  

# Fonction pour prédire les buts avec Poisson en prenant en compte plus de variables  
def predict_goals(xG_home, xG_away, conversion_home, conversion_away, pressing_home, pressing_away, max_goals=5):  
    adj_xG_home = xG_home * conversion_home * (1 + pressing_home / 50)  
    adj_xG_away = xG_away * conversion_away * (1 + pressing_away / 50)  
    
    home_probs = [poisson_prob(adj_xG_home, i) for i in range(max_goals + 1)]  
    away_probs = [poisson_prob(adj_xG_away, i) for i in range(max_goals + 1)]  
    
    win_home, win_away, draw = 0, 0, 0  
    for home_goals in range(max_goals + 1):  
        for away_goals in range(max_goals + 1):  
            if home_goals > away_goals:  
                win_home += home_probs[home_goals] * away_probs[away_goals]  
            elif home_goals < away_goals:  
                win_away += home_probs[home_goals] * away_probs[away_goals]  
            else:  
                draw += home_probs[home_goals] * away_probs[away_goals]  
    
    return win_home, win_away, draw  

# Interface Streamlit  
st.title("⚽ Prédiction de Match de Football")  

# Organisation en colonnes  
col1, col2 = st.columns(2)  

# Entrée des données équipe A (Domicile)  
with col1:  
    st.header("🏠 Équipe A")  
    xG_home = st.number_input("xG (Équipe A) 🏆", value=1.5)  
    shots_on_target_home = st.number_input("Tirs Cadrés (Équipe A) 🎯", value=5)  
    conversion_home = st.number_input("Conversion des Tirs (%) (Équipe A) 🎯", value=30)  
    touches_in_box_home = st.number_input("Touches Surface (Équipe A) ⚽", value=15)  
    key_passes_home = st.number_input("Passes Clés (Équipe A) 🔑", value=5)  
    interceptions_home = st.number_input("Interceptions (Équipe A) 🛡️", value=8)  
    defensive_duels_home = st.number_input("Duels Défensifs Gagnés (Équipe A) 💪", value=18)  
    xGA_home = st.number_input("xGA (Équipe A) 🚫", value=1.2)  
    goalkeeper_saves_home = st.number_input("Arrêts du Gardien (Équipe A) 🧤", value=4)  
    recent_form_home = st.number_input("Forme Récente (Points sur 5 derniers matchs) (Équipe A) 📊", value=10)  
days_rest_home = st.number_input("Jours de repos avant le match (Équipe A) ⏳", value=5)  

# Entrée des données équipe B (Extérieur)  
with col2:  
    st.header("🏟️ Équipe B")  
    xG_away = st.number_input("xG (Équipe B) 🏆", value=1.0)  
    shots_on_target_away = st.number_input("Tirs Cadrés (Équipe B) 🎯", value=3)  
    conversion_away = st.number_input("Conversion des Tirs (%) (Équipe B) 🎯", value=25)  
    touches_in_box_away = st.number_input("Touches Surface (Équipe B) ⚽", value=10)  
    key_passes_away = st.number_input("Passes Clés (Équipe B) 🔑", value=4)  
    interceptions_away = st.number_input("Interceptions (Équipe B) 🛡️", value=7)  
    defensive_duels_away = st.number_input("Duels Défensifs Gagnés (Équipe B) 💪", value=15)  
    xGA_away = st.number_input("xGA (Équipe B) 🚫", value=1.5)  
    goalkeeper_saves_away = st.number_input("Arrêts du Gardien (Équipe B) 🧤", value=5)  
    recent_form_away = st.number_input("Forme Récente (Points sur 5 derniers matchs) (Équipe B) 📊", value=8)  
days_rest_away = st.number_input("Jours de repos avant le match (Équipe B) ⏳", value=4)  

# Bouton de prédiction  
if st.button("🔮 Prédire le Résultat"):  
    # Prédiction avec Poisson en intégrant plus de variables  
    win_home, win_away, draw = predict_goals(xG_home, xG_away, conversion_home/100, conversion_away/100, key_passes_home/100, key_passes_away/100)  
    
    st.write(f"🏠 Probabilité de victoire Équipe A : {win_home:.2%}")  
    st.write(f"🏟️ Probabilité de victoire Équipe B : {win_away:.2%}")  
    st.write(f"🤝 Probabilité de match nul : {draw:.2%}")  
    
    # Mise à jour de la régression logistique avec toutes les variables  
    logistic_model = LogisticRegression()  
    X_train = np.random.rand(100, 14)  
    y_train = np.random.randint(0, 2, 100)  
    logistic_model.fit(X_train, y_train)  
    input_features = np.array([[xG_home, shots_on_target_home, conversion_home, touches_in_box_home, key_passes_home, interceptions_home, defensive_duels_home, xGA_home, goalkeeper_saves_home, recent_form_home, days_rest_home, xG_away, shots_on_target_away, conversion_away, touches_in_box_away, key_passes_away, interceptions_away, defensive_duels_away, xGA_away, goalkeeper_saves_away, recent_form_away, days_rest_away]])  
    proba_logistic = logistic_model.predict_proba(input_features)[0][1]  
    result_logistic = "Victoire Équipe A" if proba_logistic > 0.5 else "Victoire Équipe B"  
    
    st.write(f"📊 Prédiction Régression Logistique : {result_logistic} avec {proba_logistic*100:.2f}% de confiance")
