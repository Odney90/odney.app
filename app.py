import streamlit as st  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  

# Charger le modèle (à remplacer par ton modèle entraîné)  
model = RandomForestClassifier(n_estimators=100, random_state=42)  

# Fonction pour prédire le résultat  
def predict_match(teamA_stats, teamB_stats):  
    # Créer un DataFrame avec les statistiques des deux équipes  
    data = {**teamA_stats, **teamB_stats}  
    df = pd.DataFrame([data])  
    
    # Faire la prédiction  
    prediction = model.predict(df)[0]  
    return prediction  

# Interface Streamlit  
st.title("⚽ Prédiction de Matchs de Football")  

# Formulaire pour saisir les statistiques  
teamA_goals = st.number_input("Buts de l'Équipe A", min_value=0)  
teamB_goals = st.number_input("Buts de l'Équipe B", min_value=0)  
teamA_possession = st.number_input("Possession de l'Équipe A (%)", min_value=0, max_value=100)  
teamB_possession = st.number_input("Possession de l'Équipe B (%)", min_value=0, max_value=100)  
teamA_xg = st.number_input("xG de l'Équipe A", min_value=0.0)  
teamB_xg = st.number_input("xG de l'Équipe B", min_value=0.0)  

# Bouton pour prédire  
if st.button("Prédire le Résultat"):  
    teamA_stats = {  
        'teamA_goals': teamA_goals,  
        'teamA_possession': teamA_possession,  
        'teamA_xg': teamA_xg  
    }  
    teamB_stats = {  
        'teamB_goals': teamB_goals,  
        'teamB_possession': teamB_possession,  
        'teamB_xg': teamB_xg  
    }  
    
    # Faire la prédiction  
    prediction = predict_match(teamA_stats, teamB_stats)  
    result = {0: "Victoire de l'Équipe A", 1: "Victoire de l'Équipe B", 2: "Match nul"}[prediction]  
    
    # Afficher le résultat  
    st.success(f"Résultat prédit : {result}")
