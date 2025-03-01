import streamlit as st  
import numpy as np  
import pandas as pd  
import altair as alt  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import math  
import random  

# Exemple de données d'entraînement ajustées  
data = {  
    'xG_domicile': [1.5, 2.0, 1.2, 1.8, 2.5],  
    'tirs_cadres_domicile': [5, 6, 4, 7, 8],  
    'touches_surface_domicile': [15, 20, 10, 18, 25],  
    'xGA_domicile': [1.0, 1.5, 1.3, 1.2, 2.0],  
    'interceptions_domicile': [10, 12, 8, 9, 15],  
    'duels_defensifs_domicile': [20, 25, 15, 18, 30],  
    'possession_domicile': [55, 60, 50, 58, 62],  
    'passes_cles_domicile': [3, 4, 2, 5, 6],  
    'forme_recente_domicile': [10, 12, 8, 9, 15],  
    'buts_domicile': [2, 3, 1, 2, 4],  
    'buts_encais_domicile': [1, 2, 1, 0, 3],  
    'blessures_domicile': [1, 0, 2, 1, 0],  
    'xG_exterieur': [1.0, 1.5, 1.3, 1.2, 2.0],  
    'tirs_cadres_exterieur': [3, 4, 5, 2, 6],  
    'touches_surface_exterieur': [10, 15, 12, 8, 20],  
    'xGA_exterieur': [1.2, 1.0, 1.5, 1.3, 2.1],  
    'interceptions_exterieur': [8, 10, 9, 7, 12],  
    'duels_defensifs_exterieur': [15, 20, 18, 12, 25],  
    'possession_exterieur': [45, 40, 50, 42, 38],  
    'passes_cles_exterieur': [2, 3, 4, 1, 5],  
    'forme_recente_exterieur': [8, 7, 9, 6, 10],  
    'buts_exterieur': [1, 2, 1, 0, 3],  
    'buts_encais_exterieur': [2, 1, 3, 1, 2],  
    'blessures_exterieur': [0, 1, 1, 0, 2],  
    'resultat': [1, 1, 0, 1, 1]  # 1 = Victoire Domicile, 0 = Victoire Extérieure  
}  

df = pd.DataFrame(data)  

# Fonction pour calculer les probabilités de buts avec la méthode de Poisson  
def poisson_prob(lam, k):  
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)  

# Fonction pour prédire les buts avec Poisson  
def predict_goals(xG_domicile, xG_exterieur, max_goals=5):  
    home_probs = [poisson_prob(xG_domicile, i) for i in range(max_goals + 1)]  
    away_probs = [poisson_prob(xG_exterieur, i) for i in range(max_goals + 1)]  
    
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

st.sidebar.header("🔢 Entrer les données du match")  
user_input = {}  
for column in df.columns[:-1]:  
    user_input[column] = st.sidebar.number_input(column.replace('_', ' ').capitalize(), value=float(df[column].mean()))  

# Entrée des cotes des bookmakers  
st.sidebar.subheader("📊 Cotes des bookmakers")  
cote_domicile = st.sidebar.number_input("Cote Victoire Domicile", value=2.0)  
cote_nul = st.sidebar.number_input("Cote Match Nul", value=3.2)  
cote_exterieur = st.sidebar.number_input("Cote Victoire Extérieure", value=3.5)  

# Probabilités implicites des cotes  
p_imp_domicile = 1 / cote_domicile  
p_imp_nul = 1 / cote_nul  
p_imp_exterieur = 1 / cote_exterieur  

if st.sidebar.button("🔮 Prédire le Résultat"):  
    xG_domicile = user_input['xG_domicile']  
    xG_exterieur = user_input['xG_exterieur']  
    win_home, win_away, draw = predict_goals(xG_domicile, xG_exterieur)  
    
    st.write(f"🏠 Probabilité de victoire Domicile : {win_home:.2%}")  
    st.write(f"🏟️ Probabilité de victoire Extérieure : {win_away:.2%}")  
    st.write(f"🤝 Probabilité de match nul : {draw:.2%}")  

    # Comparaison avec les cotes  
    st.subheader("📉 Comparaison avec les cotes des bookmakers")  
    df_cotes = pd.DataFrame({  
        "Résultat": ["Domicile", "Nul", "Extérieur"],  
        "Probabilité Modèle": [win_home, draw, win_away],  
        "Probabilité Implicite": [p_imp_domicile, p_imp_nul, p_imp_exterieur]  
    })  
    st.dataframe(df_cotes)  

    chart = alt.Chart(df_cotes).mark_bar().encode(x="Résultat", y="Probabilité Modèle", color="Résultat")  
    st.altair_chart(chart, use_container_width=True)
