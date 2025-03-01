import streamlit as st  
import numpy as np  
import pandas as pd  
import altair as alt  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
import math  

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

# Fonction pour prédire les buts  
def predict_goals(xG_domicile, xG_exterieur, max_goals=5):  
    home_probs = [poisson_prob(xG_domicile, i) for i in range(max_goals + 1)]  
    away_probs = [poisson_prob(xG_exterieur, i) for i in range(max_goals + 1)]  
    
    win_home = 0  
    win_away = 0  
    draw = 0  

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

# Initialisation de session_state  
if 'model_trained' not in st.session_state:  
    st.session_state.model_trained = False  
    st.session_state.rf_model = None  
    st.session_state.xgb_model = None  

# Entrée des données utilisateur  
st.sidebar.header("🔢 Entrer les données du match")  
user_input = {}  
for column in df.columns[:-1]:  # Exclure la colonne 'resultat'  
    user_input[column] = st.sidebar.number_input(column.replace('_', ' ').capitalize(), value=float(df[column].mean()))  

# Entraînement des modèles si ce n'est pas déjà fait  
if not st.session_state.model_trained:  
    X = df.drop('resultat', axis=1)  
    y = df['resultat']  
    st.session_state.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
    st.session_state.rf_model.fit(X, y)  
    st.session_state.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  
    st.session_state.xgb_model.fit(X, y)  
    st.session_state.model_trained = True  

# Bouton de prédiction  
if st.sidebar.button("🔮 Prédire le Résultat"):  
    input_data = np.array(list(user_input.values())).reshape(1, -1)  
    
    # Vérification des valeurs d'entrée  
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):  
        st.error("Les données saisies contiennent des valeurs manquantes ou infinies.")  
    else:  
        proba_rf = st.session_state.rf_model.predict_proba(input_data)[0][1]  
        proba_xgb = st.session_state.xgb_model.predict_proba(input_data)[0][1]  

        result_rf = "Victoire Domicile" if proba_rf > 0.5 else "Victoire Extérieure"  
        result_xgb = "Victoire Domicile" if proba_xgb > 0.5 else "Victoire Extérieure"  

        st.write(f"🔮 Résultat (Random Forest): {result_rf} avec {proba_rf*100:.2f}% de confiance")  
        st.write(f"🔮 Résultat (XGBoost): {result_xgb} avec {proba_xgb*100:.2f}% de confiance")  

        # Prédiction des buts  
        xG_domicile = user_input['xG_domicile']  
        xG_exterieur = user_input['xG_exterieur']  
        win_home, win_away, draw = predict_goals(xG_domicile, xG_exterieur)  

        st.write(f"🏠 Probabilité de victoire Domicile : {win_home:.2%}")  
        st.write(f"🏟️ Probabilité de victoire Extérieure : {win_away:.2%}")  
        st.write(f"🤝 Probabilité de match nul : {draw:.2%}")  

        # Graphique Altair  
        chart_data = pd.DataFrame({"Modèle": ["RandomForest", "XGBoost"], "Probabilité Victoire Domicile": [proba_rf, proba_xgb]})  
        chart = alt.Chart(chart_data).mark_bar().encode(x="Modèle", y="Probabilité Victoire Domicile", color="Modèle")  
        st.altair_chart(chart, use_container_width=True)  
