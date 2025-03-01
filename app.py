import streamlit as st  
import numpy as np  
import pandas as pd  
import altair as alt  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  

# Exemple de données d'entraînement  
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

# Séparation en train/test  
X = df.drop('result', axis=1)  
y = df['result']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Entraînement des modèles  
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X_train, y_train)  
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  
xgb_model.fit(X_train, y_train)  

# Précision des modèles  
y_pred_rf = rf_model.predict(X_test)  
y_pred_xgb = xgb_model.predict(X_test)  
accuracy_rf = accuracy_score(y_test, y_pred_rf)  
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)  

# Interface Streamlit  
st.title("⚽ Prédiction de Match de Football")  
st.subheader(f"🎯 Précision des Modèles: RandomForest {accuracy_rf*100:.2f}%, XGBoost {accuracy_xgb*100:.2f}%")  

# Entrée des données utilisateur  
st.sidebar.header("🔢 Entrer les données du match")  
user_input = {}  
for column in X.columns:  
    user_input[column] = st.sidebar.number_input(column, value=float(df[column].mean()))  

# Bouton de prédiction  
if st.sidebar.button("🔮 Prédire le Résultat"):  
    input_data = np.array(list(user_input.values())).reshape(1, -1)  
    proba_rf = rf_model.predict_proba(input_data)[0][1]  
    proba_xgb = xgb_model.predict_proba(input_data)[0][1]  
    
    result_rf = "Victoire Domicile" if proba_rf > 0.5 else "Victoire Extérieure"  
    result_xgb = "Victoire Domicile" if proba_xgb > 0.5 else "Victoire Extérieure"  
    
    st.write(f"🔮 Résultat (Random Forest): {result_rf} avec {proba_rf*100:.2f}% de confiance")  
    st.write(f"🔮 Résultat (XGBoost): {result_xgb} avec {proba_xgb*100:.2f}% de confiance")  
    
    # Graphique Altair  
    chart_data = pd.DataFrame({"Modèle": ["RandomForest", "XGBoost"], "Probabilité Victoire Domicile": [proba_rf, proba_xgb]})  
    chart = alt.Chart(chart_data).mark_bar().encode(x="Modèle", y="Probabilité Victoire Domicile", color="Modèle")  
    st.altair_chart(chart, use_container_width=True)  
