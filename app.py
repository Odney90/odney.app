import streamlit as st  
import numpy as np  
import pandas as pd  
import altair as alt  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.linear_model import LogisticRegression  
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
    st.session_state.logistic_model = None  

# Entrée des données utilisateur  
st.sidebar.header("🔢 Entrer les données du match")  

# Créer deux colonnes pour Équipe A et Équipe B  
col1, col2 = st.columns(2)  

with col1:  
    st.subheader("Équipe A")  
    xG_domicile = st.number_input("xG Domicile", value=float(df['xG_domicile'].mean()))  
    tirs_cadres_domicile = st.number_input("Tirs Cadres Domicile", value=int(df['tirs_cadres_domicile'].mean()))  
    touches_surface_domicile = st.number_input("Touches Surface Domicile", value=int(df['touches_surface_domicile'].mean()))  
    xGA_domicile = st.number_input("xGA Domicile", value=float(df['xGA_domicile'].mean()))  
    interceptions_domicile = st.number_input("Interceptions Domicile", value=int(df['interceptions_domicile'].mean()))  
    duels_defensifs_domicile = st.number_input("Duels Défensifs Domicile", value=int(df['duels_defensifs_domicile'].mean()))  
    possession_domicile = st.number_input("Possession Domicile (%)", value=float(df['possession_domicile'].mean()))  
    passes_cles_domicile = st.number_input("Passes Clés Domicile", value=int(df['passes_cles_domicile'].mean()))  
    forme_recente_domicile = st.number_input("Forme Récente Domicile", value=int(df['forme_recente_domicile'].mean()))  
    blessures_domicile = st.number_input("Blessures Domicile", value=int(df['blessures_domicile'].mean()))  

with col2:  
    st.subheader("Équipe B")  
    xG_exterieur = st.number_input("xG Extérieur", value=float(df['xG_exterieur'].mean()))  
    tirs_cadres_exterieur = st.number_input("Tirs Cadres Extérieur", value=int(df['tirs_cadres_exterieur'].mean()))  
    touches_surface_exterieur = st.number_input("Touches Surface Extérieur", value=int(df['touches_surface_exterieur'].mean()))  
    xGA_exterieur = st.number_input("xGA Extérieur", value=float(df['xGA_exterieur'].mean()))  
    interceptions_exterieur = st.number_input("Interceptions Extérieur", value=int(df['interceptions_exterieur'].mean()))  
    duels_defensifs_exterieur = st.number_input("Duels Défensifs Extérieur", value=int(df['duels_defensifs_exterieur'].mean()))  
    possession_exterieur = st.number_input("Possession Extérieur (%)", value=float(df['possession_exterieur'].mean()))  
    passes_cles_exterieur = st.number_input("Passes Clés Extérieur", value=int(df['passes_cles_exterieur'].mean()))  
    forme_recente_exterieur = st.number_input("Forme Récente Extérieur", value=int(df['forme_recente_exterieur'].mean()))  
    blessures_exterieur = st.number_input("Blessures Extérieur", value=int(df['blessures_exterieur'].mean()))  

# Entraînement des modèles si ce n'est pas déjà fait  
if not st.session_state.model_trained:  
    X = df.drop('resultat', axis=1)  
    y = df['resultat']  
    st.session_state.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
    st.session_state.rf_model.fit(X, y)  
    st.session_state.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  
    st.session_state.xgb_model.fit(X, y)  
    st.session_state.logistic_model = LogisticRegression(random_state=42)  
    st.session_state.logistic_model.fit(X, y)  
    st.session_state.model_trained = True  

# Bouton de prédiction  
if st.button("🔮 Prédire le Résultat"):  
    input_data = np.array([  
        xG_domicile, tirs_cadres_domicile, touches_surface_domicile, xGA_domicile,  
        interceptions_domicile, duels_defensifs_domicile, possession_domicile,  
        passes_cles_domicile, forme_recente_domicile, blessures_domicile,  
        xG_exterieur, tirs_cadres_exterieur, touches_surface_exterieur, xGA_exterieur,  
        interceptions_exterieur, duels_defensifs_exterieur, possession_exterieur,  
        passes_cles_exterieur, forme_recente_exterieur, blessures_exterieur  
    ]).reshape(1, -1)  
    
    # Vérification des valeurs d'entrée  
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):  
        st.error("Les données saisies contiennent des valeurs manquantes ou infinies.")  
    else:  
        # Vérification de la forme de input_data  
        if input_data.shape[1] != X.shape[1]:  
            st.error("Le nombre de caractéristiques dans les données d'entrée ne correspond pas au modèle.")  
        else:  
            proba_rf = st.session_state.rf_model.predict_proba(input_data)[0][1]  
            proba_xgb = st.session_state.xgb_model.predict_proba(input_data)[0][1]  
            proba_logistic = st.session_state.logistic_model.predict_proba(input_data)[0][1]  

            result_rf = "Victoire Domicile" if proba_rf > 0.5 else "Victoire Extérieure"  
            result_xgb = "Victoire Domicile" if proba_xgb > 0.5 else "Victoire Extérieure"  
            result_logistic = "Victoire Domicile" if proba_logistic > 0.5 else "Victoire Extérieure"  

            st.write(f"🔮 Résultat (Random Forest): {result_rf} avec {proba_rf*100:.2f}% de confiance")  
            st.write(f"🔮 Résultat (XGBoost): {result_xgb} avec {proba_xgb*100:.2f}% de confiance")  
            st.write(f"🔮 Résultat (Régression Logistique): {result_logistic} avec {proba_logistic*100:.2f}% de confiance")  

            # Prédiction des buts  
            win_home, win_away, draw = predict_goals(xG_domicile, xG_exterieur)  

            st.write(f"🏠 Probabilité de victoire Domicile : {win_home:.2%}")  
            st.write(f"🏟️ Probabilité de victoire Extérieure : {win_away:.2%}")  
            st.write(f"🤝 Probabilité de match nul : {draw:.2%}")  

            # Graphique Altair  
            chart_data = pd.DataFrame({  
                "Modèle": ["RandomForest", "XGBoost", "Régression Logistique"],  
                "Probabilité Victoire Domicile": [proba_rf, proba_xgb, proba_logistic]  
            })  
            chart = alt.Chart(chart_data).mark_bar().encode(x="Modèle", y="Probabilité Victoire Domicile", color="Modèle")  
            st.altair_chart(chart, use_container_width=True)  
