import streamlit as st  
import numpy as np  
import pandas as pd  
import altair as alt  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.linear_model import LogisticRegression  
import math  

# [Gardez toutes les fonctions précédentes : poisson_match_analysis, poisson_prob]  

# Interface Streamlit  
st.title("⚽ Prédiction de Match de Football")  

# Initialisation de session_state  
if 'model_trained' not in st.session_state:  
    st.session_state.model_trained = False  
    st.session_state.rf_model = None  
    st.session_state.xgb_model = None  
    st.session_state.logistic_model = None  

# Exemple de données d'entraînement  
data = {  
    # [Votre dataset précédent]  
    'resultat': [1, 1, 0, 1, 1]  # 1 = Victoire Domicile, 0 = Victoire Extérieure  
}  

df = pd.DataFrame(data)  

# Entrée des données utilisateur  
st.sidebar.header("🔢 Paramètres du Match")  

# Colonnes pour Équipe A et Équipe B  
col1, col2 = st.columns(2)  

with col1:  
    st.subheader("Équipe A (Domicile)")  
    xG_domicile = st.number_input("xG Domicile", min_value=0.0, max_value=5.0, value=1.5, step=0.1)  
    tirs_cadres_domicile = st.number_input("Tirs Cadrés Domicile", min_value=0, max_value=20, value=5)  
    touches_surface_domicile = st.number_input("Touches Surface Domicile", min_value=0, max_value=50, value=15)  
    xGA_domicile = st.number_input("xGA Domicile", min_value=0.0, max_value=5.0, value=1.0, step=0.1)  
    interceptions_domicile = st.number_input("Interceptions Domicile", min_value=0, max_value=30, value=10)  
    duels_defensifs_domicile = st.number_input("Duels Défensifs Domicile", min_value=0, max_value=50, value=20)  
    possession_domicile = st.number_input("Possession Domicile (%)", min_value=0, max_value=100, value=55)  
    passes_cles_domicile = st.number_input("Passes Clés Domicile", min_value=0, max_value=10, value=3)  
    forme_recente_domicile = st.number_input("Forme Récente Domicile", min_value=0, max_value=20, value=10)  
    blessures_domicile = st.number_input("Blessures Domicile", min_value=0, max_value=5, value=1)  

with col2:  
    st.subheader("Équipe B (Extérieur)")  
    xG_exterieur = st.number_input("xG Extérieur", min_value=0.0, max_value=5.0, value=1.2, step=0.1)  
    tirs_cadres_exterieur = st.number_input("Tirs Cadrés Extérieur", min_value=0, max_value=20, value=4)  
    touches_surface_exterieur = st.number_input("Touches Surface Extérieur", min_value=0, max_value=50, value=12)  
    xGA_exterieur = st.number_input("xGA Extérieur", min_value=0.0, max_value=5.0, value=1.3, step=0.1)  
    interceptions_exterieur = st.number_input("Interceptions Extérieur", min_value=0, max_value=30, value=9)  
    duels_defensifs_exterieur = st.number_input("Duels Défensifs Extérieur", min_value=0, max_value=50, value=18)  
    possession_exterieur = st.number_input("Possession Extérieur (%)", min_value=0, max_value=100, value=45)  
    passes_cles_exterieur = st.number_input("Passes Clés Extérieur", min_value=0, max_value=10, value=2)  
    forme_recente_exterieur = st.number_input("Forme Récente Extérieur", min_value=0, max_value=20, value=8)  
    blessures_exterieur = st.number_input("Blessures Extérieur", min_value=0, max_value=5, value=0)  

# Cotes de marché (sidebar)  
st.sidebar.header("💰 Cotes de Marché")  
market_home_odds = st.sidebar.number_input("Cote Victoire Domicile", min_value=1.0, max_value=10.0, value=2.0, step=0.1)  
market_away_odds = st.sidebar.number_input("Cote Victoire Extérieure", min_value=1.0, max_value=10.0, value=3.5, step=0.1)  
market_draw_odds = st.sidebar.number_input("Cote Match Nul", min_value=1.0, max_value=10.0, value=3.2, step=0.1)  

# Bouton de prédiction  
if st.button("🔮 Analyser le Match"):  
    # Préparation des données pour le modèle  
    input_data = np.array([  
        xG_domicile,   
        tirs_cadres_domicile,   
        touches_surface_domicile,   
        xGA_domicile,  
        interceptions_domicile,   
        duels_defensifs_domicile,   
        possession_domicile,  
        passes_cles_domicile,   
        forme_recente_domicile,   
        df['buts_domicile'].mean() if 'buts_domicile' in df.columns else 2,  
        df['buts_encais_domicile'].mean() if 'buts_encais_domicile' in df.columns else 1,  
        blessures_domicile,  
        
        xG_exterieur,   
        tirs_cadres_exterieur,   
        touches_surface_exterieur,   
        xGA_exterieur,  
        interceptions_exterieur,   
        duels_defensifs_exterieur,   
        possession_exterieur,  
        passes_cles_exterieur,   
        forme_recente_exterieur,   
        df['buts_exterieur'].mean() if 'buts_exterieur' in df.columns else 1,  
        df['buts_encais_exterieur'].mean() if 'buts_encais_exterieur' in df.columns else 2,  
        blessures_exterieur  
    ]).reshape(1, -1)  
    
    # Analyse Poisson  
    poisson_analysis = poisson_match_analysis(xG_domicile, xG_exterieur)  
    
    # Création de plusieurs onglets pour organiser les résultats  
    tab1, tab2, tab3 = st.tabs(["🏆 Résultats", "📊 Analyse Probabiliste", "💡 Recommandations Paris"])  
    
    with tab1:  
        st.subheader("Prédictions des Modèles")  
        # Modèles de prédiction (si les modèles sont entraînés)  
        if not st.session_state.model_trained:  
            # Entraînement rapide si nécessaire  
            X = df.drop('resultat', axis=1)  
            y = df['resultat']  
            st.session_state.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
            st.session_state.rf_model.fit(X, y)  
            st.session_state.model_trained = True  
        
        proba_rf = st.session_state.rf_model.predict_proba(input_data)[0][1]  
        
        st.metric("Probabilité Victoire Domicile", f"{proba_rf*100:.2f}%")  
        
        # Graphique des probabilités de résultat  
        chart_data = pd.DataFrame({  
            "Résultat": ["Victoire Domicile", "Victoire Extérieure", "Match Nul"],  
            "Probabilité": [  
                poisson_analysis['probabilities']['home_win'],  
                poisson_analysis['probabilities']['away_win'],  
                poisson_analysis['probabilities']['draw']  
            ]  
        })  
        
        chart = alt.Chart(chart_data).mark_bar().encode(  
            x="Résultat",  
            y="Probabilité",  
            color="Résultat"  
        )  
        st.altair_chart(chart, use_container_width=True)  
    
    with tab2:  
        st.subheader("Analyse Probabiliste Détaillée")  
        col1, col2, col3 = st.columns(3)  
        
        with col1:  
            st.metric("Victoire Domicile", f"{poisson_analysis['probabilities']['home_win']:.2%}")  
        with col2:  
            st.metric("Victoire Extérieure", f"{poisson_analysis['probabilities']['away_win']:.2%}")  
        with col3:  
            st.metric("Match Nul", f"{poisson_analysis['probabilities']['draw']:.2%}")  
    
    with tab3:  
        st.subheader("Recommandations de Paris")  
        
        # Comparaison des cotes  
        st.write("### Comparaison des Cotes")  
        
        cols = st.columns(3)  
        bet_results = ['home', 'away', 'draw']  
        labels = ['Domicile', 'Extérieure', 'Nul']  
        
        for i, (result, label) in enumerate(zip(bet_results, labels)):  
            with cols[i]:  
                value_bet = poisson_analysis['value_bets'][result]  
                st.metric(  
                    f"Cote {label}",   
                    f"{value_bet['recommendation']} (Edge: {value_bet['edge']:.2f}%)"  
                )  

# Configuration finale de Streamlit  
st.sidebar.info("🤖 Prédictions basées sur des modèles statistiques et l'analyse Poisson")  
