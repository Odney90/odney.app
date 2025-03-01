import streamlit as st  
import numpy as np  
import pandas as pd  
import altair as alt  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.linear_model import LogisticRegression  
import math  

# Fonction Poisson améliorée avec détection de valeur bet  
def poisson_match_analysis(xG_domicile, xG_exterieur, max_goals=5):  
    # Calcul des probabilités de buts pour chaque équipe  
    home_probs = [poisson_prob(xG_domicile, i) for i in range(max_goals + 1)]  
    away_probs = [poisson_prob(xG_exterieur, i) for i in range(max_goals + 1)]  
    
    # Matrice des résultats possibles  
    match_matrix = np.zeros((max_goals + 1, max_goals + 1))  
    
    # Variables pour stocker les probabilités de résultat  
    win_home = 0  
    win_away = 0  
    draw = 0  
    
    # Calcul des probabilités de résultat  
    for home_goals in range(max_goals + 1):  
        for away_goals in range(max_goals + 1):  
            prob = home_probs[home_goals] * away_probs[away_goals]  
            match_matrix[home_goals, away_goals] = prob  
            
            if home_goals > away_goals:  
                win_home += prob  
            elif home_goals < away_goals:  
                win_away += prob  
            else:  
                draw += prob  
    
    # Calcul des cotes implicites  
    def calculate_implied_probability(odds):  
        return 1 / odds  
    
    def calculate_fair_odds(probability):  
        return 1 / probability if probability > 0 else np.inf  
    
    # Exemple de cotes du marché (à remplacer par des cotes réelles)  
    market_home_odds = 2.0  # Cote victoire domicile  
    market_away_odds = 3.5  # Cote victoire extérieure  
    market_draw_odds = 3.2  # Cote match nul  
    
    # Calcul des probabilités et cotes du modèle  
    model_home_prob = win_home  
    model_away_prob = win_away  
    model_draw_prob = draw  
    
    model_home_odds = calculate_fair_odds(model_home_prob)  
    model_away_odds = calculate_fair_odds(model_away_prob)  
    model_draw_odds = calculate_fair_odds(model_draw_prob)  
    
    # Détection de valeur bet  
    def detect_value_bet(market_odds, model_odds, market_prob, model_prob):  
        edge = (1 / market_odds - 1 / model_odds) * 100  
        return {  
            "edge": edge,  
            "recommendation": "Pari value" if edge > 0 else "Pas de valeur"  
        }  
    
    home_value = detect_value_bet(market_home_odds, model_home_odds,   
                                   calculate_implied_probability(market_home_odds), model_home_prob)  
    away_value = detect_value_bet(market_away_odds, model_away_odds,   
                                   calculate_implied_probability(market_away_odds), model_away_prob)  
    draw_value = detect_value_bet(market_draw_odds, model_draw_odds,   
                                   calculate_implied_probability(market_draw_odds), model_draw_prob)  
    
    return {  
        "probabilities": {  
            "home_win": win_home,  
            "away_win": win_away,  
            "draw": draw  
        },  
        "model_odds": {  
            "home_odds": model_home_odds,  
            "away_odds": model_away_odds,  
            "draw_odds": model_draw_odds  
        },  
        "market_odds": {  
            "home_odds": market_home_odds,  
            "away_odds": market_away_odds,  
            "draw_odds": market_draw_odds  
        },  
        "value_bets": {  
            "home": home_value,  
            "away": away_value,  
            "draw": draw_value  
        },  
        "match_matrix": match_matrix  
    }  

# Fonction de probabilité de Poisson  
def poisson_prob(lam, k):  
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)  

# Reste du code Streamlit (identique à la version précédente)  
# ...  

# Modification de la partie prédiction  
if st.button("🔮 Prédire le Résultat"):  
    # Récupérer X après l'entraînement  
    X = df.drop('resultat', axis=1)  

    # Créer input_data avec TOUTES les colonnes nécessaires  
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
        df['buts_domicile'].mean(),  
        df['buts_encais_domicile'].mean(),  
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
        df['buts_exterieur'].mean(),  
        df['buts_encais_exterieur'].mean(),  
        blessures_exterieur  
    ]).reshape(1, -1)  
    
    # Analyse Poisson détaillée  
    poisson_analysis = poisson_match_analysis(xG_domicile, xG_exterieur)  
    
    # Affichage discret des informations  
    st.text(f"Caractéristiques utilisées : {input_data.shape[1]}")  
    
    # Résultats des modèles  
    proba_rf = st.session_state.rf_model.predict_proba(input_data)[0][1]  
    proba_xgb = st.session_state.xgb_model.predict_proba(input_data)[0][1]  
    proba_logistic = st.session_state.logistic_model.predict_proba(input_data)[0][1]  

    # Affichage des résultats Poisson  
    st.subheader("🎲 Analyse Probabiliste du Match")  
    
    # Probabilités de résultat  
    st.write(f"🏠 Victoire Domicile : {poisson_analysis['probabilities']['home_win']:.2%}")  
    st.write(f"🏟️ Victoire Extérieure : {poisson_analysis['probabilities']['away_win']:.2%}")  
    st.write(f"🤝 Match Nul : {poisson_analysis['probabilities']['draw']:.2%}")  
    
    # Analyse des paris  
    st.subheader("💰 Analyse des Paris")  
    
    # Cotes du modèle vs Marché  
    st.write("Cotes Modèle:")  
    st.write(f"Domicile: {poisson_analysis['model_odds']['home_odds']:.2f}")  
    st.write(f"Extérieur: {poisson_analysis['model_odds']['away_odds']:.2f}")  
    st.write(f"Nul: {poisson_analysis['model_odds']['draw_odds']:.2f}")  
    
    # Détection de valeur bet  
    st.write("Opportunités de Pari:")  
    for result, value in poisson_analysis['value_bets'].items():  
        st.write(f"{result.capitalize()}: {value['recommendation']} (Edge: {value['edge']:.2f}%)")  

    # Graphique des probabilités  
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
