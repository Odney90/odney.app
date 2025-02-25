import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import cross_val_score  
import matplotlib.pyplot as plt  
from io import BytesIO  

# Fonction pour prédire avec le modèle Poisson  
def poisson_prediction(home_goals, away_goals):  
    home_prob = np.exp(-home_goals) * (home_goals ** home_goals) / np.math.factorial(home_goals)  
    away_prob = np.exp(-away_goals) * (away_goals ** away_goals) / np.math.factorial(away_goals)  
    return home_prob, away_prob  

# Fonction pour calculer les paris double chance  
def double_chance_probabilities(home_prob, away_prob):  
    return {  
        "1X": home_prob + (1 - away_prob),  
        "X2": away_prob + (1 - home_prob),  
        "12": home_prob + away_prob  
    }  

# Fonction pour gérer la bankroll  
def kelly_criterion(probability, odds):  
    return (probability * odds - 1) / (odds - 1)  

# Fonction pour télécharger les résultats  
def download_results(results):  
    df = pd.DataFrame([results])  
    buffer = BytesIO()  
    df.to_csv(buffer, index=False)  
    buffer.seek(0)  
    return buffer  

# Interface utilisateur  
st.title("Analyse de Matchs de Football et Prédictions de Paris Sportifs")  

# Saisie des données des équipes  
st.header("Saisie des données des équipes")  
home_team = st.text_input("Nom de l'équipe à domicile")  
away_team = st.text_input("Nom de l'équipe à l'extérieur")  

# Variables d'entrée pour l'équipe à domicile  
st.subheader("Statistiques de l'équipe à domicile")  
home_stats = {  
    "moyenne_buts_marques": st.number_input("Moyenne de buts marqués par match (domicile)", min_value=0.0, key="home_moyenne_buts_marques"),  
    "xG": st.number_input("xG (Expected Goals) (domicile)", min_value=0.0, key="home_xG"),  
    "moyenne_buts_encais": st.number_input("Moyenne de buts encaissés par match (domicile)", min_value=0.0, key="home_moyenne_buts_encais"),  
    "XGA": st.number_input("xGA (Expected Goals Against) (domicile)", min_value=0.0, key="home_XGA"),  
    "tires_par_match": st.number_input("Nombres de tirs par match (domicile)", min_value=0, key="home_tires_par_match"),  
    "passes_menant_a_tir": st.number_input("Nombres de passes menant à un tir (domicile)", min_value=0, key="home_passes_menant_a_tir"),  
    "tirs_cadres": st.number_input("Tirs cadrés par match (domicile)", min_value=0, key="home_tirs_cadres"),  
    "tires_concedes": st.number_input("Nombres de tirs concédés par match (domicile)", min_value=0, key="home_tires_concedes"),  
    "duels_defensifs": st.number_input("Duels défensifs gagnés (%) (domicile)", min_value=0.0, max_value=100.0, key="home_duels_defensifs"),  
    "possession": st.number_input("Possession moyenne (%) (domicile)", min_value=0.0, max_value=100.0, key="home_possession"),  
    "passes_reussies": st.number_input("Passes réussies (%) (domicile)", min_value=0.0, max_value=100.0, key="home_passes_reussies"),  
    "touches_surface": st.number_input("Balles touchées dans la surface adverse (domicile)", min_value=0, key="home_touches_surface"),  
    "forme_recente": st.number_input("Forme récente (points sur les 5 derniers matchs) (domicile)", min_value=0, key="home_forme_recente"),  
}  

# Variables d'entrée pour l'équipe à l'extérieur  
st.subheader("Statistiques de l'équipe à l'extérieur")  
away_stats = {  
    "moyenne_buts_marques": st.number_input("Moyenne de buts marqués par match (extérieur)", min_value=0.0, key="away_moyenne_buts_marques"),  
    "xG": st.number_input("xG (Expected Goals) (extérieur)", min_value=0.0, key="away_xG"),  
    "moyenne_buts_encais": st.number_input("Moyenne de buts encaissés par match (extérieur)", min_value=0.0, key="away_moyenne_buts_encais"),  
    "XGA": st.number_input("xGA (Expected Goals Against) (extérieur)", min_value=0.0, key="away_XGA"),  
    "tires_par_match": st.number_input("Nombres de tirs par match (extérieur)", min_value=0, key="away_tires_par_match"),  
    "passes_menant_a_tir": st.number_input("Nombres de passes menant à un tir (extérieur)", min_value=0, key="away_passes_menant_a_tir"),  
    "tirs_cadres": st.number_input("Tirs cadrés par match (extérieur)", min_value=0, key="away_tirs_cadres"),  
    "tires_concedes": st.number_input("Nombres de tirs concédés par match (extérieur)", min_value=0, key="away_tires_concedes"),  
    "duels_defensifs": st.number_input("Duels défensifs gagnés (%) (extérieur)", min_value=0.0, max_value=100.0, key="away_duels_defensifs"),  
    "possession": st.number_input("Possession moyenne (%) (extérieur)", min_value=0.0, max_value=100.0, key="away_possession"),  
    "passes_reussies": st.number_input("Passes réussies (%) (extérieur)", min_value=0.0, max_value=100.0, key="away_passes_reussies"),  
    "touches_surface": st.number_input("Balles touchées dans la surface adverse (extérieur)", min_value=0, key="away_touches_surface"),  
    "forme_recente": st.number_input("Forme récente (points sur les 5 derniers matchs) (extérieur)", min_value=0, key="away_forme_recente"),  
}  

# Prédictions  
if st.button("Prédire les résultats"):  
    home_goals = home_stats["moyenne_buts_marques"] + home_stats["xG"] - away_stats["moyenne_buts_encais"]  
    away_goals = away_stats["moyenne_buts_marques"] + away_stats["xG"] - home_stats["moyenne_buts_encais"]  
    
    home_prob, away_prob = poisson_prediction(home_goals, away_goals)  
    
    st.write(f"Probabilité de victoire de {home_team}: {home_prob:.2f}")  
    st.write(f"Probabilité de victoire de {away_team}: {away_prob:.2f}")  

    # Calcul des paris double chance  
    double_chance = double_chance_probabilities(home_prob, away_prob)  
    st.write("Probabilités des paris double chance:")  
    for bet, prob in double_chance.items():  
        st.write(f"{bet}: {prob:.2f}")  

    # Visualisation des résultats  
    st.subheader("Visualisation des résultats")  
    fig, ax = plt.subplots()  
    ax.bar(["Domicile", "Extérieur"], [home_prob, away_prob])  
    ax.set_ylabel("Probabilités")  
    ax.set_title("Probabilités des résultats")  
    st.pyplot(fig)  

    # Gestion de la bankroll  
    st.subheader("Gestion de la bankroll")  
    odds_home = st.number_input("Cote pour l'équipe à domicile", min_value=1.0, key="odds_home")  
    odds_away = st.number_input("Cote pour l'équipe à l'extérieur", min_value=1.0, key="odds_away")  

    if odds_home > 1.0 and odds_away > 1.0:  
        kelly_home = kelly_criterion(home_prob, odds_home)  
        kelly_away = kelly_criterion(away_prob, odds_away)  
        st.write(f"Mise recommandée selon Kelly pour {home_team}: {kelly_home:.2f}")  
        st.write(f"Mise recommandée selon Kelly pour {away_team}: {kelly_away:.2f}")  

    # Option de téléchargement des résultats  
    results = {  
        "Équipe Domicile": home_team,  
        "Équipe Extérieure": away_team,  
        "Probabilité Domicile": home_prob,  
        "Probabilité Extérieure": away_prob,  
        "Paris Double Chance 1X": double_chance["1X"],  
        "Paris Double Chance X2": double_chance["X2"],  
        "Paris Double Chance 12": double_chance["12"],  
    }  

    if st.button("Télécharger les résultats"):  
        buffer = download_results(results)  
        st.download_button(  
            label="Télécharger les résultats",  
            data=buffer,  
            file_name="predictions.csv",  
            mime="text/csv"  
        )
