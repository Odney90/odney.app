import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from io import BytesIO  
from scipy.special import factorial  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  

# Fonction pour prédire avec le modèle Poisson  
def poisson_prediction(home_goals, away_goals):  
    home_prob = np.exp(-home_goals) * (home_goals ** home_goals) / factorial(home_goals)  
    away_prob = np.exp(-away_goals) * (away_goals ** away_goals) / factorial(away_goals)  
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

# Fonction pour entraîner et prédire avec les modèles  
def train_and_predict(home_stats, away_stats):  
    # Créer un DataFrame avec les statistiques  
    data = pd.DataFrame({  
        'home_goals': [home_stats['moyenne_buts_marques']],  
        'away_goals': [away_stats['moyenne_buts_marques']],  
        'home_xG': [home_stats['xG']],  
        'away_xG': [away_stats['xG']],  
        'home_defense': [home_stats['moyenne_buts_encais']],  
        'away_defense': [away_stats['moyenne_buts_encais']]  
    })  

    # Modèle de régression logistique  
    log_reg = LogisticRegression()  
    log_reg.fit(data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']], [1])  # Dummy target  
    log_reg_prob = log_reg.predict_proba(data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']])[:, 1]  

    # Modèle Random Forest  
    rf = RandomForestClassifier()  
    rf.fit(data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']], [1])  # Dummy target  
    rf_prob = rf.predict_proba(data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']])[:, 1]  

    # Modèle XGBoost  
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  
    xgb.fit(data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']], [1])  # Dummy target  
    xgb_prob = xgb.predict_proba(data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']])[:, 1]  

    return log_reg_prob, rf_prob, xgb_prob  

# Interface utilisateur  
st.title("🏆 Analyse de Matchs de Football et Prédictions de Paris Sportifs")  

# Saisie des données des équipes  
st.header("Saisie des données des équipes")  
col1, col2 = st.columns(2)  

with col1:  
    home_team = st.text_input("Nom de l'équipe à domicile", value="Équipe A")  
    st.subheader("Statistiques de l'équipe à domicile")  
    home_stats = {  
        "moyenne_buts_marques": st.number_input("Moyenne de buts marqués par match (domicile)", min_value=0.0, value=2.5),  
        "xG": st.number_input("xG (Expected Goals) (domicile)", min_value=0.0, value=2.0),  
        "moyenne_buts_encais": st.number_input("Moyenne de buts encaissés par match (domicile)", min_value=0.0, value=1.0),  
        "XGA": st.number_input("xGA (Expected Goals Against) (domicile)", min_value=0.0, value=1.5),  
        "tires_par_match": st.number_input("Nombres de tirs par match (domicile)", min_value=0, value=15),  
        "passes_menant_a_tir": st.number_input("Nombres de passes menant à un tir (domicile)", min_value=0, value=10),  
        "tirs_cadres": st.number_input("Tirs cadrés par match (domicile)", min_value=0, value=5),  
        "tires_concedes": st.number_input("Nombres de tirs concédés par match (domicile)", min_value=0, value=8),  
        "duels_defensifs": st.number_input("Duels défensifs gagnés (%) (domicile)", min_value=0.0, max_value=100.0, value=60.0),  
        "possession": st.number_input("Possession moyenne (%) (domicile)", min_value=0.0, max_value=100.0, value=55.0),  
        "passes_reussies": st.number_input("Passes réussies (%) (domicile)", min_value=0.0, max_value=100.0, value=80.0),  
        "touches_surface": st.number_input("Balles touchées dans la surface adverse (domicile)", min_value=0, value=20),  
        "forme_recente": st.number_input("Forme récente (points sur les 5 derniers matchs) (domicile)", min_value=0, value=10),  
    }  

with col2:  
    away_team = st.text_input("Nom de l'équipe à l'extérieur", value="Équipe B")  
    st.subheader("Statistiques de l'équipe à l'extérieur")  
    away_stats = {  
        "moyenne_buts_marques": st.number_input("Moyenne de buts marqués par match (extérieur)", min_value=0.0, value=1.5),  
        "xG": st.number_input("xG (Expected Goals) (extérieur)", min_value=0.0, value=1.8),  
        "moyenne_buts_encais": st.number_input("Moyenne de buts encaissés par match (extérieur)", min_value=0.0, value=2.0),  
        "XGA": st.number_input("xGA (Expected Goals Against) (extérieur)", min_value=0.0, value=2.5),  
        "tires_par_match": st.number_input("Nombres de tirs par match (extérieur)", min_value=0, value=12),  
        "passes_menant_a_tir": st.number_input("Nombres de passes menant à un tir (extérieur)", min_value=0, value=8),  
        "tirs_cadres": st.number_input("Tirs cadrés par match (extérieur)", min_value=0, value=4),  
        "tires_concedes": st.number_input("Nombres de tirs concédés par match (extérieur)", min_value=0, value=10),  
        "duels_defensifs": st.number_input("Duels défensifs gagnés (%) (extérieur)", min_value=0.0, max_value=100.0, value=50.0),  
        "possession": st.number_input("Possession moyenne (%) (extérieur)", min_value=0.0, max_value=100.0, value=45.0),  
        "passes_reussies": st.number_input("Passes réussies (%) (extérieur)", min_value=0.0, max_value=100.0, value=75.0),  
        "touches_surface": st.number_input("Balles touchées dans la surface adverse (extérieur)", min_value=0, value=15),  
        "forme_recente": st.number_input("Forme récente (points sur les 5 derniers matchs) (extérieur)", min_value=0, value=8),  
    }  

# Prédictions  
if st.button("🔍 Prédire les résultats"):  
    home_goals = home_stats["moyenne_buts_marques"] + home_stats["xG"] - away_stats["moyenne_buts_encais"]  
    away_goals = away_stats["moyenne_buts_marques"] + away_stats["xG"] - home_stats["moyenne_buts_encais"]  
    
    home_prob, away_prob = poisson_prediction(home_goals, away_goals)  

    # Prédictions avec les autres modèles  
    log_reg_prob, rf_prob, xgb_prob = train_and_predict(home_stats, away_stats)  

    # Affichage des résultats  
    st.subheader("📊 Résultats des Prédictions")  
    st.write(f"**Nombre de buts prédit pour {home_team} :** {home_goals:.2f} ({home_prob * 100:.2f}%)")  
    st.write(f"**Nombre de buts prédit pour {away_team} :** {away_goals:.2f} ({away_prob * 100:.2f}%)")  
    
    st.write(f"**Probabilité de victoire selon la régression logistique pour {home_team} :** {log_reg_prob[0] * 100:.2f}%")  
    st.write(f"**Probabilité de victoire selon Random Forest pour {home_team} :** {rf_prob[0] * 100:.2f}%")  
    st.write(f"**Probabilité de victoire selon XGBoost pour {home_team} :** {xgb_prob[0] * 100:.2f}%")  

    # Calcul des paris double chance  
    double_chance = double_chance_probabilities(home_prob, away_prob)  
    st.write("**Probabilités des paris double chance :**")  
    for bet, prob in double_chance.items():  
        st.write(f"{bet}: {prob:.2f}")  

    # Visualisation des résultats  
    st.subheader("📈 Visualisation des résultats")  
    fig, ax = plt.subplots()  
    ax.bar(["Domicile", "Extérieur"], [home_prob, away_prob], color=['blue', 'orange'])  
    ax.set_ylabel("Probabilités")  
    ax.set_title("Probabilités des résultats")  
    st.pyplot(fig)  

    # Gestion de la bankroll  
    st.subheader("💰 Gestion de la bankroll")  
    odds_home = st.number_input("Cote pour l'équipe à domicile", min_value=1.0, value=1.8)  
    odds_away = st.number_input("Cote pour l'équipe à l'extérieur", min_value=1.0, value=2.2)  

    if odds_home > 1.0 and odds_away > 1.0:  
        kelly_home = kelly_criterion(home_prob, odds_home)  
        kelly_away = kelly_criterion(away_prob, odds_away)  
        st.write(f"**Mise recommandée selon Kelly pour {home_team}:** {kelly_home:.2f}")  
        st.write(f"**Mise recommandée selon Kelly pour {away_team}:** {kelly_away:.2f}")  

    # Option de téléchargement des résultats  
    results = {  
        "Équipe Domicile": home_team,  
        "Équipe Extérieure": away_team,  
        "Buts Prédit Domicile": home_goals,  
        "Buts Prédit Extérieur": away_goals,  
        "Probabilité Domicile": home_prob,  
        "Probabilité Extérieure": away_prob,  
        "Probabilité Régression Logistique": log_reg_prob[0],  
        "Probabilité Random Forest": rf_prob[0],  
        "Probabilité XGBoost": xgb_prob[0],  
        "Paris Double Chance 1X": double_chance["1X"],  
        "Paris Double Chance X2": double_chance["X2"],  
        "Paris Double Chance 12": double_chance["12"],  
    }  

    if st.button("📥 Télécharger les résultats"):  
        buffer = download_results(results)  
        st.download_button(  
            label="Télécharger les résultats",  
            data=buffer,  
            file_name="predictions.csv",  
            mime="text/csv"  
        )
