import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from io import BytesIO  
from scipy.special import factorial  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from docx import Document  

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

# Fonction pour créer un document Word avec les résultats  
def create_doc(results):  
    doc = Document()  
    doc.add_heading('Analyse de Matchs de Football et Prédictions de Paris Sportifs', level=1)  

    # Ajout des données des équipes  
    doc.add_heading('Données des Équipes', level=2)  
    doc.add_paragraph(f"Équipe Domicile: {results['Équipe Domicile']}")  
    doc.add_paragraph(f"Équipe Extérieure: {results['Équipe Extérieure']}")  
    doc.add_paragraph(f"Buts Prédit Domicile: {results['Buts Prédit Domicile']:.2f}")  
    doc.add_paragraph(f"Buts Prédit Extérieur: {results['Buts Prédit Extérieur']:.2f}")  
    doc.add_paragraph(f"Probabilité Domicile: {results['Probabilité Domicile']:.2f}")  
    doc.add_paragraph(f"Probabilité Nul: {results['Probabilité Nul']:.2f}")  
    doc.add_paragraph(f"Probabilité Extérieure: {results['Probabilité Extérieure']:.2f}")  

    # Ajout des probabilités des paris double chance  
    doc.add_heading('Probabilités des Paris Double Chance', level=2)  
    for bet, prob in results.items():  
        if "Paris Double Chance" in bet:  
            doc.add_paragraph(f"{bet}: {prob:.2f}")  

    # Ajout des probabilités des modèles  
    doc.add_heading('Probabilités des Modèles', level=2)  
    doc.add_paragraph(f"Probabilité Régression Logistique: {results['Probabilité Régression Logistique']:.2f}")  
    doc.add_paragraph(f"Probabilité Random Forest: {results['Probabilité Random Forest']:.2f}")  
    doc.add_paragraph(f"Probabilité XGBoost: {results['Probabilité XGBoost']:.2f}")  

    # Enregistrement du document  
    buffer = BytesIO()  
    doc.save(buffer)  
    buffer.seek(0)  
    return buffer  

# Fonction pour entraîner et prédire avec les modèles  
def train_models():  
    # Créer un ensemble de données d'entraînement fictif  
    data = pd.DataFrame({  
        'home_goals': np.random.randint(0, 5, size=100),  
        'away_goals': np.random.randint(0, 5, size=100),  
        'home_xG': np.random.uniform(0, 3, size=100),  
        'away_xG': np.random.uniform(0, 3, size=100),  
        'home_defense': np.random.randint(0, 5, size=100),  
        'away_defense': np.random.randint(0, 5, size=100),  
        'result': np.random.choice([0, 1, 2], size=100)  # 0 pour victoire extérieure, 1 pour match nul, 2 pour victoire domicile  
    })  

    # Séparer les caractéristiques et la cible  
    X = data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']]  
    y = data['result']  

    # Modèle de régression logistique  
    log_reg = LogisticRegression()  
    log_reg.fit(X, y)  

    # Modèle Random Forest  
    rf = RandomForestClassifier()  
    rf.fit(X, y)  

    # Modèle XGBoost  
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  
    xgb.fit(X, y)  

    # Stocker les modèles dans l'état de session  
    st.session_state.log_reg_model = log_reg  
    st.session_state.rf_model = rf  
    st.session_state.xgb_model = xgb  

# Vérifier si les modèles sont déjà chargés dans l'état de session  
if 'log_reg_model' not in st.session_state:  
    train_models()  

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

    # Prédictions avec les modèles  
    log_reg_model = st.session_state.log_reg_model  
    rf_model = st.session_state.rf_model  
    xgb_model = st.session_state.xgb_model  

    # Vérification des données d'entrée  
    input_data = [[home_stats['moyenne_buts_marques'], away_stats['moyenne_buts_marques'], home_stats['xG'], away_stats['xG'], home_stats['moyenne_buts_encais'], away_stats['moyenne_buts_encais']]]  
    st.write("Données d'entrée pour les prédictions :", input_data)  

    try:  
        log_reg_prob = log_reg_model.predict_proba(input_data)[0]  
        rf_prob = rf_model.predict_proba(input_data)[0]  
        xgb_prob = xgb_model.predict_proba(input_data)[0]  
    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        log_reg_prob, rf_prob, xgb_prob = None, None, None  

    # Affichage des résultats  
    st.subheader("📊 Résultats des Prédictions")  

    # Vérification de la structure des probabilités  
    if log_reg_prob is not None and len(log_reg_prob) == 3:  
        st.write(f"**Nombre de buts prédit pour {home_team} :** {home_goals:.2f} ({home_prob * 100:.2f}%)")  
        st.write(f"**Nombre de buts prédit pour {away_team} :** {away_goals:.2f} ({away_prob * 100:.2f}%)")  
        
        st.write(f"**Probabilité de victoire selon la régression logistique pour {home_team} :** {log_reg_prob[2] * 100:.2f}% (Victoire Domicile)")  
        st.write(f"**Probabilité de match nul :** {log_reg_prob[1] * 100:.2f}% (Match Nul)")  
        st.write(f"**Probabilité de victoire selon la régression logistique pour {away_team} :** {log_reg_prob[0] * 100:.2f}% (Victoire Extérieure)")  

        # Affichage des probabilités des autres modèles  
        st.write(f"**Probabilité de victoire selon Random Forest pour {home_team} :** {rf_prob[2] * 100:.2f}% (Victoire Domicile)")  
        st.write(f"**Probabilité de match nul :** {rf_prob[1] * 100:.2f}% (Match Nul)")  
        st.write(f"**Probabilité de victoire selon Random Forest pour {away_team} :** {rf_prob[0] * 100:.2f}% (Victoire Extérieure)")  

        st.write(f"**Probabilité de victoire selon XGBoost pour {home_team} :** {xgb_prob[2] * 100:.2f}% (Victoire Domicile)")  
        st.write(f"**Probabilité de match nul :** {xgb_prob[1] * 100:.2f}% (Match Nul)")  
        st.write(f"**Probabilité de victoire selon XGBoost pour {away_team} :** {xgb_prob[0] * 100:.2f}% (Victoire Extérieure)")  
    else:  
        st.error("Erreur dans les probabilités prédites. Vérifiez les modèles et les données d'entrée.")  

    # Calcul des paris double chance  
    if home_prob is not None and away_prob is not None:  
        double_chance = double_chance_probabilities(home_prob, away_prob)  
        st.write("**Probabilités des paris double chance :**")  
        for bet, prob in double_chance.items():  
            st.write(f"{bet}: {prob:.2f}")  

    # Visualisation des résultats  
    if log_reg_prob is not None and len(log_reg_prob) == 3:  
        fig, ax = plt.subplots()  
        ax.bar(["Domicile", "Nul", "Extérieur"], [log_reg_prob[2], log_reg_prob[1], log_reg_prob[0]], color=['blue', 'orange', 'green'])  
        ax.set_ylabel("Probabilités")  
        ax.set_title("Probabilités des résultats")  
        st.pyplot(fig)  
    else:  
        st.error("Impossible de visualiser les résultats en raison d'une erreur dans les probabilités prédites.")  

    # Gestion de la bankroll  
    st.subheader("💰 Gestion de la bankroll")  
    odds_home = st.number_input("Cote pour l'équipe à domicile", min_value=1.0, value=1.8)  
    odds_away = st.number_input("Cote pour l'équipe à l'extérieur", min_value=1.0, value=2.2)  

    # Vérification des probabilités avant de calculer les mises  
    if log_reg_prob is not None and len(log_reg_prob) == 3:  
        kelly_home = kelly_criterion(log_reg_prob[2], odds_home)  # Victoire Domicile  
        kelly_away = kelly_criterion(log_reg_prob[0], odds_away)  # Victoire Extérieure  
        st.write(f"**Mise recommandée selon Kelly pour {home_team}:** {kelly_home:.2f}")  
        st.write(f"**Mise recommandée selon Kelly pour {away_team}:** {kelly_away:.2f}")  
    else:  
        st.error("Impossible de calculer les mises recommandées en raison d'une erreur dans les probabilités prédites.")  

    # Option de téléchargement des résultats  
    results = {  
        "Équipe Domicile": home_team,  
        "Équipe Extérieure": away_team,  
        "Buts Prédit Domicile": home_goals,  
        "Buts Prédit Extérieur": away_goals,  
        "Probabilité Domicile": log_reg_prob[2] if log_reg_prob is not None else None,  
        "Probabilité Nul": log_reg_prob[1] if log_reg_prob is not None else None,  
        "Probabilité Extérieure": log_reg_prob[0] if log_reg_prob is not None else None,  
        "Probabilité Régression Logistique": log_reg_prob,  
        "Probabilité Random Forest": rf_prob,  
        "Probabilité XGBoost": xgb_prob,  
        "Paris Double Chance 1X": double_chance["1X"] if 'double_chance' in locals() else None,  
        "Paris Double Chance X2": double_chance["X2"] if 'double_chance' in locals() else None,  
        "Paris Double Chance 12": double_chance["12"] if 'double_chance' in locals() else None,  
    }  

    if st.button("📥 Télécharger les résultats en DOC"):  
        buffer = create_doc(results)  
        st.download_button(  
            label="Télécharger les résultats",  
            data=buffer,  
            file_name="predictions.docx",  
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"  
        )
