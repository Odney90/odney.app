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

    # Ajout des probabilités des modèles  
    doc.add_heading('Probabilités des Modèles', level=2)  
    doc.add_paragraph(f"Probabilité Domicile: {results['Probabilité Domicile']:.2f}")  
    doc.add_paragraph(f"Probabilité Nul: {results['Probabilité Nul']:.2f}")  
    doc.add_paragraph(f"Probabilité Extérieure: {results['Probabilité Extérieure']:.2f}")  

    # Ajout des probabilités des paris double chance  
    doc.add_heading('Probabilités des Paris Double Chance', level=2)  
    for bet, prob in results.items():  
        if "Paris Double Chance" in bet:  
            doc.add_paragraph(f"{bet}: {prob:.2f}")  

    # Enregistrement du document  
    buffer = BytesIO()  
    doc.save(buffer)  
    buffer.seek(0)  
    return buffer  

# Fonction pour entraîner et prédire avec les modèles  
@st.cache_resource  
def train_models():  
    # Créer un ensemble de données d'entraînement fictif plus petit  
    data = pd.DataFrame({  
        'home_goals': np.random.randint(0, 3, size=50),  # Réduire la taille à 50  
        'away_goals': np.random.randint(0, 3, size=50),  
        'home_xG': np.random.uniform(0, 2, size=50),  
        'away_xG': np.random.uniform(0, 2, size=50),  
        'home_defense': np.random.randint(0, 3, size=50),  
        'away_defense': np.random.randint(0, 3, size=50),  
        'result': np.random.choice([0, 1, 2], size=50)  # 0 pour victoire extérieure, 1 pour match nul, 2 pour victoire domicile  
    })  

    # Séparer les caractéristiques et la cible  
    X = data[['home_goals', 'away_goals', 'home_xG', 'away_xG', 'home_defense', 'away_defense']]  
    y = data['result']  

    # Modèle de régression logistique  
    log_reg = LogisticRegression(max_iter=100)  # Limiter le nombre d'itérations  
    log_reg.fit(X, y)  

    # Modèle Random Forest  
    rf = RandomForestClassifier(n_estimators=10)  # Réduire le nombre d'estimateurs  
    rf.fit(X, y)  

    # Modèle XGBoost  
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=10)  # Réduire le nombre d'estimateurs  
    xgb.fit(X, y)  

    return log_reg, rf, xgb  

# Vérifier si les modèles sont déjà chargés dans l'état de session  
if 'models' not in st.session_state:  
    st.session_state.models = train_models()  

log_reg_model, rf_model, xgb_model = st.session_state.models  

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

# Saisie des cotes des bookmakers  
st.header("Saisie des cotes des bookmakers")  
odds_home = st.number_input("Cote pour l'équipe à domicile", min_value=1.0, value=1.8)  
odds_away = st.number_input("Cote pour l'équipe à l'extérieur", min_value=1.0, value=2.2)  

# Prédictions  
if st.button("🔍 Prédire les résultats"):  
    home_goals = home_stats["moyenne_buts_marques"] + home_stats["xG"] - away_stats["moyenne_buts_encais"]  
    away_goals = away_stats["moyenne_buts_marques"] + away_stats["xG"] - home_stats["moyenne_buts_encais"]  
    
    home_prob, away_prob = poisson_prediction(home_goals, away_goals)  

    # Prédictions avec les modèles  
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

    # Section pour les buts prédit  
    st.markdown("### Nombre de buts prédit")  
    st.write(f"**Nombre de buts prédit pour {home_team} :** {home_goals:.2f} ({home_prob * 100:.2f}%)")  
    st.write(f"**Nombre de buts prédit pour {away_team} :** {away_goals:.2f} ({away_prob * 100:.2f}%)")  

    # Section pour les probabilités des modèles  
    st.markdown("### Probabilités des Modèles")  
    model_results = {  
        "Modèle": ["Régression Logistique", "Random Forest", "XGBoost"],  
        "Victoire Domicile (%)": [log_reg_prob[2] * 100 if log_reg_prob is not None else None,  
                                  rf_prob[2] * 100 if rf_prob is not None else None,  
                                  xgb_prob[2] * 100 if xgb_prob is not None else None],  
        "Match Nul (%)": [log_reg_prob[1] * 100 if log_reg_prob is not None else None,  
                          rf_prob[1] * 100 if rf_prob is not None else None,  
                          xgb_prob[1] * 100 if xgb_prob is not None else None],  
        "Victoire Extérieure (%)": [log_reg_prob[0] * 100 if log_reg_prob is not None else None,  
                                    rf_prob[0] * 100 if rf_prob is not None else None,  
                                    xgb_prob[0] * 100 if xgb_prob is not None else None],  
    }  
    model_results_df = pd.DataFrame(model_results)  
    st.dataframe(model_results_df)  

    # Calcul des paris double chance  
    if home_prob is not None and away_prob is not None:  
        double_chance = double_chance_probabilities(home_prob, away_prob)  
        st.markdown("### Probabilités des Paris Double Chance")  
        double_chance_df = pd.DataFrame(double_chance.items(), columns=["Paris", "Probabilité (%)"])  
        st.dataframe(double_chance_df)  

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

    # Conversion des cotes en probabilités implicites  
    implied_prob_home = 1 / odds_home  
    implied_prob_away = 1 / odds_away  
    st.write(f"**Probabilité implicite pour {home_team} :** {implied_prob_home * 100:.2f}%")  
    st.write(f"**Probabilité implicite pour {away_team} :** {implied_prob_away * 100:.2f}%")  

    # Comparaison avec les probabilités des modèles  
    st.subheader("🔍 Comparaison des Probabilités")  
    if log_reg_prob is not None and len(log_reg_prob) == 3:  
        st.write(f"**Probabilité de victoire selon la régression logistique pour {home_team} :** {log_reg_prob[2] * 100:.2f}%")  
        st.write(f"**Probabilité de victoire selon la régression logistique pour {away_team} :** {log_reg_prob[0] * 100:.2f}%")  

        # Détection de valeur de pari  
        if log_reg_prob[2] > implied_prob_home:  
            st.success(f"Opportunité de pari sur {home_team} !")  
        if log_reg_prob[0] > implied_prob_away:  
            st.success(f"Opportunité de pari sur {away_team} !")  

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
