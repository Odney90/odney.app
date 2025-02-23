import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.model_selection import StratifiedKFold, cross_val_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc  
import plotly.express as px  
import matplotlib.pyplot as plt  
import traceback  
from docx import Document  

# Configuration de la page  
st.set_page_config(page_title="⚽ Analyse de Match de Football", page_icon="⚽", layout="wide")  

# Initialisation de st.session_state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  
if 'historique' not in st.session_state:  
    st.session_state.historique = []  

# Fonction pour convertir une cote en probabilité implicite  
def cote_en_probabilite(cote):  
    return 1 / cote  

# Fonction pour générer un rapport DOC  
def generer_rapport(prediction, poids_criteres):  
    doc = Document()  
    doc.add_heading("Rapport de Prédiction", level=1)  
    doc.add_paragraph(f"Prédiction : {prediction}")  
    doc.add_heading("Poids des Critères", level=2)  
    for critere, poids in poids_criteres.items():  
        doc.add_paragraph(f"{critere} : {poids:.2f}")  
    return doc  

# Formulaire de collecte des données  
with st.form("data_form"):  
    st.markdown("### 🏁 Entrez les Statistiques des Équipes")  

    # Équipe A  
    st.markdown("#### Équipe A")  
    col1, col2 = st.columns(2)  
    with col1:  
        st.session_state.data['score_rating_A'] = st.number_input("⭐ Score Rating", value=70.6, format="%.2f", key="rating_A")  
        st.session_state.data['buts_par_match_A'] = st.number_input("⚽ Buts Marqués", value=1.5, format="%.2f", key="buts_A")  
        st.session_state.data['buts_concedes_par_match_A'] = st.number_input("🥅 Buts Concédés", value=1.0, format="%.2f", key="concedes_A")  
        st.session_state.data['possession_moyenne_A'] = st.number_input("🎯 Possession Moyenne", value=55.0, format="%.2f", key="possession_A")  
    with col2:  
        st.session_state.data['expected_but_A'] = st.number_input("📊 Expected Goals (xG)", value=1.5, format="%.2f", key="xG_A")  
        st.session_state.data['expected_concedes_A'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.2, format="%.2f", key="xGA_A")  
        st.session_state.data['tirs_cadres_A'] = st.number_input("🎯 Tirs Cadrés", value=120.0, format="%.2f", key="tirs_A")  
        st.session_state.data['grandes_chances_A'] = st.number_input("🔥 Grandes Chances", value=25.0, format="%.2f", key="chances_A")  

    # Équipe B  
    st.markdown("#### Équipe B")  
    col3, col4 = st.columns(2)  
    with col3:  
        st.session_state.data['score_rating_B'] = st.number_input("⭐ Score Rating", value=65.7, format="%.2f", key="rating_B")  
        st.session_state.data['buts_par_match_B'] = st.number_input("⚽ Buts Marqués", value=1.0, format="%.2f", key="buts_B")  
        st.session_state.data['buts_concedes_par_match_B'] = st.number_input("🥅 Buts Concédés", value=1.5, format="%.2f", key="concedes_B")  
        st.session_state.data['possession_moyenne_B'] = st.number_input("🎯 Possession Moyenne", value=45.0, format="%.2f", key="possession_B")  
    with col4:  
        st.session_state.data['expected_but_B'] = st.number_input("📊 Expected Goals (xG)", value=1.2, format="%.2f", key="xG_B")  
        st.session_state.data['expected_concedes_B'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.8, format="%.2f", key="xGA_B")  
        st.session_state.data['tirs_cadres_B'] = st.number_input("🎯 Tirs Cadrés", value=100.0, format="%.2f", key="tirs_B")  
        st.session_state.data['grandes_chances_B'] = st.number_input("🔥 Grandes Chances", value=20.0, format="%.2f", key="chances_B")  

    # Nouveaux critères  
    st.markdown("#### 🆕 Nouveaux Critères")  
    col5, col6 = st.columns(2)  
    with col5:  
        st.session_state.data['absences_A'] = st.number_input("🚑 Absences (Équipe A)", value=2, key="absences_A")  
        st.session_state.data['forme_recente_A'] = st.number_input("📈 Forme Récente (Équipe A)", value=7.5, format="%.2f", key="forme_A")  
    with col6:  
        st.session_state.data['absences_B'] = st.number_input("🚑 Absences (Équipe B)", value=3, key="absences_B")  
        st.session_state.data['forme_recente_B'] = st.number_input("📈 Forme Récente (Équipe B)", value=6.0, format="%.2f", key="forme_B")  

    # Cotes des bookmakers  
    st.markdown("#### 📊 Cotes des Bookmakers")  
    col7, col8, col9 = st.columns(3)  
    with col7:  
        st.session_state.data['cote_bookmaker_A'] = st.number_input("Cote Victoire A", value=2.0, format="%.2f", key="cote_A")  
    with col8:  
        st.session_state.data['cote_bookmaker_B'] = st.number_input("Cote Victoire B", value=3.0, format="%.2f", key="cote_B")  
    with col9:  
        st.session_state.data['cote_bookmaker_Nul'] = st.number_input("Cote Match Nul", value=3.5, format="%.2f", key="cote_Nul")  

    # Bouton de soumission du formulaire  
    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Section d'analyse et de prédiction  
if submitted:  
    try:  
        # Génération de données synthétiques pour la validation croisée  
        n_samples = 100  # Nombre d'échantillons synthétiques  
        np.random.seed(42)  

        # Données synthétiques pour l'équipe A  
        data_A = {  
            'score_rating_A': np.random.normal(st.session_state.data['score_rating_A'], 5, n_samples),  
            'buts_par_match_A': np.random.normal(st.session_state.data['buts_par_match_A'], 0.5, n_samples),  
            'buts_concedes_par_match_A': np.random.normal(st.session_state.data['buts_concedes_par_match_A'], 0.5, n_samples),  
            'possession_moyenne_A': np.random.normal(st.session_state.data['possession_moyenne_A'], 5, n_samples),  
            'expected_but_A': np.random.normal(st.session_state.data['expected_but_A'], 0.5, n_samples),  
            'expected_concedes_A': np.random.normal(st.session_state.data['expected_concedes_A'], 0.5, n_samples),  
            'tirs_cadres_A': np.random.normal(st.session_state.data['tirs_cadres_A'], 10, n_samples),  
            'grandes_chances_A': np.random.normal(st.session_state.data['grandes_chances_A'], 5, n_samples),  
            'absences_A': np.random.normal(st.session_state.data['absences_A'], 1, n_samples),  
            'forme_recente_A': np.random.normal(st.session_state.data['forme_recente_A'], 1, n_samples),  
        }  

        # Données synthétiques pour l'équipe B  
        data_B = {  
            'score_rating_B': np.random.normal(st.session_state.data['score_rating_B'], 5, n_samples),  
            'buts_par_match_B': np.random.normal(st.session_state.data['buts_par_match_B'], 0.5, n_samples),  
            'buts_concedes_par_match_B': np.random.normal(st.session_state.data['buts_concedes_par_match_B'], 0.5, n_samples),  
            'possession_moyenne_B': np.random.normal(st.session_state.data['possession_moyenne_B'], 5, n_samples),  
            'expected_but_B': np.random.normal(st.session_state.data['expected_but_B'], 0.5, n_samples),  
            'expected_concedes_B': np.random.normal(st.session_state.data['expected_concedes_B'], 0.5, n_samples),  
            'tirs_cadres_B': np.random.normal(st.session_state.data['tirs_cadres_B'], 10, n_samples),  
            'grandes_chances_B': np.random.normal(st.session_state.data['grandes_chances_B'], 5, n_samples),  
            'absences_B': np.random.normal(st.session_state.data['absences_B'], 1, n_samples),  
            'forme_recente_B': np.random.normal(st.session_state.data['forme_recente_B'], 1, n_samples),  
        }  

        # Création du DataFrame synthétique  
        df_A = pd.DataFrame(data_A)  
        df_B = pd.DataFrame(data_B)  
        df = pd.concat([df_A, df_B], axis=1)  

        # Création de la variable cible (1 si l'équipe A gagne, 0 sinon)  
        y = (df['buts_par_match_A'] > df['buts_par_match_B'].astype(int))  

        # Modèle Poisson  
        lambda_A = (  
            st.session_state.data['expected_but_A'] +  
            st.session_state.data['buts_par_match_A'] +  
            st.session_state.data['tirs_cadres_A'] * 0.1 +  
            st.session_state.data['grandes_chances_A'] * 0.2  
        )  

        lambda_B = (  
            st.session_state.data['expected_but_B'] +  
            st.session_state.data['buts_par_match_B'] +  
            st.session_state.data['tirs_cadres_B'] * 0.1 +  
            st.session_state.data['grandes_chances_B'] * 0.2  
        )  

        # Prédiction des buts avec Poisson  
        buts_A = poisson.rvs(mu=lambda_A, size=1000)  
        buts_B = poisson.rvs(mu=lambda_B, size=1000)  

        # Résultats Poisson  
        st.subheader("📊 Prédiction des Buts (Poisson)")  
        col_poisson_A, col_poisson_B = st.columns(2)  
        with col_poisson_A:  
            st.metric("⚽ Buts Moyens (Équipe A)", f"{np.mean(buts_A):.2f}")  
            st.metric("⚽ Buts Prévus (75e percentile)", f"{np.percentile(buts_A, 75):.2f}", help="75% des simulations prévoient moins de buts que cette valeur.")  
        with col_poisson_B:  
            st.metric("⚽ Buts Moyens (Équipe B)", f"{np.mean(buts_B):.2f}")  
            st.metric("⚽ Buts Prévus (75e percentile)", f"{np.percentile(buts_B, 75):.2f}", help="75% des simulations prévoient moins de buts que cette valeur.")  

        # Modèles de classification avec validation croisée  
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

        # Logistic Regression  
        log_reg = LogisticRegression()  
        log_reg_scores = cross_val_score(log_reg, df, y, cv=skf, scoring='accuracy')  
        log_reg_mean_score = np.mean(log_reg_scores)  

        # Random Forest Classifier  
        rf_clf = RandomForestClassifier()  
        rf_scores = cross_val_score(rf_clf, df, y, cv=skf, scoring='accuracy')  
        rf_mean_score = np.mean(rf_scores)  

        # Comparaison des probabilités de victoire  
        proba_A = np.mean(buts_A) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_B = np.mean(buts_B) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_Nul = 1 - (proba_A + proba_B)  

        # Cotes prédites  
        cote_predite_A = 1 / proba_A  
        cote_predite_B = 1 / proba_B  
        cote_predite_Nul = 1 / proba_Nul  

        # Convertisseur de cote implicite en probabilité  
        st.subheader("📊 Convertisseur de Cote Implicite en Probabilité")  
        cote_input = st.number_input("Entrez une cote pour convertir en probabilité implicite", value=2.0, format="%.2f")  
        probabilite_implicite = cote_en_probabilite(cote_input)  
        st.metric("Probabilité Implicite", f"{probabilite_implicite:.2%}")  

        # Tableau synthétique des résultats  
        st.subheader("📊 Tableau Synthétique des Résultats")  
        data = {  
            "Équipe": ["Équipe A", "Équipe B", "Match Nul"],  
            "Probabilité Prédite": [f"{proba_A:.2%}", f"{proba_B:.2%}", f"{proba_Nul:.2%}"],  
            "Cote Prédite": [f"{cote_predite_A:.2f}", f"{cote_predite_B:.2f}", f"{cote_predite_Nul:.2f}"],  
            "Cote Bookmaker": [  
                f"{st.session_state.data['cote_bookmaker_A']:.2f}",  
                f"{st.session_state.data['cote_bookmaker_B']:.2f}",  
                f"{st.session_state.data['cote_bookmaker_Nul']:.2f}",  
            ],  
            "Value Bet": [  
                "✅" if cote_predite_A < st.session_state.data['cote_bookmaker_A'] else "❌",  
                "✅" if cote_predite_B < st.session_state.data['cote_bookmaker_B'] else "❌",  
                "✅" if cote_predite_Nul < st.session_state.data['cote_bookmaker_Nul'] else "❌",  
            ],  
        }  
        df_resultats = pd.DataFrame(data)  
        st.table(df_resultats)  

        # Message rappel sur le Value Bet  
        st.markdown("""  
        ### 💡 Qu'est-ce qu'un Value Bet ?  
        Un **Value Bet** est un pari où la cote prédite par le modèle est **inférieure** à la cote proposée par le bookmaker.   
        Cela indique que le bookmaker sous-estime la probabilité de cet événement, ce qui en fait une opportunité potentiellement rentable.  
        """)  

    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(traceback.format_exc())  

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  
- **📊 Prédiction des Buts (Poisson)** : Les buts moyens prévus pour chaque équipe sont calculés à partir des statistiques d'entrée.  
- **🤖 Performance des Modèles** : Les précisions des modèles de régression logistique et de forêt aléatoire sont affichées.  
- **📈 Comparateur de Cotes** : Les cotes prédites et les cotes des bookmakers sont comparées pour identifier les **Value Bets**.  
⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""")
