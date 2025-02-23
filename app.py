import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.model_selection import StratifiedKFold, cross_val_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score  
import traceback  

# Configuration de la page  
st.set_page_config(page_title="⚽ Analyse de Match de Football", page_icon="⚽", layout="wide")  

# Initialisation de st.session_state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Formulaire de collecte des données  
with st.form("data_form"):  
    # Équipe A  
    st.markdown("### Équipe A")  
    st.session_state.data['score_rating_A'] = st.number_input("⭐ Score Rating", value=70.6, format="%.2f", key="rating_A")  
    st.session_state.data['buts_par_match_A'] = st.number_input("⚽ Buts Marqués", value=1.5, format="%.2f", key="buts_A")  
    st.session_state.data['buts_concedes_par_match_A'] = st.number_input("🥅 Buts Concédés", value=1.0, format="%.2f", key="concedes_A")  
    st.session_state.data['possession_moyenne_A'] = st.number_input("🎯 Possession Moyenne", value=55.0, format="%.2f", key="possession_A")  
    st.session_state.data['expected_but_A'] = st.number_input("📊 Expected Goals (xG)", value=1.5, format="%.2f", key="xG_A")  
    st.session_state.data['expected_concedes_A'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.2, format="%.2f", key="xGA_A")  
    st.session_state.data['tirs_cadres_A'] = st.number_input("🎯 Tirs Cadrés", value=120.0, format="%.2f", key="tirs_A")  
    st.session_state.data['grandes_chances_A'] = st.number_input("🔥 Grandes Chances", value=25.0, format="%.2f", key="chances_A")  
    st.session_state.data['corners_A'] = st.number_input("🔄 Corners", value=60.0, format="%.2f", key="corners_A")  
    st.session_state.data['fautes_A'] = st.number_input("⚠️ Fautes", value=15.0, format="%.2f", key="fautes_A")  
    st.session_state.data['motivation_A'] = st.number_input("💪 Motivation", value=4.0, format="%.2f", key="motivation_A")  
    st.session_state.data['jours_repos_A'] = st.number_input("⏳ Jours de Repos", value=4.0, format="%.2f", key="repos_A")  

    # Équipe B  
    st.markdown("### Équipe B")  
    st.session_state.data['score_rating_B'] = st.number_input("⭐ Score Rating", value=65.7, format="%.2f", key="rating_B")  
    st.session_state.data['buts_par_match_B'] = st.number_input("⚽ Buts Marqués", value=1.0, format="%.2f", key="buts_B")  
    st.session_state.data['buts_concedes_par_match_B'] = st.number_input("🥅 Buts Concédés", value=1.5, format="%.2f", key="concedes_B")  
    st.session_state.data['possession_moyenne_B'] = st.number_input("🎯 Possession Moyenne", value=45.0, format="%.2f", key="possession_B")  
    st.session_state.data['expected_but_B'] = st.number_input("📊 Expected Goals (xG)", value=1.2, format="%.2f", key="xG_B")  
    st.session_state.data['expected_concedes_B'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.8, format="%.2f", key="xGA_B")  
    st.session_state.data['tirs_cadres_B'] = st.number_input("🎯 Tirs Cadrés", value=100.0, format="%.2f", key="tirs_B")  
    st.session_state.data['grandes_chances_B'] = st.number_input("🔥 Grandes Chances", value=20.0, format="%.2f", key="chances_B")  
    st.session_state.data['corners_B'] = st.number_input("🔄 Corners", value=50.0, format="%.2f", key="corners_B")  
    st.session_state.data['fautes_B'] = st.number_input("⚠️ Fautes", value=20.0, format="%.2f", key="fautes_B")  
    st.session_state.data['motivation_B'] = st.number_input("💪 Motivation", value=3.0, format="%.2f", key="motivation_B")  
    st.session_state.data['jours_repos_B'] = st.number_input("⏳ Jours de Repos", value=3.0, format="%.2f", key="repos_B")  

    # Bouton de soumission du formulaire  
    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Section d'analyse et de prédiction  
if submitted:  
    try:  
        # Préparation des données pour les modèles  
        features = pd.DataFrame({  
            'score_rating_A': [st.session_state.data['score_rating_A']],  
            'buts_par_match_A': [st.session_state.data['buts_par_match_A']],  
            'buts_concedes_par_match_A': [st.session_state.data['buts_concedes_par_match_A']],  
            'possession_moyenne_A': [st.session_state.data['possession_moyenne_A']],  
            'expected_but_A': [st.session_state.data['expected_but_A']],  
            'expected_concedes_A': [st.session_state.data['expected_concedes_A']],  
            'tirs_cadres_A': [st.session_state.data['tirs_cadres_A']],  
            'grandes_chances_A': [st.session_state.data['grandes_chances_A']],  
            'corners_A': [st.session_state.data['corners_A']],  
            'fautes_A': [st.session_state.data['fautes_A']],  
            'motivation_A': [st.session_state.data['motivation_A']],  
            'jours_repos_A': [st.session_state.data['jours_repos_A']],  
            'score_rating_B': [st.session_state.data['score_rating_B']],  
            'buts_par_match_B': [st.session_state.data['buts_par_match_B']],  
            'buts_concedes_par_match_B': [st.session_state.data['buts_concedes_par_match_B']],  
            'possession_moyenne_B': [st.session_state.data['possession_moyenne_B']],  
            'expected_but_B': [st.session_state.data['expected_but_B']],  
            'expected_concedes_B': [st.session_state.data['expected_concedes_B']],  
            'tirs_cadres_B': [st.session_state.data['tirs_cadres_B']],  
            'grandes_chances_B': [st.session_state.data['grandes_chances_B']],  
            'corners_B': [st.session_state.data['corners_B']],  
            'fautes_B': [st.session_state.data['fautes_B']],  
            'motivation_B': [st.session_state.data['motivation_B']],  
            'jours_repos_B': [st.session_state.data['jours_repos_B']],  
        })  

        # Modèle Poisson  
        lambda_A = (  
            st.session_state.data['expected_but_A'] +  
            st.session_state.data['buts_par_match_A'] +  
            st.session_state.data['tirs_cadres_A'] * 0.1 +  
            st.session_state.data['grandes_chances_A'] * 0.2 +  
            st.session_state.data['corners_A'] * 0.05  
        )  

        lambda_B = (  
            st.session_state.data['expected_but_B'] +  
            st.session_state.data['buts_par_match_B'] +  
            st.session_state.data['tirs_cadres_B'] * 0.1 +  
            st.session_state.data['grandes_chances_B'] * 0.2 +  
            st.session_state.data['corners_B'] * 0.05  
        )  

        # Prédiction des buts avec Poisson  
        buts_A = poisson.rvs(mu=lambda_A, size=1000)  
        buts_B = poisson.rvs(mu=lambda_B, size=1000)  

        # Résultats Poisson  
        st.subheader("📊 Prédiction des Buts (Poisson)")  
        col_poisson_A, col_poisson_B = st.columns(2)  
        with col_poisson_A:  
            st.metric("⚽ Buts Moyens (Équipe A)", f"{np.mean(buts_A):.2f}")  
            st.metric("⚽ Buts Prévus (Équipe A)", f"{np.percentile(buts_A, 75):.2f} (75e percentile)")  
        with col_poisson_B:  
            st.metric("⚽ Buts Moyens (Équipe B)", f"{np.mean(buts_B):.2f}")  
            st.metric("⚽ Buts Prévus (Équipe B)", f"{np.percentile(buts_B, 75):.2f} (75e percentile)")  

        # Modèles de classification  
        X = features  
        y = np.array([1 if st.session_state.data['buts_par_match_A'] > st.session_state.data['buts_par_match_B'] else 0])  

        # Validation croisée  
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

        # Logistic Regression  
        log_reg = LogisticRegression()  
        log_reg_scores = cross_val_score(log_reg, X, y, cv=skf, scoring='accuracy')  
        log_reg_mean_score = np.mean(log_reg_scores)  

        # Random Forest Classifier  
        rf_clf = RandomForestClassifier()  
        rf_scores = cross_val_score(rf_clf, X, y, cv=skf, scoring='accuracy')  
        rf_mean_score = np.mean(rf_scores)  

        # Affichage des résultats des modèles  
        st.subheader("📈 Résultats des Modèles de Prédiction")  
        st.write(f"**Logistic Regression Accuracy**: {log_reg_mean_score:.2%}")  
        st.write(f"**Random Forest Classifier Accuracy**: {rf_mean_score:.2%}")  

        # Comparaison des probabilités de victoire  
        proba_A = np.mean(buts_A) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_B = np.mean(buts_B) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_Nul = 1 - (proba_A + proba_B)  

        st.write("##### Probabilités Prédites (Modèle Poisson)")  
        col_predite_A, col_predite_B, col_predite_Nul = st.columns(3)  
        with col_predite_A:  
            st.metric("Victoire A", f"{proba_A:.2%}")  
        with col_predite_B:  
            st.metric("Victoire B", f"{proba_B:.2%}")  
        with col_predite_Nul:  
            st.metric("Match Nul", f"{proba_Nul:.2%}")  

    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(traceback.format_exc())  

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  
- **📊 Prédiction des Buts (Poisson)** : Les buts moyens prévus pour chaque équipe sont calculés à partir des statistiques d'entrée.  
- **🤖 Performance des Modèles** : Les précisions des modèles de régression logistique et de forêt aléatoire sont affichées.  
- **📈 Comparaison des Performances des Modèles** : Les probabilités de victoire sont calculées à partir des résultats des simulations.  
⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""")
