import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.model_selection import StratifiedKFold, cross_val_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
import plotly.express as px  
from docx import Document  
import io  
import altair as alt  

# Configuration de la page  
st.set_page_config(page_title="⚽ Analyse de Match de Football", page_icon="⚽", layout="wide")  

# Initialisation de st.session_state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  
if 'historique' not in st.session_state:  
    st.session_state.historique = []  
if 'poids_criteres' not in st.session_state:  
    st.session_state.poids_criteres = []  

# Fonction pour générer un rapport DOC  
def generer_rapport(predictions):  
    doc = Document()  
    doc.add_heading("Rapport de Prédiction", level=1)  
    for prediction in predictions:  
        doc.add_paragraph(f"Équipe A: {prediction['proba_A']:.2%}, Équipe B: {prediction['proba_B']:.2%}, Match Nul: {prediction['proba_Nul']:.2%}")  
    return doc  

# Saisie des Données  
st.markdown("### 🏁 Entrez les Statistiques des Équipes")  

with st.form(key='match_form'):  
    # Équipe A  
    st.markdown("#### Équipe A")  
    col1, col2 = st.columns(2)  
    with col1:  
        st.session_state.data['score_rating_A'] = st.number_input("⭐ Score Rating", value=70.6, format="%.2f", key="rating_A")  
        st.session_state.data['buts_par_match_A'] = st.number_input("⚽ Buts Marqués", value=1.5, format="%.2f", key="buts_A")  
        st.session_state.data['buts_concedes_par_match_A'] = st.number_input("🥅 Buts Concédés", value=1.0, format="%.2f", key="concedes_A")  
        st.session_state.data['possession_moyenne_A'] = st.number_input("🎯 Possession Moyenne", value=55.0, format="%.2f", key="possession_A")  
        st.session_state.data['expected_but_A'] = st.number_input("📊 Expected Goals (xG)", value=1.5, format="%.2f", key="xG_A")  
        st.session_state.data['expected_concedes_A'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.2, format="%.2f", key="xGA_A")  
        st.session_state.data['tirs_cadres_A'] = st.number_input("🎯 Tirs Cadrés", value=120.0, format="%.2f", key="tirs_A")  
        st.session_state.data['grandes_chances_A'] = st.number_input("🔥 Grandes Chances", value=25.0, format="%.2f", key="chances_A")  
    with col2:  
        st.session_state.data['victoires_domicile_A'] = st.number_input("🏠 Victoires à Domicile", value=5, key="victoires_domicile_A")  
        st.session_state.data['victoires_exterieur_A'] = st.number_input("✈️ Victoires à l'Extérieur", value=3, key="victoires_exterieur_A")  
        st.session_state.data['joueurs_absents_A'] = st.number_input("🚫 Joueurs Absents", value=2, key="joueurs_absents_A")  
        st.session_state.data['joueurs_cles_absents_A'] = st.number_input("🔑 Joueurs Clés Absents", value=1, key="joueurs_cles_absents_A")  
        st.session_state.data['degagements_A'] = st.number_input("🚀 Dégagements", value=50, key="degagements_A")  
        st.session_state.data['balles_touchees_camp_adverse_A'] = st.number_input("⚽ Balles Touchées Camp Adverse", value=30, key="balles_touchees_A")  
        st.session_state.data['motivation_A'] = st.number_input("💪 Motivation (1-10)", value=8, key="motivation_A")  
        st.session_state.data['face_a_face_A'] = st.number_input("🤝 Face à Face (Victoires)", value=3, key="face_a_face_A")  

    # Équipe B  
    st.markdown("#### Équipe B")  
    col3, col4 = st.columns(2)  
    with col3:  
        st.session_state.data['score_rating_B'] = st.number_input("⭐ Score Rating", value=65.7, format="%.2f", key="rating_B")  
        st.session_state.data['buts_par_match_B'] = st.number_input("⚽ Buts Marqués", value=1.0, format="%.2f", key="buts_B")  
        st.session_state.data['buts_concedes_par_match_B'] = st.number_input("🥅 Buts Concédés", value=1.5, format="%.2f", key="concedes_B")  
        st.session_state.data['possession_moyenne_B'] = st.number_input("🎯 Possession Moyenne", value=45.0, format="%.2f", key="possession_B")  
        st.session_state.data['expected_but_B'] = st.number_input("📊 Expected Goals (xG)", value=1.2, format="%.2f", key="xG_B")  
        st.session_state.data['expected_concedes_B'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.8, format="%.2f", key="xGA_B")  
        st.session_state.data['tirs_cadres_B'] = st.number_input("🎯 Tirs Cadrés", value=100.0, format="%.2f", key="tirs_B")  
        st.session_state.data['grandes_chances_B'] = st.number_input("🔥 Grandes Chances", value=20.0, format="%.2f", key="chances_B")  
    with col4:  
        st.session_state.data['victoires_domicile_B'] = st.number_input("🏠 Victoires à Domicile", value=4, key="victoires_domicile_B")  
        st.session_state.data['victoires_exterieur_B'] = st.number_input("✈️ Victoires à l'Extérieur", value=2, key="victoires_exterieur_B")  
        st.session_state.data['joueurs_absents_B'] = st.number_input("🚫 Joueurs Absents", value=1, key="joueurs_absents_B")  
        st.session_state.data['joueurs_cles_absents_B'] = st.number_input("🔑 Joueurs Clés Absents", value=0, key="joueurs_cles_absents_B")  
        st.session_state.data['degagements_B'] = st.number_input("🚀 Dégagements", value=40, key="degagements_B")  
        st.session_state.data['balles_touchees_camp_adverse_B'] = st.number_input("⚽ Balles Touchées Camp Adverse", value=25, key="balles_touchees_B")  
        st.session_state.data['motivation_B'] = st.number_input("💪 Motivation (1-10)", value=7, key="motivation_B")  
        st.session_state.data['face_a_face_B'] = st.number_input("🤝 Face à Face (Victoires)", value=2, key="face_a_face_B")  

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

# Analyse des résultats  
if submitted:  
    try:  
        # Création d'un DataFrame pour les données  
        data = {  
            'score_rating': [st.session_state.data['score_rating_A'], st.session_state.data['score_rating_B']],  
            'buts_par_match': [st.session_state.data['buts_par_match_A'], st.session_state.data['buts_par_match_B']],  
            'buts_concedes_par_match': [st.session_state.data['buts_concedes_par_match_A'], st.session_state.data['buts_concedes_par_match_B']],  
            'possession_moyenne': [st.session_state.data['possession_moyenne_A'], st.session_state.data['possession_moyenne_B']],  
            'expected_but': [st.session_state.data['expected_but_A'], st.session_state.data['expected_but_B']],  
            'expected_concedes': [st.session_state.data['expected_concedes_A'], st.session_state.data['expected_concedes_B']],  
            'tirs_cadres': [st.session_state.data['tirs_cadres_A'], st.session_state.data['tirs_cadres_B']],  
            'grandes_chances': [st.session_state.data['grandes_chances_A'], st.session_state.data['grandes_chances_B']],  
            'victoires_domicile': [st.session_state.data['victoires_domicile_A'], st.session_state.data['victoires_domicile_B']],  
            'victoires_exterieur': [st.session_state.data['victoires_exterieur_A'], st.session_state.data['victoires_exterieur_B']],  
            'joueurs_absents': [st.session_state.data['joueurs_absents_A'], st.session_state.data['joueurs_absents_B']],  
            'joueurs_cles_absents': [st.session_state.data['joueurs_cles_absents_A'], st.session_state.data['joueurs_cles_absents_B']],  
            'degagements': [st.session_state.data['degagements_A'], st.session_state.data['degagements_B']],  
            'balles_touchees_camp_adverse': [st.session_state.data['balles_touchees_camp_adverse_A'], st.session_state.data['balles_touchees_camp_adverse_B']],  
            'motivation': [st.session_state.data['motivation_A'], st.session_state.data['motivation_B']],  
            'face_a_face': [st.session_state.data['face_a_face_A'], st.session_state.data['face_a_face_B']],  
        }  
        df = pd.DataFrame(data, index=['Équipe A', 'Équipe B'])  

        # Création de la variable cible (1 si l'équipe A gagne, 0 sinon)  
        y = np.array([1, 0])  # Équipe A gagne par défaut  

        # Modèle Poisson  
        lambda_A = (  
            st.session_state.data['expected_but_A'] +  
            st.session_state.data['buts_par_match_A'] +  
            st.session_state.data['tirs_cadres_A'] * 0.1 +  
            st.session_state.data['grandes_chances_A'] * 0.2 +  
            (st.session_state.data['victoires_domicile_A'] * 0.3) - (st.session_state.data['joueurs_absents_B'] * 0.2)  # Impact des victoires et joueurs absents  
        )  

        lambda_B = (  
            st.session_state.data['expected_but_B'] +  
            st.session_state.data['buts_par_match_B'] +  
            st.session_state.data['tirs_cadres_B'] * 0.1 +  
            st.session_state.data['grandes_chances_B'] * 0.2 +  
            (st.session_state.data['victoires_domicile_B'] * 0.3) - (st.session_state.data['joueurs_absents_A'] * 0.2)  # Impact des victoires et joueurs absents  
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

        # Entraînement du modèle Random Forest pour obtenir les poids des critères  
        rf_clf.fit(df, y)  
        st.session_state.poids_criteres = rf_clf.feature_importances_  

        # Affichage des résultats des modèles  
        st.subheader("🤖 Performance des Modèles")  
        col_log_reg, col_rf = st.columns(2)  
        with col_log_reg:  
            st.metric("📈 Régression Logistique (Précision)", f"{log_reg_mean_score:.2%}")  
        with col_rf:  
            st.metric("🌲 Forêt Aléatoire (Précision)", f"{rf_mean_score:.2%}")  

        # Comparaison des probabilités de victoire  
        proba_A = np.mean(buts_A) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_B = np.mean(buts_B) / (np.mean(buts_A) + np.mean(buts_B))  
        proba_Nul = 1 - (proba_A + proba_B)  

        # Cotes prédites  
        cote_predite_A = 1 / proba_A  
        cote_predite_B = 1 / proba_B  
        cote_predite_Nul = 1 / proba_Nul  

        # Stockage des prédictions dans l'historique  
        st.session_state.historique.append({  
            'proba_A': proba_A,  
            'proba_B': proba_B,  
            'proba_Nul': proba_Nul,  
        })  

        # Génération du rapport DOC  
        if st.button("📄 Télécharger le Rapport"):  
            doc = generer_rapport(st.session_state.historique)  
            buffer = io.BytesIO()  
            doc.save(buffer)  
            buffer.seek(0)  
            st.download_button("Télécharger le rapport", buffer, "rapport_predictions.docx")  

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
        Cela indique que le bookmaker sous-estime la probabilité de cet événement.  
        """)  

    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  
- **📊 Prédiction des Buts (Poisson)** : Les buts moyens prévus pour chaque équipe sont calculés à partir des statistiques d'entrée.  
- **🤖 Performance des Modèles** : Les précisions des modèles de régression logistique et de forêt aléatoire sont affichées.  
- **📈 Comparateur de Cotes** : Les cotes prédites et les cotes des bookmakers sont comparées pour identifier les **Value Bets**.  
⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""") 

# Fin de l'application  
if __name__ == "__main__":  
    st.write("Merci d'utiliser l'application d'analyse de match de football !")
