import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.model_selection import StratifiedKFold, cross_val_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc  
import matplotlib.pyplot as plt  
import plotly.express as px  
import altair as alt  
from docx import Document  
import io  

# Configuration de la page  
st.set_page_config(page_title="⚽ Analyse de Match de Football", page_icon="⚽", layout="wide")  

# Initialisation de st.session_state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  
if 'historique' not in st.session_state:  
    st.session_state.historique = []  
if 'poids_criteres' not in st.session_state:  
    st.session_state.poids_criteres = {}  

# Fonction pour convertir une cote en probabilité implicite  
def cote_en_probabilite(cote):  
    return 1 / cote  

# Fonction pour générer un rapport DOC  
def generer_rapport(predictions):  
    doc = Document()  
    doc.add_heading("Rapport de Prédiction", level=1)  
    for prediction in predictions:  
        doc.add_paragraph(f"Équipe A: {prediction['proba_A']:.2%}, Équipe B: {prediction['proba_B']:.2%}, Match Nul: {prediction['proba_Nul']:.2%}")  
    return doc  

# Onglets pour l'application  
tab1, tab2 = st.tabs(["Analyse de Match", "Poids des Critères"])  

# Formulaire de collecte des données  
with tab1:  
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

    # Curseurs pour ajuster les poids des critères  
    st.markdown("#### ⚖️ Ajustez les Poids des Critères")  
    poids_xG_A = st.slider("Poids pour xG Équipe A", 0.0, 2.0, 1.0)  
    poids_xG_B = st.slider("Poids pour xG Équipe B", 0.0, 2.0, 1.0)  

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
        y = (df['buts_par_match_A'] > df['buts_par_match_B']).astype(int)  

        # Modèle Poisson  
        lambda_A = (  
            st.session_state.data['expected_but_A'] * poids_xG_A +  
            st.session_state.data['buts_par_match_A'] +  
            st.session_state.data['tirs_cadres_A'] * 0.1 +  
            st.session_state.data['grandes_chances_A'] * 0.2  
        )  

        lambda_B = (  
            st.session_state.data['expected_but_B'] * poids_xG_B +  
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
        Cela indique que le bookmaker sous-estime la probabilité de cet événement, ce qui en fait une opportunité potentiellement rentable.  
        """)  

# Onglet pour afficher les poids des critères  
with tab2:  
    st.subheader("📊 Poids des Critères du Modèle Random Forest")  
    if st.session_state.poids_criteres:  
        poids_df = pd.DataFrame({  
            'Critères': [  
                'Score Rating A', 'Buts Marqués A', 'Buts Concédés A', 'Possession Moyenne A',  
                'Expected Goals A', 'Expected Goals Against A', 'Tirs Cadrés A', 'Grandes Chances A',  
                'Absences A', 'Forme Récente A', 'Score Rating B', 'Buts Marqués B', 'Buts Concédés B',  
                'Possession Moyenne B', 'Expected Goals B', 'Expected Goals Against B', 'Tirs Cadrés B',  
                'Grandes Chances B', 'Absences B', 'Forme Récente B'  
            ],  
            'Poids': st.session_state.poids_criteres  
        })
                # Affichage des poids des critères dans un tableau  
        st.table(poids_df)  

        # Visualisation des poids des critères avec Plotly  
        fig = px.bar(poids_df, x='Critères', y='Poids', title='Poids des Critères du Modèle Random Forest',   
                      labels={'Poids': 'Poids', 'Critères': 'Critères'}, color='Poids')  
        st.plotly_chart(fig)  

        # Visualisation des poids des critères avec Altair  
        alt_chart = alt.Chart(poids_df).mark_bar().encode(  
            x=alt.X('Critères:N', sort='-y'),  
            y='Poids:Q',  
            color='Poids:Q'  
        ).properties(  
            title='Poids des Critères du Modèle Random Forest'  
        )  
        st.altair_chart(alt_chart, use_container_width=True)  

# Section de face-à-face entre les équipes  
st.markdown("### 🤝 Face-à-Face entre les Équipes")  
st.markdown("#### Critères d'Attaque et de Défense")  

# Collecte des données pour les critères d'attaque et de défense  
for i in range(1, 16):  
    col_attack_A, col_attack_B = st.columns(2)  
    with col_attack_A:  
        st.session_state.data[f'attaque_A_{i}'] = st.number_input(f"Attaque Critère {i} Équipe A", value=0.0, format="%.2f", key=f'attaque_A_{i}')  
    with col_attack_B:  
        st.session_state.data[f'attaque_B_{i}'] = st.number_input(f"Attaque Critère {i} Équipe B", value=0.0, format="%.2f", key=f'attaque_B_{i}')  

for i in range(1, 16):  
    col_defense_A, col_defense_B = st.columns(2)  
    with col_defense_A:  
        st.session_state.data[f'defense_A_{i}'] = st.number_input(f"Défense Critère {i} Équipe A", value=0.0, format="%.2f", key=f'defense_A_{i}')  
    with col_defense_B:  
        st.session_state.data[f'defense_B_{i}'] = st.number_input(f"Défense Critère {i} Équipe B", value=0.0, format="%.2f", key=f'defense_B_{i}')  

# Bouton de soumission pour les critères d'attaque et de défense  
if st.button("🔍 Analyser le Face-à-Face"):  
    # Récupération des données d'attaque et de défense  
    attaque_A = [st.session_state.data[f'attaque_A_{i}'] for i in range(1, 16)]  
    attaque_B = [st.session_state.data[f'attaque_B_{i}'] for i in range(1, 16)]  
    defense_A = [st.session_state.data[f'defense_A_{i}'] for i in range(1, 16)]  
    defense_B = [st.session_state.data[f'defense_B_{i}'] for i in range(1, 16)]  

    # Calcul des moyennes pour les critères d'attaque et de défense  
    moyenne_attaque_A = np.mean(attaque_A)  
    moyenne_attaque_B = np.mean(attaque_B)  
    moyenne_defense_A = np.mean(defense_A)  
    moyenne_defense_B = np.mean(defense_B)  

    # Affichage des résultats du face-à-face  
    st.subheader("📊 Résultats du Face-à-Face")  
    st.metric("Moyenne Attaque Équipe A", f"{moyenne_attaque_A:.2f}")  
    st.metric("Moyenne Attaque Équipe B", f"{moyenne_attaque_B:.2f}")  
    st.metric("Moyenne Défense Équipe A", f"{moyenne_defense_A:.2f}")  
    st.metric("Moyenne Défense Équipe B", f"{moyenne_defense_B:.2f}")  

    # Visualisation des résultats du face-à-face avec Plotly  
    df_face_a_face = pd.DataFrame({  
        'Critères': ['Attaque Équipe A', 'Attaque Équipe B', 'Défense Équipe A', 'Défense Équipe B'],  
        'Moyenne': [moyenne_attaque_A, moyenne_attaque_B, moyenne_defense_A, moyenne_defense_B]  
    })  

    fig_face_a_face = px.bar(df_face_a_face, x='Critères', y='Moyenne', title='Face-à-Face entre les Équipes',   
                             labels={'Moyenne': 'Moyenne', 'Critères': 'Critères'}, color='Moyenne')  
    st.plotly_chart(fig_face_a_face)  

    # Visualisation des résultats du face-à-face avec Altair  
    alt_face_a_face = alt.Chart(df_face_a_face).mark_bar().encode(  
        x=alt.X('Critères:N', sort='-y'),  
        y='Moyenne:Q',  
        color='Moyenne:Q'  
    ).properties(  
        title='Face-à-Face entre les Équipes'  
    )  
    st.altair_chart(alt_face_a_face, use_container_width=True)  

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  
- **📊 Prédiction des Buts (Poisson)** : Les buts moyens prévus pour chaque équipe sont calculés à partir des statistiques d'entrée.  
- **🤖 Performance des Modèles** : Les précisions des modèles de régression logistique et de forêt aléatoire sont affichées.  
- **📈 Comparateur de Cotes** : Les cotes prédites et les cotes des bookmakers sont comparées pour identifier les **Value Bets**.  
⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""")
