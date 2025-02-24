# Imports  
import streamlit as st  
import pandas as pd  
import numpy as np  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import KFold, cross_val_score  # Remplacement de StratifiedKFold par KFold  
import io  
import altair as alt  
import plotly.express as px  

# Fonction pour générer le rapport DOC  
def generer_rapport(historique):  
    from docx import Document  
    doc = Document()  
    doc.add_heading('Rapport de Prédictions', 0)  
    for i, pred in enumerate(historique):  
        doc.add_heading(f"Prédiction {i+1}", level=1)  
        doc.add_paragraph(f"Probabilité Équipe A: {pred['proba_A']:.2%}")  
        doc.add_paragraph(f"Probabilité Équipe B: {pred['proba_B']:.2%}")  
        doc.add_paragraph(f"Probabilité Match Nul: {pred['proba_Nul']:.2%}")  
    return doc  

# Initialisation de la session pour la persistance des données  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        'score_rating_A': 85.0,  
        'score_rating_B': 78.0,  
        'buts_par_match_A': 1.8,  
        'buts_par_match_B': 1.5,  
        'buts_concedes_par_match_A': 1.2,  
        'buts_concedes_par_match_B': 1.4,  
        'possession_moyenne_A': 58.0,  
        'possession_moyenne_B': 52.0,  
        'expected_but_A': 1.7,  
        'expected_but_B': 1.4,  
        'expected_concedes_A': 1.1,  
        'expected_concedes_B': 1.3,  
        'tirs_cadres_A': 5.2,  
        'tirs_cadres_B': 4.8,  
        'grandes_chances_A': 2.5,  
        'grandes_chances_B': 2.0,  
        'victoires_domicile_A': 8,  
        'victoires_domicile_B': 6,  
        'victoires_exterieur_A': 5,  
        'victoires_exterieur_B': 4,  
        'joueurs_absents_A': 1,  
        'joueurs_absents_B': 2,  
        'joueurs_cles_absents_A': 0,  
        'joueurs_cles_absents_B': 1,  
        'degagements_A': 12,  
        'degagements_B': 15,  
        'balles_touchees_camp_adverse_A': 25,  
        'balles_touchees_camp_adverse_B': 20,  
        'motivation_A': 8,  
        'motivation_B': 7,  
        'face_a_face_A': 3,  
        'face_a_face_B': 2,  
        'cote_bookmaker_A': 2.5,  
        'cote_bookmaker_B': 3.0,  
        'cote_bookmaker_Nul': 3.2,  
    }  

if 'historique' not in st.session_state:  
    st.session_state.historique = []  

if 'poids_criteres' not in st.session_state:  
    st.session_state.poids_criteres = None  

# Formulaire flottant pour la saisie des données  
with st.form("formulaire_saisie"):  
    st.header("📊 Saisie des Données")  
    col1, col2 = st.columns(2)  

    with col1:  
        st.subheader("Équipe A")  
        st.session_state.data['score_rating_A'] = st.number_input("Score de Performance (Équipe A)", value=float(st.session_state.data['score_rating_A']), step=1.0)  
        st.session_state.data['buts_par_match_A'] = st.number_input("Buts par Match (Équipe A)", value=float(st.session_state.data['buts_par_match_A']), step=0.1)  
        st.session_state.data['victoires_domicile_A'] = st.number_input("Victoires à Domicile (Équipe A)", value=int(st.session_state.data['victoires_domicile_A']), step=1)  
        st.session_state.data['joueurs_absents_A'] = st.number_input("Joueurs Absents (Équipe A)", value=int(st.session_state.data['joueurs_absents_A']), step=1)  
        st.session_state.data['expected_but_A'] = st.number_input("Expected Goals (Équipe A)", value=float(st.session_state.data['expected_but_A']), step=0.1)  

    with col2:  
        st.subheader("Équipe B")  
        st.session_state.data['score_rating_B'] = st.number_input("Score de Performance (Équipe B)", value=float(st.session_state.data['score_rating_B']), step=1.0)  
        st.session_state.data['buts_par_match_B'] = st.number_input("Buts par Match (Équipe B)", value=float(st.session_state.data['buts_par_match_B']), step=0.1)  
        st.session_state.data['victoires_domicile_B'] = st.number_input("Victoires à Domicile (Équipe B)", value=int(st.session_state.data['victoires_domicile_B']), step=1)  
        st.session_state.data['joueurs_absents_B'] = st.number_input("Joueurs Absents (Équipe B)", value=int(st.session_state.data['joueurs_absents_B']), step=1)  
        st.session_state.data['expected_but_B'] = st.number_input("Expected Goals (Équipe B)", value=float(st.session_state.data['expected_but_B']), step=0.1)  

    # Bouton pour soumettre le formulaire  
    submitted = st.form_submit_button("Mettre à jour les données")  

# Données existantes pour l'Équipe A et l'Équipe B  
data = {  
    'Équipe': ['Équipe A', 'Équipe B'],  
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

# Création du DataFrame  
df = pd.DataFrame(data)  

# Vérification des données avant l'entraînement  
if df.drop(columns=['Équipe']).isnull().values.any():  
    st.error("Les données contiennent des valeurs manquantes. Veuillez vérifier les entrées.")  
else:  
    # Création de la variable cible (1 si l'équipe A gagne, 0 sinon)  
    y = np.array([1, 0])  # Équipe A gagne par défaut  

    # Modèles de classification  
    log_reg = LogisticRegression()  
    rf_clf = RandomForestClassifier()  

    # Validation croisée avec KFold  
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  

    # Évaluation des modèles  
    log_reg_scores = cross_val_score(log_reg, df.drop(columns=['Équipe']), y, cv=kf)  
    rf_scores = cross_val_score(rf_clf, df.drop(columns=['Équipe']), y, cv=kf)  

    # Calcul des scores moyens  
    log_reg_score = np.mean(log_reg_scores)  
    rf_score = np.mean(rf_scores)  

    # Entraînement des modèles sur l'ensemble des données  
    log_reg.fit(df.drop(columns=['Équipe']), y)  
    rf_clf.fit(df.drop(columns=['Équipe']), y)  

    # Entraînement du modèle Random Forest pour obtenir les poids des critères  
    st.session_state.poids_criteres = rf_clf.feature_importances_  

    # Modèle Poisson  
    lambda_A = (  
        data['expected_but'][0] +  
        data['buts_par_match'][0] +  
        data['tirs_cadres'][0] * 0.1 +  
        data['grandes_chances'][0] * 0.2 +  
        (data['victoires_domicile'][0] * 0.3) - (data['joueurs_absents'][1] * 0.2)  # Impact des victoires et joueurs absents  
    )  

    lambda_B = (  
        data['expected_but'][1] +  
        data['buts_par_match'][1] +  
        data['tirs_cadres'][1] * 0.1 +  
        data['grandes_chances'][1] * 0.2 +  
        (data['victoires_domicile'][1] * 0.3) - (data['joueurs_absents'][0] * 0.2)  # Impact des victoires et joueurs absents  
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

    # Affichage des résultats des modèles  
    st.subheader("🤖 Performance des Modèles")  
    col_log_reg, col_rf = st.columns(2)  
    with col_log_reg:  
        st.metric("📈 Régression Logistique (Précision)", f"{log_reg_score:.2%}")  
    with col_rf:  
        st.metric("🌲 Forêt Aléatoire (Précision)", f"{rf_score:.2%}")  

    # Visualisation avec Altair  
    st.subheader("📊 Visualisation des Probabilités de Victoire")  
    df_probabilites = pd.DataFrame({  
        'Équipe': ['Équipe A', 'Équipe B', 'Match Nul'],  
        'Probabilité': [proba_A, proba_B, proba_Nul]  
    })  

    chart = alt.Chart(df_probabilites).mark_bar().encode(  
        x='Équipe',  
        y='Probabilité',  
        color='Équipe'  
    ).properties(  
        title='Probabilités de Victoire'  
    )  

    st.altair_chart(chart, use_container_width=True)  

    # Tableau interactif avec Plotly  
    st.subheader("📊 Tableau des Résultats")  
    data_resultats = {  
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
    df_resultats = pd.DataFrame(data_resultats)  

    # Utilisation de Plotly pour afficher le tableau  
    fig = px.data.tips()  
    fig = px.table(df_resultats, title="Tableau Synthétique des Résultats")  
    st.plotly_chart(fig)  

    # Message rappel sur le Value Bet  
    st.markdown("""  
    ### 💡 Qu'est-ce qu'un Value Bet ?  
    Un **Value Bet** est un pari où la cote prédite par le modèle est **inférieure** à la cote proposée par le bookmaker.   
    Cela indique que le bookmaker sous-estime la probabilité de cet événement.  
    """)  

    # Affichage des poids des critères (si disponible)  
    if st.session_state.poids_criteres is not None:  
        st.subheader("📊 Poids des Critères (Forêt Aléatoire)")  
        df_poids = pd.DataFrame({  
            'Critère': df.drop(columns=['Équipe']).columns,  
            'Poids': st.session_state.poids_criteres  
        })  
        st.bar_chart(df_poids.set_index('Critère'))  

    # Section pour l'historique des prédictions  
    if st.session_state.historique:  
        st.subheader("📜 Historique des Prédictions")  
        for i, pred in enumerate(st.session_state.historique):  
            st.markdown(f"**Prédiction {i+1}**")  
            st.markdown(f"- Probabilité Équipe A: {pred['proba_A']:.2%}")  
            st.markdown(f"- Probabilité Équipe B: {pred['proba_B']:.2%}")  
            st.markdown(f"- Probabilité Match Nul: {pred['proba_Nul']:.2%}")  
            st.markdown("---")  

    # Bouton pour réinitialiser l'historique  
    if st.button("🔄 Réinitialiser l'Historique"):  
        st.session_state.historique = []  
        st.success("L'historique des prédictions a été réinitialisé.")
