import streamlit as st  
import pandas as pd  
import numpy as np  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from docx import Document  
import matplotlib.pyplot as plt  

# Configuration de la page  
st.set_page_config(page_title="Prédictions de Match", layout="wide")  

# Fonction pour convertir les valeurs en float de manière sécurisée  
def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

# Initialisation des données dans st.session_state  
if "data" not in st.session_state:  
    st.session_state.data = {  
        # Équipe A  
        "score_rating_A": 70.0,  
        "buts_par_match_A": 1.5,  
        "buts_concedes_par_match_A": 1.0,  
        "possession_moyenne_A": 55.0,  
        "expected_but_A": 1.8,  
        "expected_concedes_A": 1.2,  
        "tirs_cadres_A": 120.0,  
        "grandes_chances_A": 25.0,  
        "passes_reussies_A": 400.0,  
        "corners_A": 60.0,  
        "interceptions_A": 50.0,  
        "tacles_reussis_A": 40.0,  
        "fautes_A": 15.0,  
        "cartons_jaunes_A": 5.0,  
        "cartons_rouges_A": 1.0,  
        "joueurs_cles_absents_A": 0,  
        "motivation_A": 3,  
        "clean_sheets_gardien_A": 2.0,  
        "ratio_tirs_arretes_A": 0.75,  
        "victoires_domicile_A": 60.0,  
        "passes_longues_A": 50.0,  
        "dribbles_reussis_A": 10.0,  
        "ratio_tirs_cadres_A": 0.4,  
        "grandes_chances_manquees_A": 5.0,  
        "fautes_zones_dangereuses_A": 3.0,  
        "buts_corners_A": 2.0,  
        "jours_repos_A": 4.0,  
        "matchs_30_jours_A": 8.0,  
        # Équipe B  
        "score_rating_B": 65.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "possession_moyenne_B": 45.0,  
        "expected_but_B": 1.2,  
        "expected_concedes_B": 1.8,  
        "tirs_cadres_B": 100.0,  
        "grandes_chances_B": 20.0,  
        "passes_reussies_B": 350.0,  
        "corners_B": 50.0,  
        "interceptions_B": 40.0,  
        "tacles_reussis_B": 35.0,  
        "fautes_B": 20.0,  
        "cartons_jaunes_B": 6.0,  
        "cartons_rouges_B": 2.0,  
        "joueurs_cles_absents_B": 0,  
        "motivation_B": 3,  
        "clean_sheets_gardien_B": 1.0,  
        "ratio_tirs_arretes_B": 0.65,  
        "victoires_exterieur_B": 40.0,  
        "passes_longues_B": 40.0,  
        "dribbles_reussis_B": 8.0,  
        "ratio_tirs_cadres_B": 0.35,  
        "grandes_chances_manquees_B": 6.0,  
        "fautes_zones_dangereuses_B": 4.0,  
        "buts_corners_B": 1.0,  
        "jours_repos_B": 3.0,  
        "matchs_30_jours_B": 9.0,  
        # Conditions du Match  
        "conditions_match": "",  
        # Historique Face-à-Face  
        "face_a_face": "",  
        # Forme Récente  
        "forme_recente_A": ["V", "V", "V", "V", "V"],  # Données fictives pour l'Équipe A  
        "forme_recente_B": ["D", "N", "V", "D", "V"],  # Données fictives pour l'Équipe B  
        # Cotes  
        "cote_victoire_A": 2.0,  
        "cote_nul": 3.0,  
        "cote_victoire_B": 4.0,  
        # Bankroll  
        "bankroll": 1000.0,  
    }  

# Menu de Navigation  
st.sidebar.title("Navigation")  
option = st.sidebar.radio("Choisissez une option", ["📊 Statistiques", "🌦️ Conditions et Motivation", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"])  

# 📊 Statistiques  
if option == "📊 Statistiques":  
    st.title("📊 Statistiques des Matchs")  
    st.write("Analyse des statistiques des équipes et des matchs.")  

    # Exemple de graphique  
    st.subheader("Graphique des Performances")  
    fig, ax = plt.subplots(figsize=(6, 4))  
    ax.bar(["Équipe A", "Équipe B"], [3, 5])  
    ax.set_ylabel("Nombre de Victoires")  
    ax.set_title("Performances des Équipes")  
    st.pyplot(fig)  

# 🌦️ Conditions et Motivation  
elif option == "🌦️ Conditions et Motivation":  
    st.title("🌦️ Conditions et Motivation")  
    st.write("Analyse des conditions météorologiques et de la motivation des équipes.")  

    # Formulaire de saisie  
    with st.form("conditions_form"):  
        col1, col2 = st.columns(2)  
        with col1:  
            st.selectbox("Conditions Météo Équipe A", ["Ensoleillé", "Pluvieux", "Nuageux"], key="meteo_A")  
            st.slider("Motivation Équipe A (1-10)", 1, 10, key="motivation_A")  
        with col2:  
            st.selectbox("Conditions Météo Équipe B", ["Ensoleillé", "Pluvieux", "Nuageux"], key="meteo_B")  
            st.slider("Motivation Équipe B (1-10)", 1, 10, key="motivation_B")  
        if st.form_submit_button("Analyser"):  
            st.success("Analyse en cours...")  

# 🔮 Prédictions  
elif option == "🔮 Prédictions":  
    st.title("🔮 Prédictions de Match")  

    # Formulaire de saisie des données  
    with st.form("formulaire_simple"):  
        col1, col2 = st.columns(2)  
        with col1:  
            st.text_input("Nom de l'Équipe A", key="nom_A")  
            st.number_input("Score de Rating Équipe A", key="score_rating_A")  
            st.number_input("Buts par Match Équipe A", key="buts_par_match_A")  
            st.number_input("Possession Moyenne Équipe A", key="possession_moyenne_A")  
            st.number_input("Motivation Équipe A", key="motivation_A")  
            st.number_input("Tirs Cadrés Équipe A", key="tirs_cadres_A")  
            st.number_input("Interceptions Équipe A", key="interceptions_A")  
        with col2:  
            st.text_input("Nom de l'Équipe B", key="nom_B")  
            st.number_input("Score de Rating Équipe B", key="score_rating_B")  
            st.number_input("Buts par Match Équipe B", key="buts_par_match_B")  
            st.number_input("Possession Moyenne Équipe B", key="possession_moyenne_B")  
            st.number_input("Motivation Équipe B", key="motivation_B")  
            st.number_input("Tirs Cadrés Équipe B", key="tirs_cadres_B")  
            st.number_input("Interceptions Équipe B", key="interceptions_B")  

        if st.form_submit_button("Prédire le Résultat"):  
            st.success("Prédiction en cours...")  

    # Prédiction des Buts avec Poisson  
    with st.expander("📊 Prédiction des Scores avec la Loi de Poisson"):  
        avg_goals_A = safe_float(st.session_state.data["buts_par_match_A"])  
        avg_goals_B = safe_float(st.session_state.data["buts_par_match_B"])  
        if avg_goals_A <= 0 or avg_goals_B <= 0:  
            st.error("Les moyennes de buts doivent être positives.")  
        else:  
            prob_0_0 = poisson.pmf(0, avg_goals_A) * poisson.pmf(0, avg_goals_B)  
            prob_1_1 = poisson.pmf(1, avg_goals_A) * poisson.pmf(1, avg_goals_B)  
            prob_2_2 = poisson.pmf(2, avg_goals_A) * poisson.pmf(2, avg_goals_B)  
            st.write(f"📉 **Probabilité de 0-0** : {prob_0_0:.2%}")  
            st.write(f"📉 **Probabilité de 1-1** : {prob_1_1:.2%}")  
            st.write(f"📉 **Probabilité de 2-2** : {prob_2_2:.2%}")  

    # Régression Logistique  
    with st.expander("📈 Prédiction avec Régression Logistique"):  
        X_lr = np.array([  
            [  
                safe_float(st.session_state.data["score_rating_A"]),  
                safe_float(st.session_state.data["score_rating_B"]),  
                safe_float(st.session_state.data["possession_moyenne_A"]),  
                safe_float(st.session_state.data["possession_moyenne_B"]),  
                safe_float(st.session_state.data["motivation_A"]),  
                safe_float(st.session_state.data["motivation_B"]),  
                safe_float(st.session_state.data["tirs_cadres_A"]),  
                safe_float(st.session_state.data["tirs_cadres_B"]),  
                safe_float(st.session_state.data["interceptions_A"]),  
                safe_float(st.session_state.data["interceptions_B"]),  
            ],  
            [  
                safe_float(st.session_state.data["score_rating_B"]),  
                safe_float(st.session_state.data["score_rating_A"]),  
                safe_float(st.session_state.data["possession_moyenne_B"]),  
                safe_float(st.session_state.data["possession_moyenne_A"]),  
                safe_float(st.session_state.data["motivation_B"]),  
                safe_float(st.session_state.data["motivation_A"]),  
                safe_float(st.session_state.data["tirs_cadres_B"]),  
                safe_float(st.session_state.data["tirs_cadres_A"]),  
                safe_float(st.session_state.data["interceptions_B"]),  
                safe_float(st.session_state.data["interceptions_A"]),  
            ]  
        ])  
        y_lr = np.array([1, 0])  # Deux classes : 1 pour Équipe A, 0 pour Équipe B  
        model_lr = LogisticRegression()  
        model_lr.fit(X_lr, y_lr)  
        prediction_lr = model_lr.predict(X_lr)  
        st.write(f"📊 **Résultat** : {'Équipe A' if prediction_lr[0] == 1 else 'Équipe B'}")  

    # Random Forest  
    with st.expander("🌲 Prédiction avec Random Forest"):  
        X_rf = np.array([  
            [  
                safe_float(st.session_state.data[key])  
                for key in st.session_state.data  
                if (key.endswith("_A") or key.endswith("_B")) and isinstance(st.session_state.data[key], (int, float))  
                and not key.startswith("cote_") and not key.startswith("bankroll")  
            ],  
            [  
                safe_float(st.session_state.data[key])  
                for key in st.session_state.data  
                if (key.endswith("_B") or key.endswith("_A")) and isinstance(st.session_state.data[key], (int, float))  
                and not key.startswith("cote_") and not key.startswith("bankroll")  
            ]  
        ])  
        y_rf = np.array([1, 0])  # Deux classes : 1 pour Équipe A, 0 pour Équipe B  
        model_rf = RandomForestClassifier()  
        model_rf.fit(X_rf, y_rf)  
        prediction_rf = model_rf.predict(X_rf)  
        st.write(f"📊 **Résultat** : {'Équipe A' if prediction_rf[0] == 1 else 'Équipe B'}")  

    # Prédiction des Paris Double Chance  
    with st.expander("🎰 Prédiction des Paris Double Chance"):  
        prob_victoire_A = prediction_lr[0]  # Probabilité de victoire de l'Équipe A  
        prob_victoire_B = 1 - prediction_lr[0]  # Probabilité de victoire de l'Équipe B  
        prob_nul = prob_1_1  # Probabilité de match nul (basée sur Poisson)  

        # Probabilités Double Chance  
        prob_1X = prob_victoire_A + prob_nul  
        prob_12 = prob_victoire_A + prob_victoire_B  
        prob_X2 = prob_nul + prob_victoire_B  

        st.write(f"📊 **Probabilité 1X (Équipe A gagne ou match nul)** : {prob_1X:.2%}")  
        st.write(f"📊 **Probabilité 12 (Équipe A gagne ou Équipe B gagne)** : {prob_12:.2%}")  
        st.write(f"📊 **Probabilité X2 (Match nul ou Équipe B gagne)** : {prob_X2:.2%}")  

        # Graphique des probabilités  
        fig, ax = plt.subplots(figsize=(6, 4))  
        ax.bar(["1X", "12", "X2"], [prob_1X, prob_12, prob_X2])  
        ax.set_ylabel("Probabilité")  
        ax.set_title("Probabilités des Paris Double Chance")  
        st.pyplot(fig)  

    # Téléchargement des données  
    st.subheader("📥 Téléchargement des Données")  
    data = {  
        "Données Équipe A": {  
            key.replace("_A", ""): [st.session_state.data[key]]  
            for key in st.session_state.data  
            if key.endswith("_A") and isinstance(st.session_state.data[key], (int, float, str))  
            and not key.startswith("cote_") and not key.startswith("bankroll")  
        },  
        "Données Équipe B": {  
            key.replace("_B", ""): [st.session_state.data[key]]  
            for key in st.session_state.data  
            if key.endswith("_B") and isinstance(st.session_state.data[key], (int, float, str))  
            and not key.startswith("cote_") and not key.startswith("bankroll")  
        },  
    }  

    # Conversion en DataFrame et téléchargement  
    excel_data = pd.DataFrame(data["Données Équipe A"]).to_csv(index=False).encode("utf-8")  
    doc = Document()  
    doc.add_heading("Données des Équipes", level=1)  
    doc.add_heading("Données Équipe A", level=2)  
    for key, value in data["Données Équipe A"].items():  
        doc.add_paragraph(f"{key}: {value[0]}")  
    doc.add_heading("Données Équipe B", level=2)  
    for key, value in data["Données Équipe B"].items():  
        doc.add_paragraph(f"{key}: {value[0]}")  
    doc_filename = "donnees_equipes.docx"  
    doc.save(doc_filename)  

    col1, col2 = st.columns(2)  
    with col1:  
        st.download_button(  
            label="📥 Télécharger en Excel",  
            data=excel_data,  
            file_name="donnees_equipes.xlsx",  
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  
        )  
    with col2:  
        with open(doc_filename, "rb") as f:  
            st.download_button(  
                label="📥 Télécharger en DOC",  
                data=f,  
                file_name=doc_filename,  
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",  
            )  

# 🎰 Cotes et Value Bet  
elif option == "🎰 Cotes et Value Bet":  
    st.title("🎰 Cotes et Value Bet")  
    st.write("Analyse des cotes et identification des value bets.")  

    # Formulaire de saisie des cotes  
    with st.form("cotes_form"):  
        col1, col2 = st.columns(2)  
        with col1:  
            st.number_input("Cote Équipe A", key="cote_A")  
        with col2:  
            st.number_input("Cote Équipe B", key="cote_B")  
        if st.form_submit_button("Analyser"):  
            st.success("Analyse en cours...")  

# 💰 Système de Mise  
elif option == "💰 Système de Mise":  
    st.title("💰 Système de Mise")  
    st.write("Gestion des mises et optimisation des gains.")  

        # Formulaire de saisie des mises  
    with st.form("mises_form"):  
        col1, col2 = st.columns(2)  
        with col1:  
            st.number_input("Mise sur Équipe A", key="mise_A")  
        with col2:  
            st.number_input("Mise sur Équipe B", key="mise_B")  
        if st.form_submit_button("Calculer"):  
            st.success("Calcul en cours...")  

    # Calcul des gains potentiels  
    with st.expander("📈 Gains Potentiels"):  
        mise_A = safe_float(st.session_state.data["mise_A"])  
        mise_B = safe_float(st.session_state.data["mise_B"])  
        cote_A = safe_float(st.session_state.data["cote_victoire_A"])  
        cote_B = safe_float(st.session_state.data["cote_victoire_B"])  

        if mise_A > 0 or mise_B > 0:  
            gain_A = mise_A * cote_A if mise_A > 0 else 0  
            gain_B = mise_B * cote_B if mise_B > 0 else 0  
            st.write(f"💰 **Gain potentiel sur Équipe A** : {gain_A:.2f} €")  
            st.write(f"💰 **Gain potentiel sur Équipe B** : {gain_B:.2f} €")  
        else:  
            st.warning("Veuillez entrer une mise pour calculer les gains.")  

    # Gestion de la bankroll  
    with st.expander("💼 Gestion de la Bankroll"):  
        bankroll = safe_float(st.session_state.data["bankroll"])  
        st.write(f"💼 **Bankroll actuelle** : {bankroll:.2f} €")  

        # Mise à jour de la bankroll  
        mise_totale = mise_A + mise_B  
        if mise_totale > 0:  
            bankroll_restante = bankroll - mise_totale  
            st.write(f"💼 **Bankroll restante après mise** : {bankroll_restante:.2f} €")  
            if bankroll_restante < 0:  
                st.error("⚠️ Attention : Votre bankroll est insuffisante pour cette mise.")  
            else:  
                st.success("✅ Votre bankroll est suffisante pour cette mise.")
