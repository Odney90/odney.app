import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
import matplotlib.pyplot as plt  
import io  
from docx import Document  
import openpyxl  # Module nécessaire pour exporter en Excel  

# Configuration de la page  
st.set_page_config(page_title="Prédictions Football", page_icon="⚽")  

# Fonction pour vérifier et convertir en float  
def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

# Fonction pour quantifier la forme récente  
def quantifier_forme_recente(forme_recente):  
    score = 0  
    for resultat in forme_recente:  
        if resultat == "V":  
            score += 3  # Victoire = 3 points  
        elif resultat == "N":  
            score += 1  # Nul = 1 point  
        elif resultat == "D":  
            score += 0  # Défaite = 0 point  
    return score  

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

# Création des onglets  
tab1, tab2, tab3, tab4, tab5 = st.tabs(  
    ["📊 Statistiques", "🌦️ Conditions et Motivation", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"]  
)  

# Onglet 1 : Statistiques  
with tab1:  
    st.header("📊 Statistiques des Équipes")  
    col_a, col_b = st.columns(2)  

    # Statistiques de l'Équipe A  
    with col_a:  
        st.subheader("Équipe A")  
        for key in st.session_state.data:  
            if key.endswith("_A"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    value=safe_float(st.session_state.data[key]),  
                    min_value=-1e6,  
                    max_value=1e6,  
                )  

    # Statistiques de l'Équipe B  
    with col_b:  
        st.subheader("Équipe B")  
        for key in st.session_state.data:  
            if key.endswith("_B"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    value=safe_float(st.session_state.data[key]),  
                    min_value=-1e6,  
                    max_value=1e6,  
                )  

# Onglet 2 : Conditions et Motivation  
with tab2:  
    st.header("🌦️ Conditions du Match et Motivation")  
    st.session_state.data["conditions_match"] = st.text_input(  
        "🌧️ Conditions du Match (ex : pluie, terrain sec)",  
        value=st.session_state.data["conditions_match"],  
    )  
    st.session_state.data["face_a_face"] = st.text_area(  
        "📅 Historique des Face-à-Face (ex : 3 victoires A, 2 nuls, 1 victoire B)",  
        value=st.session_state.data["face_a_face"],  
    )  

    st.subheader("Forme Récente (5 derniers matchs)")  
    col_a, col_b = st.columns(2)  

    # Vérification et initialisation de la forme récente  
    if "forme_recente_A" not in st.session_state.data or not isinstance(st.session_state.data["forme_recente_A"], list):  
        st.session_state.data["forme_recente_A"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    if "forme_recente_B" not in st.session_state.data or not isinstance(st.session_state.data["forme_recente_B"], list):  
        st.session_state.data["forme_recente_B"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    # Vérification de la longueur des listes  
    if len(st.session_state.data["forme_recente_A"]) != 5:  
        st.session_state.data["forme_recente_A"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    if len(st.session_state.data["forme_recente_B"]) != 5:  
        st.session_state.data["forme_recente_B"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    with col_a:  
        st.write("Équipe A")  
        for i in range(5):  
            current_value = st.session_state.data["forme_recente_A"][i]  
            if current_value not in ["V", "N", "D"]:  
                current_value = "V"  # Valeur par défaut  
            st.session_state.data["forme_recente_A"][i] = st.select_slider(  
                f"Match {i+1}", options=["V", "N", "D"], value=current_value, key=f"match_A_{i}"  
            )  
    
    with col_b:  
        st.write("Équipe B")  
        for i in range(5):  
            current_value = st.session_state.data["forme_recente_B"][i]  
            if current_value not in ["V", "N", "D"]:  
                current_value = "V"  # Valeur par défaut  
            st.session_state.data["forme_recente_B"][i] = st.select_slider(  
                f"Match {i+1}", options=["V", "N", "D"], value=current_value, key=f"match_B_{i}"  
            )  

    # Quantification de la forme récente  
    score_forme_A = quantifier_forme_recente(st.session_state.data["forme_recente_A"])  
    score_forme_B = quantifier_forme_recente(st.session_state.data["forme_recente_B"])  
    st.write(f"📊 **Score Forme Récente Équipe A** : {score_forme_A}")  
    st.write(f"📊 **Score Forme Récente Équipe B** : {score_forme_B}")  
    
# Onglet 3 : Prédictions  
with tab3:  
    st.header("🔮 Prédictions")  
    if st.button("Prédire le résultat"): 
        try:  
            # Prédiction des Buts avec Poisson  
            avg_goals_A = safe_float(st.session_state.data["buts_par_match_A"])  
            avg_goals_B = safe_float(st.session_state.data["buts_par_match_B"]) 
            avg_goals_A = safe_float(st.session_state.data["buts_concedes_par_match_A"])
            avg_goals_A = safe_float(st.session_state.data["buts_concedes_par_match_B"])
                                     
            if avg_goals_A <= 0 or avg_goals_B <= 0:  
                raise ValueError("Les moyennes de buts doivent être positives.")  
            prob_0_0 = poisson.pmf(0, avg_goals_A) * poisson.pmf(0, avg_goals_B)  
            prob_1_1 = poisson.pmf(1, avg_goals_A) * poisson.pmf(1, avg_goals_B)  
            prob_2_2 = poisson.pmf(2, avg_goals_A) * poisson.pmf(2, avg_goals_B)  

            # Explication Poisson  
            st.subheader("📊 Prédiction des Scores avec la Loi de Poisson")  
            st.write(  
                "La loi de Poisson est utilisée pour prédire la probabilité des scores en fonction des moyennes de buts des équipes. "  
                "Elle est particulièrement utile pour estimer les scores les plus probables."  
            )  
            st.write(f"📉 **Probabilité de 0-0** : {prob_0_0:.2%}")  
            st.write(f"📉 **Probabilité de 1-1** : {prob_1_1:.2%}")  
            st.write(f"📉 **Probabilité de 2-2** : {prob_2_2:.2%}")  

            # Régression Logistique avec critères supplémentaires  
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
                    score_forme_A,  # Intégration de la forme récente  
                    score_forme_B,  # Intégration de la forme récente  
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
                    score_forme_B,  # Intégration de la forme récente  
                    score_forme_A,  # Intégration de la forme récente  
                ]  
            ])  
            y_lr = np.array([1, 0])  # Deux classes : 1 pour Équipe A, 0 pour Équipe B  
            model_lr = LogisticRegression()  
            model_lr.fit(X_lr, y_lr)  
            prediction_lr = model_lr.predict(X_lr)  

            # Explication Régression Logistique  
            st.subheader("📈 Prédiction avec Régression Logistique")  
            st.write(  
                "La régression logistique est un modèle de classification qui prédit la probabilité de victoire d'une équipe "  
                "en fonction de plusieurs critères, tels que le score de rating, la possession, la motivation, et la forme récente."  
            )  
            st.write(f"📊 **Résultat** : {'Équipe A' if prediction_lr[0] == 1 else 'Équipe B'}")  

            # Random Forest (exclut les données des onglets 4 et 5 et # Cotes)  
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

            # Explication Random Forest  
            st.subheader("🌲 Prédiction avec Random Forest")  
            st.write(  
                "Le Random Forest est un modèle d'apprentissage automatique qui utilise plusieurs arbres de décision pour prédire le résultat. "  
                "Il est robuste et prend en compte de nombreuses variables pour améliorer la précision."  
            )  
            st.write(f"📊 **Résultat** : {'Équipe A' if prediction_rf[0] == 1 else 'Équipe B'}")  

            # Prédiction des Paris Double Chance  
            st.subheader("🎰 Prédiction des Paris Double Chance")  
            st.write(  
                "Les paris Double Chance permettent de couvrir deux des trois résultats possibles : "  
                "1X (Équipe A gagne ou match nul), 12 (Équipe A gagne ou Équipe B gagne), X2 (Match nul ou Équipe B gagne)."  
            )  

            # Calcul des probabilités Double Chance  
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
            st.subheader("📉 Graphique des Probabilités")  
            fig, ax = plt.subplots()  
            ax.bar(["1X", "12", "X2"], [prob_1X, prob_12, prob_X2])  
            ax.set_ylabel("Probabilité")  
            ax.set_title("Probabilités des Paris Double Chance")  
            st.pyplot(fig)  

            # Téléchargement des données organisées par équipe  
            st.subheader("📥 Téléchargement des Données des Équipes")  

            # Création d'un document Word  
            doc = Document()  
            doc.add_heading("Données des Équipes", level=1)  

            # Ajout des données de l'Équipe A  
            doc.add_heading("Données Équipe A", level=2)  
            for key, value in st.session_state.data.items():  
                if key.endswith("_A"):  
                    doc.add_paragraph(f"{key.replace('_A', '')}: {value}")  

            # Ajout des données de l'Équipe B  
            doc.add_heading("Données Équipe B", level=2)  
            for key, value in st.session_state.data.items():  
                if key.endswith("_B"):  
                    doc.add_paragraph(f"{key.replace('_B', '')}: {value}")  

            # Sauvegarde du document Word  
            doc_filename = "donnees_equipes.docx"  
            doc.save(doc_filename)  

            # Téléchargement du document Word  
            with open(doc_filename, "rb") as f:  
                st.download_button(  
                    label="📥 Télécharger les données des équipes en DOC",  
                    data=f,  
                    file_name=doc_filename,  
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",  
                )  

        except Exception as e:  
            st.error(f"Une erreur s'est produite lors de la prédiction : {e}")
# Onglet 4 : Cotes et Value Bet  
with tab4:  
    st.header("🎰 Cotes et Value Bet")  
    st.subheader("Convertisseur de Cotes Implicites")  
    cote_victoire_A = st.number_input("Cote Victoire A", value=safe_float(st.session_state.data["cote_victoire_A"]))  
    cote_nul = st.number_input("Cote Nul", value=safe_float(st.session_state.data["cote_nul"]))  
    cote_victoire_B = st.number_input("Cote Victoire B", value=safe_float(st.session_state.data["cote_victoire_B"]))  

    # Calcul de la marge bookmaker  
    marge_bookmaker = (1 / cote_victoire_A) + (1 / cote_nul) + (1 / cote_victoire_B) - 1  
    st.write(f"📉 **Marge Bookmaker** : {marge_bookmaker:.2%}")  

    # Calcul des cotes réelles  
    cote_reelle_victoire_A = cote_victoire_A / (1 + marge_bookmaker)  
    cote_reelle_nul = cote_nul / (1 + marge_bookmaker)  
    cote_reelle_victoire_B = cote_victoire_B / (1 + marge_bookmaker)  
    st.write(f"📊 **Cote Réelle Victoire A** : {cote_reelle_victoire_A:.2f}")  
    st.write(f"📊 **Cote Réelle Nul** : {cote_reelle_nul:.2f}")  
    st.write(f"📊 **Cote Réelle Victoire B** : {cote_reelle_victoire_B:.2f}")  

    st.subheader("Calculateur de Paris Combiné")  
    cote_equipe_1 = st.number_input("Cote Équipe 1", value=1.5)  
    cote_equipe_2 = st.number_input("Cote Équipe 2", value=2.0)  
    cote_equipe_3 = st.number_input("Cote Équipe 3", value=2.5)  
    cote_finale = cote_equipe_1 * cote_equipe_2 * cote_equipe_3  
    st.write(f"📈 **Cote Finale** : {cote_finale:.2f}")  

# Onglet 5 : Système de Mise  
with tab5:  
    st.header("💰 Système de Mise")  
    bankroll = st.number_input("Bankroll (€)", value=safe_float(st.session_state.data["bankroll"]))  
    niveau_kelly = st.slider("Niveau de Kelly (1 à 5)", min_value=1, max_value=5, value=3)  
    probabilite_victoire = st.number_input("Probabilité de Victoire (%)", value=50.0) / 100  
    cote = st.number_input("Cote", value=2.0)  

    # Calcul de la mise selon Kelly  
    mise_kelly = (probabilite_victoire * (cote - 1) - (1 - probabilite_victoire)) / (cote - 1)  
    mise_kelly = max(0, mise_kelly)  # Éviter les valeurs négatives  
    mise_kelly *= niveau_kelly / 5  # Ajuster selon le niveau de Kelly  
    mise_finale = bankroll * mise_kelly  

    st.write(f"📊 **Mise Kelly** : {mise_kelly:.2%}")  
    st.write(f"💰 **Mise Finale** : {mise_finale:.2f} €")
