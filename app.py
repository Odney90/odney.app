import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
import matplotlib.pyplot as plt  
import io  

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
    if "forme_recente_A" not in st.session_state.data or len(st.session_state.data["forme_recente_A"]) != 5:  
        st.session_state.data["forme_recente_A"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    if "forme_recente_B" not in st.session_state.data or len(st.session_state.data["forme_recente_B"]) != 5:  
        st.session_state.data["forme_recente_B"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    with col_a:  
        st.write("Équipe A")  
        for i in range(5):  
            current_value = st.session_state.data["forme_recente_A"][i]  
            if current_value not in ["V", "N", "D"]:  
                current_value = "V"  # Valeur par défaut  
            st.session_state.data["forme_recente_A"][i] = st.select_slider(  
                f"Match {i+1}", options=["V", "N", "D"], value=current_value  
            )  
    
    with col_b:  
        st.write("Équipe B")  
        for i in range(5):  
            current_value = st.session_state.data["forme_recente_B"][i]  
            if current_value not in ["V", "N", "D"]:  
                current_value = "V"  # Valeur par défaut  
            st.session_state.data["forme_recente_B"][i] = st.select_slider(  
                f"Match {i+1}", options=["V", "N", "D"], value=current_value  
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
            prob_0_0 = poisson.pmf(0, avg_goals_A) * poisson.pmf(0, avg_goals_B)  
            prob_1_1 = poisson.pmf(1, avg_goals_A) * poisson.pmf(1, avg_goals_B)  
            prob_2_2 = poisson.pmf(2, avg_goals_A) * poisson.pmf(2, avg_goals_B)  

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

            # Random Forest (exclut les données des onglets 4 et 5)  
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

            # Affichage des résultats  
            st.subheader("Résultats des Prédictions")  
            st.write(f"📊 **Probabilité de 0-0 (Poisson)** : {prob_0_0:.2%}")  
            st.write(f"📊 **Probabilité de 1-1 (Poisson)** : {prob_1_1:.2%}")  
            st.write(f"📊 **Probabilité de 2-2 (Poisson)** : {prob_2_2:.2%}")  
            st.write(f"📊 **Régression Logistique** : {'Équipe A' if prediction_lr[0] == 1 else 'Équipe B'}")  
            st.write(f"🌲 **Random Forest** : {'Équipe A' if prediction_rf[0] == 1 else 'Équipe B'}")  

            # Graphique des probabilités  
            fig, ax = plt.subplots()  
            ax.bar(["0-0", "1-1", "2-2"], [prob_0_0, prob_1_1, prob_2_2])  
            ax.set_ylabel("Probabilité")  
            ax.set_title("Probabilités des Scores")  
            st.pyplot(fig)  

            # Téléchargement des données  
            data = {  
                "Probabilité 0-0": [prob_0_0],  
                "Probabilité 1-1": [prob_1_1],  
                "Probabilité 2-2": [prob_2_2],  
                "Régression Logistique": [prediction_lr[0]],  
                "Random Forest": [prediction_rf[0]],  
            }  
            df = pd.DataFrame(data)  
            towrite = io.BytesIO()  
            df.to_excel(towrite, index=False)  
            towrite.seek(0)  
            st.download_button(  
                label="📥 Télécharger les données",  
                data=towrite,  
                file_name="predictions.xlsx",  
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  
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
