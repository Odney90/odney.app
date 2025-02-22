import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
import matplotlib.pyplot as plt  

# Initialisation des données par défaut  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        # Statistiques de l'Équipe A  
        "score_rating_A": 70.0,  
        "buts_totaux_A": 50.0,  
        "buts_par_match_A": 1.5,  
        "buts_concedes_par_match_A": 1.0,  
        "buts_concedes_totaux_A": 30.0,  
        "possession_moyenne_A": 55.0,  
        "aucun_but_encaisse_A": 10,  
        "expected_but_A": 1.8,  
        "expected_concedes_A": 1.2,  
        "tirs_cadres_A": 120,  
        "grandes_chances_A": 25,  
        "grandes_chances_manquees_A": 10,  
        "passes_reussies_A": 400,  
        "passes_longues_A": 80,  
        "centres_reussis_A": 30,  
        "penalties_obtenues_A": 5,  
        "balles_surface_A": 150,  
        "corners_A": 60,  
        "interceptions_A": 50,  
        "degagements_A": 70,  
        "tacles_reussis_A": 40,  
        "penalties_concedes_A": 3,  
        "arrets_A": 45,  
        "fautes_A": 15,  
        "cartons_jaunes_A": 5,  
        "cartons_rouges_A": 1,  

        # Statistiques de l'Équipe B  
        "score_rating_B": 65.0,  
        "buts_totaux_B": 40.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "buts_concedes_totaux_B": 35.0,  
        "possession_moyenne_B": 45.0,  
        "aucun_but_encaisse_B": 8,  
        "expected_but_B": 1.2,  
        "expected_concedes_B": 1.8,  
        "tirs_cadres_B": 100,  
        "grandes_chances_B": 20,  
        "grandes_chances_manquees_B": 15,  
        "passes_reussies_B": 350,  
        "passes_longues_B": 60,  
        "centres_reussis_B": 25,  
        "penalties_obtenues_B": 3,  
        "balles_surface_B": 120,  
        "corners_B": 50,  
        "interceptions_B": 40,  
        "degagements_B": 60,  
        "tacles_reussis_B": 35,  
        "penalties_concedes_B": 4,  
        "arrets_B": 30,  
        "fautes_B": 20,  
        "cartons_jaunes_B": 6,  
        "cartons_rouges_B": 2,  

        "recent_form_A": [1, 2, 1, 0, 3],  
        "recent_form_B": [0, 1, 2, 1, 1],  
        "head_to_head": {"victoires_A": 0, "nuls": 0, "victoires_B": 0},  
        "conditions_match": "",  
    }  

# Configuration de la page  
st.set_page_config(page_title="Prédiction de Matchs de Football", page_icon="⚽", layout="wide")  

# En-tête  
st.header("⚽ Analyse de Prédiction de Matchs de Football")  

# Onglets pour les différentes sections  
tab1, tab2, tab3 = st.tabs(["📊 Statistiques des Équipes", "🌦️ Conditions du Match", "🔮 Prédictions"])  

with tab1:  
    st.subheader("📊 Statistiques des Équipes")  

    col_a, col_b = st.columns(2)  

    with col_a:  
        st.subheader("Équipe A 🟡")  
        for key in st.session_state.data:  
            if key.endswith("_A"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    min_value=0.0,  
                    value=float(st.session_state.data[key]),  
                    key=key  
                )  

    with col_b:  
        st.subheader("Équipe B 🔴")  
        for key in st.session_state.data:  
            if key.endswith("_B"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    min_value=0.0,  
                    value=float(st.session_state.data[key]),  
                    key=key  
                )  

with tab2:  
    st.subheader("🌦️ Conditions du Match")  
    st.session_state.data["conditions_match"] = st.text_input("Conditions du Match (ex : pluie, terrain sec, etc.)", value=st.session_state.data["conditions_match"], key="conditions_match")  

    col_h2h, col_recent_form = st.columns(2)  

    with col_h2h:  
        st.subheader("📅 Historique des Confrontations")  
        col_a, col_b, col_c = st.columns(3)  
        with col_a:  
            victoires_A = st.number_input("Victoires de l'Équipe A", min_value=0, value=st.session_state.data["head_to_head"]["victoires_A"])  
        with col_b:  
            nuls = st.number_input("Nuls", min_value=0, value=st.session_state.data["head_to_head"]["nuls"])  
        with col_c:  
            victoires_B = st.number_input("Victoires de l'Équipe B", min_value=0, value=st.session_state.data["head_to_head"]["victoires_B"])  

        if st.button("Mettre à jour l'historique"):  
            st.session_state.data["head_to_head"]["victoires_A"] = victoires_A  
            st.session_state.data["head_to_head"]["nuls"] = nuls  
            st.session_state.data["head_to_head"]["victoires_B"] = victoires_B  
            st.success("Historique mis à jour avec succès !")  

        st.subheader("Résultats des Confrontations")  
        st.write(f"Victoires de l'Équipe A : {st.session_state.data['head_to_head']['victoires_A']}")  
        st.write(f"Nuls : {st.session_state.data['head_to_head']['nuls']}")  
        st.write(f"Victoires de l'Équipe B : {st.session_state.data['head_to_head']['victoires_B']}")  

    with col_recent_form:  
        st.subheader("📈 Forme Récente")  
        col_a, col_b = st.columns(2)  
        with col_a:  
            st.write("Forme Récente de l'Équipe A")  
            for i in range(5):  
                st.session_state.data["recent_form_A"][i] = st.number_input(  
                    f"Match {i + 1} (Buts Marqués)",  
                    min_value=0,  
                    value=int(st.session_state.data["recent_form_A"][i]),  
                    key=f"recent_form_A_{i}"  
                )  
        with col_b:  
            st.write("Forme Récente de l'Équipe B")  
            for i in range(5):  
                st.session_state.data["recent_form_B"][i] = st.number_input(  
                    f"Match {i + 1} (Buts Marqués)",  
                    min_value=0,  
                    value=int(st.session_state.data["recent_form_B"][i]),  
                    key=f"recent_form_B_{i}"  
                )  

with tab3:  
    st.subheader("🔮 Prédiction du Résultat du Match")  

    if st.button("Prédire le Résultat du Match"):  
        try:  
            # Méthode de Poisson  
            avg_goals_A = st.session_state.data["expected_but_A"]  
            avg_goals_B = st.session_state.data["expected_but_B"]  

            goals_A = [poisson.pmf(i, avg_goals_A) for i in range(6)]  
            goals_B = [poisson.pmf(i, avg_goals_B) for i in range(6)]  

            results = pd.DataFrame(np.zeros((6, 6)), columns=[f"Équipe B: {i}" for i in range(6)], index=[f"Équipe A: {i}" for i in range(6)])  
            for i in range(6):  
                for j in range(6):  
                    results.iloc[i, j] = goals_A[i] * goals_B[j]  

            # Conversion en pourcentages  
            results_percentage = results * 100  

            st.subheader("📊 Résultats de la Méthode de Poisson (en %)")  
            st.write(results_percentage)  

            # Amélioration du schéma  
            plt.figure(figsize=(10, 6))  
            plt.imshow(results_percentage, cmap='Blues', interpolation='nearest')  
            plt.colorbar(label='Probabilité (%)')  
            plt.xticks(ticks=np.arange(6), labels=[f"Équipe B: {i}" for i in range(6)])  
            plt.yticks(ticks=np.arange(6), labels=[f"Équipe A: {i}" for i in range(6)])  
            plt.title("Probabilités des Résultats (Méthode de Poisson)")  
            st.pyplot(plt)  

            # Random Forest  
            # Préparation des données  
            X = []  
            y = []  
            for _ in range(1000):  # Génération de données d'entraînement  
                stats_A = [np.random.rand() * 100 for _ in range(26)]  # 26 statistiques pour l'Équipe A  
                stats_B = [np.random.rand() * 100 for _ in range(26)]  # 26 statistiques pour l'Équipe B  
                X.append(stats_A + stats_B)  
                y.append(np.random.randint(0, 2))  # Cible binaire  

            X = np.array(X)  
            y = np.array(y)  

            # Division en ensembles d'entraînement et de test  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

            # Entraînement du modèle  
            model_rf = RandomForestClassifier()  
            model_rf.fit(X_train, y_train)  

            # Prédiction  
            y_pred_rf = model_rf.predict(X_test)  

            # Évaluation du modèle  
            accuracy_rf = accuracy_score(y_test, y_pred_rf)  
            cm_rf = confusion_matrix(y_test, y_pred_rf)  

            # Vérification que y_pred est bien défini avant d'appeler classification_report  
            if y_pred_rf is not None:  
                report_rf = classification_report(y_test, y_pred_rf, output_dict=True)  
            else:  
                st.error("La prédiction n'a pas pu être générée.")  

            # Affichage des résultats dans un tableau  
            st.subheader("📈 Résultats de la Random Forest")  
            st.write(f"Précision du modèle : {accuracy_rf:.2%}")  

            # Tableau des résultats  
            if y_pred_rf is not None:  
                report_df_rf = pd.DataFrame(report_rf).transpose()  
                st.table(report_df_rf)  

                # Explication des résultats  
                st.markdown("""  
                **Explication des résultats :**  
                - **Précision (Precision)** : Proportion de prédictions positives correctes.  
                - **Rappel (Recall)** : Proportion de cas positifs correctement identifiés.  
                - **F1-Score** : Moyenne harmonique de la précision et du rappel.  
                - **Support** : Nombre d'échantillons pour chaque classe.  
                """)  
            else:  
                st.error("Impossible de générer le rapport de classification.")  

            # Prédiction du match actuel  
            current_match_features = np.array([[st.session_state.data[key] for key in st.session_state.data if key.endswith("_A") or key.endswith("_B")]])  

            # Prédiction avec Random Forest  
            prediction_rf = model_rf.predict(current_match_features)  
            prediction_proba_rf = model_rf.predict_proba(current_match_features)  

            # Détermination du pari double chance  
            if abs(prediction_proba_rf[0][0] - prediction_proba_rf[0][1]) < 0.1:  
                st.success("🔔 Résultat serré : Pari Double Chance recommandé (1X ou X2) 🔔")  
                if prediction_rf[0] == 1:  
                    st.info("Pari Double Chance : 1X (Équipe A ou Match Nul)")  
                else:  
                    st.info("Pari Double Chance : X2 (Match Nul ou Équipe B)")  
            else:  
                if prediction_rf[0] == 1:  
                    st.success("Prédiction : L'Équipe A gagne 🎉")  
                else:  
                    st.success("Prédiction : L'Équipe B gagne 🎉")  

            # Affichage des probabilités  
            st.subheader("📊 Probabilités des Prédictions")  
            st.write(f"Probabilité Équipe A : {prediction_proba_rf[0][1]:.2%}")  
            st.write(f"Probabilité Équipe B : {prediction_proba_rf[0][0]:.2%}")  

        except Exception as e:  
            st.error(f"Une erreur s'est produite lors de la prédiction : {str(e)}")  

# Pied de page simulé avec HTML  
st.markdown(  
    """  
    <div style="text-align: center; padding: 10px; background-color: #f0f0f0; margin-top: 20px;">  
        ⚽ Application de Prédiction de Matchs de Football - Version 1.0  
    </div>  
    """,  
    unsafe_allow_html=True  
)
