import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
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
        "head_to_head": [],  
        "conditions_match": "",  
    }  

# Configuration de la page  
st.set_page_config(page_title="Prédiction de Matchs de Football", page_icon="⚽", layout="wide")  

# En-tête  
st.header("Analyse de Prédiction de Matchs de Football")  

# Onglets pour les différentes sections  
tab1, tab2, tab3 = st.tabs(["Statistiques des Équipes", "Conditions du Match", "Prédictions"])  

with tab1:  
    st.subheader("Statistiques des Équipes")  

    col_a, col_b = st.columns(2)  

    with col_a:  
        st.subheader("Équipe A")  
        st.number_input("Score Rating", min_value=0.0, value=float(st.session_state.data["score_rating_A"]), key="score_rating_A")  
        st.number_input("Buts Totaux", min_value=0.0, value=float(st.session_state.data["buts_totaux_A"]), key="buts_totaux_A")  
        st.number_input("Buts par Match", min_value=0.0, value=float(st.session_state.data["buts_par_match_A"]), key="buts_par_match_A")  
        st.number_input("Buts Concédés par Match", min_value=0.0, value=float(st.session_state.data["buts_concedes_par_match_A"]), key="buts_concedes_par_match_A")  
        st.number_input("Buts Concédés Totaux", min_value=0.0, value=float(st.session_state.data["buts_concedes_totaux_A"]), key="buts_concedes_totaux_A")  
        st.number_input("Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.data["possession_moyenne_A"]), key="possession_moyenne_A")  
        st.number_input("Clean Sheets", min_value=0, value=int(st.session_state.data["aucun_but_encaisse_A"]), key="aucun_but_encaisse_A")  

    with col_b:  
        st.subheader("Équipe B")  
        st.number_input("Score Rating", min_value=0.0, value=float(st.session_state.data["score_rating_B"]), key="score_rating_B")  
        st.number_input("Buts Totaux", min_value=0.0, value=float(st.session_state.data["buts_totaux_B"]), key="buts_totaux_B")  
        st.number_input("Buts par Match", min_value=0.0, value=float(st.session_state.data["buts_par_match_B"]), key="buts_par_match_B")  
        st.number_input("Buts Concédés par Match", min_value=0.0, value=float(st.session_state.data["buts_concedes_par_match_B"]), key="buts_concedes_par_match_B")  
        st.number_input("Buts Concédés Totaux", min_value=0.0, value=float(st.session_state.data["buts_concedes_totaux_B"]), key="buts_concedes_totaux_B")  
        st.number_input("Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.data["possession_moyenne_B"]), key="possession_moyenne_B")  
        st.number_input("Clean Sheets", min_value=0, value=int(st.session_state.data["aucun_but_encaisse_B"]), key="aucun_but_encaisse_B")  

with tab2:  
    st.subheader("Conditions du Match")  
    st.text_input("Conditions du Match (ex : pluie, terrain sec, etc.)", value=st.session_state.data["conditions_match"], key="conditions_match")  

    col_h2h, col_recent_form = st.columns(2)  

    with col_h2h:  
        st.subheader("Historique des Confrontations")  
        col_a, col_b = st.columns(2)  
        with col_a:  
            equipe_A_buts = st.number_input("Buts de l'Équipe A", min_value=0)  
        with col_b:  
            equipe_B_buts = st.number_input("Buts de l'Équipe B", min_value=0)  

        if st.button("Ajouter un Résultat"):  
            st.session_state.data["head_to_head"].append((equipe_A_buts, equipe_B_buts))  
            st.success("Résultat ajouté avec succès !")  

        if st.session_state.data["head_to_head"]:  
            st.subheader("Résultats des Confrontations")  
            for index, (buts_A, buts_B) in enumerate(st.session_state.data["head_to_head"]):  
                st.write(f"Match {index + 1}: Équipe A {buts_A} - Équipe B {buts_B}")  

    with col_recent_form:  
        st.subheader("Forme Récente")  
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
    st.subheader("Prédiction du Résultat du Match")  

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

            st.subheader("Résultats de la Méthode de Poisson")  
            st.write(results)  

            plt.figure(figsize=(10, 6))  
            plt.imshow(results, cmap='Blues', interpolation='nearest')  
            plt.colorbar(label='Probabilité')  
            plt.xticks(ticks=np.arange(6), labels=[f"Équipe B: {i}" for i in range(6)])  
            plt.yticks(ticks=np.arange(6), labels=[f"Équipe A: {i}" for i in range(6)])  
            plt.title("Probabilités des Résultats (Méthode de Poisson)")  
            st.pyplot(plt)  

            # Régression Logistique  
            # Génération de données d'entraînement  
            np.random.seed(0)  
            X = np.random.rand(100, 10)  # 100 échantillons, 10 caractéristiques  
            y = np.random.randint(0, 2, 100)  # Cible binaire  

            # Division en ensembles d'entraînement et de test  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

            # Entraînement du modèle  
            model = LogisticRegression()  
            model.fit(X_train, y_train)  

            # Prédiction  
            prediction = model.predict(X_test)  

            # Évaluation du modèle  
            accuracy = accuracy_score(y_test, prediction)  
            cm = confusion_matrix(y_test, prediction)  
            report = classification_report(y_test, prediction)  

            st.subheader("Résultats de la Régression Logistique")  
            st.write(f"Précision : {accuracy:.2f}")  
            st.write("Matrice de Confusion :")  
            st.write(cm)  
            st.write("Rapport de Classification :")  
            st.write(report)  

            # Prédiction du match actuel  
            current_match_features = np.array([[st.session_state.data["score_rating_A"], st.session_state.data["buts_par_match_A"], st.session_state.data["buts_concedes_par_match_A"],  
                                               st.session_state.data["possession_moyenne_A"], st.session_state.data["expected_but_A"],  
                                               st.session_state.data["score_rating_B"], st.session_state.data["buts_par_match_B"], st.session_state.data["buts_concedes_par_match_B"],  
                                               st.session_state.data["possession_moyenne_B"], st.session_state.data["expected_but_B"]]])  

            current_prediction = model.predict(current_match_features)  
            if current_prediction[0] == 1:  
                st.success("Prédiction : L'Équipe A gagne !")  
            else:  
                st.success("Prédiction : L'Équipe B gagne !")  

        except Exception as e:  
            st.error(f"Une erreur s'est produite lors de la prédiction : {str(e)}")  

# Pied de page  
st.footer("Application de Prédiction de Matchs de Football - Version 1.0")
