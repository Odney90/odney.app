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
        # ... (autres données)  
        "head_to_head": {"victoires_A": 0, "nuls": 0, "victoires_B": 0},  # Initialisation correcte  
        # ... (autres données)  
    }  

# Vérification de sécurité avant l'accès  
if "head_to_head" not in st.session_state.data:  
    st.session_state.data["head_to_head"] = {"victoires_A": 0, "nuls": 0, "victoires_B": 0}  

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
        st.session_state.data["score_rating_A"] = st.number_input("Score Rating", min_value=0.0, value=float(st.session_state.data["score_rating_A"]), key="score_rating_A")  
        st.session_state.data["buts_totaux_A"] = st.number_input("Buts Totaux", min_value=0.0, value=float(st.session_state.data["buts_totaux_A"]), key="buts_totaux_A")  
        st.session_state.data["buts_par_match_A"] = st.number_input("Buts par Match", min_value=0.0, value=float(st.session_state.data["buts_par_match_A"]), key="buts_par_match_A")  
        st.session_state.data["buts_concedes_par_match_A"] = st.number_input("Buts Concédés par Match", min_value=0.0, value=float(st.session_state.data["buts_concedes_par_match_A"]), key="buts_concedes_par_match_A")  
        st.session_state.data["buts_concedes_totaux_A"] = st.number_input("Buts Concédés Totaux", min_value=0.0, value=float(st.session_state.data["buts_concedes_totaux_A"]), key="buts_concedes_totaux_A")  
        st.session_state.data["possession_moyenne_A"] = st.number_input("Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.data["possession_moyenne_A"]), key="possession_moyenne_A")  
        st.session_state.data["aucun_but_encaisse_A"] = st.number_input("Clean Sheets", min_value=0, value=int(st.session_state.data["aucun_but_encaisse_A"]), key="aucun_but_encaisse_A")  

    with col_b:  
        st.subheader("Équipe B 🔴")  
        st.session_state.data["score_rating_B"] = st.number_input("Score Rating", min_value=0.0, value=float(st.session_state.data["score_rating_B"]), key="score_rating_B")  
        st.session_state.data["buts_totaux_B"] = st.number_input("Buts Totaux", min_value=0.0, value=float(st.session_state.data["buts_totaux_B"]), key="buts_totaux_B")  
        st.session_state.data["buts_par_match_B"] = st.number_input("Buts par Match", min_value=0.0, value=float(st.session_state.data["buts_par_match_B"]), key="buts_par_match_B")  
        st.session_state.data["buts_concedes_par_match_B"] = st.number_input("Buts Concédés par Match", min_value=0.0, value=float(st.session_state.data["buts_concedes_par_match_B"]), key="buts_concedes_par_match_B")  
        st.session_state.data["buts_concedes_totaux_B"] = st.number_input("Buts Concédés Totaux", min_value=0.0, value=float(st.session_state.data["buts_concedes_totaux_B"]), key="buts_concedes_totaux_B")  
        st.session_state.data["possession_moyenne_B"] = st.number_input("Possession Moyenne (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.data["possession_moyenne_B"]), key="possession_moyenne_B")  
        st.session_state.data["aucun_but_encaisse_B"] = st.number_input("Clean Sheets", min_value=0, value=int(st.session_state.data["aucun_but_encaisse_B"]), key="aucun_but_encaisse_B")  

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

            st.subheader("📊 Résultats de la Méthode de Poisson")  
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

            st.subheader("📈 Résultats de la Régression Logistique")  
            st.write(f"Précision du modèle : {accuracy:.2%}")  
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
            prediction_proba = model.predict_proba(current_match_features)  

            if current_prediction[0] == 1:  
                st.success(f"Prédiction : L'Équipe A gagne avec une probabilité de {prediction_proba[0][1]:.2%} 🎉")  
            else:  
                st.success(f"Prédiction : L'Équipe B gagne avec une probabilité de {prediction_proba[0][0]:.2%} 🎉")  

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
