import streamlit as st  
import numpy as np  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
import pandas as pd  
import matplotlib.pyplot as plt  

# Initialisation des données par défaut  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        # Statistiques de l'Équipe A  
        "score_rating_A": 70.0,  
        "buts_par_match_A": 1.5,  
        "buts_concedes_par_match_A": 1.0,  
        "possession_moyenne_A": 55.0,  
        "expected_but_A": 1.8,  
        "expected_concedes_A": 1.2,  
        "tirs_cadres_A": 120,  
        "grandes_chances_A": 25,  
        "passes_reussies_A": 400,  
        "corners_A": 60,  
        "interceptions_A": 50,  
        "tacles_reussis_A": 40,  
        "fautes_A": 15,  
        "cartons_jaunes_A": 5,  
        "cartons_rouges_A": 1,  

        # Statistiques de l'Équipe B  
        "score_rating_B": 65.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "possession_moyenne_B": 45.0,  
        "expected_but_B": 1.2,  
        "expected_concedes_B": 1.8,  
        "tirs_cadres_B": 100,  
        "grandes_chances_B": 20,  
        "passes_reussies_B": 350,  
        "corners_B": 50,  
        "interceptions_B": 40,  
        "tacles_reussis_B": 35,  
        "fautes_B": 20,  
        "cartons_jaunes_B": 6,  
        "cartons_rouges_B": 2,  

        "recent_form_A": [1, 2, 1, 0, 3],  
        "recent_form_B": [0, 1, 2, 1, 1],  
        "head_to_head": {"victoires_A": 0, "nuls": 0, "victoires_B": 0},  
    }  

# Fonctions pour les outils de paris  
def cotes_vers_probabilite(cote):  
    """Convertit une cote décimale en probabilité implicite."""  
    return 1 / cote  

def enlever_marge(probabilites, marge):  
    """Enlève la marge du bookmaker des probabilités."""  
    total_probabilite = sum(probabilites)  
    facteur_correction = (1 - marge) / total_probabilite  
    return [p * facteur_correction for p in probabilites]  

def kelly_criterion(cote, probabilite, bankroll, kelly_fraction):  
    """Calcule la mise de Kelly."""  
    probabilite_avantage = probabilite * cote - 1  
    fraction = kelly_fraction * probabilite_avantage / (cote - 1)  
    return bankroll * fraction  

# Configuration de la page  
st.set_page_config(page_title="Prédiction de Matchs de Football", page_icon="⚽", layout="wide")  

# En-tête  
st.header("⚽ Analyse de Prédiction de Matchs de Football")  

# Onglets pour les différentes sections  
tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistiques des Équipes", "🌦️ Conditions du Match", "🔮 Prédictions", "🛠️ Outils de Paris"])  

with tab1:  
    st.subheader("📊 Statistiques des Équipes")  

    col_a, col_b = st.columns(2)  

    with col_a:  
        st.subheader("Équipe A 🟡")  
        for key in st.session_state.data:  
            if key.endswith("_A") and isinstance(st.session_state.data[key], (int, float)):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    min_value=0.0,  
                    value=float(st.session_state.data[key]),  
                    key=key  
                )  

    with col_b:  
        st.subheader("Équipe B 🔴")  
        for key in st.session_state.data:  
            if key.endswith("_B") and isinstance(st.session_state.data[key], (int, float)):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    min_value=0.0,  
                    value=float(st.session_state.data[key]),  
                    key=key  
                )  

with tab2:  
    st.subheader("🌦️ Conditions du Match")  
    st.session_state.data["conditions_match"] = st.text_input("Conditions du Match (ex : pluie, terrain sec, etc.)", value="", key="conditions_match")  

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

            # Régression Logistique  
            # Critères importants pour la régression logistique  
            X_lr = np.array([  
                [  
                    st.session_state.data["score_rating_A"],  
                    st.session_state.data["buts_par_match_A"],  
                    st.session_state.data["buts_concedes_par_match_A"],  
                    st.session_state.data["possession_moyenne_A"],  
                    st.session_state.data["expected_but_A"],  
                    st.session_state.data["score_rating_B"],  
                    st.session_state.data["buts_par_match_B"],  
                    st.session_state.data["buts_concedes_par_match_B"],  
                    st.session_state.data["possession_moyenne_B"],  
                    st.session_state.data["expected_but_B"]  
                ]  
            ])  

            # Génération de données d'entraînement  
            np.random.seed(0)  
            X_train_lr = np.random.rand(100, 10)  # 100 échantillons, 10 caractéristiques  
            y_train_lr = np.random.randint(0, 2, 100)  # Cible binaire  

            # Entraînement du modèle  
            model_lr = LogisticRegression()  
            model_lr.fit(X_train_lr, y_train_lr)  

            # Prédiction  
            prediction_lr = model_lr.predict(X_lr)  
            prediction_proba_lr = model_lr.predict_proba(X_lr)  

            # Affichage des résultats  
            st.subheader("📈 Résultats de la Régression Logistique")  
            st.write(f"Probabilité Équipe A : {prediction_proba_lr[0][1]:.2%}")  
            st.write(f"Probabilité Équipe B : {prediction_proba_lr[0][0]:.2%}")  

            # Random Forest  
            # Utilisation de toutes les statistiques disponibles (uniquement les valeurs numériques)  
            X_rf = np.array([[st.session_state.data[key] for key in st.session_state.data if key.endswith("_A") or key.endswith("_B") and isinstance(st.session_state.data[key], (int, float))]])  

            # Vérification que toutes les valeurs sont numériques  
            if X_rf.shape[1] != 52:  # Assurez-vous que vous avez le bon nombre de caractéristiques  
                raise ValueError("Le nombre de caractéristiques doit être de 52.")  

            # Génération de données d'entraînement  
            np.random.seed(0)  
            X_train_rf = np.random.rand(100, 52)  # 100 échantillons, 52 caractéristiques  
            y_train_rf = np.random.randint(0, 2, 100)  # Cible binaire  

            # Entraînement du modèle  
            model_rf = RandomForestClassifier()  
            model_rf.fit(X_train_rf, y_train_rf)  

            # Prédiction  
            prediction_rf = model_rf.predict(X_rf)  
            prediction_proba_rf = model_rf.predict_proba(X_rf)  

            # Affichage des résultats  
            st.subheader("📈 Résultats de la Random Forest")  
            st.write(f"Probabilité Équipe A : {prediction_proba_rf[0][1]:.2%}")  
            st.write(f"Probabilité Équipe B : {prediction_proba_rf[0][0]:.2%}")  

            # Comparaison des modèles  
            st.subheader("📊 Comparaison des Modèles")  
            comparison_df = pd.DataFrame({  
                "Modèle": ["Poisson", "Régression Logistique", "Random Forest"],  
                "Probabilité Équipe A": [results_percentage.iloc[1:].sum().sum(), prediction_proba_lr[0][1], prediction_proba_rf[0][1]],  
                "Probabilité Équipe B": [results_percentage.iloc[:, 1:].sum().sum(), prediction_proba_lr[0][0], prediction_proba_rf[0][0]]  
            })  
            st.table(comparison_df)  

            # Détermination du pari double chance  
            if abs(prediction_proba_lr[0][0] - prediction_proba_lr[0][1]) < 0.1 or abs(prediction_proba_rf[0][0] - prediction_proba_rf[0][1]) < 0.1:  
                st.success("🔔 Résultat serré : Pari Double Chance recommandé (1X ou X2) 🔔")  
                if prediction_lr[0] == 1 and prediction_rf[0] == 1:  
                    st.info("Pari Double Chance : 1X (Équipe A ou Match Nul)")  
                elif prediction_lr[0] == 0 and prediction_rf[0] == 0:  
                    st.info("Pari Double Chance : X2 (Match Nul ou Équipe B)")  
                else:  
                    st.info("Pari Double Chance : 1X ou X2 (Résultat trop incertain)")  
            else:  
                if prediction_lr[0] == 1 and prediction_rf[0] == 1:  
                    st.success("Prédiction : L'Équipe A gagne 🎉")  
                elif prediction_lr[0] == 0 and prediction_rf[0] == 0:  
                    st.success("Prédiction : L'Équipe B gagne 🎉")  
                else:  
                    st.warning("Prédiction : Match Nul ou Résultat Incertain 🤔")  

        except Exception as e:  
            st.error(f"Une erreur s'est produite lors de la prédiction : {str(e)}")  

with tab4:  
    st.title("🛠️ Outils de Paris")  

    # Convertisseur de cotes  
    st.header("🧮 Convertisseur de Cotes")  
    cote_decimale = st.number_input("Cote décimale", min_value=1.01, value=2.0)  
    probabilite_implicite = cotes_vers_probabilite(cote_decimale)  
    st.write(f"Probabilité implicite : **{probabilite_implicite:.2%}**")  

    # Calcul de value bets  
    st.header("💰 Calcul de Value Bets")  
    cote_value_bet = st.number_input("Cote (value bet)", min_value=1.01, value=2.0, key="cote_value_bet")  
    probabilite_estimee = st.slider("Probabilité estimée (%)", min_value=1, max_value=100, value=50) / 100  
    probabilite_implicite_value_bet = cotes_vers_probabilite(cote_value_bet)  
    if probabilite_estimee > probabilite_implicite_value_bet:  
        st.success("Value bet détecté !")  
    else:  
        st.warning("Pas de value bet.")  

    # Simulateur de paris combinés  
    st.header("➕ Simulateur de Paris Combinés")  
    nombre_equipes = st.slider("Nombre d'équipes (jusqu'à 3)", min_value=1, max_value=3, value=2)  
    probabilites = []  
    for i in range(nombre_equipes):  
        probabilite = st.slider(f"Probabilité implicite équipe {i + 1} (%)", min_value=1, max_value=100, value=50, key=f"probabilite_{i}") / 100  
        probabilites.append(probabilite)  
    probabilite_combinee = np.prod(probabilites)  
    st.write(f"Probabilité combinée : **{probabilite_combinee:.2%}**")  

    # Mise de Kelly  
    st.header("🏦 Mise de Kelly")  
    cote_kelly = st.number_input("Cote (Kelly)", min_value=1.01, value=2.0, key="cote_kelly")  
    probabilite_kelly = st.slider("Probabilité estimée (Kelly) (%)", min_value=1, max_value=100, value=50, key="probabilite_kelly") / 100  
    bankroll = st.number_input("Bankroll", min_value=1, value=1000)  
    kelly_fraction = st.slider("Fraction de Kelly (1 à 5)", min_value=1, max_value=5, value=1) / 5  
    mise_kelly = kelly_criterion(cote_kelly, probabilite_kelly, bankroll, kelly_fraction)  
    st.write(f"Mise de Kelly recommandée : **{mise_kelly:.2f}**")  

    # Analyse de la marge du bookmaker  
    st.header("📊 Analyse de la Marge du Bookmaker")  
    cotes = []  
    nombre_cotes = st.slider("Nombre de cotes à analyser", min_value=1, max_value=10, value=3)  
    for i in range(nombre_cotes):  
        cote = st.number_input(f"Cote {i + 1}", min_value=1.01, value=2.0, key=f"cote_{i}")  
        cotes.append(cote)  
    probabilites_implicites = [cotes_vers_probabilite(cote) for cote in cotes]  
    marge_bookmaker = sum(probabilites_implicites) - 1  
    st.write(f"Marge du bookmaker : **{marge_bookmaker:.2%}**")
    probabilites_corrigees = enlever_marge(probabilites_implicites, marge_bookmaker)  
    st.write("Probabilités corrigées (sans marge) :")  
    for i, p in enumerate(probabilites_corrigees):  
        st.write(f"Événement {i + 1} : **{p:.2%}**")  

# Pied de page simulé avec HTML  
st.markdown(  
    """  
    <div style="text-align: center; padding: 10px; background-color: #f0f0f0; margin-top: 20px;">  
        ⚽ Application de Prédiction de Matchs de Football - Version 1.0  
    </div>  
    """,  
    unsafe_allow_html=True  
)
