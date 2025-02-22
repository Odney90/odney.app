import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
import matplotlib.pyplot as plt  

# Initialisation des données dans session_state  
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
        
        # Conditions du match  
        "conditions_match": "",  
    }  

# Titre de l'application  
st.title("⚽ Analyse de Prédiction de Matchs de Football")  
st.markdown("Renseignez les critères suivants pour prédire le résultat du match :")  

# Section 1 : Statistiques des Équipes  
st.header("📊 Statistiques des Équipes")  

col_a, col_b = st.columns(2)  

# Équipe A  
with col_a:  
    st.subheader("Équipe A 🟡")  
    st.session_state.data["score_rating_A"] = st.number_input(  
        "⭐ Score Rating (Équipe A)",  
        min_value=0.0,  
        value=float(st.session_state.data["score_rating_A"]),  
        key="score_rating_A_input"  
    )  
    st.session_state.data["buts_par_match_A"] = st.number_input(  
        "⚽ Buts par match (Équipe A)",  
        min_value=0.0,  
        value=float(st.session_state.data["buts_par_match_A"]),  
        key="buts_par_match_A_input"  
    )  
    st.session_state.data["buts_concedes_par_match_A"] = st.number_input(  
        "🥅 Buts concédés par match (Équipe A)",  
        min_value=0.0,  
        value=float(st.session_state.data["buts_concedes_par_match_A"]),  
        key="buts_concedes_par_match_A_input"  
    )  
    st.session_state.data["possession_moyenne_A"] = st.number_input(  
        "🔄 Possession moyenne (Équipe A)",  
        min_value=0.0,  
        max_value=100.0,  
        value=float(st.session_state.data["possession_moyenne_A"]),  
        key="possession_moyenne_A_input"  
    )  
    st.session_state.data["joueurs_cles_absents_A"] = st.number_input(  
        "🚑 Joueurs clés absents (Équipe A)",  
        min_value=0,  
        value=int(st.session_state.data["joueurs_cles_absents_A"]),  
        key="joueurs_cles_absents_A_input"  
    )  

# Équipe B  
with col_b:  
    st.subheader("Équipe B 🔴")  
    st.session_state.data["score_rating_B"] = st.number_input(  
        "⭐ Score Rating (Équipe B)",  
        min_value=0.0,  
        value=float(st.session_state.data["score_rating_B"]),  
        key="score_rating_B_input"  
    )  
    st.session_state.data["buts_par_match_B"] = st.number_input(  
        "⚽ Buts par match (Équipe B)",  
        min_value=0.0,  
        value=float(st.session_state.data["buts_par_match_B"]),  
        key="buts_par_match_B_input"  
    )  
    st.session_state.data["buts_concedes_par_match_B"] = st.number_input(  
        "🥅 Buts concédés par match (Équipe B)",  
        min_value=0.0,  
        value=float(st.session_state.data["buts_concedes_par_match_B"]),  
        key="buts_concedes_par_match_B_input"  
    )  
    st.session_state.data["possession_moyenne_B"] = st.number_input(  
        "🔄 Possession moyenne (Équipe B)",  
        min_value=0.0,  
        max_value=100.0,  
        value=float(st.session_state.data["possession_moyenne_B"]),  
        key="possession_moyenne_B_input"  
    )  
    st.session_state.data["joueurs_cles_absents_B"] = st.number_input(  
        "🚑 Joueurs clés absents (Équipe B)",  
        min_value=0,  
        value=int(st.session_state.data["joueurs_cles_absents_B"]),  
        key="joueurs_cles_absents_B_input"  
    )  

# Section 2 : Conditions du Match et Motivation  
st.header("🌦️ Conditions du Match et Motivation")  
col_a, col_b = st.columns(2)  

# Conditions du Match  
with col_a:  
    st.session_state.data["conditions_match"] = st.text_input(  
        "🌧️ Conditions du Match (ex : pluie, terrain sec, etc.)",  
        value=st.session_state.data["conditions_match"],  
        key="conditions_match_input"  
    )  

# Motivation  
with col_b:  
    st.session_state.data["motivation_A"] = st.slider(  
        "💪 Motivation de l'Équipe A (1 à 5)",  
        min_value=1,  
        max_value=5,  
        value=int(st.session_state.data["motivation_A"]),  
        key="motivation_A_slider"  
    )  
    st.session_state.data["motivation_B"] = st.slider(  
        "💪 Motivation de l'Équipe B (1 à 5)",  
        min_value=1,  
        max_value=5,  
        value=int(st.session_state.data["motivation_B"]),  
        key="motivation_B_slider"  
    )  

# Section 3 : Prédictions  
st.header("🔮 Prédictions")  
if st.button("Prédire le résultat"):  
    # Prédiction des Buts avec Poisson  
    avg_goals_A = st.session_state.data["buts_par_match_A"]  
    avg_goals_B = st.session_state.data["buts_par_match_B"]  
    prob_0_0 = poisson.pmf(0, avg_goals_A) * poisson.pmf(0, avg_goals_B)  
    prob_1_1 = poisson.pmf(1, avg_goals_A) * poisson.pmf(1, avg_goals_B)  
    prob_2_2 = poisson.pmf(2, avg_goals_A) * poisson.pmf(2, avg_goals_B)  

    # Génération de données d'entraînement factices pour la régression logistique  
    np.random.seed(42)  
    X_lr = np.random.rand(100, 6) * 100  # 100 échantillons, 6 caractéristiques  
    y_lr = np.random.randint(0, 2, 100)  # 100 labels binaires (0 ou 1)  

    # Régression Logistique  
    model_lr = LogisticRegression()  
    model_lr.fit(X_lr, y_lr)  
    X_lr_pred = np.array([  
        [  
            st.session_state.data["score_rating_A"],  
            st.session_state.data["score_rating_B"],  
            st.session_state.data["possession_moyenne_A"],  
            st.session_state.data["possession_moyenne_B"],  
            st.session_state.data["motivation_A"],  
            st.session_state.data["motivation_B"],  
        ]  
    ])  
    prediction_lr = model_lr.predict(X_lr_pred)  

    # Génération de données d'entraînement factices pour Random Forest  
    X_rf = np.random.rand(100, 50) * 100  # 100 échantillons, 50 caractéristiques  
    y_rf = np.random.randint(0, 2, 100)  # 100 labels binaires (0 ou 1)  

    # Random Forest  
    model_rf = RandomForestClassifier()  
    model_rf.fit(X_rf, y_rf)  
    X_rf_pred = np.array([  
        [  
            st.session_state.data["score_rating_A"],  
            st.session_state.data["score_rating_B"],  
            st.session_state.data["buts_par_match_A"],  
            st.session_state.data["buts_par_match_B"],  
            st.session_state.data["possession_moyenne_A"],  
            st.session_state.data["possession_moyenne_B"],  
            st.session_state.data["joueurs_cles_absents_A"],  
            st.session_state.data["joueurs_cles_absents_B"],  
            st.session_state.data["motivation_A"],  
            st.session_state.data["motivation_B"],  
            st.session_state.data["expected_but_A"],  
            st.session_state.data["expected_but_B"],  
            st.session_state.data["tirs_cadres_A"],  
            st.session_state.data["tirs_cadres_B"],  
            st.session_state.data["grandes_chances_A"],  
            st.session_state.data["grandes_chances_B"],  
            st.session_state.data["passes_reussies_A"],  
            st.session_state.data["passes_reussies_B"],  
            st.session_state.data["corners_A"],  
            st.session_state.data["corners_B"],  
            st.session_state.data["interceptions_A"],  
            st.session_state.data["interceptions_B"],  
            st.session_state.data["tacles_reussis_A"],  
            st.session_state.data["tacles_reussis_B"],  
            st.session_state.data["fautes_A"],  
            st.session_state.data["fautes_B"],  
            st.session_state.data["cartons_jaunes_A"],  
            st.session_state.data["cartons_jaunes_B"],  
            st.session_state.data["cartons_rouges_A"],  
            st.session_state.data["cartons_rouges_B"],  
            st.session_state.data["clean_sheets_gardien_A"],  
            st.session_state.data["clean_sheets_gardien_B"],  
            st.session_state.data["ratio_tirs_arretes_A"],  
            st.session_state.data["ratio_tirs_arretes_B"],  
            st.session_state.data["victoires_domicile_A"],  
            st.session_state.data["victoires_exterieur_B"],  
            st.session_state.data["passes_longues_A"],  
            st.session_state.data["passes_longues_B"],  
            st.session_state.data["dribbles_reussis_A"],  
            st.session_state.data["dribbles_reussis_B"],  
            st.session_state.data["ratio_tirs_cadres_A"],  
            st.session_state.data["ratio_tirs_cadres_B"],  
            st.session_state.data["grandes_chances_manquees_A"],  
            st.session_state.data["grandes_chances_manquees_B"],  
            st.session_state.data["fautes_zones_dangereuses_A"],  
            st.session_state.data["fautes_zones_dangereuses_B"],  
            st.session_state.data["buts_corners_A"],  
            st.session_state.data["buts_corners_B"],  
            st.session_state.data["jours_repos_A"],  
            st.session_state.data["jours_repos_B"],  
            st.session_state.data["matchs_30_jours_A"],  
            st.session_state.data["matchs_30_jours_B"],  
        ]  
    ])  
    prediction_rf = model_rf.predict(X_rf_pred)  

    # Affichage des résultats  
    st.subheader("Résultats des Prédictions")  
    st.write(f"📊 **Probabilité de 0-0 (Poisson)** : {prob_0_0:.2%}")  
    st.write(f"📊 **Probabilité de 1-1 (Poisson)** : {prob_1_1:.2%}")  
    st.write(f"📊 **Probabilité de 2-2 (Poisson)** : {prob_2_2:.2%}")  
    st.write(f"📊 **Régression Logistique** : {'Équipe A' if prediction_lr[0] == 1 else 'Équipe B'}")  
    st.write(f"🌲 **Random Forest** : {'Équipe A' if prediction_rf[0] == 1 else 'Équipe B'}")  

# Section 4 : Bankroll et Value Bet  
st.header("💶 Bankroll et Value Bet")  
col_a, col_b = st.columns(2)  

# Cote Implicite et Value Bet  
with col_a:  
    cote = st.number_input(  
        "📈 Cote offerte",  
        min_value=1.0,  
        value=2.0,  
        key="cote_input"  
    )  
    prob_victoire = st.number_input(  
        "🎯 Probabilité de victoire (en %)",  
        min_value=0.0,  
        max_value=100.0,  
        value=50.0,  
        key="prob_victoire_input"  
    )  
    cote_implicite = 100 / prob_victoire  
    value_bet = (cote * prob_victoire / 100) - 1  
    st.write(f"📊 **Cote Implicite** : {cote_implicite:.2f}")  
    st.write(f"📊 **Value Bet** : {value_bet:.2%}")  

# Mise de Kelly  
with col_b:  
    bankroll = st.number_input(  
        "💰 Bankroll (en €)",  
        min_value=0.0,  
        value=100.0,  
        key="bankroll_input"  
    )  
    mise_kelly = (prob_victoire / 100 * (cote - 1) - (1 - prob_victoire / 100)) / (cote - 1)  
    mise_kelly = max(0, mise_kelly)  # Éviter les valeurs négatives  
    st.write(f"💶 **Mise de Kelly recommandée** : {mise_kelly * bankroll:.2f} €")  

# Section 5 : Visuels  
st.header("📈 Visuels")  
col_a, col_b = st.columns(2)  

# Graphique des buts  
with col_a:  
    fig, ax = plt.subplots()  
    ax.bar(["Équipe A", "Équipe B"], [st.session_state.data["buts_par_match_A"], st.session_state.data["buts_par_match_B"]], color=["yellow", "red"])  
    ax.set_title("⚽ Buts par match")  
    st.pyplot(fig)  

# Graphique de possession  
with col_b:  
    fig, ax = plt.subplots()  
    ax.bar(["Équipe A", "Équipe B"], [st.session_state.data["possession_moyenne_A"], st.session_state.data["possession_moyenne_B"]], color=["yellow", "red"])  
    ax.set_title("🔄 Possession moyenne")  
    st.pyplot(fig)
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
            X_rf = np.array([[st.session_state.data[key] for key in st.session_state.data if (key.endswith("_A") or key.endswith("_B")) and isinstance(st.session_state.data[key], (int, float))]])  

          # Vérification de la forme de X_rf_pred  
if X_rf_pred.shape[1] != 52:  
    st.error(f"Erreur : Le nombre de caractéristiques doit être de 52. Actuellement : {X_rf_pred.shape[1]}")  
else:  
    prediction_rf = model_rf.predict(X_rf_pred)  
    st.write(f"🌲 **Random Forest** : {'Équipe A' if prediction_rf[0] == 1 else 'Équipe B'}")  
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

    # Calcul des probabilités implicites  
    probabilites_implicites = [cotes_vers_probabilite(cote) for cote in cotes]  
    marge_bookmaker = sum(probabilites_implicites) - 1  
    st.write(f"Marge du bookmaker : **{marge_bookmaker:.2%}**")  

    # Enlève la marge pour obtenir les probabilités corrigées  
    probabilites_corrigees = enlever_marge(probabilites_implicites, marge_bookmaker)  
    for i, proba in enumerate(probabilites_corrigees):  
        st.write(f"Probabilité corrigée pour la cote {cotes[i]} : **{proba:.2%}**")
