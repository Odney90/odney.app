import streamlit as st  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from scipy.stats import poisson  

# Fonction de sécurité pour les conversions  
def safe_float(value):  
    try:  
        return float(value)  
    except:  
        return 0.0  

def safe_int(value):  
    try:  
        return int(value)  
    except:  
        return 0  

# Initialisation de la session state  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        "score_rating_A": 0.0,  
        "score_rating_B": 0.0,  
        "tirs_cadres_par_match_A": 0.0,  
        "tirs_cadres_par_match_B": 0.0,  
        "grandes_chances_A": 0,  
        "grandes_chances_B": 0,  
        "passes_reussies_par_match_A": 0.0,  
        "passes_reussies_par_match_B": 0.0,  
        "centres_reussies_par_match_A": 0.0,  
        "centres_reussies_par_match_B": 0.0,  
        "dribbles_reussis_par_match_A": 0.0,  
        "dribbles_reussis_par_match_B": 0.0,  
        "buts_attendus_concedes_A": 0.0,  
        "buts_attendus_concedes_B": 0.0,  
        "interceptions_A": 0.0,  
        "interceptions_B": 0.0,  
        "tacles_reussis_par_match_A": 0.0,  
        "tacles_reussis_par_match_B": 0.0,  
        "penalties_concedees_A": 0,  
        "penalties_concedees_B": 0,  
        "fautes_par_match_A": 0.0,  
        "fautes_par_match_B": 0.0,  
        "motivation_A": 0,  
        "motivation_B": 0,  
        "buts_marques_A": 0.0,  
        "buts_marques_B": 0.0,  
        "tirs_par_match_A": 0.0,  
        "tirs_par_match_B": 0.0,  
        "possession_moyenne_A": 0.0,  
        "possession_moyenne_B": 0.0,  
        "corners_par_match_A": 0.0,  
        "corners_par_match_B": 0.0,  
        "forme_recente_A_victoires": 0,  
        "forme_recente_A_nuls": 0,  
        "forme_recente_A_defaites": 0,  
        "forme_recente_B_victoires": 0,  
        "forme_recente_B_nuls": 0,  
        "forme_recente_B_defaites": 0,  
        "cote_victoire_X": 0.0,  
        "cote_nul": 0.0,  
        "cote_victoire_Z": 0.0,  
        "bankroll": 0.0,  
        "historique_victoires_A": 0,  
        "historique_victoires_B": 0,  
        "historique_nuls": 0  
    }  

# Configuration de la page  
st.set_page_config(page_title="Prédiction de Matchs", layout="wide", page_icon="⚽")  

# Création des onglets  
tab1, tab2, tab3, tab4 = st.tabs(["Données des Équipes", "Prédictions", "Cotes et Value Bet", "Système de Mise"])  

# Onglet 1 : Données des Équipes  
with tab1:  
    st.header("📊 Données des Équipes")  

    # Colonnes pour les données  
    col1, col2 = st.columns(2)  

    # Formulaire pour l'Équipe A  
    with col1:  
        with st.form("form_equipe_A"):  
            st.subheader("🏆 Équipe A")  
            st.session_state.data["score_rating_A"] = st.number_input(  
                "⭐ Score Rating (A)",  
                value=safe_float(st.session_state.data.get("score_rating_A", 0.0)),  
                key="score_rating_A_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_cadres_par_match_A"] = st.number_input(  
                "🎯 Tirs Cadres par Match (A)",  
                value=safe_float(st.session_state.data.get("tirs_cadres_par_match_A", 0.0)),  
                key="tirs_cadres_A_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_A"] = st.number_input(  
                "🔥 Grandes Chances (A)",  
                value=safe_int(st.session_state.data.get("grandes_chances_A", 0)),  
                key="grandes_chances_A_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_A"] = st.number_input(  
                "🔄 Passes Réussies par Match (A)",  
                value=safe_float(st.session_state.data.get("passes_reussies_par_match_A", 0.0)),  
                key="passes_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_A"] = st.number_input(  
                "🎯 Centres Réussies par Match (A)",  
                value=safe_float(st.session_state.data.get("centres_reussies_par_match_A", 0.0)),  
                key="centres_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_A"] = st.number_input(  
                "🏃♂️ Dribbles Réussis par Match (A)",  
                value=safe_float(st.session_state.data.get("dribbles_reussis_par_match_A", 0.0)),  
                key="dribbles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_A"] = st.number_input(  
                "🚫 Buts Attendus Concédés (A)",  
                value=safe_float(st.session_state.data.get("buts_attendus_concedes_A", 0.0)),  
                key="buts_attendus_concedes_A_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_A"] = st.number_input(  
                "🛑 Interceptions (A)",  
                value=safe_float(st.session_state.data.get("interceptions_A", 0.0)),  
                key="interceptions_A_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_A"] = st.number_input(  
                "🦶 Tacles Réussis par Match (A)",  
                value=safe_float(st.session_state.data.get("tacles_reussis_par_match_A", 0.0)),  
                key="tacles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_A"] = st.number_input(  
                "⚠️ Penalties Concédées (A)",  
                value=safe_int(st.session_state.data.get("penalties_concedees_A", 0)),  
                key="penalties_concedees_A_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_A"] = st.number_input(  
                "🚩 Fautes par Match (A)",  
                value=safe_float(st.session_state.data.get("fautes_par_match_A", 0.0)),  
                key="fautes_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_A"] = st.number_input(  
                "💪 Motivation (A)",  
                value=safe_int(st.session_state.data.get("motivation_A", 0)),  
                key="motivation_A_input",  
                step=1  
            )  
            st.session_state.data["buts_marques_A"] = st.number_input(  
                "⚽ Buts Marqués par Match (A)",  
                value=safe_float(st.session_state.data.get("buts_marques_A", 0.0)),  
                key="buts_marques_A_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_A"] = st.number_input(  
                "🎯 Tirs par Match (A)",  
                value=safe_float(st.session_state.data.get("tirs_par_match_A", 0.0)),  
                key="tirs_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["possession_moyenne_A"] = st.number_input(  
                "⏳ Possession (%) (A)",  
                value=safe_float(st.session_state.data.get("possession_moyenne_A", 0.0)),  
                key="possession_moyenne_A_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_A"] = st.number_input(  
                "🔄 Corners par Match (A)",  
                value=safe_float(st.session_state.data.get("corners_par_match_A", 0.0)),  
                key="corners_par_match_A_input",  
                step=0.01  
            )  
            st.subheader("📅 Forme Récente (Équipe A)")  
            st.session_state.data["forme_recente_A_victoires"] = st.number_input(  
                "✅ Victoires (A) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data.get("forme_recente_A_victoires", 0)),  
                key="forme_recente_A_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_nuls"] = st.number_input(  
                "➖ Nuls (A) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data.get("forme_recente_A_nuls", 0)),  
                key="forme_recente_A_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_defaites"] = st.number_input(  
                "❌ Défaites (A) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data.get("forme_recente_A_defaites", 0)),  
                key="forme_recente_A_defaites_input",  
                step=1  
            )  
            st.subheader("📊 Historique des Confrontations")  
            st.session_state.data["historique_victoires_A"] = st.number_input(  
                "✅ Victoires Équipe A contre Équipe B",  
                value=safe_int(st.session_state.data.get("historique_victoires_A", 0)),  
                key="historique_victoires_A_input",  
                step=1  
            )  
            st.session_state.data["historique_victoires_B"] = st.number_input(  
                "✅ Victoires Équipe B contre Équipe A",  
                value=safe_int(st.session_state.data.get("historique_victoires_B", 0)),  
                key="historique_victoires_B_input",  
                step=1  
            )  
            st.session_state.data["historique_nuls"] = st.number_input(  
                "➖ Nuls entre Équipe A et Équipe B",  
                value=safe_int(st.session_state.data.get("historique_nuls", 0)),  
                key="historique_nuls_input",  
                step=1  
            )  
            submitted_A = st.form_submit_button("💾 Enregistrer les données Équipe A")  

            if submitted_A:  
                st.success("Données de l'Équipe A enregistrées avec succès !")  

    # Formulaire pour l'Équipe B  
    with col2:  
        with st.form("form_equipe_B"):  
            st.subheader("🏆 Équipe B")  
            st.session_state.data["score_rating_B"] = st.number_input(  
                "⭐ Score Rating (B)",  
                value=safe_float(st.session_state.data.get("score_rating_B", 0.0)),  
                key="score_rating_B_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_cadres_par_match_B"] = st.number_input(  
                "🎯 Tirs Cadres par Match (B)",  
                value=safe_float(st.session_state.data.get("tirs_cadres_par_match_B", 0.0)),  
                key="tirs_cadres_B_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_B"] = st.number_input(  
                "🔥 Grandes Chances (B)",  
                value=safe_int(st.session_state.data.get("grandes_chances_B", 0)),  
                key="grandes_chances_B_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_B"] = st.number_input(  
                "🔄 Passes Réussies par Match (B)",  
                value=safe_float(st.session_state.data.get("passes_reussies_par_match_B", 0.0)),  
                key="passes_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_B"] = st.number_input(  
                "🎯 Centres Réussies par Match (B)",  
                value=safe_float(st.session_state.data.get("centres_reussies_par_match_B", 0.0)),  
                key="centres_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_B"] = st.number_input(  
                "🏃♂️ Dribbles Réussis par Match (B)",  
                value=safe_float(st.session_state.data.get("dribbles_reussis_par_match_B", 0.0)),  
                key="dribbles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_B"] = st.number_input(  
                "🚫 Buts Attendus Concédés (B)",  
                value=safe_float(st.session_state.data.get("buts_attendus_concedes_B", 0.0)),  
                key="buts_attendus_concedes_B_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_B"] = st.number_input(  
                "🛑 Interceptions (B)",  
                value=safe_float(st.session_state.data.get("interceptions_B", 0.0)),  
                key="interceptions_B_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_B"] = st.number_input(  
                "🦶 Tacles Réussis par Match (B)",  
                value=safe_float(st.session_state.data.get("tacles_reussis_par_match_B", 0.0)),  
                key="tacles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_B"] = st.number_input(  
                "⚠️ Penalties Concédées (B)",  
                value=safe_int(st.session_state.data.get("penalties_concedees_B", 0)),  
                key="penalties_concedees_B_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_B"] = st.number_input(  
                "🚩 Fautes par Match (B)",  
                value=safe_float(st.session_state.data.get("fautes_par_match_B", 0.0)),  
                key="fautes_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_B"] = st.number_input(  
                "💪 Motivation (B)",  
                value=safe_int(st.session_state.data.get("motivation_B", 0)),  
                key="motivation_B_input",  
                step=1  
            )  
            st.session_state.data["buts_marques_B"] = st.number_input(  
                "⚽ Buts Marqués par Match (B)",  
                value=safe_float(st.session_state.data.get("buts_marques_B", 0.0)),  
                key="buts_marques_B_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_B"] = st.number_input(  
                "🎯 Tirs par Match (B)",  
                value=safe_float(st.session_state.data.get("tirs_par_match_B", 0.0)),  
                key="tirs_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["possession_moyenne_B"] = st.number_input(  
                "⏳ Possession (%) (B)",  
                value=safe_float(st.session_state.data.get("possession_moyenne_B", 0.0)),  
                key="possession_moyenne_B_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_B"] = st.number_input(  
                "🔄 Corners par Match (B)",  
                value=safe_float(st.session_state.data.get("corners_par_match_B", 0.0)),  
                key="corners_par_match_B_input",  
                step=0.01  
            )  
            st.subheader("📅 Forme Récente (Équipe B)")  
            st.session_state.data["forme_recente_B_victoires"] = st.number_input(  
                "✅ Victoires (B) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data.get("forme_recente_B_victoires", 0)),  
                key="forme_recente_B_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_nuls"] = st.number_input(  
                "➖ Nuls (B) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data.get("forme_recente_B_nuls", 0)),  
                key="forme_recente_B_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_defaites"] = st.number_input(  
                "❌ Défaites (B) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data.get("forme_recente_B_defaites", 0)),  
                key="forme_recente_B_defaites_input",  
                step=1  
            )  
            submitted_B = st.form_submit_button("💾 Enregistrer les données Équipe B")  

            if submitted_B:  
                st.success("Données de l'Équipe B enregistrées avec succès !")
            )  
            submitted_B = st.form_submit_button("💾 Enregistrer les données Équipe B")  

            if submitted_B:  
                st.success("Données de l'Équipe B enregistrées avec succès !")  

# Onglet 2 : Prédictions  
with tab2:  
    st.header("🔮 Prédictions du Match")  

    # Calcul du score de forme récente  
    score_forme_A = (st.session_state.data["forme_recente_A_victoires"] * 3 +  
                     st.session_state.data["forme_recente_A_nuls"] * 1 +  
                     st.session_state.data["forme_recente_A_defaites"] * 0)  
    score_forme_B = (st.session_state.data["forme_recente_B_victoires"] * 3 +  
                     st.session_state.data["forme_recente_B_nuls"] * 1 +  
                     st.session_state.data["forme_recente_B_defaites"] * 0)  

    # Méthode 1 : Régression Logistique  
    try:  
        # Préparation des données pour la régression logistique  
        X_lr = np.array([  
            [  
                safe_float(st.session_state.data["score_rating_A"]),  
                safe_float(st.session_state.data["score_rating_B"]),  
                safe_float(st.session_state.data["possession_moyenne_A"]),  
                safe_float(st.session_state.data["possession_moyenne_B"]),  
                safe_int(st.session_state.data["motivation_A"]),  
                safe_int(st.session_state.data["motivation_B"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
                safe_float(st.session_state.data["interceptions_A"]),  
                safe_float(st.session_state.data["interceptions_B"]),  
                score_forme_A,  
                score_forme_B,  
                safe_float(st.session_state.data["historique_victoires_A"]),  
                safe_float(st.session_state.data["historique_victoires_B"]),  
                safe_float(st.session_state.data["historique_nuls"])  
            ]  
        ])  
        
        # Modèle de Régression Logistique  
        model_lr = LogisticRegression()  
        model_lr.fit(X_lr, [1])  # Dummy fit for structure  
        prediction_lr = model_lr.predict(X_lr)[0]  

    except Exception as e:  
        prediction_lr = "Erreur"  

    # Méthode 2 : Random Forest  
    try:  
        model_rf = RandomForestClassifier()  
        model_rf.fit(X_lr, [1])  # Dummy fit for structure  
        prediction_rf = model_rf.predict(X_lr)[0]  

    except Exception as e:  
        prediction_rf = "Erreur"  

    # Méthode 3 : Modèle de Poisson  
    try:  
        # Calcul des buts attendus  
        lambda_A = (st.session_state.data["buts_marques_A"] + score_forme_A) / 2  
        lambda_B = (st.session_state.data["buts_marques_B"] + score_forme_B) / 2  

        # Prédiction des buts  
        proba_A = poisson.pmf(range(6), lambda_A)  
        proba_B = poisson.pmf(range(6), lambda_B)  

        prediction_poisson = (np.argmax(proba_A), np.argmax(proba_B))  # Nombre de buts prédit  
    except Exception as e:  
        prediction_poisson = "Erreur"  

    # Affichage des résultats  
    st.subheader("📈 Résultats des Prédictions")  
    st.write(f"Régression Logistique : {prediction_lr}")  
    st.write(f"Random Forest : {prediction_rf}")  
    st.write(f"Modèle de Poisson : Équipe A {prediction_poisson[0]} - Équipe B {prediction_poisson[1]}")  

# Onglet 3 : Cotes et Value Bet  
with tab3:  
    st.header("💰 Cotes et Value Bet")  

    # Formulaire pour les cotes  
    with st.form("Cotes et Value Bet"):  
        st.session_state.data["cote_victoire_X"] = st.number_input(  
            "🏆 Cote Victoire Équipe A",  
            value=safe_float(st.session_state.data.get("cote_victoire_X", 0.0)),  
            key="cote_victoire_X_input",  
            step=0.01  
        )  
        st.session_state.data["cote_nul"] = st.number_input(  
            "➖ Cote Match Nul",  
            value=safe_float(st.session_state.data.get("cote_nul", 0.0)),  
            key="cote_nul_input",  
            step=0.01  
        )  
        st.session_state.data["cote_victoire_Z"] = st.number_input(  
            "🏆 Cote Victoire Équipe B",  
            value=safe_float(st.session_state.data.get("cote_victoire_Z", 0.0)),  
            key="cote_victoire_Z_input",  
            step=0.01  
        )  
        st.form_submit_button("💾 Enregistrer les cotes")  

    # Calcul des value bets  
    if prediction_lr != "Erreur" and prediction_poisson != "Erreur":  
        # Probabilités prédites  
        proba_victoire_A = prediction_poisson[0] / (prediction_poisson[0] + prediction_poisson[1])  
        proba_nul = 1 - (proba_victoire_A + (prediction_poisson[1] / (prediction_poisson[0] + prediction_poisson[1])))  
        proba_victoire_B = prediction_poisson[1] / (prediction_poisson[0] + prediction_poisson[1])  

        # Calcul des value bets  
        value_bet_A = (proba_victoire_A * st.session_state.data["cote_victoire_X"]) - 1  
        value_bet_nul = (proba_nul * st.session_state.data["cote_nul"]) - 1  
        value_bet_B = (proba_victoire_B * st.session_state.data["cote_victoire_Z"]) - 1  

        # Affichage des value bets  
        st.subheader("💎 Value Bets")  
        st.write(f"Value Bet Équipe A : {value_bet_A:.2f}")  
        st.write(f"Value Bet Match Nul : {value_bet_nul:.2f}")  
        st.write(f"Value Bet Équipe B : {value_bet_B:.2f}")  

        # Recommandation de value bet  
        if value_bet_A > 0:  
            st.success("✅ Value Bet détectée pour l'Équipe A !")  
        if value_bet_nul > 0:  
            st.success("✅ Value Bet détectée pour le Match Nul !")  
        if value_bet_B > 0:  
            st.success("✅ Value Bet détectée pour l'Équipe B !")  

# Onglet 4 : Système de Mise  
with tab4:  
    st.header("💰 Système de Mise")  
    st.write("Cette section calcule la mise optimale selon la méthode de Kelly.")  

    # Formulaire pour la bankroll  
    with st.form("Système de Mise"):  
        st.session_state.data["bankroll"] = st.number_input(  
            "💵 Bankroll",  
            value=safe_float(st.session_state.data.get("bankroll", 0.0)),  
            key="bankroll_input",  
            step=10.0  
        )  
        st.form_submit_button("💾 Enregistrer la bankroll")  

    # Fonction pour calculer la mise Kelly  
    def mise_kelly(probabilite, cote, bankroll):  
        if cote <= 1 or probabilite <= 0:  
            return 0  # Pas de mise si la cote est <= 1 ou la probabilité est <= 0  
        return (bankroll * (probabilite * cote - 1)) / (cote - 1)  

    # Calcul de la mise Kelly  
    if prediction_lr != "Erreur" and prediction_poisson != "Erreur":  
        # Probabilités  
        proba_victoire_A = prediction_poisson[0] / (prediction_poisson[0] + prediction_poisson[1])  
        proba_nul = 1 - (proba_victoire_A + (prediction_poisson[1] / (prediction_poisson[0] + prediction_poisson[1])))  
        proba_victoire_B = prediction_poisson[1] / (prediction_poisson[0] + prediction_poisson[1])  

        # Calcul des mises  
        mise_A = mise_kelly(proba_victoire_A, st.session_state.data["cote_victoire_X"], st.session_state.data["bankroll"])  
        mise_nul = mise_kelly(proba_nul, st.session_state.data["cote_nul"], st.session_state.data["bankroll"])  
        mise_B = mise_kelly(proba_victoire_B, st.session_state.data["cote_victoire_Z"], st.session_state.data["bankroll"])  

        # Affichage des mises  
        st.subheader("💰 Mises Recommandées")  
        st.write(f"Mise sur l'Équipe A : {mise_A:.2f}")  
        st.write(f"Mise sur le Match Nul : {mise_nul:.2f}")  
        st.write(f"Mise sur l'Équipe B : {mise_B:.2f}")  

        # Conseils de mise  
        st.subheader("📌 Conseils de Mise")  
        if mise_A > 0:  
            st.success(f"💰 Mise conseillée sur l'Équipe A : {mise_A:.2f}")  
        if mise_nul > 0:  
            st.success(f"💰 Mise conseillée sur le Match Nul : {mise_nul:.2f}")  
        if mise_B > 0:  
            st.success(f"💰 Mise conseillée sur l'Équipe B : {mise_B:.2f}")
