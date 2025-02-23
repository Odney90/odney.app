import streamlit as st  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from scipy.stats import poisson  

# Initialisation de st.session_state.data  
if "data" not in st.session_state:  
    st.session_state.data = {  
        "score_rating_A": 0.0,  
        "score_rating_B": 0.0,  
        "possession_moyenne_A": 0.0,  
        "possession_moyenne_B": 0.0,  
        "motivation_A": 0,  
        "motivation_B": 0,  
        "tirs_cadres_par_match_A": 0.0,  
        "tirs_cadres_par_match_B": 0.0,  
        "interceptions_A": 0.0,  
        "interceptions_B": 0.0,  
        "forme_recente_A_victoires": 0,  
        "forme_recente_A_nuls": 0,  
        "forme_recente_A_defaites": 0,  
        "forme_recente_B_victoires": 0,  
        "forme_recente_B_nuls": 0,  
        "forme_recente_B_defaites": 0,  
        "cote_victoire_X": 1.0,  
        "cote_nul": 1.0,  
        "cote_victoire_Z": 1.0,  
        "bankroll": 100.0,  
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
        "tacles_reussis_par_match_A": 0.0,  
        "tacles_reussis_par_match_B": 0.0,  
        "penalties_concedees_A": 0,  
        "penalties_concedees_B": 0,  
        "fautes_par_match_A": 0.0,  
        "fautes_par_match_B": 0.0,  
        "buts_marques_A": 0.0,  
        "buts_marques_B": 0.0,  
        "tirs_par_match_A": 0.0,  
        "tirs_par_match_B": 0.0,  
        "corners_par_match_A": 0.0,  
        "corners_par_match_B": 0.0,  
        "face_a_face_A": 0,  
        "face_a_face_B": 0,  
        "tirs_cadres_par_match_A": 0.0,  
        "tirs_cadres_par_match_B": 0.0,  
        "interceptions_A": 0.0,  
        "interceptions_B": 0.0  
    }  

# Titre de l'application  
st.title("⚽ Prédiction de Match de Football")  

# Onglets  
tab1, tab2, tab3, tab4 = st.tabs(["📊 Données", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"])  

# Fonctions de sécurité  
def safe_float(value):  
    try:  
        return float(value)  
    except (TypeError, ValueError):  
        return 0.0  

def safe_int(value):  
    try:  
        return int(value)  
    except (TypeError, ValueError):  
        return 0  

# Onglet 1 : Données  
with tab1:  
    st.header("📝 Entrez les données du match")  

    # Formulaire pour les Équipes A et B  
    with st.form("Statistiques des Équipes"):  
        col1, col2 = st.columns(2)  

        # Équipe A  
        with col1:  
            st.subheader("🏆 Équipe A")  
            st.session_state.data["score_rating_A"] = st.number_input(  
                "⭐ Score Rating (A)",  
                value=safe_float(st.session_state.data["score_rating_A"]),  
                key="score_rating_A_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_cadres_par_match_A"] = st.number_input(  
                "🎯 Tirs Cadres par Match (A)",  
                value=safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
                key="tirs_cadres_A_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_A"] = st.number_input(  
                "🔥 Grandes Chances (A)",  
                value=safe_int(st.session_state.data["grandes_chances_A"]),  
                key="grandes_chances_A_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_A"] = st.number_input(  
                "🔄 Passes Réussies par Match (A)",  
                value=safe_float(st.session_state.data["passes_reussies_par_match_A"]),  
                key="passes_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_A"] = st.number_input(  
                "🎯 Centres Réussies par Match (A)",  
                value=safe_float(st.session_state.data["centres_reussies_par_match_A"]),  
                key="centres_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_A"] = st.number_input(  
                "🏃‍♂️ Dribbles Réussis par Match (A)",  
                value=safe_float(st.session_state.data["dribbles_reussis_par_match_A"]),  
                key="dribbles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_A"] = st.number_input(  
                "🚫 Buts Attendus Concédés (A)",  
                value=safe_float(st.session_state.data["buts_attendus_concedes_A"]),  
                key="buts_attendus_concedes_A_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_A"] = st.number_input(  
                "🛑 Interceptions (A)",  
                value=safe_float(st.session_state.data["interceptions_A"]),  
                key="interceptions_A_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_A"] = st.number_input(  
                "🦶 Tacles Réussis par Match (A)",  
                value=safe_float(st.session_state.data["tacles_reussis_par_match_A"]),  
                key="tacles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_A"] = st.number_input(  
                "⚠️ Penalties Concédées (A)",  
                value=safe_int(st.session_state.data["penalties_concedees_A"]),  
                key="penalties_concedees_A_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_A"] = st.number_input(  
                "🚩 Fautes par Match (A)",  
                value=safe_float(st.session_state.data["fautes_par_match_A"]),  
                key="fautes_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_A"] = st.number_input(  
                "💪 Motivation (A)",  
                value=safe_int(st.session_state.data["motivation_A"]),  
                key="motivation_A_input",  
                step=1  
            )  
            st.session_state.data["buts_marques_A"] = st.number_input(  
                "⚽ Buts Marqués par Match (A)",  
                value=safe_float(st.session_state.data["buts_marques_A"]),  
                key="buts_marques_A_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_A"] = st.number_input(  
                "🎯 Tirs par Match (A)",  
                value=safe_float(st.session_state.data["tirs_par_match_A"]),  
                key="tirs_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["possession_moyenne_A"] = st.number_input(  
                "⏳ Possession (%) (A)",  
                value=safe_float(st.session_state.data["possession_moyenne_A"]),  
                key="possession_moyenne_A_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_A"] = st.number_input(  
                "🔄 Corners par Match (A)",  
                value=safe_float(st.session_state.data["corners_par_match_A"]),  
                key="corners_par_match_A_input",  
                step=0.01  
            )  
            st.subheader("📅 Forme Récente (Équipe A)")  
            st.session_state.data["forme_recente_A_victoires"] = st.number_input(  
                "✅ Victoires (A) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data["forme_recente_A_victoires"]),  
                key="forme_recente_A_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_nuls"] = st.number_input(  
                "➖ Nuls (A) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data["forme_recente_A_nuls"]),  
                key="forme_recente_A_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_defaites"] = st.number_input(  
                "❌ Défaites (A) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data["forme_recente_A_defaites"]),  
                key="forme_recente_A_defaites_input",  
                step=1  
            )  

        # Équipe B  
        with col2:  
            st.subheader("🏆 Équipe B")  
            st.session_state.data["score_rating_B"] = st.number_input(  
                "⭐ Score Rating (B)",  
                value=safe_float(st.session_state.data["score_rating_B"]),  
                key="score_rating_B_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_cadres_par_match_B"] = st.number_input(  
                "🎯 Tirs Cadres par Match (B)",  
                value=safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
                key="tirs_cadres_B_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_B"] = st.number_input(  
                "🔥 Grandes Chances (B)",  
                value=safe_int(st.session_state.data["grandes_chances_B"]),  
                key="grandes_chances_B_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_B"] = st.number_input(  
                "🔄 Passes Réussies par Match (B)",  
                value=safe_float(st.session_state.data["passes_reussies_par_match_B"]),  
                key="passes_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_B"] = st.number_input(  
                "🎯 Centres Réussies par Match (B)",  
                value=safe_float(st.session_state.data["centres_reussies_par_match_B"]),  
                key="centres_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_B"] = st.number_input(  
                "🏃‍♂️ Dribbles Réussis par Match (B)",  
                value=safe_float(st.session_state.data["dribbles_reussis_par_match_B"]),  
                key="dribbles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_B"] = st.number_input(  
                "🚫 Buts Attendus Concédés (B)",  
                value=safe_float(st.session_state.data["buts_attendus_concedes_B"]),  
                key="buts_attendus_concedes_B_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_B"] = st.number_input(  
                "🛑 Interceptions (B)",  
                value=safe_float(st.session_state.data["interceptions_B"]),  
                key="interceptions_B_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_B"] = st.number_input(  
                "🦶 Tacles Réussis par Match (B)",  
                value=safe_float(st.session_state.data["tacles_reussis_par_match_B"]),  
                key="tacles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_B"] = st.number_input(  
                "⚠️ Penalties Concédées (B)",  
                value=safe_int(st.session_state.data["penalties_concedees_B"]),  
                key="penalties_concedees_B_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_B"] = st.number_input(  
                "🚩 Fautes par Match (B)",  
                value=safe_float(st.session_state.data["fautes_par_match_B"]),  
                key="fautes_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_B"] = st.number_input(  
                "💪 Motivation (B)",  
                value=safe_int(st.session_state.data["motivation_B"]),  
                key="motivation_B_input",  
                step=1  
            )  
            st.session_state.data["buts_marques_B"] = st.number_input(  
                "⚽ Buts Marqués par Match (B)",  
                value=safe_float(st.session_state.data["buts_marques_B"]),  
                key="buts_marques_B_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_B"] = st.number_input(  
                "🎯 Tirs par Match (B)",  
                value=safe_float(st.session_state.data["tirs_par_match_B"]),  
                key="tirs_par_match_B_input",  
                step=0.01  
            )  
            
           st.session_state.data["possession_moyenne_B"] = st.number_input(  
               "⏳ Possession (%) (B)",  
               value=safe_float(st.session_state.data["possession_moyenne_B"]),  
               key="possession_moyenne_B_input",  
               step=0.01  
            )  
            st.session_state.data["corners_par_match_B"] = st.number_input(  
                "🔄 Corners par Match (B)",  
                value=safe_float(st.session_state.data["corners_par_match_B"]),  
                key="corners_par_match_B_input",  
                step=0.01  
            )  
            st.subheader("📅 Forme Récente (Équipe B)")  
            st.session_state.data["forme_recente_B_victoires"] = st.number_input(  
                "✅ Victoires (B) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data["forme_recente_B_victoires"]),  
                key="forme_recente_B_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_nuls"] = st.number_input(  
                "➖ Nuls (B) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data["forme_recente_B_nuls"]),  
                key="forme_recente_B_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_defaites"] = st.number_input(  
                "❌ Défaites (B) sur les 5 derniers matchs",  
                value=safe_int(st.session_state.data["forme_recente_B_defaites"]),  
                key="forme_recente_B_defaites_input",  
                step=1  
            )  

        # Bouton de soumission  
        if st.form_submit_button("💾 Enregistrer les données"):  
            st.success("Données enregistrées avec succès !")  

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
                score_forme_A,  # Intégration de la forme récente  
                score_forme_B   # Intégration de la forme récente  
            ],  
            [  
                safe_float(st.session_state.data["score_rating_B"]),  
                safe_float(st.session_state.data["score_rating_A"]),  
                safe_float(st.session_state.data["possession_moyenne_B"]),  
                safe_float(st.session_state.data["possession_moyenne_A"]),  
                safe_int(st.session_state.data["motivation_B"]),  
                safe_int(st.session_state.data["motivation_A"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
                safe_float(st.session_state.data["interceptions_B"]),  
                safe_float(st.session_state.data["interceptions_A"]),  
                score_forme_B,  # Intégration de la forme récente  
                score_forme_A   # Intégration de la forme récente  
            ]  
        ])  

        y_lr = np.array([0, 1])  # 0 pour l'Équipe A, 1 pour l'Équipe B  

        model_lr = LogisticRegression(random_state=42)  
        model_lr.fit(X_lr, y_lr)  

        prediction_lr = model_lr.predict(X_lr)[0]  
        probabilite_lr = model_lr.predict_proba(X_lr)[0][prediction_lr] * 100  
    except ValueError as e:  
        st.error(f"Erreur lors de l'entraînement de la Régression Logistique : {e}")  
        prediction_lr = "Erreur"  
        probabilite_lr = 0  
    except Exception as e:  
        st.error(f"Erreur inattendue lors de la prédiction : {e}")  
        prediction_lr = "Erreur"  
        probabilite_lr = 0  

    st.subheader("📈 Régression Logistique")  
    st.write(f"Prédiction : {'Équipe A' if prediction_lr == 0 else 'Équipe B'}")  
    st.write(f"Probabilité : {probabilite_lr:.2f}%")  

    # Méthode 2 : Random Forest  
    try:  
        # Préparation des données  
        X_rf = np.array([  
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
                score_forme_A,  # Intégration de la forme récente  
                score_forme_B,  # Intégration de la forme récente  
            ],  
            [  
                safe_float(st.session_state.data["score_rating_B"]),  
                safe_float(st.session_state.data["score_rating_A"]),  
                safe_float(st.session_state.data["possession_moyenne_B"]),  
                safe_float(st.session_state.data["possession_moyenne_A"]),  
                safe_int(st.session_state.data["motivation_B"]),  
                safe_int(st.session_state.data["motivation_A"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
                safe_float(st.session_state.data["interceptions_B"]),  
                safe_float(st.session_state.data["interceptions_A"]),  
                score_forme_B,  # Intégration de la forme récente  
                score_forme_A,  # Intégration de la forme récente  
            ]  
        ])  

        # Étiquettes pour Random Forest  
        y_rf = np.array([0, 1])  # 0 pour l'Équipe A, 1 pour l'Équipe B  

        # Entraînement du modèle  
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 arbres  
        model_rf.fit(X_rf, y_rf)  

        # Prédiction  
        prediction_rf = model_rf.predict(X_rf)[0]  
        probabilite_rf = model_rf.predict_proba(X_rf)[0][prediction_rf] * 100  
    except ValueError as e:  
        st.error(f"Erreur lors de l'entraînement de Random Forest : {e}")  
        prediction_rf = "Erreur"  
        probabilite_rf = 0  
    except Exception as e:  
        st.error(f"Erreur inattendue lors de la prédiction : {e}")  
        prediction_rf = "Erreur"  
        probabilite_rf = 0  

    # Affichage des résultats de Random Forest  
    st.subheader("🌳 Random Forest")  
    st.write(f"Prédiction : {'Équipe A' if prediction_rf == 0 else 'Équipe B'}")  
    st.write(f"Probabilité : {probabilite_rf:.2f}%")  

    # Méthode 3 : Modèle de Poisson  
    try:  
        # Paramètres  
        avg_goals_A = safe_float(st.session_state.data["buts_marques_A"])  
        avg_goals_B = safe_float(st.session_state.data["buts_marques_B"])  
        avg_conceded_A = safe_float(st.session_state.data["buts_attendus_concedes_A"])  
        avg_conceded_B = safe_float(st.session_state.data["buts_attendus_concedes_B"])  

        # Calcul des buts attendus  
        lambda_A = avg_goals_A * avg_conceded_B  
        lambda_B = avg_goals_B * avg_conceded_A  

        # Probabilité de chaque résultat  
        proba_A = sum(poisson.pmf(k, lambda_A) for k in range(1, 5))  # Victoire de A (1-4 buts)  
        proba_B = sum(poisson.pmf(k, lambda_B) for k in range(1, 5))  # Victoire de B (1-4 buts)  
        proba_nul = poisson.pmf(0, lambda_A) * poisson.pmf(0, lambda_B)  # Match nul (0-0)  

        # Normalisation des probabilités  
        total_proba = proba_A + proba_B + proba_nul  
        proba_A /= total_proba  
        proba_B /= total_proba  
        proba_nul /= total_proba  

        # Prédiction basée sur la probabilité la plus élevée  
        if proba_A > proba_B and proba_A > proba_nul:  
            prediction_poisson = 0  # Équipe A  
            probabilite_poisson = proba_A * 100  
        elif proba_B > proba_A and proba_B > proba_nul:  
            prediction_poisson = 1  # Équipe B  
            probabilite_poisson = proba_B * 100  
        else:  
            prediction_poisson = 2  # Match Nul  
            probabilite_poisson = proba_nul * 100  
    except ValueError as e:  
        st.error(f"Erreur lors du calcul du Modèle de Poisson : {e}")  
        prediction_poisson = "Erreur"  
        probabilite_poisson = 0  
    except Exception as e:  
        st.error(f"Erreur inattendue lors de la prédiction : {e}")  
        prediction_poisson = "Erreur"  
        probabilite_poisson = 0  

    # Affichage des résultats du Modèle de Poisson  
    st.subheader("🐟 Modèle de Poisson")  
    if prediction_poisson == 0:  
        st.write(f"Prédiction : Victoire de l'Équipe A")  
    elif prediction_poisson == 1:  
        st.write(f"Prédiction : Victoire de l'Équipe B")  
    else:  
        st.write(f"Prédiction : Match Nul")  
    st.write(f"Probabilité : {probabilite_poisson:.2f}%")  

# Onglet 3 : Cotes et Value Bet  
with tab3:  
    st.header("🎰 Cotes et Value Bet")  
    st.write("Cette section calcule les cotes et identifie les value bets.")  

    # Formulaire pour les cotes  
    with st.form("Cotes et Value Bet"):  
        st.session_state.data["cote_victoire_X"] = st.number_input(  
            "🏆 Cote Victoire Équipe A",  
            value=safe_float(st.session_state.data["cote_victoire_X"]),  
            key="cote_victoire_X_input",  
            step=0.01  
        )  
        st.session_state.data["cote_nul"] = st.number_input(  
            "➖ Cote Match Nul",  
            value=safe_float(st.session_state.data["cote_nul"]),  
            key="cote_nul_input",  
            step=0.01  
        )  
        st.session_state.data["cote_victoire_Z"] = st.number_input(  
            "🏆 Cote Victoire Équipe B",  
            value=safe_float(st.session_state.data["cote_victoire_Z"]),  
            key="cote_victoire_Z_input",  
            step=0.01  
        )  
        st.form_submit_button("💾 Enregistrer les cotes")  

    # Calcul des value bets  
    if prediction_lr != "Erreur" and prediction_poisson != "Erreur":  
        # Probabilités prédites  
        proba_victoire_A = probabilite_lr / 100  
        proba_nul = (probabilite_poisson / 100) if prediction_poisson == 2 else 0.33  # Valeur par défaut si non calculée  
        proba_victoire_B = probabilite_poisson / 100 if prediction_poisson == 1 else (1 - proba_victoire_A - proba_nul)  

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
            value=safe_float(st.session_state.data["bankroll"]),  
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
