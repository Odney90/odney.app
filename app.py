import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  

# Fonction pour convertir les valeurs en float de manière sécurisée  
def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

# Fonction pour la Mise Kelly  
def mise_kelly(probabilite, cote, bankroll):  
    if cote <= 1:  
        return 0  
    return (probabilite * (cote - 1) - (1 - probabilite)) / (cote - 1) * bankroll  
# Initialisation de st.session_state.data  
if "data" not in st.session_state:  
    st.session_state.data = {  
        "grandes_chances_A": 0,  # Initialisation de la clé manquante  
        "grandes_chances_B": 0,  
        "score_rating_A": 0.0,  
        "score_rating_B": 0.0,  
        "possession_moyenne_A": 0.0,  
        "possession_moyenne_B": 0.0,  
        "motivation_A": 0.0,  
        "motivation_B": 0.0,  
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
    }  

# Fonction pour vérifier et obtenir une valeur  
def safe_int(value):  
    try:  
        return int(value)  
    except (TypeError, ValueError):  
        return 0  # Retourne 0 si la valeur est invalide  

# Utilisation de safe_int pour éviter les erreurs  
grandes_chances_A = safe_int(st.session_state.data["grandes_chances_A"])

# Création des onglets  
tab1, tab2, tab3, tab4 = st.tabs(  
    ["📊 Statistiques", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"]  
)  

# Onglet 1 : Statistiques  
with tab1:  
    st.header("📊 Statistiques des Équipes")  
    st.write("Veuillez saisir les données pour chaque critère.")  

    # Formulaire pour les Équipes A et B  
    with st.form("Statistiques des Équipes"):  
        col1, col2 = st.columns(2)  

        # Équipe A  
        with col1:  
            st.subheader("🏆 Équipe A")  
            st.session_state.data["tirs_cadres_par_match_A"] = st.number_input(  
                "🎯 Tirs Cadres par Match (A)",  
                value=st.session_state.data["tirs_cadres_par_match_A"],  
                key="tirs_cadres_A_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_A"] = st.number_input(  
                "🔥 Grandes Chances (A)",  
                value=int(st.session_state.data["grandes_chances_A"]),  
                key="grandes_chances_A_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_A"] = st.number_input(  
                "🔄 Passes Réussies par Match (A)",  
                value=st.session_state.data["passes_reussies_par_match_A"],  
                key="passes_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_A"] = st.number_input(  
                "🎯 Centres Réussies par Match (A)",  
                value=st.session_state.data["centres_reussies_par_match_A"],  
                key="centres_reussies_A_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_A"] = st.number_input(  
                "🏃‍♂️ Dribbles Réussis par Match (A)",  
                value=st.session_state.data["dribbles_reussis_par_match_A"],  
                key="dribbles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_A"] = st.number_input(  
                "🚫 Buts Attendus Concédés (A)",  
                value=st.session_state.data["buts_attendus_concedes_A"],  
                key="buts_attendus_concedes_A_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_A"] = st.number_input(  
                "🛑 Interceptions (A)",  
                value=st.session_state.data["interceptions_A"],  
                key="interceptions_A_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_A"] = st.number_input(  
                "🦶 Tacles Réussis par Match (A)",  
                value=st.session_state.data["tacles_reussis_par_match_A"],  
                key="tacles_reussis_A_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_A"] = st.number_input(  
                "⚠️ Penalties Concédées (A)",  
                value=int(st.session_state.data["penalties_concedees_A"]),  
                key="penalties_concedees_A_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_A"] = st.number_input(  
                "🚩 Fautes par Match (A)",  
                value=st.session_state.data["fautes_par_match_A"],  
                key="fautes_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_A"] = st.number_input(  
                "💪 Motivation (A)",  
                value=int(st.session_state.data["motivation_A"]),  
                key="motivation_A_input",  
                step=1  
            )  
            st.session_state.data["buts_marques_A"] = st.number_input(  
                "⚽ Buts Marqués par Match (A)",  
                value=st.session_state.data["buts_marques_A"],  
                key="buts_marques_A_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_A"] = st.number_input(  
                "🎯 Tirs par Match (A)",  
                value=st.session_state.data["tirs_par_match_A"],  
                key="tirs_par_match_A_input",  
                step=0.01  
            )  
            st.session_state.data["possession_moyenne_A"] = st.number_input(  
                "⏳ Possession (%) (A)",  
                value=st.session_state.data["possession_moyenne_A"],  
                key="possession_moyenne_A_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_A"] = st.number_input(  
                "🔄 Corners par Match (A)",  
                value=st.session_state.data["corners_par_match_A"],  
                key="corners_par_match_A_input",  
                step=0.01  
            )  
            st.subheader("📅 Forme Récente (Équipe A)")  
            st.session_state.data["forme_recente_A_victoires"] = st.number_input(  
                "✅ Victoires (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_victoires"],  
                key="forme_recente_A_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_nuls"] = st.number_input(  
                "➖ Nuls (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_nuls"],  
                key="forme_recente_A_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_A_defaites"] = st.number_input(  
                "❌ Défaites (A) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_A_defaites"],  
                key="forme_recente_A_defaites_input",  
                step=1  
            )  

        # Équipe B  
        with col2:  
            st.subheader("🏆 Équipe B")  
            st.session_state.data["tirs_cadres_par_match_B"] = st.number_input(  
                "🎯 Tirs Cadres par Match (B)",  
                value=st.session_state.data["tirs_cadres_par_match_B"],  
                key="tirs_cadres_B_input",  
                step=0.01  
            )  
            st.session_state.data["grandes_chances_B"] = st.number_input(  
                "🔥 Grandes Chances (B)",  
                value=int(st.session_state.data["grandes_chances_B"]),  
                key="grandes_chances_B_input",  
                step=1  
            )  
            st.session_state.data["passes_reussies_par_match_B"] = st.number_input(  
                "🔄 Passes Réussies par Match (B)",  
                value=st.session_state.data["passes_reussies_par_match_B"],  
                key="passes_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["centres_reussies_par_match_B"] = st.number_input(  
                "🎯 Centres Réussies par Match (B)",  
                value=st.session_state.data["centres_reussies_par_match_B"],  
                key="centres_reussies_B_input",  
                step=0.01  
            )  
            st.session_state.data["dribbles_reussis_par_match_B"] = st.number_input(  
                "🏃‍♂️ Dribbles Réussis par Match (B)",  
                value=st.session_state.data["dribbles_reussis_par_match_B"],  
                key="dribbles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["buts_attendus_concedes_B"] = st.number_input(  
                "🚫 Buts Attendus Concédés (B)",  
                value=st.session_state.data["buts_attendus_concedes_B"],  
                key="buts_attendus_concedes_B_input",  
                step=0.01  
            )  
            st.session_state.data["interceptions_B"] = st.number_input(  
                "🛑 Interceptions (B)",  
                value=st.session_state.data["interceptions_B"],  
                key="interceptions_B_input",  
                step=0.01  
            )  
            st.session_state.data["tacles_reussis_par_match_B"] = st.number_input(  
                "🦶 Tacles Réussis par Match (B)",  
                value=st.session_state.data["tacles_reussis_par_match_B"],  
                key="tacles_reussis_B_input",  
                step=0.01  
            )  
            st.session_state.data["penalties_concedees_B"] = st.number_input(  
                "⚠️ Penalties Concédées (B)",  
                value=int(st.session_state.data["penalties_concedees_B"]),  
                key="penalties_concedees_B_input",  
                step=1  
            )  
            st.session_state.data["fautes_par_match_B"] = st.number_input(  
                "🚩 Fautes par Match (B)",  
                value=st.session_state.data["fautes_par_match_B"],  
                key="fautes_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["motivation_B"] = st.number_input(  
                "💪 Motivation (B)",  
                value=int(st.session_state.data["motivation_B"]),  
                key="motivation_B_input",  
                step=1  
            )  
            st.session_state.data["buts_marques_B"] = st.number_input(  
                "⚽ Buts Marqués par Match (B)",  
                value=st.session_state.data["buts_marques_B"],  
                key="buts_marques_B_input",  
                step=0.01  
            )  
            st.session_state.data["tirs_par_match_B"] = st.number_input(  
                "🎯 Tirs par Match (B)",  
                value=st.session_state.data["tirs_par_match_B"],  
                key="tirs_par_match_B_input",  
                step=0.01  
            )  
            st.session_state.data["possession_moyenne_B"] = st.number_input(  
                "⏳ Possession (%) (B)",  
                value=st.session_state.data["possession_moyenne_B"],  
                key="possession_moyenne_B_input",  
                step=0.01  
            )  
            st.session_state.data["corners_par_match_B"] = st.number_input(  
                "🔄 Corners par Match (B)",  
                value=st.session_state.data["corners_par_match_B"],  
                key="corners_par_match_B_input",  
                step=0.01  
            )  
            st.subheader("📅 Forme Récente (Équipe B)")  
            st.session_state.data["forme_recente_B_victoires"] = st.number_input(  
                "✅ Victoires (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_victoires"],  
                key="forme_recente_B_victoires_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_nuls"] = st.number_input(  
                "➖ Nuls (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_nuls"],  
                key="forme_recente_B_nuls_input",  
                step=1  
            )  
            st.session_state.data["forme_recente_B_defaites"] = st.number_input(  
                "❌ Défaites (B) sur les 5 derniers matchs",  
                value=st.session_state.data["forme_recente_B_defaites"],  
                key="forme_recente_B_defaites_input",  
                step=1  
            )  

        # Face-à-face  
        st.subheader("🤝 Face-à-Face")  
        st.session_state.data["face_a_face_A"] = st.number_input(  
            "✅ Victoires de l'Équipe A (Face-à-Face)",  
            value=int(st.session_state.data["face_a_face_A"]),  
            key="face_a_face_A_input",  
            step=1  
        )  
        st.session_state.data["face_a_face_B"] = st.number_input(  
            "✅ Victoires de l'Équipe B (Face-à-Face)",  
            value=int(st.session_state.data["face_a_face_B"]),  
            key="face_a_face_B_input",  
            step=1  
        )  

        # Bouton de soumission  
        if st.form_submit_button("💾 Enregistrer les données"):  
            st.success("Données enregistrées avec succès !")  

# Onglet 2 : Prédictions  
with tab2:  
    st.header("🔮 Prédictions")  
    st.write("Cette section prédit le gagnant entre l'Équipe A et l'Équipe B en utilisant plusieurs méthodes de prédiction.")  

    # Calcul des scores de forme récente  
    score_forme_A = (  
        safe_float(st.session_state.data["forme_recente_A_victoires"]) * 3  
        + safe_float(st.session_state.data["forme_recente_A_nuls"]) * 1  
        - safe_float(st.session_state.data["forme_recente_A_defaites"]) * 2  
    )  
    score_forme_B = (  
        safe_float(st.session_state.data["forme_recente_B_victoires"]) * 3  
        + safe_float(st.session_state.data["forme_recente_B_nuls"]) * 1  
        - safe_float(st.session_state.data["forme_recente_B_defaites"]) * 2  
    )  

    # Préparation des données pour la Régression Logistique  
    X_lr = np.array([  
        [  
            safe_float(st.session_state.data["score_rating_A"]),  
            safe_float(st.session_state.data["score_rating_B"]),  
            safe_float(st.session_state.data["possession_moyenne_A"]),  
            safe_float(st.session_state.data["possession_moyenne_B"]),  
            safe_float(st.session_state.data["motivation_A"]),  
            safe_float(st.session_state.data["motivation_B"]),  
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
            safe_float(st.session_state.data["motivation_B"]),  
            safe_float(st.session_state.data["motivation_A"]),  
            safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
            safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
            safe_float(st.session_state.data["interceptions_B"]),  
            safe_float(st.session_state.data["interceptions_A"]),  
            score_forme_B,  # Intégration de la forme récente  
            score_forme_A,  # Intégration de la forme récente  
        ]  
    ])  

    # Étiquettes pour la Régression Logistique  
    y_lr = np.array([0, 1])  # 0 pour l'Équipe A, 1 pour l'Équipe B  

    # Méthode 1 : Régression Logistique  
    try:  
        model_lr = LogisticRegression()  
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

    # Affichage des résultats de la Régression Logistique  
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
            safe_float(st.session_state.data["motivation_A"]),  
            safe_float(st.session_state.data["motivation_B"]),  
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
            safe_float(st.session_state.data["motivation_B"]),  
            safe_float(st.session_state.data["motivation_A"]),  
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
