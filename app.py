import streamlit as st  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from scipy.stats import poisson  

# Fonctions utilitaires pour convertir les valeurs en entiers ou flottants de manière sécurisée  
def safe_int(value):  
    try:  
        return int(value)  
    except (ValueError, TypeError):  
        return 0  

def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

# Fonction pour convertir les cotes en probabilités implicites  
def cote_to_probabilite_implicite(cote):  
    return (1 / cote) * 100  

# Initialisation des variables de prédiction  
prediction_lr = None  
prediction_rf = None  
prediction_poisson = None  

# Titre de l'application  
st.title("📊 Prédictions de Matchs de Football")  

# Formulaire pour les données des équipes  
with st.form("Données des Équipes"):  
    st.subheader("⚽ Données de l'Équipe A")  
    st.session_state.data = {  
        "score_rating_A": st.number_input("📊 Score Rating (A)", value=0.0, key="score_rating_A_input"),  
        "tirs_cadres_par_match_A": st.number_input("🎯 Tirs Cadrés par Match (A)", value=0.0, key="tirs_cadres_par_match_A_input"),  
        "grandes_chances_A": st.number_input("🔥 Grandes Chances (A)", value=0, key="grandes_chances_A_input"),  
        "passes_reussies_par_match_A": st.number_input("🎯 Passes Réussies par Match (A)", value=0.0, key="passes_reussies_par_match_A_input"),  
        "centres_reussies_par_match_A": st.number_input("🔄 Centres Réussies par Match (A)", value=0.0, key="centres_reussies_par_match_A_input"),  
        "dribbles_reussis_par_match_A": st.number_input("⚡ Dribbles Réussis par Match (A)", value=0.0, key="dribbles_reussis_par_match_A_input"),  
        "buts_attendus_concedes_A": st.number_input("🚫 Buts Attendus Concédés (A)", value=0.0, key="buts_attendus_concedes_A_input"),  
        "interceptions_A": st.number_input("🛑 Interceptions par Match (A)", value=0.0, key="interceptions_A_input"),  
        "tacles_reussis_par_match_A": st.number_input("🛡️ Tacles Réussis par Match (A)", value=0.0, key="tacles_reussis_par_match_A_input"),  
        "penalties_concedees_A": st.number_input("⚠️ Pénalties Concédées (A)", value=0, key="penalties_concedees_A_input"),  
        "fautes_par_match_A": st.number_input("🚩 Fautes par Match (A)", value=0.0, key="fautes_par_match_A_input"),  
        "motivation_A": st.number_input("💪 Motivation (A)", value=0, key="motivation_A_input"),  
        "buts_marques_A": st.number_input("⚽ Buts Marqués par Match (A)", value=0.0, key="buts_marques_A_input"),  
        "tirs_par_match_A": st.number_input("🎯 Tirs par Match (A)", value=0.0, key="tirs_par_match_A_input"),  
        "possession_moyenne_A": st.number_input("⏳ Possession (%) (A)", value=0.0, key="possession_moyenne_A_input"),  
        "corners_par_match_A": st.number_input("🔄 Corners par Match (A)", value=0.0, key="corners_par_match_A_input"),  
        "forme_recente_A_victoires": st.number_input("✅ Victoires (A) sur les 5 derniers matchs", value=0, key="forme_recente_A_victoires_input"),  
        "forme_recente_A_nuls": st.number_input("➖ Nuls (A) sur les 5 derniers matchs", value=0, key="forme_recente_A_nuls_input"),  
        "forme_recente_A_defaites": st.number_input("❌ Défaites (A) sur les 5 derniers matchs", value=0, key="forme_recente_A_defaites_input"),  
        "historique_victoires_A": st.number_input("🏆 Victoires Historiques (A)", value=0, key="historique_victoires_A_input"),  
    }  

    st.subheader("⚽ Données de l'Équipe B")  
    st.session_state.data.update({  
        "score_rating_B": st.number_input("📊 Score Rating (B)", value=0.0, key="score_rating_B_input"),  
        "tirs_cadres_par_match_B": st.number_input("🎯 Tirs Cadrés par Match (B)", value=0.0, key="tirs_cadres_par_match_B_input"),  
        "grandes_chances_B": st.number_input("🔥 Grandes Chances (B)", value=0, key="grandes_chances_B_input"),  
        "passes_reussies_par_match_B": st.number_input("🎯 Passes Réussies par Match (B)", value=0.0, key="passes_reussies_par_match_B_input"),  
        "centres_reussies_par_match_B": st.number_input("🔄 Centres Réussies par Match (B)", value=0.0, key="centres_reussies_par_match_B_input"),  
        "dribbles_reussis_par_match_B": st.number_input("⚡ Dribbles Réussis par Match (B)", value=0.0, key="dribbles_reussis_par_match_B_input"),  
        "buts_attendus_concedes_B": st.number_input("🚫 Buts Attendus Concédés (B)", value=0.0, key="buts_attendus_concedes_B_input"),  
        "interceptions_B": st.number_input("🛑 Interceptions par Match (B)", value=0.0, key="interceptions_B_input"),  
        "tacles_reussis_par_match_B": st.number_input("🛡️ Tacles Réussis par Match (B)", value=0.0, key="tacles_reussis_par_match_B_input"),  
        "penalties_concedees_B": st.number_input("⚠️ Pénalties Concédées (B)", value=0, key="penalties_concedees_B_input"),  
        "fautes_par_match_B": st.number_input("🚩 Fautes par Match (B)", value=0.0, key="fautes_par_match_B_input"),  
        "motivation_B": st.number_input("💪 Motivation (B)", value=0, key="motivation_B_input"),  
        "buts_marques_B": st.number_input("⚽ Buts Marqués par Match (B)", value=0.0, key="buts_marques_B_input"),  
        "tirs_par_match_B": st.number_input("🎯 Tirs par Match (B)", value=0.0, key="tirs_par_match_B_input"),  
        "possession_moyenne_B": st.number_input("⏳ Possession (%) (B)", value=0.0, key="possession_moyenne_B_input"),  
        "corners_par_match_B": st.number_input("🔄 Corners par Match (B)", value=0.0, key="corners_par_match_B_input"),  
        "forme_recente_B_victoires": st.number_input("✅ Victoires (B) sur les 5 derniers matchs", value=0, key="forme_recente_B_victoires_input"),  
        "forme_recente_B_nuls": st.number_input("➖ Nuls (B) sur les 5 derniers matchs", value=0, key="forme_recente_B_nuls_input"),  
        "forme_recente_B_defaites": st.number_input("❌ Défaites (B) sur les 5 derniers matchs", value=0, key="forme_recente_B_defaites_input"),  
        "historique_victoires_B": st.number_input("🏆 Victoires Historiques (B)", value=0, key="historique_victoires_B_input"),  
        "historique_nuls": st.number_input("➖ Nuls Historiques", value=0, key="historique_nuls_input"),  
    })  

    st.subheader("🎰 Cotes du Match")  
    st.session_state.data.update({  
        "cote_victoire_X": st.number_input("💰 Cote Victoire Équipe A", value=0.0, key="cote_victoire_X_input"),  
        "cote_nul": st.number_input("💰 Cote Match Nul", value=0.0, key="cote_nul_input"),  
        "cote_victoire_Z": st.number_input("💰 Cote Victoire Équipe B", value=0.0, key="cote_victoire_Z_input"),  
        "bankroll": st.number_input("💰 Bankroll", value=0.0, key="bankroll_input"),  
    })  

    submitted = st.form_submit_button("💾 Enregistrer les Données")  

# Bouton pour lancer les prédictions  
if st.button("🔮 Lancer les Prédictions"):  
    try:  
        # Calcul du score de forme récente  
        score_forme_A = (st.session_state.data["forme_recente_A_victoires"] * 3 +  
                         st.session_state.data["forme_recente_A_nuls"] * 1)  
        score_forme_B = (st.session_state.data["forme_recente_B_victoires"] * 3 +  
                         st.session_state.data["forme_recente_B_nuls"] * 1)  

        # Préparation des données pour les modèles  
        X_lr = np.array([  
            [  
                safe_float(st.session_state.data["score_rating_A"]),  
                safe_float(st.session_state.data["score_rating_B"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
                safe_int(st.session_state.data["grandes_chances_A"]),  
                safe_int(st.session_state.data["grandes_chances_B"]),  
                safe_float(st.session_state.data["passes_reussies_par_match_A"]),  
                safe_float(st.session_state.data["passes_reussies_par_match_B"]),  
                safe_float(st.session_state.data["centres_reussies_par_match_A"]),  
                safe_float(st.session_state.data["centres_reussies_par_match_B"]),  
                safe_float(st.session_state.data["dribbles_reussis_par_match_A"]),  
                safe_float(st.session_state.data["dribbles_reussis_par_match_B"]),  
                safe_float(st.session_state.data["buts_attendus_concedes_A"]),  
                safe_float(st.session_state.data["buts_attendus_concedes_B"]),  
                safe_float(st.session_state.data["interceptions_A"]),  
                safe_float(st.session_state.data["interceptions_B"]),  
                safe_float(st.session_state.data["tacles_reussis_par_match_A"]),  
                safe_float(st.session_state.data["tacles_reussis_par_match_B"]),  
                safe_int(st.session_state.data["penalties_concedees_A"]),  
                safe_int(st.session_state.data["penalties_concedees_B"]),  
                safe_float(st.session_state.data["fautes_par_match_A"]),  
                safe_float(st.session_state.data["fautes_par_match_B"]),  
                safe_int(st.session_state.data["motivation_A"]),  
                safe_int(st.session_state.data["motivation_B"]),  
                safe_float(st.session_state.data["possession_moyenne_A"]),  
                safe_float(st.session_state.data["possession_moyenne_B"]),  
                safe_float(st.session_state.data["corners_par_match_A"]),  
                safe_float(st.session_state.data["corners_par_match_B"]),  
                score_forme_A,  
                score_forme_B,  
                safe_int(st.session_state.data["historique_victoires_A"]),  
                safe_int(st.session_state.data["historique_victoires_B"]),  
                safe_int(st.session_state.data["historique_nuls"]),  
            ]  
        ])  

        # Modèle de Régression Logistique  
        try:  
            # Exemple de données d'entraînement (à remplacer par vos données réelles)  
            X_train = np.array([  
                [1.0, 2.0, 50.0, 50.0, 5, 5, 5.0, 5.0, 3.0, 3.0, 10.0, 10.0, 3.0, 2.0, 1.0, 10, 10],  
                [2.0, 1.0, 60.0, 40.0, 4, 6, 6.0, 4.0, 4.0, 2.0, 8.0, 8.0, 2.0, 3.0, 1.0, 8, 12],  
            ])  
            y_train = np.array([1, 0])  # Exemple de labels (à remplacer par vos vraies étiquettes)  

            model_lr = LogisticRegression()  
            model_lr.fit(X_train, y_train)  # Entraînement du modèle  
            prediction_lr = model_lr.predict(X_lr.reshape(1, -1))[0]  # Prédiction  
        except Exception as e:  
            prediction_lr = "Erreur"  
            st.error(f"Erreur dans le modèle de régression logistique : {e}")  

        # Modèle Random Forest  
        try:  
            model_rf = RandomForestClassifier()  
            model_rf.fit(X_train, y_train)  # Entraînement du modèle  
            prediction_rf = model_rf.predict(X_lr.reshape(1, -1))[0]  # Prédiction  
        except Exception as e:  
            prediction_rf = "Erreur"  
            st.error(f"Erreur dans le modèle Random Forest : {e}")  

        # Modèle de Poisson  
        try:  
            # Calcul des buts attendus avec la distribution de Poisson  
            lambda_A = safe_float(st.session_state.data["buts_marques_A"])  
            lambda_B = safe_float(st.session_state.data["buts_marques_B"])  

            # Prédiction des buts avec la distribution de Poisson  
            buts_predits_A = poisson.rvs(lambda_A)  # Buts prédits pour l'équipe A  
            buts_predits_B = poisson.rvs(lambda_B)  # Buts prédits pour l'équipe B  

            # Calcul des probabilités en pourcentage  
            proba_buts_A = [poisson.pmf(k, lambda_A) * 100 for k in range(5)]  # Probabilité de 0 à 4 buts pour l'équipe A  
            proba_buts_B = [poisson.pmf(k, lambda_B) * 100 for k in range(5)]  # Probabilité de 0 à 4 buts pour l'équipe B  

            prediction_poisson = {  
                "buts_predits_A": buts_predits_A,  
                "buts_predits_B": buts_predits_B,  
                "proba_buts_A": proba_buts_A,  
                "proba_buts_B": proba_buts_B,  
            }  
        except Exception as e:  
            prediction_poisson = "Erreur"  
            st.error(f"Erreur dans le modèle de Poisson : {e}")  

        # Affichage des prédictions  
        st.subheader("📈 Résultats des Prédictions")  
        if prediction_lr != "Erreur":  
            st.write(f"Régression Logistique : Équipe A {prediction_lr} - Équipe B {1 - prediction_lr}")  
        if prediction_rf != "Erreur":  
            st.write(f"Random Forest : Équipe A {prediction_rf} - Équipe B {1 - prediction_rf}")  
        if prediction_poisson != "Erreur":  
            st.write("Modèle de Poisson :")  
            st.write(f"⚽ Buts Prédits - Équipe A : {prediction_poisson['buts_predits_A']}")  
            st.write(f"⚽ Buts Prédits - Équipe B : {prediction_poisson['buts_predits_B']}")  
            st.write("📊 Probabilité de Buts (en %) :")  

            # Affichage des probabilités dans un tableau  
            st.table({  
                "Nombre de Buts": [0, 1, 2, 3, 4],  
                "Équipe A (%)": [f"{p:.2f}%" for p in prediction_poisson["proba_buts
