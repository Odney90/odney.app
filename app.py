import streamlit as st  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from scipy.stats import poisson  

# Fonctions utilitaires  
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

def cote_to_probabilite_implicite(cote):  
    try:  
        return (1 / cote) * 100  
    except (ValueError, ZeroDivisionError):  
        return 0.0  

# Initialisation des variables globales  
prediction_lr = "Erreur"  
prediction_rf = "Erreur"  
prediction_poisson = "Erreur"  

# Configuration de la page Streamlit  
st.set_page_config(page_title="Prédictions de Matchs", page_icon="⚽")  
st.title("📊 Prédictions de Matchs de Football")  

# Formulaire pour les données des équipes  
with st.form("Données des Équipes"):  
    # Section pour les données de l'Équipe A  
    st.subheader("⚽ Données de l'Équipe A")  
    
    # Initialisation du dictionnaire de données  
    if 'data' not in st.session_state:  
        st.session_state.data = {}  

    # Collecte des données pour l'Équipe A  
    st.session_state.data.update({  
        "buts_marques_A": st.number_input("⚽ Buts Marqués par Match (A)", value=1.5, key="buts_marques_A_input"),  
        "tirs_par_match_A": st.number_input("🎯 Tirs par Match (A)", value=10.0, key="tirs_par_match_A_input"),  
        "grandes_chances_A": st.number_input("🔥 Grandes Chances (A)", value=5, key="grandes_chances_A_input"),  
        "score_rating_A": st.number_input("📊 Score Rating (A)", value=7.5, key="score_rating_A_input"),  
        "passes_reussies_par_match_A": st.number_input("🎯 Passes Réussies par Match (A)", value=400.0, key="passes_reussies_par_match_A_input"),  
        "dribbles_reussis_par_match_A": st.number_input("⚡ Dribbles Réussis par Match (A)", value=10.0, key="dribbles_reussis_par_match_A_input"),  
        "possession_moyenne_A": st.number_input("⏳ Possession (%) (A)", value=55.0, key="possession_moyenne_A_input"),  
        "motivation_A": st.number_input("💪 Motivation (A)", value=8, key="motivation_A_input"),  
        "tirs_cadres_par_match_A": st.number_input("🎯 Tirs Cadrés par Match (A)", value=5.0, key="tirs_cadres_par_match_A_input"),  
        "centres_reussies_par_match_A": st.number_input("🔄 Centres Réussies par Match (A)", value=20.0, key="centres_reussies_par_match_A_input"),  
        "buts_attendus_concedes_A": st.number_input("🚫 Buts Attendus Concédés (A)", value=1.0, key="buts_attendus_concedes_A_input"),  
        "interceptions_A": st.number_input("🛑 Interceptions par Match (A)", value=10.0, key="interceptions_A_input"),  
        "tacles_reussis_par_match_A": st.number_input("🛡️ Tacles Réussis par Match (A)", value=15.0, key="tacles_reussis_par_match_A_input"),  
        "penalties_concedees_A": st.number_input("⚠️ Pénalties Concédées (A)", value=1, key="penalties_concedees_A_input"),  
        "fautes_par_match_A": st.number_input("🚩 Fautes par Match (A)", value=12.0, key="fautes_par_match_A_input"),  
        "corners_par_match_A": st.number_input("🔄 Corners par Match (A)", value=6.0, key="corners_par_match_A_input"),  
        "forme_recente_A_victoires": st.number_input("✅ Victoires (A) sur les 5 derniers matchs", value=3, key="forme_recente_A_victoires_input"),  
        "forme_recente_A_nuls": st.number_input("➖ Nuls (A) sur les 5 derniers matchs", value=1, key="forme_recente_A_nuls_input"),  
        "forme_recente_A_defaites": st.number_input("❌ Défaites (A) sur les 5 derniers matchs", value=1, key="forme_recente_A_defaites_input"),  
        "historique_victoires_A": st.number_input("🏆 Victoires Historiques (A)", value=5, key="historique_victoires_A_input"),  
    })  

    # Section pour les données de l'Équipe B  
    st.subheader("⚽ Données de l'Équipe B")  
    st.session_state.data.update({  
        "buts_marques_B": st.number_input("⚽ Buts Marqués par Match (B)", value=1.2, key="buts_marques_B_input"),  
        "tirs_par_match_B": st.number_input("🎯 Tirs par Match (B)", value=8.0, key="tirs_par_match_B_input"),  
        "grandes_chances_B": st.number_input("🔥 Grandes Chances (B)", value=4, key="grandes_chances_B_input"),  
        "score_rating_B": st.number_input("📊 Score Rating (B)", value=6.5, key="score_rating_B_input"),  
        "passes_reussies_par_match_B": st.number_input("🎯 Passes Réussies par Match (B)", value=350.0, key="passes_reussies_par_match_B_input"),  
        "dribbles_reussis_par_match_B": st.number_input("⚡ Dribbles Réussis par Match (B)", value=8.0, key="dribbles_reussis_par_match_B_input"),  
        "possession_moyenne_B": st.number_input("⏳ Possession (%) (B)", value=45.0, key="possession_moyenne_B_input"),  
        "motivation_B": st.number_input("💪 Motivation (B)", value=7, key="motivation_B_input"),  
        "tirs_cadres_par_match_B": st.number_input("🎯 Tirs Cadrés par Match (B)", value=4.0, key="tirs_cadres_par_match_B_input"),  
        "centres_reussies_par_match_B": st.number_input("🔄 Centres Réussies par Match (B)", value=15.0, key="centres_reussies_par_match_B_input"),  
        "buts_attendus_concedes_B": st.number_input("🚫 Buts Attendus Concédés (B)", value=1.5, key="buts_attendus_concedes_B_input"),  
        "interceptions_B": st.number_input("🛑 Interceptions par Match (B)", value=8.0, key="interceptions_B_input"),  
        "tacles_reussis_par_match_B": st.number_input("🛡️ Tacles Réussis par Match (B)", value=12.0, key="tacles_reussis_par_match_B_input"),  
        "penalties_concedees_B": st.number_input("⚠️ Pénalties Concédées (B)", value=2, key="penalties_concedees_B_input"),  
        "fautes_par_match_B": st.number_input("🚩 Fautes par Match (B)", value=15.0, key="fautes_par_match_B_input"),  
        "corners_par_match_B": st.number_input("🔄 Corners par Match (B)", value=5.0, key="corners_par_match_B_input"),  
        "forme_recente_B_victoires": st.number_input("✅ Victoires (B) sur les 5 derniers matchs", value=2, key="forme_recente_B_victoires_input"),  
        "forme_recente_B_nuls": st.number_input("➖ Nuls (B) sur les 5 derniers matchs", value=2, key="forme_recente_B_nuls_input"),  
        "forme_recente_B_defaites": st.number_input("❌ Défaites (B) sur les 5 derniers matchs", value=1, key="forme_recente_B_defaites_input"),  
        "historique_victoires_B": st.number_input("🏆 Victoires Historiques (B)", value=3, key="historique_victoires_B_input"),  
        "historique_nuls": st.number_input("➖ Nuls Historiques", value=2, key="historique_nuls_input"),  
    })  

    # Section pour les cotes du match  
    st.subheader("🎰 Cotes du Match")  
    st.session_state.data.update({  
        "cote_victoire_X": st.number_input("💰 Cote Victoire Équipe A", value=2.0, key="cote_victoire_X_input"),  
        "cote_nul": st.number_input("💰 Cote Match Nul", value=3.0, key="cote_nul_input"),  
        "cote_victoire_Z": st.number_input("💰 Cote Victoire Équipe B", value=4.0, key="cote_victoire_Z_input"),  
        "bankroll": st.number_input("💰 Bankroll", value=100.0, key="bankroll_input"),  
    })  

    # Bouton de soumission du formulaire  
    submitted = st.form_submit_button("💾 Enregistrer les Données")  

# Bouton pour lancer les prédictions  
if st.button("🔮 Lancer les Prédictions"):  
    try:  
        # Préparation des données pour les modèles  
        score_forme_A = (st.session_state.data["forme_recente_A_victoires"] * 3 +  
                         st.session_state.data["forme_recente_A_nuls"] * 1)  
        score_forme_B = (st.session_state.data["forme_recente_B_victoires"] * 3 +  
                         st.session_state.data["forme_recente_B_nuls"] * 1)  

        # Données pour la Régression Logistique  
        X_lr = np.array([  
            [  
                safe_float(st.session_state.data["score_rating_A"]),  
                safe_float(st.session_state.data["score_rating_B"]),  
                safe_float(st.session_state.data["passes_reussies_par_match_A"]),  
                safe_float(st.session_state.data["passes_reussies_par_match_B"]),  
                safe_float(st.session_state.data["dribbles_reussis_par_match_A"]),  
                safe_float(st.session_state.data["dribbles_reussis_par_match_B"]),  
                safe_float(st.session_state.data["possession_moyenne_A"]),  
                safe_float(st.session_state.data["possession_moyenne_B"]),  
                safe_int(st.session_state.data["motivation_A"]),  
                safe_int(st.session_state.data["motivation_B"]),  
            ]  
        ])  

        # Données pour Random Forest  
        X_rf = np.array([  
            [  
                safe_float(st.session_state.data["tirs_cadres_par_match_A"]),  
                safe_float(st.session_state.data["tirs_cadres_par_match_B"]),  
                safe_float(st.session_state.data["centres_reussies_par_match_A"]),  
                safe_float(st.session_state.data["centres_reussies_par_match_B"]),  
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
            X_train_lr = np.array([  
                [1.0, 2.0, 50.0, 50.0, 5, 5, 5.0, 5.0, 3.0, 3.0],  
                [2.0, 1.0, 60.0, 40.0, 4, 6, 6.0, 4.0, 4.0, 2.0],  
            ])  
            y_train_lr = np.array([1, 0])  

            model_lr = LogisticRegression()  
            model_lr.fit(X_train_lr, y_train_lr)  
            prediction_lr = model_lr.predict(X_lr.reshape(1, -1))[0]  
        except Exception as e:  
            st.error(f"Erreur dans le modèle de régression logistique : {e}")  
            prediction_lr = "Erreur"  

        # Modèle Random Forest  
        try:  
            X_train_rf = np.array([  
                [1.0, 2.0, 50.0, 50.0, 5, 5, 10, 10, 3.0, 3.0, 1, 1, 5.0, 5.0, 4.0,4.0, 15, 10, 5, 5, 2],  
                [2.0, 1.0, 60.0, 40.0, 4, 6, 8, 12, 4.0, 2.0, 2, 0, 6.0, 4.0, 3.0, 5.0, 12, 15, 4, 6, 3],  
            ])  
            y_train_rf = np.array([1, 0])  

            model_rf = RandomForestClassifier()  
            model_rf.fit(X_train_rf, y_train_rf)  
            prediction_rf = model_rf.predict(X_rf.reshape(1, -1))[0]  
        except Exception as e:  
            st.error(f"Erreur dans le modèle Random Forest : {e}")  
            prediction_rf = "Erreur"  

        # Modèle de Poisson  
        try:  
            lambda_A = safe_float(st.session_state.data["buts_marques_A"])  
            lambda_B = safe_float(st.session_state.data["buts_marques_B"])  

            if lambda_A <= 0 or lambda_B <= 0:  
                st.error("Erreur : Les valeurs de lambda (buts marqués) doivent être positives.")  
                prediction_poisson = "Erreur"  
            else:  
                buts_predits_A = poisson.rvs(lambda_A)  
                buts_predits_B = poisson.rvs(lambda_B)  

                proba_buts_A = [poisson.pmf(k, lambda_A) * 100 for k in range(5)]  
                proba_buts_B = [poisson.pmf(k, lambda_B) * 100 for k in range(5)]  

                prediction_poisson = {  
                    "buts_predits_A": buts_predits_A,  
                    "buts_predits_B": buts_predits_B,  
                    "proba_buts_A": proba_buts_A,  
                    "proba_buts_B": proba_buts_B,  
                }  
        except Exception as e:  
            st.error(f"Erreur dans le modèle de Poisson : {e}")  
            prediction_poisson = "Erreur"  

        # Affichage des résultats  
        st.subheader("📊 Résultats des Prédictions")  

        col1, col2, col3 = st.columns(3)  

        with col1:  
            st.markdown("**Régression Logistique**")  
            if prediction_lr != "Erreur":  
                st.success(f"Victoire de l'Équipe A" if prediction_lr == 1 else "Victoire de l'Équipe B")  
            else:  
                st.error("Erreur")  

        with col2:  
            st.markdown("**Random Forest**")  
            if prediction_rf != "Erreur":  
                st.success(f"Victoire de l'Équipe A" if prediction_rf == 1 else "Victoire de l'Équipe B")  
            else:  
                st.error("Erreur")  

        with col3:  
            st.markdown("**Modèle de Poisson**")  
            if prediction_poisson != "Erreur":  
                st.write(f"**Buts prédits**")  
                st.write(f"Équipe A : {prediction_poisson['buts_predits_A']}")  
                st.write(f"Équipe B : {prediction_poisson['buts_predits_B']}")  
            else:  
                st.error("Erreur")  

        # Affichage des probabilités de buts  
        if prediction_poisson != "Erreur":  
            st.markdown("---")  
            st.subheader("📈 Probabilités de Buts")  

            col4, col5 = st.columns(2)  

            with col4:  
                st.markdown("**Équipe A**")  
                for k, proba in enumerate(prediction_poisson["proba_buts_A"]):  
                    st.write(f"{k} but(s) : {proba:.2f}%")  

            with col5:  
                st.markdown("**Équipe B**")  
                for k, proba in enumerate(prediction_poisson["proba_buts_B"]):  
                    st.write(f"{k} but(s) : {proba:.2f}%")  

        # Comparaison des probabilités prédites vs implicites  
        st.markdown("---")  
        st.subheader("📊 Comparaison des Probabilités")  

        try:  
            # Calcul des probabilités implicites  
            cote_A = safe_float(st.session_state.data["cote_victoire_X"])  
            cote_nul = safe_float(st.session_state.data["cote_nul"])  
            cote_B = safe_float(st.session_state.data["cote_victoire_Z"])  

            proba_implicite_A = cote_to_probabilite_implicite(cote_A)  
            proba_implicite_nul = cote_to_probabilite_implicite(cote_nul)  
            proba_implicite_B = cote_to_probabilite_implicite(cote_B)  

            # Calcul des probabilités prédites  
            proba_predite_A = prediction_poisson["proba_buts_A"][1] if prediction_poisson != "Erreur" else 0  
            proba_predite_B = prediction_poisson["proba_buts_B"][1] if prediction_poisson != "Erreur" else 0  

            # Affichage des probabilités  
            col6, col7, col8 = st.columns(3)  

            with col6:  
                st.markdown("**Équipe A**")  
                st.write(f"Prédite : {proba_predite_A:.2f}%")  
                st.write(f"Implicite : {proba_implicite_A:.2f}%")  

            with col7:  
                st.markdown("**Match Nul**")  
                st.write(f"Prédite : N/A")  
                st.write(f"Implicite : {proba_implicite_nul:.2f}%")  

            with col8:  
                st.markdown("**Équipe B**")  
                st.write(f"Prédite : {proba_predite_B:.2f}%")  
                st.write(f"Implicite : {proba_implicite_B:.2f}%")  

            # Détermination de la Value Bet  
            st.markdown("---")  
            st.subheader("💰 Value Bet")  

            value_bet_A = proba_predite_A - proba_implicite_A  
            value_bet_B = proba_predite_B - proba_implicite_B  

            if value_bet_A > 0:  
                st.success(f"**Value Bet Équipe A** : +{value_bet_A:.2f}%")  
            else:  
                st.warning(f"**Value Bet Équipe A** : {value_bet_A:.2f}%")  

            if value_bet_B > 0:  
                st.success(f"**Value Bet Équipe B** : +{value_bet_B:.2f}%")  
            else:  
                st.warning(f"**Value Bet Équipe B** : {value_bet_B:.2f}%")  

        except Exception as e:  
            st.error(f"Erreur lors de la comparaison des probabilités : {e}")  

    except Exception as e:  
        st.error(f"Erreur lors du traitement des données : {e}")
