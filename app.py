import streamlit as st  
import numpy as np  
import pandas as pd  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
import matplotlib.pyplot as plt  
import io  
from docx import Document  
import openpyxl  # Module nécessaire pour exporter en Excel  

# Configuration de la page  
st.set_page_config(page_title="Prédictions Football", page_icon="⚽")  

# Fonction pour vérifier et convertir en float  
def safe_float(value):  
    try:  
        return float(value)  
    except (ValueError, TypeError):  
        return 0.0  

# Fonction pour quantifier la forme récente  
def quantifier_forme_recente(forme_recente):  
    score = 0  
    for resultat in forme_recente:  
        if resultat == "V":  
            score += 3  # Victoire = 3 points  
        elif resultat == "N":  
            score += 1  # Nul = 1 point  
        elif resultat == "D":  
            score += 0  # Défaite = 0 point  
    return score  

# Initialisation des données dans st.session_state  
if "data" not in st.session_state:  
    st.session_state.data = {  
        # Équipe A  
        "score_rating_A": 65.0,  
        "buts_par_match_A": 1.0,  
        "buts_concedes_par_match_A": 1.5,  
        "possession_moyenne_A": 45.0,  
        "clean_sheets_gardien_A": 1.0,
        "Buts_attendus_A": 1.2,
        "tirs_cadres_A": 100,
        "tirs_cadres_par_match_A": 0.35, 
        "grandes_chances_A": 20,
        "grandes_chances_manquées_A": 50,
        "passes_reussies_A": 350,
        "passes_reussies_par_match_A": 12.0,
        "passes_longues_par_match_A": 40.9,
        "centres_réussies_par_match_A": 4.8,
        "penelaties_obtenues_A": 5, 
        "balle_toucher_dans_la_surface_adverse_A": 48, 
        "corners_A": 50,
        "Buts_attendus_concedes_A": 1.8,  
        "interceptions_A": 40.0,  
        "tacles_reussis_par_match_A": 35.0,
        "dégagements_par_match_A": 12.6,
        "penalities_concedees_A": 5, 
        "tirs_arretes_par_match_A": 0.65,  
        "fautes_par_match_A": 20.0,  
        "cartons_jaunes_A": 6,  
        "cartons_rouges_A": 2,  
        "dribbles_reussis_par_match_A": 13.9, 
        "nombre_de_joueurs_cles_absents_A": 0,
        "victoires_exterieur_A": 4,  
        "jours_repos_A": 3., 
        "matchs-sur_30_jours_A": 9, 
        "motivation_A": 3,
        
        # Équipe B  
        "score_rating_B": 65.0,  
        "buts_par_match_B": 1.0,  
        "buts_concedes_par_match_B": 1.5,  
        "possession_moyenne_B": 45.0,  
        "clean_sheets_gardien_B": 1.0,
        "Buts_attendus_B": 1.2,
        "tirs_cadres_B": 100,
        "tirs_cadres_par_match_B": 0.35, 
        "grandes_chances_B": 20,
        "grandes_chances_manquées_B": 50,
        "passes_reussies_B": 350,
        "passes_reussies_par_match_B": 12.0,
        "passes_longues_par_match_B": 40.9,
        "centres_réussies_par_match_B": 4.8,
        "penelaties_obtenues_B": 5, 
        "balle_toucher_dans_la_surface_adverse_B": 48, 
        "corners_B": 50,
        "Buts_attendus_concedes_B": 1.8,  
        "interceptions_B": 40.0,  
        "tacles_reussis_par_match_B": 35.0,
        "dégagements_par_match_B": 12.6,
        "penalities_concedees_B": 5, 
        "tirs_arretes_par_match_B": 0.65,  
        "fautes_par_match_B": 20.0,  
        "cartons_jaunes_B": 6,  
        "cartons_rouges_B": 2,  
        "dribbles_reussis_par_match_B": 13.9, 
        "nombre_de_joueurs_cles_absents_B": 0,
        "victoires_exterieur_B": 4,  
        "jours_repos_B": 3., 
        "matchs-sur_30_jours_B": 9, 
        "motivation_B": 3,  
       
        # Cotes  
        "cote_victoire_X": 2.0,  
        "cote_nul": 3.0,  
        "cote_victoire_Z": 4.0,
    
        # Bankroll  
        "bankroll": 1000.0,  
        
        # Conditions du Match  
        "conditions_match": "",  
        # Historique Face-à-Face  
        "face_a_face": "",  
        # Forme Récente  
        "forme_recente_A": ["V", "V", "V", "V", "V"],  # Données fictives pour l'Équipe A  
        "forme_recente_B": ["D", "N", "V", "D", "V"],  # Données fictives pour l'Équipe B 
       
        
    }  

# Création des onglets  
tab1, tab2, tab3, tab4, tab5 = st.tabs(  
    ["📊 Statistiques", "🌦️ Conditions et Motivation", "🔮 Prédictions", "🎰 Cotes et Value Bet", "💰 Système de Mise"]  
)  

# Onglet 1 : Statistiques  
with tab1:  
    st.header("📊 Statistiques des Équipes")  
    col_a, col_b = st.columns(2)  

    # Statistiques de l'Équipe A  
    with col_a:  
        st.subheader("Équipe A")  
        for key in st.session_state.data:
            if key.endswith("_A"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    value=safe_float(st.session_state.data[key]),  
                    min_value=-1e6,  
                    max_value=1e6,  
                )  

    # Statistiques de l'Équipe B  
    with col_b:  
        st.subheader("Équipe B")  
        for key in st.session_state.data:  
            if key.endswith("_B"):  
                st.session_state.data[key] = st.number_input(  
                    key.replace("_", " ").title(),  
                    value=safe_float(st.session_state.data[key]),  
                    min_value=-1e6,  
                    max_value=1e6,  
                )  

# Onglet 2 : Conditions et Motivation  
with tab2:  
    st.header("🌦️ Conditions du Match et Motivation")  
    st.session_state.data["conditions_match"] = st.text_input(  
        "🌧️ Conditions du Match (ex : pluie, terrain sec)",  
        value=st.session_state.data["conditions_match"],  
    )  
    st.session_state.data["face_a_face"] = st.text_area(  
        "📅 Historique des Face-à-Face (ex : 3 victoires A, 2 nuls, 1 victoire B)",  
        value=st.session_state.data["face_a_face"],  
    )  

    st.subheader("Forme Récente (5 derniers matchs)")  
    col_a, col_b = st.columns(2)  

    # Vérification et initialisation de la forme récente  
    if "forme_recente_A" not in st.session_state.data or not isinstance(st.session_state.data["forme_recente_A"], list):  
        st.session_state.data["forme_recente_A"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    if "forme_recente_B" not in st.session_state.data or not isinstance(st.session_state.data["forme_recente_B"], list):  
        st.session_state.data["forme_recente_B"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    # Vérification de la longueur des listes  
    if len(st.session_state.data["forme_recente_A"]) != 5:  
        st.session_state.data["forme_recente_A"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    if len(st.session_state.data["forme_recente_B"]) != 5:  
        st.session_state.data["forme_recente_B"] = ["V", "V", "V", "V", "V"]  # Réinitialiser avec des valeurs par défaut  

    with col_a:  
        st.write("Équipe A")  
        for i in range(5):  
            current_value = st.session_state.data["forme_recente_A"][i]  
            if current_value not in ["V", "N", "D"]:  
                current_value = "V"  # Valeur par défaut  
            st.session_state.data["forme_recente_A"][i] = st.select_slider(  
                f"Match {i+1}", options=["V", "N", "D"], value=current_value, key=f"match_A_{i}"  
            )  
    
    with col_b:  
        st.write("Équipe B")  
        for i in range(5):  
            current_value = st.session_state.data["forme_recente_B"][i]  
            if current_value not in ["V", "N", "D"]:  
                current_value = "V"  # Valeur par défaut  
            st.session_state.data["forme_recente_B"][i] = st.select_slider(  
                f"Match {i+1}", options=["V", "N", "D"], value=current_value, key=f"match_B_{i}"  
            )  

    # Quantification de la forme récente  
    score_forme_A = quantifier_forme_recente(st.session_state.data["forme_recente_A"])  
    score_forme_B = quantifier_forme_recente(st.session_state.data["forme_recente_B"])  
    st.write(f"📊 **Score Forme Récente Équipe A** : {score_forme_A}")  
    st.write(f"📊 **Score Forme Récente Équipe B** : {score_forme_B}")  
    
# Onglet 3 : Prédictions  
with tab3:  
    st.header("🔮 Prédictions")  
    if st.button("Prédire le résultat"): 
        try:  
            # Prédiction des Buts avec Poisson  
            avg_goals_A = safe_float(st.session_state.data["buts_par_match_A"])  
            avg_goals_B = safe_float(st.session_state.data["buts_par_match_B"]) 
            avg_goals_A = safe_float(st.session_state.data["buts_concedes_par_match_A"])
            avg_goals_B = safe_float(st.session_state.data["buts_concedes_par_match_B"])
            avg_goals_A = safe_float(st.session_state.data["balle_toucher_dans_la_surface_adverse_A"])
            avg_goals_B = safe_float(st.session_state.data["balle_toucher_dans_la_surface_adverse_B"])
            avg_goals_A = safe_float(st.session_state.data["Buts_attendus_A"])
            avg_goals_B = safe_float(st.session_state.data["Buts_attendus_B"])
                                     
            if avg_goals_A <= 0 or avg_goals_B <= 0:  
                raise ValueError("Les moyennes de buts doivent être positives.")  
            prob_0_0 = poisson.pmf(0, avg_goals_A) * poisson.pmf(0, avg_goals_B)  
            prob_1_1 = poisson.pmf(1, avg_goals_A) * poisson.pmf(1, avg_goals_B)  
            prob_2_2 = poisson.pmf(2, avg_goals_A) * poisson.pmf(2, avg_goals_B)  

            # Explication Poisson  
            st.subheader("📊 Prédiction des Scores avec la Loi de Poisson")  
            st.write(  
                "La loi de Poisson est utilisée pour prédire la probabilité des scores en fonction des moyennes de buts des équipes. "  
                "Elle est particulièrement utile pour estimer les scores les plus probables."  
            )  
            st.write(f"📉 **Probabilité de 0-0** : {prob_0_0:.2%}")  
            st.write(f"📉 **Probabilité de 1-1** : {prob_1_1:.2%}")  
            st.write(f"📉 **Probabilité de 2-2** : {prob_2_2:.2%}")  

            # Régression Logistique avec critères supplémentaires  
            X_lr = np.array([  
                [  
                    safe_float(st.session_state.data["score_rating_A"]),  
                    safe_float(st.session_state.data["score_rating_B"]),  
                    safe_float(st.session_state.data["possession_moyenne_A"]),  
                    safe_float(st.session_state.data["possession_moyenne_B"]),  
                    safe_float(st.session_state.data["motivation_A"]),  
                    safe_float(st.session_state.data["motivation_B"]),  
                    safe_float(st.session_state.data["tirs_cadres_A"]),  
                    safe_float(st.session_state.data["tirs_cadres_B"]),  
                    safe_float(st.session_state.data["interceptions_A"]),  
                    safe_float(st.session_state.data["interceptions_B"]),
                    safe_float(st.session_state.data["Buts_attendus_A"]),
                    safe_float(st.session_state.data["Buts_attendus_B"]),
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
                    safe_float(st.session_state.data["tirs_cadres_B"]),  
                    safe_float(st.session_state.data["tirs_cadres_A"]),  
                    safe_float(st.session_state.data["interceptions_B"]),  
                    safe_float(st.session_state.data["interceptions_A"]),
                    safe_float(st.session_state.data["balle_toucher_dans_la_surface_adverse_A"]),
                    safe_float(st.session_state.data["balle_toucher_dans_la_surface_adverse_B"]),
                    score_forme_B,  # Intégration de la forme récente  
                    score_forme_A,  # Intégration de la forme récente  
                ]  
            ])  
            y_lr = np.array([1, 0])  # Deux classes : 1 pour Équipe A, 0 pour Équipe B  
            model_lr = LogisticRegression()  
            model_lr.fit(X_lr, y_lr)  
            prediction_lr = model_lr.predict(X_lr)  

            # Explication Régression Logistique  
            st.subheader("📈 Prédiction avec Régression Logistique")  
            st.write(  
                "La régression logistique est un modèle de classification qui prédit la probabilité de victoire d'une équipe "  
                "en fonction de plusieurs critères, tels que le score de rating, la possession, la motivation, et la forme récente."  
            )  
            st.write(f"📊 **Résultat** : {'Équipe A' if prediction_lr[0] == 1 else 'Équipe B'}")  
 
       # Préparation des caractéristiques pour les équipes A et B  
    features_A = [  
        st.session_state.data["tirs_cadres_par_match_A"],  
        st.session_state.data["grandes_chances_A"],  
        st.session_state.data["grandes_chances_manquées_A"],  
        st.session_state.data["passes_reussies_A"],  
        st.session_state.data["passes_reussies_par_match_A"],  
        st.session_state.data["passes_longues_par_match_A"],  
        st.session_state.data["centres_réussies_par_match_A"],  
        st.session_state.data["penelaties_obtenues_A"],  
        st.session_state.data["balle_toucher_dans_la_surface_adverse_A"],  
        st.session_state.data["corners_A"],  
        st.session_state.data["Buts_attendus_concedes_A"],  
        st.session_state.data["interceptions_A"],  
        st.session_state.data["tacles_reussis_par_match_A"],  
        st.session_state.data["dégagements_par_match_A"],  
        st.session_state.data["penalities_concedees_A"],  
        st.session_state.data["tirs_arretes_par_match_A"],  
        st.session_state.data["fautes_par_match_A"],  
        st.session_state.data["cartons_jaunes_A"],  
        st.session_state.data["cartons_rouges_A"],  
        st.session_state.data["dribbles_reussis_par_match_A"],  
        st.session_state.data["nombre_de_joueurs_cles_absents_A"],  
        st.session_state.data["victoires_exterieur_A"],  
        st.session_state.data["jours_repos_A"],  
        st.session_state.data["matchs-sur_30_jours_A"],  
        st.session_state.data["motivation_A"],  
    ]  

    features_B = [  
        st.session_state.data["tirs_cadres_par_match_B"],  
        st.session_state.data["grandes_chances_B"],  
        st.session_state.data["grandes_chances_manquées_B"],  
        st.session_state.data["passes_reussies_B"],  
        st.session_state.data["passes_reussies_par_match_B"],  
        st.session_state.data["passes_longues_par_match_B"],  
        st.session_state.data["centres_réussies_par_match_B"],  
        st.session_state.data["penelaties_obtenues_B"],  
        st.session_state.data["balle_toucher_dans_la_surface_adverse_B"],  
        st.session_state.data["corners_B"],  
        st.session_state.data["Buts_attendus_concedes_B"],  
        st.session_state.data["interceptions_B"],  
        st.session_state.data["tacles_reussis_par_match_B"],  
        st.session_state.data["dégagements_par_match_B"],  
        st.session_state.data["penalities_concedees_B"],  
        st.session_state.data["tirs_arretes_par_match_B"],  
        st.session_state.data["fautes_par_match_B"],  
        st.session_state.data["cartons_jaunes_B"],  
        st.session_state.data["cartons_rouges_B"],  
        st.session_state.data["dribbles_reussis_par_match_B"],  
        st.session_state.data["nombre_de_joueurs_cles_absents_B"],  
        st.session_state.data["victoires_exterieur_B"],  
        st.session_state.data["jours_repos_B"],  
        st.session_state.data["matchs-sur_30_jours_B"],  
        st.session_state.data["motivation_B"],  
    ]  

    # Données d'entraînement fictives (à remplacer par vos données réelles)  
    X_train = np.array([features_A, features_B])  # Caractéristiques des équipes A et B  
    y_train = np.array([1, 0])  # 1 pour Équipe A, 0 pour Équipe B  

    # Entraînement du modèle Random Forest  
    model_rf = RandomForestClassifier()  
    model_rf.fit(X_train, y_train)  

    # Prédiction pour les équipes A et B  
    prediction_A = model_rf.predict([features_A])  
    prediction_B = model_rf.predict([features_B])  

    # Affichage du gagnant prédit  
    if prediction_A == 1 and prediction_B == 0:  
        st.write("📊 **Gagnant Prédit** : Équipe A")  
    elif prediction_A == 0 and prediction_B == 1:  
        st.write("📊 **Gagnant Prédit** : Équipe B")  
    else:  
        st.write("📊 **Gagnant Prédit** : Match Nul")  

    # Probabilités de prédiction  
    probabilities_A = model_rf.predict_proba([features_A])  
    probabilities_B = model_rf.predict_proba([features_B])  
    st.write(f"📊 **Probabilité de victoire de l'Équipe A** : {probabilities_A[0][1]:.2%}")  
    st.write(f"📊 **Probabilité de victoire de l'Équipe B** : {probabilities_B[0][0]:.2%}")  



            # Téléchargement des données organisées par équipe  
            st.subheader("📥 Téléchargement des Données des Équipes")  

            # Création d'un document Word  
            doc = Document()  
            doc.add_heading("Données des Équipes", level=1)  

            # Ajout des données de l'Équipe A  
            doc.add_heading("Données Équipe A", level=2)  
            for key, value in st.session_state.data.items():  
                if key.endswith("_A"):  
                    doc.add_paragraph(f"{key.replace('_A', '')}: {value}")  

            # Ajout des données de l'Équipe B  
            doc.add_heading("Données Équipe B", level=2)  
            for key, value in st.session_state.data.items():  
                if key.endswith("_B"):  
                    doc.add_paragraph(f"{key.replace('_B', '')}: {value}")  

            # Sauvegarde du document Word  
            doc_filename = "donnees_equipes.docx"  
            doc.save(doc_filename)  

            # Téléchargement du document Word  
            with open(doc_filename, "rb") as f:  
                st.download_button(  
                    label="📥 Télécharger les données des équipes en DOC",  
                    data=f,  
                    file_name=doc_filename,  
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",  
                )  

        except Exception as e:  
            st.error(f"Une erreur s'est produite lors de la prédiction : {e}")
# Onglet 4 : Cotes et Value Bet  
with tab4:  
    st.header("🎰 Cotes et Value Bet")  
    st.subheader("Convertisseur de Cotes Implicites")  
    cote_victoire_X= st.number_input("Cote Victoire X", value=safe_float(st.session_state.data["cote_victoire_X"]))  
    cote_nul = st.number_input("Cote Nul", value=safe_float(st.session_state.data["cote_nul"]))  
    cote_victoire_Z = st.number_input("Cote Victoire Z", value=safe_float(st.session_state.data["cote_victoire_Z"]))  

    # Calcul de la marge bookmaker  
    marge_bookmaker = (1 / cote_victoire_X) + (1 / cote_nul) + (1 / cote_victoire_Z) - 1  
    st.write(f"📉 **Marge Bookmaker** : {marge_bookmaker:.2%}")  

    # Calcul des cotes réelles  
    cote_reelle_victoire_X = cote_victoire_X / (1 + marge_bookmaker)  
    cote_reelle_nul = cote_nul / (1 + marge_bookmaker)  
    cote_reelle_victoire_Z = cote_victoire_Z / (1 + marge_bookmaker)  
    st.write(f"📊 **Cote Réelle Victoire X** : {cote_reelle_victoire_X:.2f}")  
    st.write(f"📊 **Cote Réelle Nul** : {cote_reelle_nul:.2f}")  
    st.write(f"📊 **Cote Réelle Victoire Z** : {cote_reelle_victoire_Z:.2f}")  

    st.subheader("Calculateur de Paris Combiné")  
    cote_equipe_1 = st.number_input("Cote Équipe 1", value=1.5)  
    cote_equipe_2 = st.number_input("Cote Équipe 2", value=2.0)  
    cote_equipe_3 = st.number_input("Cote Équipe 3", value=2.5)  
    cote_finale = cote_equipe_1 * cote_equipe_2 * cote_equipe_3  
    st.write(f"📈 **Cote Finale** : {cote_finale:.2f}")  

# Onglet 5 : Système de Mise  
with tab5:  
    st.header("💰 Système de Mise")  
    bankroll = st.number_input("Bankroll (€)", value=safe_float(st.session_state.data["bankroll"]))  
    niveau_kelly = st.slider("Niveau de Kelly (1 à 5)", min_value=1, max_value=5, value=3)  
    probabilite_victoire = st.number_input("Probabilité de Victoire (%)", value=50.0) / 100  
    cote = st.number_input("Cote", value=2.0)  

    # Calcul de la mise selon Kelly  
    mise_kelly = (probabilite_victoire * (cote - 1) - (1 - probabilite_victoire)) / (cote - 1)  
    mise_kelly = max(0, mise_kelly)  # Éviter les valeurs négatives  
    mise_kelly *= niveau_kelly / 5  # Ajuster selon le niveau de Kelly  
    mise_finale = bankroll * mise_kelly  

    st.write(f"📊 **Mise Kelly** : {mise_kelly:.2%}")  
    st.write(f"💰 **Mise Finale** : {mise_finale:.2f} €")
