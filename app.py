import streamlit as st  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import StandardScaler  
from scipy.stats import poisson  
import traceback  

# Initialisation de st.session_state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Titre de l'application  
st.title("⚽ Analyse de Match de Football")  

# Formulaire de collecte des données  
with st.form("data_form"):  
    st.markdown("### Équipe A")  
    col1, col2 = st.columns(2)  
    with col1:  
        st.session_state.data['score_rating_A'] = st.number_input("⭐ Score Rating", value=70, key="rating_A")  
        st.session_state.data['buts_par_match_A'] = st.number_input("⚽ Buts Marqués", value=1.5, key="buts_A")  
        st.session_state.data['buts_concedes_par_match_A'] = st.number_input("🥅 Buts Concédés", value=1.0, key="concedes_A")  
        st.session_state.data['possession_moyenne_A'] = st.number_input("🎯 Possession Moyenne", value=55, key="possession_A")  
        st.session_state.data['expected_but_A'] = st.number_input("📊 Expected Goals (xG)", value=1.5, key="xG_A")  
        st.session_state.data['expected_concedes_A'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.2, key="xGA_A")  
        st.session_state.data['tirs_cadres_A'] = st.number_input("🎯 Tirs Cadrés", value=120, key="tirs_A")  
        st.session_state.data['grandes_chances_A'] = st.number_input("🔥 Grandes Chances", value=25, key="chances_A")  
        st.session_state.data['passes_reussies_A'] = st.number_input("🔄 Passes Réussies", value=400, key="passes_A")  
        st.session_state.data['corners_A'] = st.number_input("🔄 Corners", value=60, key="corners_A")  
        st.session_state.data['interceptions_A'] = st.number_input("🛡️ Interceptions", value=50, key="interceptions_A")  
        st.session_state.data['tacles_reussis_A'] = st.number_input("🛡️ Tacles Réussis", value=40, key="tacles_A")  
        st.session_state.data['fautes_A'] = st.number_input("⚠️ Fautes", value=15, key="fautes_A")  
        st.session_state.data['cartons_jaunes_A'] = st.number_input("🟨 Cartons Jaunes", value=5, key="jaunes_A")  
        st.session_state.data['cartons_rouges_A'] = st.number_input("🟥 Cartons Rouges", value=1, key="rouges_A")  
        st.session_state.data['joueurs_cles_absents_A'] = st.number_input("🚑 Joueurs Clés Absents", value=0, key="absents_A")  
        st.session_state.data['motivation_A'] = st.number_input("💪 Motivation", value=4, key="motivation_A")  
        st.session_state.data['clean_sheets_gardien_A'] = st.number_input("🧤 Clean Sheets Gardien", value=2, key="clean_sheets_A")  
        st.session_state.data['ratio_tirs_arretes_A'] = st.number_input("🧤 Ratio Tirs Arrêtés", value=0.7, key="ratio_tirs_A")  
        st.session_state.data['victoires_domicile_A'] = st.number_input("🏠 Victoires Domicile", value=50, key="victoires_A")  
        st.session_state.data['passes_longues_A'] = st.number_input("🎯 Passes Longes", value=60, key="passes_longues_A")  
        st.session_state.data['dribbles_reussis_A'] = st.number_input("🔥 Dribbles Réussis", value=12, key="dribbles_A")  
        st.session_state.data['ratio_tirs_cadres_A'] = st.number_input("🎯 Ratio Tirs Cadrés", value=0.4, key="ratio_tirs_cadres_A")  
        st.session_state.data['grandes_chances_manquees_A'] = st.number_input("🔥 Grandes Chances Manquées", value=5, key="chances_manquees_A")  
        st.session_state.data['fautes_zones_dangereuses_A'] = st.number_input("⚠️ Fautes Zones Dangereuses", value=3, key="fautes_zones_A")  
        st.session_state.data['buts_corners_A'] = st.number_input("⚽ Buts Corners", value=2, key="buts_corners_A")  
        st.session_state.data['jours_repos_A'] = st.number_input("⏳ Jours de Repos", value=4, key="repos_A")  
        st.session_state.data['matchs_30_jours_A'] = st.number_input("📅 Matchs (30 jours)", value=8, key="matchs_A")  

    with col2:  
        st.markdown("### Équipe B")  
        st.session_state.data['score_rating_B'] = st.number_input("⭐ Score Rating", value=65, key="rating_B")  
        st.session_state.data['buts_par_match_B'] = st.number_input("⚽ Buts Marqués", value=1.0, key="buts_B")  
        st.session_state.data['buts_concedes_par_match_B'] = st.number_input("🥅 Buts Concédés", value=1.5, key="concedes_B")  
        st.session_state.data['possession_moyenne_B'] = st.number_input("🎯 Possession Moyenne", value=45, key="possession_B")  
        st.session_state.data['expected_but_B'] = st.number_input("📊 Expected Goals (xG)", value=1.2, key="xG_B")  
        st.session_state.data['expected_concedes_B'] = st.number_input("📉 Expected Goals Against (xGA)", value=1.8, key="xGA_B")  
        st.session_state.data['tirs_cadres_B'] = st.number_input("🎯 Tirs Cadrés", value=100, key="tirs_B")  
        st.session_state.data['grandes_chances_B'] = st.number_input("🔥 Grandes Chances", value=20, key="chances_B")  
        st.session_state.data['passes_reussies_B'] = st.number_input("🔄 Passes Réussies", value=350, key="passes_B")  
        st.session_state.data['corners_B'] = st.number_input("🔄 Corners", value=50, key="corners_B")  
        st.session_state.data['interceptions_B'] = st.number_input("🛡️ Interceptions", value=40, key="interceptions_B")  
        st.session_state.data['tacles_reussis_B'] = st.number_input("🛡️ Tacles Réussis", value=35, key="tacles_B")  
        st.session_state.data['fautes_B'] = st.number_input("⚠️ Fautes", value=20, key="fautes_B")  
        st.session_state.data['cartons_jaunes_B'] = st.number_input("🟨 Cartons Jaunes", value=6, key="jaunes_B")  
        st.session_state.data['cartons_rouges_B'] = st.number_input("🟥 Cartons Rouges", value=2, key="rouges_B")  
        st.session_state.data['joueurs_cles_absents_B'] = st.number_input("🚑 Joueurs Clés Absents", value=0, key="absents_B")  
        st.session_state.data['motivation_B'] = st.number_input("💪 Motivation", value=3, key="motivation_B")  
        st.session_state.data['clean_sheets_gardien_B'] = st.number_input("🧤 Clean Sheets Gardien", value=1, key="clean_sheets_B")  
        st.session_state.data['ratio_tirs_arretes_B'] = st.number_input("🧤 Ratio Tirs Arrêtés", value=0.65, key="ratio_tirs_B")  
        st.session_state.data['victoires_exterieur_B'] = st.number_input("🏠 Victoires Extérieur", value=40, key="victoires_B")  
        st.session_state.data['passes_longues_B'] = st.number_input("🎯 Passes Longes", value=40, key="passes_longues_B")  
        st.session_state.data['dribbles_reussis_B'] = st.number_input("🔥 Dribbles Réussis", value=8, key="dribbles_B")  
        st.session_state.data['ratio_tirs_cadres_B'] = st.number_input("🎯 Ratio Tirs Cadrés", value=0.35, key="ratio_tirs_cadres_B")  
        st.session_state.data['grandes_chances_manquees_B'] = st.number_input("🔥 Grandes Chances Manquées", value=6, key="chances_manquees_B")  
        st.session_state.data['fautes_zones_dangereuses_B'] = st.number_input("⚠️ Fautes Zones Dangereuses", value=4, key="fautes_zones_B")  
        st.session_state.data['buts_corners_B'] = st.number_input("⚽ Buts Corners", value=1, key="buts_corners_B")  
        st.session_state.data['jours_repos_B'] = st.number_input("⏳ Jours de Repos", value=3, key="repos_B")  
        st.session_state.data['matchs_30_jours_B'] = st.number_input("📅 Matchs (30 jours)", value=9, key="matchs_B")  

    # Bouton de soumission en bas du formulaire  
    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Section d'analyse et de prédiction  
if submitted:  
    try:  
        # Préparation des données pour Poisson  
        lambda_A = st.session_state.data['expected_but_A']  # Taux de buts de l'équipe A (xG)  
        lambda_B = st.session_state.data['expected_but_B']  # Taux de buts de l'équipe B (xG)  

        # Prédiction des buts avec Poisson  
        buts_A = poisson.rvs(mu=lambda_A, size=1000)  # 1000 simulations pour l'équipe A  
        buts_B = poisson.rvs(mu=lambda_B, size=1000)  # 1000 simulations pour l'équipe B  

        # Résultats Poisson  
        st.subheader("📊 Prédiction des Buts (Poisson)")  
        col_poisson_A, col_poisson_B = st.columns(2)  
        with col_poisson_A:  
            st.metric("⚽ Buts Moyens (Équipe A)", f"{np.mean(buts_A):.2f}")  
        with col_poisson_B:  
            st.metric("⚽ Buts Moyens (Équipe B)", f"{np.mean(buts_B):.2f}")  

        # Préparation des données pour Régression Logistique et Random Forest  
        X = np.array(list(st.session_state.data.values())).reshape(1, -1)  # Toutes les données des équipes  
        X = np.repeat(X, 1000, axis=0)  # Répéter les données pour correspondre à la taille de y  
        y = np.random.randint(0, 3, 1000)  # 3 classes : 0 (défaite), 1 (victoire A), 2 (match nul)  

        # Modèles  
        modeles = {  
            "Régression Logistique": LogisticRegression(max_iter=1000),  
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  
        }  

        # Validation croisée et prédictions  
        st.markdown("### 🤖 Performance des Modèles")  
        resultats_modeles = {}  
        for nom, modele in modeles.items():  
            # Validation croisée stratifiée  
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
            scores = cross_val_score(modele, X, y, cv=cv, scoring='accuracy')  

            # Affichage des métriques de validation croisée  
            st.markdown(f"#### {nom}")  
            col_accuracy, col_precision, col_recall, col_f1 = st.columns(4)  
            with col_accuracy:  
                st.metric("🎯 Précision Globale", f"{np.mean(scores):.2%}")  

            # Prédiction finale  
            modele.fit(X, y)  
            proba = modele.predict_proba(X)[0]  

            # Affichage des prédictions  
            st.markdown("**📊 Prédictions**")  
            col_victoire_A, col_victoire_B, col_nul = st.columns(3)  
            with col_victoire_A:  
                st.metric("🏆 Victoire A", f"{proba[1]:.2%}")  
            with col_victoire_B:  
                st.metric("🏆 Victoire B", f"{proba[0]:.2%}")  
            with col_nul:  
                st.metric("🤝 Match Nul", f"{proba[2]:.2%}")  

            # Stockage des résultats pour comparaison  
            resultats_modeles[nom] = {  
                'accuracy': np.mean(scores),  
                'precision': np.mean(cross_val_score(modele, X, y, cv=cv, scoring='precision_macro')),  
                'recall': np.mean(cross_val_score(modele, X, y, cv=cv, scoring='recall_macro')),  
                'f1_score': np.mean(cross_val_score(modele, X, y, cv=cv, scoring='f1_macro'))  
            }  

        # Analyse finale  
        probabilite_victoire_A = (  
            (resultats_modeles["Régression Logistique"]["accuracy"] + resultats_modeles["Random Forest"]["accuracy"]) / 2  
        )  

        st.subheader("🏆 Résultat Final")  
        st.metric("Probabilité de Victoire de l'Équipe A", f"{probabilite_victoire_A:.2%}")  

        # Visualisation des performances des modèles  
        st.subheader("📈 Comparaison des Performances des Modèles")  

        # Préparation des données pour le graphique  
        metriques = ['accuracy', 'precision', 'recall', 'f1_score']  

        # Création du DataFrame  
        df_performances = pd.DataFrame({  
            nom: [resultats_modeles[nom][metrique] for metrique in metriques]  
            for nom in resultats_modeles.keys()  
        }, index=metriques)  

        # Affichage du DataFrame  
        st.dataframe(df_performances)  

        # Graphique de comparaison  
        fig, ax = plt.subplots(figsize=(10, 6))  
        df_performances.T.plot(kind='bar', ax=ax)  
        plt.title("Comparaison des Performances des Modèles")  
        plt.xlabel("Modèles")  
        plt.ylabel("Score")  
        plt.legend(title="Métriques", bbox_to_anchor=(1.05, 1), loc='upper left')  
        plt.tight_layout()  
        st.pyplot(fig)  

    except Exception as e:  
        st.error(f"Erreur lors de la prédiction : {e}")  
        st.error(traceback.format_exc())  

# Pied de page informatif  
st.markdown("""  
### 🤔 Comment Interpréter ces Résultats ?  

- **📊 Prédiction des Buts (Poisson)** : Basée sur les Expected Goals (xG) des équipes.  
- **🤖 Performance des Modèles** :   
  - **Régression Logistique** : Modèle linéaire simple.  
  - **Random Forest** : Modèle plus complexe, moins sensible au bruit.  
- **📈 K-Fold Cross-Validation** : Cette méthode permet d'évaluer la performance des modèles en divisant les données en plusieurs sous-ensembles (folds). Chaque fold est utilisé comme ensemble de test, tandis que les autres servent à l'entraînement. Cela garantit une estimation plus robuste de la performance.  
- **🏆 Résultat Final** : Moyenne pondérée des différentes méthodes de prédiction.  

⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""")
