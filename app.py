import streamlit as st  
import numpy as np  
import pandas as pd  
import traceback  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  
from scipy.stats import poisson  
from sklearn.base import clone  

# [Toutes les fonctions précédentes restent identiques]  

# Configuration Streamlit  
st.set_page_config(page_title="Prédictions de Matchs", page_icon="⚽")  
st.title("🏆 Système de Prédiction de Matchs de Football")  

# Initialisation du state  
if 'data' not in st.session_state:  
    st.session_state.data = {}  

# Formulaire de saisie  
with st.form("Données du Match"):  
    st.subheader("📊 Saisie des Données du Match")  
    
    col1, col2 = st.columns(2)  
    
    with col1:  
        st.markdown("**Équipe A**")  
        st.session_state.data.update({  
            "score_rating_A": st.number_input("⭐ Score Rating", value=1500, key="rating_A"),  
            "classement_championnat_A": st.number_input("🏆 Classement", value=10, key="classement_A"),  
            "buts_marques_A": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_A"),  
            "tirs_cadres_par_match_A": st.number_input("🎯 Tirs Cadrés", value=5, key="tirs_A"),  
            "performance_domicile_A": st.number_input("🏠 Performance Domicile", value=70, key="domicile_A"),  
            "joueurs_blesses_A": st.number_input("🩺 Joueurs Blessés", value=2, key="blesses_A")  
        })  

    with col2:  
        st.markdown("**Équipe B**")  
        st.session_state.data.update({  
            "score_rating_B": st.number_input("⭐ Score Rating", value=1500, key="rating_B"),  
            "classement_championnat_B": st.number_input("🏆 Classement", value=10, key="classement_B"),  
            "buts_marques_B": st.number_input("⚽ Buts Marqués", value=1.5, key="buts_B"),  
            "tirs_cadres_par_match_B": st.number_input("🎯 Tirs Cadrés", value=5, key="tirs_B"),  
            "performance_domicile_B": st.number_input("🏠 Performance Domicile", value=70, key="domicile_B"),  
            "joueurs_blesses_B": st.number_input("🩺 Joueurs Blessés", value=2, key="blesses_B")  
        })  

    submitted = st.form_submit_button("🔍 Analyser le Match")  

# Bouton de prédiction  
if st.button("🏆 Prédire le Résultat"):  
    try:  
        # Génération de données historiques  
        donnees_historiques = generer_donnees_historiques_defaut()  
        
        # Préparation des données  
        X_train, y_train = preparer_donnees_entrainement(donnees_historiques)  
        
        # Préparation des données du match actuel  
        X_lr = preparer_donnees_regression_logistique(st.session_state.data)  
        X_rf = preparer_donnees_random_forest(st.session_state.data)  
        
        # Modèles  
        modeles = {  
            "Régression Logistique": LogisticRegression(max_iter=1000),  
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  
        }  
        
        # Calcul des lambda pour Poisson  
        lambda_A = calculer_lambda_attendus(st.session_state.data, 'A')  
        lambda_B = calculer_lambda_attendus(st.session_state.data, 'B')  
        
        # Résultats Poisson  
        resultats_poisson = predire_resultat_match_poisson(lambda_A, lambda_B)  
        
        # Résultats  
        st.subheader("🔮 Résultats de Prédiction")  
        
        col_poisson, col_modeles = st.columns(2)  
        
        with col_poisson:  
            st.markdown("**📊 Probabilités de Poisson**")  
            for resultat, proba in resultats_poisson.items():  
                st.metric(resultat, f"{proba:.2%}")  
        
        with col_modeles:  
            st.markdown("**🤖 Performance des Modèles**")  
            
            resultats_modeles = {}  
            for nom, modele in modeles.items():  
                # Validation croisée stratifiée  
                resultats_cv = validation_croisee_stratifiee(  
                    X_train,   
                    y_train,   
                    modele  
                )  
                
                # Stockage des résultats  
                resultats_modeles[nom] = resultats_cv  
                
                # Affichage des métriques de validation croisée  
                st.markdown(f"**{nom}**")  
                for metrique, valeur in resultats_cv.items():  
                    st.write(f"{metrique.capitalize()} : {valeur:.2%}")  
                
                # Entraînement final et prédiction  
                modele.fit(X_train, y_train)  
                proba = modele.predict_proba(X_lr if nom == "Régression Logistique" else X_rf)[0]  
                
                st.write(f"Probabilité Victoire A: {proba[1]:.2%}")  
                st.write(f"Probabilité Victoire B: {proba[0]:.2%}")  
        
        # Analyse finale  
        probabilite_victoire_A = (  
            resultats_poisson['Victoire Équipe A'] +   
            (modeles["Régression Logistique"].predict_proba(X_lr)[0][1] * 0.5) +   
            (modeles["Random Forest"].predict_proba(X_rf)[0][1] * 0.5)  
        ) / 2  
        
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

- **Probabilités de Poisson** : Basées sur les statistiques historiques et les caractéristiques des équipes  
- **Modèles de Machine Learning** :   
  - Régression Logistique : Modèle linéaire simple  
  - Random Forest : Modèle plus complexe, moins sensible au bruit  
- **Résultat Final** : Moyenne pondérée des différentes méthodes de prédiction  

⚠️ *Ces prédictions sont des estimations statistiques et ne garantissent pas le résultat réel.*  
""")  

# Informations supplémentaires et explication des métriques  
st.sidebar.title("📊 Comprendre les Métriques")  
st.sidebar.markdown("""  
### Métriques de Performance  

- **Accuracy** : Proportion de prédictions correctes  
- **Precision** : Proportion de prédictions positives qui sont réellement positives  
- **Recall** : Proportion de vrais positifs correctement identifiés  
- **F1-Score** : Moyenne harmonique de la précision et du recall  

### Modèles Utilisés  
- **Régression Logistique** : Modèle simple, bon pour les relations linéaires  
- **Random Forest** : Modèle d'ensemble, robuste aux variations des données  
""")
