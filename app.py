import streamlit as st
import numpy as np
import pandas as pd
import math
import altair as alt
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# -------------------------------
# 🎯 Initialisation du session state
# -------------------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

# -------------------------------
# 📂 Chargement des Données d'Entraînement
# -------------------------------
st.sidebar.header("📊 Données d'entraînement 📂")
fichier_csv = st.sidebar.file_uploader("📁 **Importer un fichier CSV**", type=["csv"])

# 📌 **Données par défaut améliorées** (statistiques réalistes inspirées des ligues européennes)
donnees_defaut = """
xG_A,Tirs_cadrés_A,Taux_conversion_A,Touches_surface_A,Passes_clés_A,Tirs_tentés_A,Interceptions_A,Duels_defensifs_A,xGA_A,Arrêts_gardien_A,Fautes_A,Possession_A,Corners_A,Forme_recente_A,Points_5_matchs_A,xG_B,Tirs_cadrés_B,Taux_conversion_B,Touches_surface_B,Passes_clés_B,Tirs_tentés_B,Interceptions_B,Duels_defensifs_B,xGA_B,Arrêts_gardien_B,Fautes_B,Possession_B,Corners_B,Forme_recente_B,Points_5_matchs_B,resultat
1.7,5,32,26,7,14,10,18,1.3,4,12,55,6,10,9,1.2,4,27,22,5,10,9,16,1.5,5,14,50,5,9,7,1
1.3,5,29,24,6,11,8,17,1.4,5,13,52,5,10,8,1.1,3,26,19,4,9,7,14,1.3,4,11,48,4,8,6,0
2.0,7,35,30,8,13,11,20,1.1,3,11,58,7,12,10,1.5,5,28,23,6,11,9,18,1.4,5,12,54,5,9,8,1
"""

if fichier_csv:
    df_entrainement = pd.read_csv(fichier_csv)
else:
    st.sidebar.info("📌 **Aucun fichier chargé, utilisation des données par défaut.**")
    df_entrainement = pd.read_csv(StringIO(donnees_defaut))

# Vérification des données
if df_entrainement.empty:
    st.sidebar.error("⚠️ **Le fichier CSV est vide !** Veuillez charger un fichier valide.")
else:
    st.sidebar.write("🔍 **Aperçu des données d'entraînement :**", df_entrainement.head())
    X = df_entrainement.drop(columns=["resultat"])
    y = df_entrainement["resultat"]

    # -------------------------------
    # 🏆 Entraînement des Modèles avec Validation Croisée
    # -------------------------------
    @st.cache_resource(show_spinner=False)
    def entrainer_modele_cv(model, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        return model, np.mean(scores)

    try:
        # Entraînement des modèles avec validation croisée
        modele_logistique, score_logistique = entrainer_modele_cv(LogisticRegression(), X, y)
        modele_xgb, score_xgb = entrainer_modele_cv(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), X, y)
        modele_rf, score_rf = entrainer_modele_cv(RandomForestClassifier(random_state=42), X, y)

        # Vérification des scores (éviter erreurs)
        if any(score is None for score in [score_logistique, score_xgb, score_rf]):
            st.sidebar.error("❌ **Erreur lors de l'entraînement des modèles.** Vérifiez vos données.")
        else:
            # Affichage des performances des modèles avec validation croisée
            st.sidebar.markdown("### 🌟 **Performances des modèles (CV)**")
            st.sidebar.markdown(f"- 🧮 **Régression Logistique :** {score_logistique*100:.2f}%")
            st.sidebar.markdown(f"- ⚡ **XGBoost :** {score_xgb*100:.2f}%")
            st.sidebar.markdown(f"- 🌲 **Random Forest :** {score_rf*100:.2f}%")

            # Affichage des résultats dans un tableau
            df_scores = pd.DataFrame([
                ["🧮 Régression Logistique", f"{score_logistique*100:.2f}%", modele_logistique],
                ["⚡ XGBoost", f"{score_xgb*100:.2f}%", modele_xgb],
                ["🌲 Random Forest", f"{score_rf*100:.2f}%", modele_rf]
            ], columns=["Modèle", "Précision (CV)", "Modèle entraîné"])

            st.markdown("## 📊 **Résultats des Modèles avec Validation Croisée**")
            st.table(df_scores[["Modèle", "Précision (CV)"]])

    except Exception as e:
        st.sidebar.error(f"❌ **Une erreur s'est produite lors de l'entraînement des modèles : {str(e)}**")

# -------------------------------
# 🔮 Prédiction avec Meilleur Modèle
# -------------------------------
if st.button("🔮 Prédire avec le Meilleur Modèle"):
    if df_entrainement.empty:
        st.error("⚠️ **Impossible de prédire : les données sont absentes.**")
    else:
        best_model = max([(modele_logistique, score_logistique), (modele_xgb, score_xgb), (modele_rf, score_rf)], key=lambda x: x[1])[0]
        best_model_name = "🧮 Régression Logistique" if best_model == modele_logistique else "⚡ XGBoost" if best_model == modele_xgb else "🌲 Random Forest"
        
        st.markdown(f"### 🚀 **Meilleur Modèle Sélectionné : {best_model_name}**")

        # Exemple fictif de prédiction
        prob_victoire_A, prob_match_nul, prob_victoire_B = 0.42, 0.30, 0.28  
        st.markdown(f"🎯 **Victoire Équipe A :** {prob_victoire_A*100:.2f}%")
        st.markdown(f"⚖️ **Match Nul :** {prob_match_nul*100:.2f}%")
        st.markdown(f"🎯 **Victoire Équipe B :** {prob_victoire_B*100:.2f}%")
