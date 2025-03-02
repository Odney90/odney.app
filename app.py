import streamlit as st
import numpy as np
import pandas as pd
import math
import random
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# -------------------------------
# Initialisation du session state
# -------------------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# -------------------------------
# Fonctions Utilitaires
# -------------------------------
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def calculer_value_bet(prob, cote):
    """Calcule la valeur espérée et donne une recommandation de pari."""
    ev = (prob * cote) - 1
    return ev, "✅ Value Bet" if ev > 0 else "❌ Pas de Value Bet"

# -------------------------------
# Chargement des Données d'Entraînement
# -------------------------------
st.sidebar.header("📊 Chargement des Données d'Entraînement")
fichier_csv = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"])

if fichier_csv:
    df_entrainement = pd.read_csv(fichier_csv)
    st.sidebar.write("Aperçu des données :", df_entrainement.head())

    # Vérification des colonnes obligatoires
    colonnes_obligatoires = [
        "xG_A", "Tirs_cadrés_A", "Taux_conversion_A", "Touches_surface_A", "Passes_clés_A",
        "Interceptions_A", "Duels_defensifs_A", "xGA_A", "Arrêts_gardien_A", "Forme_recente_A",
        "Points_5_matchs_A", "Buts_encaissés_5_A", "Ratio_buts_A", "Possession_A", "Corners_A",
        "Tirs_tentés_A", "Fautes_A", "Cartons_A",
        "xG_B", "Tirs_cadrés_B", "Taux_conversion_B", "Touches_surface_B", "Passes_clés_B",
        "Interceptions_B", "Duels_defensifs_B", "xGA_B", "Arrêts_gardien_B", "Forme_recente_B",
        "Points_5_matchs_B", "Buts_encaissés_5_B", "Ratio_buts_B", "Possession_B", "Corners_B",
        "Tirs_tentés_B", "Fautes_B", "Cartons_B",
        "Historique_confrontation", "resultat"
    ]

    # Vérification de la présence de toutes les colonnes
    if not all(col in df_entrainement.columns for col in colonnes_obligatoires):
        st.sidebar.error("⚠️ Fichier invalide. Vérifiez que toutes les colonnes sont présentes.")
    else:
        # Nettoyage des valeurs extrêmes
        df_entrainement = df_entrainement[(df_entrainement["xG_A"] <= 3) & (df_entrainement["xG_B"] <= 3)]
        df_entrainement = df_entrainement[(df_entrainement["Taux_conversion_A"] <= 50) & (df_entrainement["Taux_conversion_B"] <= 50)]
        
        # Séparation des variables
        X = df_entrainement.drop(columns=["resultat"])
        y = df_entrainement["resultat"]

        # -------------------------------
        # Entraînement des Modèles avec Validation Croisée
        # -------------------------------
        @st.cache_resource(show_spinner=False)
        def entrainer_modeles(X, y):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = {}

            models = {
                "Régression Logistique": LogisticRegression(),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            for name, model in models.items():
                accuracy_list = []
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(X_train, y_train)
                    accuracy_list.append(accuracy_score(y_test, model.predict(X_test)))

                scores[name] = (model, np.mean(accuracy_list))
            
            return scores

        if 'trained_models' not in st.session_state or not st.session_state.trained_models:
            st.session_state.trained_models = entrainer_modeles(X, y)

        st.sidebar.markdown("### 📈 Performances des Modèles :")
        for model_name, (model, accuracy) in st.session_state.trained_models.items():
            st.sidebar.markdown(f"**{model_name} :** {accuracy*100:.2f}%")

# -------------------------------
# Interface Utilisateur
# -------------------------------
st.title("⚽ Prédiction de Match & Analyse Value Bet")

col1, col2 = st.columns(2)

# Variables Équipe A
xG_A = col1.number_input("⚽ xG (Équipe A)", value=1.5, format="%.2f")
tirs_cadrés_A = col1.number_input("🎯 Tirs cadrés (Équipe A)", value=5, format="%.2f")
ratio_buts_A = col1.number_input("⚖️ Ratio Buts Marqués/Encaissés (Équipe A)", value=1.2, format="%.2f")

# Variables Équipe B
xG_B = col2.number_input("⚽ xG (Équipe B)", value=1.2, format="%.2f")
tirs_cadrés_B = col2.number_input("🎯 Tirs cadrés (Équipe B)", value=4, format="%.2f")
ratio_buts_B = col2.number_input("⚖️ Ratio Buts Marqués/Encaissés (Équipe B)", value=1.1, format="%.2f")

if st.button("🔮 Prédire le Match"):
    # Sélection du modèle le plus précis
    best_model = max(st.session_state.trained_models.items(), key=lambda x: x[1][1])[1][0]
    input_features = np.array([[xG_A, tirs_cadrés_A, ratio_buts_A, xG_B, tirs_cadrés_B, ratio_buts_B]])
    prediction = best_model.predict(input_features)[0]

    st.write("**🔎 Résultat Prévu :**", "🏆 Victoire Équipe A" if prediction == 1 else "🏟️ Victoire Équipe B")

