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
# Initialisation du session state (optionnel)
# -------------------------------
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

# -------------------------------
# Fonctions de base
# -------------------------------
def poisson_prob(lam, k):
    """Calcule la probabilité d'obtenir k buts selon la loi de Poisson."""
    return (np.exp(-lam) * (lam ** k)) / math.factorial(k)

def predire_resultat_match(...):  # Pas de modification ici, conserver l'existant
    # Utilisation inchangée de la loi de Poisson pour prédire les résultats
    ...

def calculer_value_bet(prob, cote):
    """Calcule la valeur espérée et fournit une recommandation de pari."""
    ev = (prob * cote) - 1
    recommendation = "✅ Value Bet" if ev > 0 else "❌ Pas de Value Bet"
    return ev, recommendation

# -------------------------------
# Validation croisée intégrée
# -------------------------------
@st.cache_resource(show_spinner=False)
def entrainer_modele_logistique_cv(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression()
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return model, np.mean(scores)

@st.cache_resource(show_spinner=False)
def entrainer_modele_xgb_cv(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return model, np.mean(scores)

@st.cache_resource(show_spinner=False)
def entrainer_modele_rf_cv(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return model, np.mean(scores)

# -------------------------------
# Chargement et entraînement des modèles
# -------------------------------
if fichier_entrainement is not None:
    modele_logistique, precision_logistique = entrainer_modele_logistique_cv(X_reel, y_reel)
    modele_xgb, precision_xgb = entrainer_modele_xgb_cv(X_reel, y_reel)
    modele_rf, precision_rf = entrainer_modele_rf_cv(X_reel, y_reel)
    st.sidebar.markdown(f"**Précision Régression Logistique (cross-validation) :** {precision_logistique*100:.2f}%")
    st.sidebar.markdown(f"**Précision XGBoost (cross-validation) :** {precision_xgb*100:.2f}%")
    st.sidebar.markdown(f"**Précision Random Forest (cross-validation) :** {precision_rf*100:.2f}%")

# -------------------------------
# Interface principale et résultats
# -------------------------------
# L’interface utilisateur pour saisir les données et afficher les résultats reste inchangée.
# Les résultats incluent désormais les scores moyens de la validation croisée.
...

