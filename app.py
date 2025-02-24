import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, mean_squared_error

# --- CONFIG INTERFACE ---
st.set_page_config(page_title="Analyse Foot & Value Bets", layout="wide")
st.sidebar.image("logo.png", width=100)
st.sidebar.title("🏆 Navigation")
page = st.sidebar.radio("Menu", ["🏟️ Analyse des Équipes", "💰 Comparateur de Cotes", "📜 Historique des Prédictions"])

# --- PERSISTANCE DES DONNÉES ---
if "data" not in st.session_state:
    st.session_state.data = None
if "historique" not in st.session_state:
    st.session_state.historique = []

# --- FONCTIONS UTILES ---
def convertir_cote_en_proba(cote):
    return 1 / cote if cote > 0 else 0

def detecter_value_bet(prob_predite, cote_bookmaker):
    prob_bookmaker = convertir_cote_en_proba(cote_bookmaker)
    value_bet = prob_predite > prob_bookmaker
    return value_bet, prob_bookmaker

def enregistrer_historique(equipe_A, equipe_B, prediction):
    historique = {"Équipe A": equipe_A, "Équipe B": equipe_B, "Prédiction": prediction}
    st.session_state.historique.append(historique)
    pd.DataFrame(st.session_state.historique).to_csv("historique_predictions.csv", index=False)

# --- ANALYSE DES ÉQUIPES ---
if page == "🏟️ Analyse des Équipes":
    st.title("Analyse des Équipes & Prédictions")
    equipe_A = st.text_input("Nom de l'équipe A", "Équipe 1")
    equipe_B = st.text_input("Nom de l'équipe B", "Équipe 2")

    if st.session_state.data is None:
        np.random.seed(42)
        st.session_state.data = pd.DataFrame({
            'Possession (%)': np.random.uniform(40, 70, 22),
            'Tirs': np.random.randint(5, 20, 22),
            'Tirs cadrés': np.random.randint(2, 10, 22),
            'Passes réussies (%)': np.random.uniform(60, 90, 22),
            'xG': np.random.uniform(0.5, 3.0, 22),
            'xGA': np.random.uniform(0.5, 3.0, 22),
            'Corners': np.random.randint(2, 10, 22),
            'Fautes': np.random.randint(5, 15, 22),
            'Cartons jaunes': np.random.randint(0, 5, 22),
            'Cartons rouges': np.random.randint(0, 2, 22),
            'Domicile': np.random.choice([0, 1], 22),
            'Forme (pts)': np.random.randint(3, 15, 22),
            'Classement': np.random.randint(1, 20, 22),
            'Buts marqués': np.random.randint(0, 4, 22),
            'Buts encaissés': np.random.randint(0, 4, 22)
        })
    data = st.session_state.data
    
    # Coloration pour les variables spécifiques au modèle de Poisson
    poisson_cols = ['xG', 'xGA']
    def color_poisson(val):
        return 'background-color: yellow' if val.name in poisson_cols else ''
    
    st.dataframe(data.style.applymap(color_poisson))
    
    # --- MODÉLISATION ---
    X = data.drop(columns=['Buts marqués'])
    y = data['Buts marqués']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    acc_log = np.mean(cross_val_score(log_reg, X_train, y_train, cv=kf, scoring='accuracy'))
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = np.mean(cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy'))
    
    poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    mse_poisson = mean_squared_error(y_test, poisson_model.predict(X_test))
    
    st.subheader("Résultats des Modèles")
    st.write(f"Régression Logistique - Accuracy : {acc_log:.2f}")
    st.write(f"Random Forest - Accuracy : {acc_rf:.2f}")
    st.write(f"Modèle de Poisson - MSE : {mse_poisson:.2f}")
    
    enregistrer_historique(equipe_A, equipe_B, {"Logistique": acc_log, "RandomForest": acc_rf, "Poisson": mse_poisson})

# --- COMPARATEUR DE COTES ---
if page == "💰 Comparateur de Cotes":
    st.title("Comparateur de Cotes & Value Bets")
    cote_bookmaker = st.number_input("Cote du Bookmaker", min_value=1.01, max_value=10.0, value=2.5, step=0.01)
    prob_predite = st.slider("Probabilité Prédite (%)", min_value=1, max_value=100, value=50) / 100
    
    value_bet, prob_bookmaker = detecter_value_bet(prob_predite, cote_bookmaker)
    st.write(f"Probabilité implicite du bookmaker : {prob_bookmaker:.2f}")
    st.write(f"Value Bet détecté : {'✅ OUI' if value_bet else '❌ NON'}")

# --- HISTORIQUE DES PRÉDICTIONS ---
if page == "📜 Historique des Prédictions":
    st.title("Historique des Prédictions")
    historique_df = pd.DataFrame(st.session_state.historique)
    st.dataframe(historique_df)
