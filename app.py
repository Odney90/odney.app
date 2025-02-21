import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def generer_donnees_fictives():
    """Génère des données fictives pour 40 matchs par équipe."""
    np.random.seed(42)
    data = {
        "Score rating": np.random.uniform(5, 10, 40),
        "Buts par match": np.random.uniform(0.5, 3, 40),
        "Buts encaissés par match": np.random.uniform(0.5, 2.5, 40),
        "Possession moyenne": np.random.uniform(40, 70, 40),
        "xG": np.random.uniform(0.5, 2.5, 40),
        "Tirs cadrés par match": np.random.uniform(2, 8, 40),
        "Pourcentage de tirs convertis": np.random.uniform(5, 25, 40),
        "Grosses occasions": np.random.randint(0, 5, 40),
        "Grosses occasions ratées": np.random.randint(0, 5, 40),
        "Passes réussies par match": np.random.uniform(200, 600, 40),
        "Corners par match": np.random.uniform(2, 10, 40),
        "Fautes par match": np.random.uniform(8, 20, 40),
        "Cartons jaunes": np.random.randint(0, 5, 40),
        "Cartons rouges": np.random.randint(0, 2, 40)
    }
    return pd.DataFrame(data)

# Charger les données
stats_equipe_A = generer_donnees_fictives()
stats_equipe_B = generer_donnees_fictives()

# Modèles de prédiction
def predire_resultat(stats_A, stats_B):
    """Prédit le résultat du match en fonction des statistiques des équipes."""
    X_train = [[2.1, 1.9, 1.0, 1.1]]  # Exemples fictifs
    y_train = [1]  # Résultat fictif

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    pred_log = log_reg.predict([[stats_A.mean()["Buts par match"], stats_B.mean()["Buts par match"],
                                 stats_A.mean()["xG"], stats_B.mean()["xG"]]])
    
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    pred_rf = rf.predict([[stats_A.mean()["Buts par match"], stats_B.mean()["Buts par match"],
                           stats_A.mean()["xG"], stats_B.mean()["xG"]]])
    
    return pred_log[0], pred_rf[0]

# Calcul de la mise avec Kelly
def mise_kelly(prob_gagner, cote, bankroll):
    """Calcule la mise optimale selon le critère de Kelly."""
    edge = (prob_gagner * cote - 1) / (cote - 1)
    return max(0, bankroll * edge)

# Interface utilisateur Streamlit
st.title("Analyse des paris sportifs")

st.subheader("Statistiques des équipes")
st.write("### Équipe A")
st.dataframe(stats_equipe_A.describe())

st.write("### Équipe B")
st.dataframe(stats_equipe_B.describe())

st.subheader("Prédictions du match")
pred_log, pred_rf = predire_resultat(stats_equipe_A, stats_equipe_B)
st.write(f"Prédiction régression logistique: {pred_log}")
st.write(f"Prédiction Random Forest: {pred_rf}")

# Calcul des mises
cote_A, cote_B = 2.0, 3.5  # Exemples fictifs
prob_A, prob_B = 1 / cote_A, 1 / cote_B  # Conversion des cotes
bankroll = 1000  # Exemple de bankroll

mise_A = mise_kelly(prob_A, cote_A, bankroll)
mise_B = mise_kelly(prob_B, cote_B, bankroll)

st.subheader("Stratégie de mise avec Kelly")
st.write(f"Mise recommandée sur l'équipe A: {mise_A:.2f} €")
st.write(f"Mise recommandée sur l'équipe B: {mise_B:.2f} €")
