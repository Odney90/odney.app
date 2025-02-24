import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Fonction pour générer les prédictions
def generate_predictions(team1_data, team2_data):
    # Modèle Poisson
    poisson_prediction = {
        'team1_goals': np.random.poisson(lam=team1_data['attack'] * 2),
        'team2_goals': np.random.poisson(lam=team2_data['attack'] * 2)
    }
    
    # Modèle Random Forest
    features = ['attack', 'defense', 'form', 'injuries']
    X = np.array([[team1_data[feature] for feature in features],
                  [team2_data[feature] for feature in features]])
    y = np.array([1, 0])  # 1 pour la victoire de l'équipe 1, 0 pour la victoire de l'équipe 2

    # Assurer que les données ne sont pas vides avant de lancer la validation croisée
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return "Erreur: Données manquantes détectées. Veuillez remplir tous les champs."

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Validation croisée K=5
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    rf.fit(X, y)
    rf_prediction = rf.predict_proba(X)[0][1]  # Probabilité que l'équipe 1 gagne

    # Modèle Régression Logistique
    logreg = LogisticRegression(random_state=42)
    logreg.fit(X, y)
    logreg_prediction = logreg.predict_proba(X)[0][1]  # Probabilité que l'équipe 1 gagne avec régression logistique
    
    return poisson_prediction, rf_prediction, logreg_prediction, cv_scores_rf.mean()

# Interface Streamlit
st.title('Analyse de Match de Football ⚽')

# Entrée des noms des équipes
team1_name = st.text_input("Nom de l'Équipe 1")
team2_name = st.text_input("Nom de l'Équipe 2")

# Variables des équipes
team1_data = {
    'name': team1_name,
    'attack': st.slider("Force d'attaque de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'defense': st.slider("Force de défense de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'form': st.slider("Forme récente de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'injuries': st.slider("Blessures dans l'équipe 1 (0-1)", 0.0, 1.0, 0.0)
}

team2_data = {
    'name': team2_name,
    'attack': st.slider("Force d'attaque de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'defense': st.slider("Force de défense de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'form': st.slider("Forme récente de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'injuries': st.slider("Blessures dans l'équipe 2 (0-1)", 0.0, 1.0, 0.0)
}

# Affichage des variables utilisées par Poisson
st.write("### Variables utilisées par le modèle Poisson (mis en valeur)")
st.write(f"- Attaque de l'Équipe 1: {team1_data['attack']}")
st.write(f"- Attaque de l'Équipe 2: {team2_data['attack']}")

# Option pour générer les prédictions
if st.button('Générer les Prédictions'):
    result = generate_predictions(team1_data, team2_data)
    
    if isinstance(result, str):  # Gestion des erreurs
        st.error(result)
    else:
        poisson_prediction, rf_prediction, logreg_prediction, cv_score_rf = result
        
        # Affichage des prédictions Poisson
        st.write("### Prédictions Poisson")
        st.write(f"{team1_name} Goals: {poisson_prediction['team1_goals']}")
        st.write(f"{team2_name} Goals: {poisson_prediction['team2_goals']}")
        
        # Affichage des prédictions Random Forest
        st.write("### Prédictions Random Forest")
        st.write(f"Probabilité de victoire de {team1_name}: {rf_prediction * 100:.2f}%")

        # Affichage des prédictions Régression Logistique
        st.write("### Prédictions Régression Logistique")
        st.write(f"Probabilité de victoire de {team1_name} (Logistic Regression): {logreg_prediction * 100:.2f}%")
        
        # Affichage de la validation croisée K=5 pour Random Forest
        st.write("### Validation Croisée K=5 (Random Forest)")
        st.write(f"Validation Croisée K=5 (moyenne des scores): {cv_score_rf * 100:.2f}%")

        # Option Double Chance
        if st.checkbox("Afficher la probabilité Double Chance"):
            double_chance_prob_rf = rf_prediction + (1 - rf_prediction)  # Double Chance pour Random Forest
            double_chance_prob_logreg = logreg_prediction + (1 - logreg_prediction)  # Double Chance pour Régression Logistique
            st.write(f"Double Chance (Random Forest): {double_chance_prob_rf * 100:.2f}%")
            st.write(f"Double Chance (Régression Logistique): {double_chance_prob_logreg * 100:.2f}%")

        # Comparaison des cotes
        st.write("### Comparateur de Cotes")
        team1_odds = st.number_input(f"Entrez la cote pour {team1_name}", min_value=1.0)
        team2_odds = st.number_input(f"Entrez la cote pour {team2_name}", min_value=1.0)

        # Conversion des cotes en probabilité implicite
        team1_prob = 1 / team1_odds
        team2_prob = 1 / team2_odds

        st.write(f"Probabilité implicite pour {team1_name}: {team1_prob * 100:.2f}%")
        st.write(f"Probabilité implicite pour {team2_name}: {team2_prob * 100:.2f}%")

        # Détection des Value Bets
        if rf_prediction > team1_prob:
            st.write(f"Value Bet pour {team1_name}: Parier sur {team1_name} pourrait être rentable.")
        elif rf_prediction < team2_prob:
            st.write(f"Value Bet pour {team2_name}: Parier sur {team2_name} pourrait être rentable.")
