import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from xgboost import XGBClassifier  
import math  # Importation de la bibliothèque math  

# Exemple de données d'entraînement (remplacez ceci par vos données réelles)  
data = {  
    'xG_home': [1.5, 2.0, 1.2, 1.8, 2.5],  
    'shots_on_target_home': [5, 6, 4, 7, 8],  
    'touches_in_box_home': [15, 20, 10, 18, 25],  
    'xGA_home': [1.0, 1.5, 1.3, 1.2, 2.0],  
    'interceptions_home': [10, 12, 8, 9, 15],  
    'defensive_duels_home': [20, 25, 15, 18, 30],  
    'possession_home': [55, 60, 50, 58, 62],  
    'key_passes_home': [3, 4, 2, 5, 6],  
    'recent_form_home': [10, 12, 8, 9, 15],  
    'home_goals': [2, 3, 1, 2, 4],  
    'home_goals_against': [1, 2, 1, 0, 3],  
    'injuries_home': [1, 0, 2, 1, 0],  # Nombre de joueurs blessés  
    'away_goals': [1, 2, 1, 0, 3],  
    
    'xG_away': [1.0, 1.5, 1.3, 1.2, 2.0],  
    'shots_on_target_away': [3, 4, 5, 2, 6],  
    'touches_in_box_away': [10, 15, 12, 8, 20],  
    'xGA_away': [1.2, 1.0, 1.5, 1.3, 2.1],  
    'interceptions_away': [8, 10, 9, 7, 12],  
    'defensive_duels_away': [15, 20, 18, 12, 25],  
    'possession_away': [45, 40, 50, 42, 38],  
    'key_passes_away': [2, 3, 4, 1, 5],  
    'recent_form_away': [8, 7, 9, 6, 10],  
    'away_goals': [1, 2, 1, 0, 3],  
    'away_goals_against': [2, 1, 3, 1, 2],  
    'injuries_away': [0, 1, 1, 0, 2],  # Nombre de joueurs blessés  
    'result': [1, 1, 0, 1, 1]  # 1 = victoire à domicile, 0 = défaite  
}  

df = pd.DataFrame(data)  

# Préparation des données  
X = df.drop('result', axis=1)  # Exclure la colonne 'result' pour les variables  
y = df['result']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Modèles  
model1 = RandomForestClassifier(n_estimators=100, random_state=42)  
model2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  
voting_clf = VotingClassifier(estimators=[('rf', model1), ('xgb', model2)], voting='hard')  

# Entraînement du modèle  
voting_clf.fit(X_train, y_train)  

# Évaluation du modèle  
y_pred = voting_clf.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  

# Interface Streamlit  
st.title("Prédiction de Match de Football")  
st.write(f"Précision du modèle : {accuracy:.2f}")  

# Gestion de l'état de session  
if 'input_data' not in st.session_state:  
    st.session_state.input_data = {}  

# Entrée des données de l'utilisateur pour l'équipe domicile  
st.header("Données de l'Équipe Domicile")  
xG_home = st.number_input("xG (Domicile)", value=st.session_state.input_data.get('xG_home', 1.5))  
shots_on_target_home = st.number_input("Tirs Cadrés (Domicile)", value=st.session_state.input_data.get('shots_on_target_home', 5))  
touches_in_box_home = st.number_input("Balles Touchées dans la Surface Adverse (Domicile)", value=st.session_state.input_data.get('touches_in_box_home', 15))  
xGA_home = st.number_input("xGA (Domicile)", value=st.session_state.input_data.get('xGA_home', 1.0))  
interceptions_home = st.number_input("Interceptions (Domicile)", value=st.session_state.input_data.get('interceptions_home', 10))  
defensive_duels_home = st.number_input("Duels Défensifs Gagnés (Domicile)", value=st.session_state.input_data.get('defensive_duels_home', 20))  
possession_home = st.number_input("Possession (%) (Domicile)", value=st.session_state.input_data.get('possession_home', 55))  
key_passes_home = st.number_input("Passes Clés (Domicile)", value=st.session_state.input_data.get('key_passes_home', 3))  
recent_form_home = st.number_input("Forme Récente (Domicile)", value=st.session_state.input_data.get('recent_form_home', 10))  
home_goals = st.number_input("Buts Marqués à Domicile", value=st.session_state.input_data.get('home_goals', 2))  
home_goals_against = st.number_input("Buts Encaissés à Domicile", value=st.session_state.input_data.get('home_goals_against', 1))  
injuries_home = st.number_input("Blessures (Domicile)", value=st.session_state.input_data.get('injuries_home', 1))  # Nombre de joueurs blessés  

# Entrée des données de l'utilisateur pour l'équipe extérieure  
st.header("Données de l'Équipe Extérieure")  
xG_away = st.number_input("xG (Extérieur)", value=st.session_state.input_data.get('xG_away', 1.0))  
shots_on_target_away = st.number_input("Tirs Cadrés (Extérieur)", value=st.session_state.input_data.get('shots_on_target_away', 3))  
touches_in_box_away = st.number_input("Balles Touchées dans la Surface Adverse (Extérieur)", value=st.session_state.input_data.get('touches_in_box_away', 10))  
xGA_away = st.number_input("xGA (Extérieur)", value=st.session_state.input_data.get('xGA_away', 1.5))  
interceptions_away = st.number_input("Interceptions (Extérieur)", value=st.session_state.input_data.get('interceptions_away', 8))  
defensive_duels_away = st.number_input("Duels Défensifs Gagnés (Extérieur)", value=st.session_state.input_data.get('defensive_duels_away', 15))  
possession_away = st.number_input("Possession (%) (Extérieur)", value=st.session_state.input_data.get('possession_away', 45))  
key_passes_away = st.number_input("Passes Clés (Extérieur)", value=st.session_state.input_data.get('key_passes_away', 2))  
recent_form_away = st.number_input("Forme Récente (Extérieur)", value=st.session_state.input_data.get('recent_form_away', 8))  
away_goals = st.number_input("Buts Marqués à l'Extérieur", value=st.session_state.input_data.get('away_goals', 1))  
away_goals_against = st.number_input("Buts Encaissés à l'Extérieur", value=st.session_state.input_data.get('away_goals_against', 1))  
injuries_away = st.number_input("Blessures (Extérieur)", value=st.session_state.input_data.get('injuries_away', 0))  # Nombre de joueurs blessés  

# Entrée des cotes (ne pas les inclure dans les variables du modèle)  
odds_home = st.number_input("Cote (Domicile)", value=st.session_state.input_data.get('odds_home', 1.5))  
odds_away = st.number_input("Cote (Extérieur)", value=st.session_state.input_data.get('odds_away', 1.5))  

# Bouton de prédiction  
if st.button("Prédire le Résultat"):  
    # Préparation des données pour la prédiction  
    input_data = np.array([[xG_home, shots_on_target_home, touches_in_box_home, xGA_home,  
                            interceptions_home, defensive_duels_home, possession_home,  
                            key_passes_home, recent_form_home, home_goals, home_goals_against,  
                            injuries_home, xG_away, shots_on_target_away, touches_in_box_away,  
                            xGA_away, interceptions_away, defensive_duels_away, possession_away,  
                            key_passes_away, recent_form_away, away_goals, away_goals_against,  
                            injuries_away]])  

    # Prédiction  
    prediction = voting_clf.predict(input_data)[0]  

    # Méthode de Poisson pour prédire les buts  
    lambda_home = xG_home  
    lambda_away = xG_away  
    prob_home_win = np.exp(-lambda_home) * (lambda_home ** 1) / math.factorial(1)  # Correction ici  
    prob_away_win = np.exp(-lambda_away) * (lambda_away ** 1) / math.factorial(1)  # Correction ici  

    # Comparaison des cotes  
    value_bet_home = (1 / odds_home) < prob_home_win  
    value_bet_away = (1 / odds_away) < prob_away_win  

    # Affichage des résultats  
    st.write(f"Prédiction du résultat : {'Victoire Domicile' if prediction == 1 else 'Victoire Extérieure'}")  
    st.write(f"Probabilité de victoire Domicile (prédite) : {prob_home_win:.2f}")  
    st.write(f"Probabilité de victoire Domicile (implicite) : {1 / odds_home:.2f}")  
    st.write(f"Probabilité de victoire Extérieure (prédite) : {prob_away_win:.2f}")  
    st.write(f"Probabilité de victoire Extérieure (implicite) : {1 / odds_away:.2f}")  
    st.write(f"Value Bet Domicile : {'Oui' if value_bet_home else 'Non'}")  
    st.write(f"Value Bet Extérieur : {'Oui' if value_bet_away else 'Non'}")  

    # Sauvegarde des données dans l'état de session  
    st.session_state.input_data = {  
        'xG_home': xG_home,  
        'shots_on_target_home': shots_on_target_home,  
        'touches_in_box_home': touches_in_box_home,  
        'xGA_home': xGA_home,  
        'interceptions_home': interceptions_home,  
        'defensive_duels_home': defensive_duels_home,  
        'possession_home': possession_home,  
        'key_passes_home': key_passes_home,  
        'recent_form_home': recent_form_home,  
        'home_goals': home_goals,  
        'home_goals_against': home_goals_against,  
        'injuries_home': injuries_home,  
        'xG_away': xG_away,  
        'shots_on_target_away': shots_on_target_away,  
        'touches_in_box_away': touches_in_box_away,  
        'xGA_away': xGA_away,  
        'interceptions_away': interceptions_away,  
        'defensive_duels_away': defensive_duels_away,  
        'possession_away': possession_away,  
        'key_passes_away': key_passes_away,  
        'recent_form_away': recent_form_away,  
        'away_goals': away_goals,  
        'away_goals_against': away_goals_against,  
        'injuries_away': injuries_away,  
        'odds_home': odds_home,  
        'odds_away': odds_away  
    }
    
