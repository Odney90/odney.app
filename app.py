import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from xgboost import XGBClassifier  

# Exemple de données d'entraînement (remplacez ceci par vos données réelles)  
data = {  
    'xG_home': [1.5, 2.0, 1.2, 1.8, 2.5],  
    'xG_away': [1.0, 1.5, 1.3, 1.2, 2.0],  
    'shots_on_target_home': [5, 6, 4, 7, 8],  
    'shots_on_target_away': [3, 4, 5, 2, 6],  
    'possession_home': [55, 60, 50, 58, 62],  
    'possession_away': [45, 40, 50, 42, 38],  
    'recent_form_home': [10, 12, 8, 9, 15],  
    'recent_form_away': [8, 7, 9, 6, 10],  
    'injuries_home': [1, 0, 2, 1, 0],  
    'injuries_away': [0, 1, 1, 0, 2],  
    'home_goals': [2, 3, 1, 2, 4],  
    'away_goals': [1, 2, 1, 0, 3],  
    'result': [1, 1, 0, 1, 1]  # 1 = victoire à domicile, 0 = défaite  
}  

df = pd.DataFrame(data)  

# Préparation des données  
X = df.drop('result', axis=1)  
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

# Entrée des données de l'utilisateur  
st.header("Données de l'Équipe Domicile")  
xG_home = st.number_input("xG (Domicile)", value=1.5)  
shots_on_target_home = st.number_input("Tirs Cadrés (Domicile)", value=5)  
possession_home = st.number_input("Possession (%) (Domicile)", value=55)  
recent_form_home = st.number_input("Forme Récente (Domicile)", value=10)  
injuries_home = st.number_input("Injuries (Domicile)", value=1)  
odds_home = st.number_input("Cote (Domicile)", value=1.5)  

st.header("Données de l'Équipe Extérieure")  
xG_away = st.number_input("xG (Extérieur)", value=1.0)  
shots_on_target_away = st.number_input("Tirs Cadrés (Extérieur)", value=3)  
possession_away = st.number_input("Possession (%) (Extérieur)", value=45)  
recent_form_away = st.number_input("Forme Récente (Extérieur)", value=8)  
injuries_away = st.number_input("Injuries (Extérieur)", value=0)  
odds_away = st.number_input("Cote (Extérieur)", value=1.5)  

# Bouton de prédiction  
if st.button("Prédire le Résultat"):  
    # Préparation des données pour la prédiction  
    input_data = np.array([[xG_home, xG_away, shots_on_target_home, shots_on_target_away,  
                            possession_home, possession_away, recent_form_home,  
                            recent_form_away, injuries_home, injuries_away]])  

    # Prédiction  
    prediction = voting_clf.predict(input_data)[0]  

    # Méthode de Poisson pour prédire les buts  
    lambda_home = xG_home  
    lambda_away = xG_away  
    prob_home_win = np.exp(-lambda_home) * (lambda_home ** 1) / np.math.factorial(1)  
    prob_away_win = np.exp(-lambda_away) * (lambda_away ** 1) / np.math.factorial(1)  

    # Comparaison des cotes  
    value_bet_home = (1 / odds_home) < prob_home_win  
    value_bet_away = (1 / odds_away) < prob_away_win  

    # Affichage des résultats  
    st.write(f"Prédiction du résultat : {'Victoire Domicile' if prediction == 1 else 'Victoire Extérieure'}")  
    st.write(f"Probabilité de victoire Domicile : {prob_home_win:.2f}")  
    st.write(f"Probabilité de victoire Extérieure : {prob_away_win:.2f}")  
    st.write(f"Value Bet Domicile : {'Oui' if value_bet_home else 'Non'}")  
    st.write(f"Value Bet Extérieur : {'Oui' if value_bet_away else 'Non'}")  

if __name__ == '__main__':  
    st.run()  
