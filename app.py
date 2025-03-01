import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.metrics import accuracy_score  
import math  
import altair as alt  

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
    'injuries_home': [1, 0, 2, 1, 0],  
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
    'injuries_away': [0, 1, 1, 0, 2],  
    'result': [1, 1, 0, 1, 1]  
}  

df = pd.DataFrame(data)  

# Vérification des valeurs manquantes dans le DataFrame initial  
if df.isnull().values.any():  
    st.error("Les données d'entraînement contiennent des valeurs manquantes.")  

# Fonction pour entraîner le modèle et calculer la précision  
def train_models(X, y):  
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)  
    model2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  
    
    # Entraînement des modèles  
    model1.fit(X, y)  
    model2.fit(X, y)  
    
    # Prédictions  
    y_pred_rf = model1.predict(X)  
    y_pred_xgb = model2.predict(X)  
    
    # Calcul de la précision  
    accuracy_rf = accuracy_score(y, y_pred_rf)  
    accuracy_xgb = accuracy_score(y, y_pred_xgb)  
    
    # Validation croisée  
    cv_scores_rf = cross_val_score(model1, X, y, cv=5)  
    cv_scores_xgb = cross_val_score(model2, X, y, cv=5)  
    
    return model1, model2, accuracy_rf, accuracy_xgb, cv_scores_rf, cv_scores_xgb  

# Vérification des données avant l'entraînement  
def validate_data(X, y):  
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):  
        missing_indices = np.where(np.isnan(X) | np.isnan(y))  
        raise ValueError(f"Les données d'entrée ou les cibles contiennent des valeurs manquantes aux indices : {missing_indices}.")  
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):  
        raise ValueError("Les données d'entrée ou les cibles contiennent des valeurs infinies.")  
    if X.shape[0] != y.shape[0]:  
        raise ValueError("Le nombre de lignes dans X et y ne correspond pas.")  

# Interface Streamlit  
st.title("⚽ Prédiction de Match de Football")  

# Gestion de l'état de session  
if 'input_data' not in st.session_state:  
    st.session_state.input_data = {}  

# Entrée des données de l'utilisateur pour l'équipe domicile  
st.header("🏠 Données de l'Équipe Domicile")  
xG_home = st.number_input("xG (Domicile) 🏆", value=st.session_state.input_data.get('xG_home', 1.5))  
shots_on_target_home = st.number_input("Tirs Cadrés (Domicile) 🎯", value=st.session_state.input_data.get('shots_on_target_home', 5))  
touches_in_box_home = st.number_input("Balles Touchées dans la Surface Adverse (Domicile) ⚽", value=st.session_state.input_data.get('touches_in_box_home', 15))  
xGA_home = st.number_input("xGA (Domicile) 🚫", value=st.session_state.input_data.get('xGA_home', 1.0))  
interceptions_home = st.number_input("Interceptions (Domicile) 🛡️", value=st.session_state.input_data.get('interceptions_home', 10))  
defensive_duels_home = st.number_input("Duels Défensifs Gagnés (Domicile) 💪", value=st.session_state.input_data.get('defensive_duels_home', 20))  
possession_home = st.number_input("Possession (%) (Domicile) 📈", value=st.session_state.input_data.get('possession_home', 55))  
key_passes_home = st.number_input("Passes Clés (Domicile) 🔑", value=st.session_state.input_data.get('key_passes_home', 3))  
recent_form_home = st.number_input("Forme Récente (Domicile) 📅", value=st.session_state.input_data.get('recent_form_home', 10))  
home_goals = st.number_input("Buts Marqués à Domicile ⚽", value=st.session_state.input_data.get('home_goals', 2))  
home_goals_against = st.number_input("Buts Encaissés à Domicile 🚫", value=st.session_state.input_data.get('home_goals_against', 1))  
injuries_home = st.number_input("Blessures (Domicile) 🚑", value=st.session_state.input_data.get('injuries_home', 1))  

# Entrée des données de l'utilisateur pour l'équipe extérieure  
st.header("🏟️ Données de l'Équipe Extérieure")  
xG_away = st.number_input("xG (Extérieur) 🏆", value=st.session_state.input_data.get('xG_away', 1.0))  
shots_on_target_away = st.number_input("Tirs Cadrés (Extérieur) 🎯", value=st.session_state.input_data.get('shots_on_target_away', 3))  
touches_in_box_away = st.number_input("Balles Touchées dans la Surface Adverse (Extérieur) ⚽", value=st.session_state.input_data.get('touches_in_box_away', 10))  
xGA_away = st.number_input("xGA (Extérieur) 🚫", value=st.session_state.input_data.get('xGA_away', 1.5))  
interceptions_away = st.number_input("Interceptions (Extérieur) 🛡️", value=st.session_state.input_data.get('interceptions_away', 8))  
defensive_duels_away = st.number_input("Duels Défensifs Gagnés (Extérieur) 💪", value=st.session_state.input_data.get('defensive_duels_away', 15))  
possession_away = st.number_input("Possession (%) (Extérieur) 📈", value=st.session_state.input_data.get('possession_away', 45))  
key_passes_away = st.number_input("Passes Clés (Extérieur) 🔑", value=st.session_state.input_data.get('key_passes_away', 2))  
recent_form_away = st.number_input("Forme Récente (Extérieur) 📅", value=st.session_state.input_data.get('recent_form_away', 8))  
away_goals = st.number_input("Buts Marqués à l'Extérieur ⚽", value=st.session_state.input_data.get('away_goals', 1))  
away_goals_against = st.number_input("Buts Encaissés à l'Extérieur 🚫", value=st.session_state.input_data.get('away_goals_against', 1))  
injuries_away = st.number_input("Blessures (Extérieur) 🚑", value=st.session_state.input_data.get('injuries_away', 0))  

# Bouton de prédiction  
if st.button("🔮 Prédire le Résultat"):  
    # Préparation des données pour la prédiction  
    input_data = np.array([[xG_home, shots_on_target_home, touches_in_box_home, xGA_home,  
                            interceptions_home, defensive_duels_home, possession_home,  
                            key_passes_home, recent_form_home, home_goals, home_goals_against,  
                            injuries_home, xG_away, shots_on_target_away, touches_in_box_away,  
                            xGA_away, interceptions_away, defensive_duels_away, possession_away,  
                            key_passes_away, recent_form_away, away_goals, away_goals_against,  
                            injuries_away]])  

    # Vérification des valeurs manquantes dans les données saisies par l'utilisateur  
    if np.any(np.isnan(input_data)):  
        st.error("Les données saisies contiennent des valeurs manquantes.")  
    else:  
        # Ajout des nouvelles données à l'ensemble de données d'entraînement  
        new_data = pd.DataFrame(input_data, columns=df.columns[:-1])  # Exclure la colonne 'result'  
        
        # Ajouter une valeur par défaut pour 'result' (par exemple, 0)  
        new_data['result'] = 0  # Vous pouvez ajuster cette valeur par défaut  
        
        df = pd.concat([df, new_data], ignore_index=True)  

        # Préparation des nouvelles cibles  
        y = df['result']  # Assurez-vous que cela correspond à vos nouvelles données  

        # Validation des données  
        X = df.drop('result', axis=1)  
        
        # Remplacer les valeurs manquantes par la moyenne  
        X = X.fillna(X.mean())  
        y = pd.Series(y).fillna(pd.Series(y).mean())  # Convertir y en Series pour utiliser fillna  
        
        st.write("X (Données d'entrée) :", X)  
        st.write("y (Cibles) :", y)  
        
        validate_data(X, y)  

        # Entraînement des modèles avec les nouvelles données  
        model1, model2, accuracy_rf, accuracy_xgb, cv_scores_rf, cv_scores_xgb = train_models(X, y)  

        # Prédiction  
        input_data_filled = np.nan_to_num(input_data)  # Remplacer NaN par 0 pour la prédiction  
        prediction_rf = model1.predict(input_data_filled)[0]  
        prediction_xgb = model2.predict(input_data_filled)[0]  

        # Méthode de Poisson pour prédire les buts  
        lambda_home = xG_home  
        lambda_away = xG_away  

        # Calcul des probabilités de buts pour chaque équipe  
        def poisson_prob(lam, k):  
            return (np.exp(-lam) * (lam ** k)) / math.factorial(k)  

        # Calcul des probabilités pour 0 à 5 buts  
        max_goals = 5  
        home_probs = [poisson_prob(lambda_home, i) for i in range(max_goals + 1)]  
        away_probs = [poisson_prob(lambda_away, i) for i in range(max_goals + 1)]  

        # Calcul des résultats possibles  
        win_home = 0  
        win_away = 0  
        draw = 0  

        for home_goals in range(max_goals + 1):  
            for away_goals in range(max_goals + 1):  
                if home_goals > away_goals:  
                    win_home += home_probs[home_goals] * away_probs[away_goals]  
                elif home_goals < away_goals:  
                    win_away += home_probs[home_goals] * away_probs[away_goals]  
                else:  
                    draw += home_probs[home_goals] * away_probs[away_goals]  

        # Affichage des résultats  
        st.write(f"🔮 Prédiction du résultat (Random Forest) : {'Victoire Domicile' if prediction_rf == 1 else 'Victoire Extérieure'}")  
        st.write(f"🔮 Prédiction du résultat (XGBoost) : {'Victoire Domicile' if prediction_xgb == 1 else 'Victoire Extérieure'}")  
        st.write(f"🏠 Probabilité de victoire Domicile : {win_home:.2%}")  
        st.write(f"🏟️ Probabilité de victoire Extérieure : {win_away:.2%}")  
        st.write(f"🤝 Probabilité de match nul : {draw:.2%}")  

        # Visualisation des résultats avec Altair  
        results_df = pd.DataFrame({  
            'Résultat': ['Victoire Domicile', 'Victoire Extérieure', 'Match Nul'],  
            'Probabilité': [win_home, win_away, draw]  
        })  

        chart = alt.Chart(results_df).mark_bar().encode(  
            x='Résultat',  
            y='Probabilité',  
            color='Résultat'  
        ).properties(  
            title='Probabilités des Résultats'  
        )  

        st.altair_chart(chart, use_container_width=True)  

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
            'defensive_duels_away': defensive_duels_away
