import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import KFold  
from sklearn.metrics import mean_squared_error  
from statsmodels.api import GLM  
import matplotlib.pyplot as plt  
import seaborn as sns  
import altair as alt  

# Initialisation des données de session si elles n'existent pas  
if 'data' not in st.session_state:  
    st.session_state.data = {  
        'score_rating_A': 85.0,  
        'score_rating_B': 78.0,  
        'buts_par_match_A': 1.8,  
        'buts_par_match_B': 1.5,  
        'buts_concedes_par_match_A': 1.2,  
        'buts_concedes_par_match_B': 1.4,  
        'possession_moyenne_A': 58.0,  
        'possession_moyenne_B': 52.0,  
        'expected_but_A': 1.7,  
        'expected_but_B': 1.4,  
        'tirs_cadres_A': 5,  
        'tirs_cadres_B': 4,  
        'grandes_chances_A': 2,  
        'grandes_chances_B': 2,  
        'victoires_domicile_A': 8,  
        'victoires_domicile_B': 6,  
        'victoires_exterieur_A': 5,  
        'victoires_exterieur_B': 4,  
        'joueurs_absents_A': 1,  
        'joueurs_absents_B': 2,  
        'moyenne_age_joueurs_A': 27.0,  
        'moyenne_age_joueurs_B': 26.5,  
        'experience_entraineur_A': 5,  
        'experience_entraineur_B': 3,  
        'nombre_blessures_A': 2,  
        'nombre_blessures_B': 1,  
        'cartons_jaunes_A': 10,  
        'cartons_jaunes_B': 8,  
        'cartons_rouges_A': 1,  
        'cartons_rouges_B': 0,  
        'taux_reussite_passes_A': 85.0,  
        'taux_reussite_passes_B': 80.0,  
        'taux_possession_A': 55.0,  
        'taux_possession_B': 50.0,  
        'nombre_tirs_totaux_A': 15,  
        'nombre_tirs_totaux_B': 12,  
        'nombre_tirs_cadres_A': 5,  
        'nombre_tirs_cadres_B': 4,  
        'ratio_buts_tirs_A': 0.12,  
        'ratio_buts_tirs_B': 0.10,  
        'ratio_buts_encais_tirs_A': 0.08,  
        'ratio_buts_encais_tirs_B': 0.09,  
        'performance_domicile_A': 2.1,  
        'performance_domicile_B': 1.8,  
        'performance_exterieur_A': 1.5,  
        'performance_exterieur_B': 1.2,  
        'historique_confrontations_A_B': 3,  
        'historique_confrontations_B_A': 1,  
        'moyenne_buts_marques_A': 2.0,  
        'moyenne_buts_marques_B': 1.5,  
        'moyenne_buts_encais_A': 1.0,  
        'moyenne_buts_encais_B': 1.2,  
        'impact_joueurs_cles_A': 0.8,  
        'impact_joueurs_cles_B': 0.7,  
        'taux_reussite_corners_A': 30.0,  
        'taux_reussite_corners_B': 25.0,  
        'nombre_degagements_A': 15,  
        'nombre_degagements_B': 12,  
        'tactique_A': 8,  
        'tactique_B': 7,  
        'motivation_A': 8,  
        'motivation_B': 7,  
        'forme_recente_A': 12,  
        'forme_recente_B': 9,  
    }  

# Fonction pour entraîner un modèle de régression linéaire avec KFold (k=3)  
def train_linear_regression_with_kfold(X, y):  
    if X.shape[0] < 3:  
        raise ValueError("Nombre d'échantillons insuffisant pour KFold (minimum 3 requis).")  
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # k=3  
    mse_scores = []  
    for train_index, test_index in kf.split(X):  
        X_train, X_test = X[train_index], X[test_index]  
        y_train, y_test = y[train_index], y[test_index]  
        model = LinearRegression()  
        model.fit(X_train, y_train)  
        y_pred = model.predict(X_test)  
        mse_scores.append(mean_squared_error(y_test, y_pred))  
    return np.mean(mse_scores)  

# Fonction pour entraîner un modèle de régression de Poisson  
def train_poisson_regression(X, y):  
    model = GLM(y, X, family=GLM.families.Poisson())  
    results = model.fit()  
    return results  

# Fonction pour entraîner un modèle Random Forest  
def train_random_forest(X, y):  
    model = RandomForestRegressor(n_estimators=100, random_state=42)  
    model.fit(X, y)  
    return model  

# Fonction pour afficher les résultats et les visuels  
def display_results_and_visuals(data):  
    st.header("📊 Résultats et Visuels Explicatifs")  
    
    # Préparation des données  
    X = np.array([data['buts_par_match_A'], data['possession_moyenne_A'], data['taux_reussite_passes_A']]).T  
    y = np.array(data['score_rating_A'])  
    
    # Vérification de la forme des données  
    if len(X.shape) == 1:  
        X = X.reshape(-1, 1)  # Convertir en tableau 2D si nécessaire  
    if X.shape[0] == 0 or X.shape[1] == 0:  
        st.error("Les données d'entrée sont vides ou mal formatées.")  
        return  
    
    # Affichage des données dans un tableau interactif  
    st.subheader("Données d'Entrée")  
    df = pd.DataFrame({  
        'Buts par Match': X[:, 0],  
        'Possession Moyenne (%)': X[:, 1] if X.shape[1] > 1 else np.nan,  
        'Taux de Réussite des Passes (%)': X[:, 2] if X.shape[1] > 2 else np.nan,  
        'Score de Performance': y  
    })  
    st.dataframe(df)  
    
    # KFold avec régression linéaire (k=3)  
    st.subheader("Validation Croisée (KFold avec k=3) avec Régression Linéaire")  
    try:  
        mse = train_linear_regression_with_kfold(X, y)  
        st.write(f"Erreur Quadratique Moyenne (MSE) : {mse:.2f}")  
    except ValueError as e:  
        st.error(f"Erreur lors de la validation croisée : {e}")  
    
    # Régression de Poisson  
    st.subheader("Régression de Poisson")  
    poisson_results = train_poisson_regression(X, y)  
    st.write(poisson_results.summary())  
    
    # Random Forest  
    st.subheader("Random Forest")  
    rf_model = train_random_forest(X, y)  
    y_pred = rf_model.predict(X)  
    st.write(f"Score de Performance Prédit : {y_pred[0]:.2f}")  
    
    # Graphique explicatif avec Altair  
    st.subheader("Graphique Explicatif (Altair)")  
    chart = alt.Chart(df).mark_circle(size=60).encode(  
        x='Buts par Match',  
        y='Score de Performance',  
        tooltip=['Buts par Match', 'Score de Performance']  
    ).interactive()  
    st.altair_chart(chart, use_container_width=True)  

# Onglets pour l'application  
tab1, tab2 = st.tabs(["📝 Saisie des Données", "📊 Résultats et Visuels"])  

with tab1:  
    # Formulaire flottant pour la saisie des données  
    with st.form("formulaire_saisie"):  
        st.header("📊 Saisie des Données d'Analyse")  
        
        # Champs pour les données de l'équipe A  
        st.subheader("Équipe A 🏆")  
        st.session_state.data['score_rating_A'] = st.number_input("Score de Performance (Équipe A)", value=float(st.session_state.data['score_rating_A']), step=0.1)  
        st.session_state.data['buts_par_match_A'] = st.number_input("Buts par Match (Équipe A)", value=float(st.session_state.data['buts_par_match_A']), step=0.1)  
        st.session_state.data['possession_moyenne_A'] = st.number_input("Possession Moyenne (%) (Équipe A)", value=float(st.session_state.data['possession_moyenne_A']), step=0.1)  
        st.session_state.data['taux_reussite_passes_A'] = st.number_input("Taux de Réussite des Passes (%) (Équipe A)", value=float(st.session_state.data['taux_reussite_passes_A']), step=0.1)  
        
        # Bouton pour soumettre le formulaire  
        submitted = st.form_submit_button("✅ Soumettre les Données")  
        if submitted:  
            st.success("Données soumises avec succès! 🎉")  

with tab2:  
    # Afficher les résultats et les visuels  
    display_results_and_visuals(st.session_state.data)  

# Fin de l'application  
st.write("Merci d'utiliser l'outil d'analyse des équipes! ⚽️")
