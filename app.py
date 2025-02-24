import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression, PoissonRegressor  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import KFold, cross_val_score  
import plotly.express as px  

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
    }  

# Formulaire flottant pour la saisie des données  
with st.form("formulaire_saisie"):  
    st.header("📊 Saisie des Données d'Analyse")  
    
    # Champs pour les données de l'équipe A  
    st.subheader("Équipe A")  
    st.session_state.data['score_rating_A'] = st.number_input("Score de Performance (Équipe A)", value=float(st.session_state.data['score_rating_A']), step=0.1)  
    st.session_state.data['buts_par_match_A'] = st.number_input("Buts par Match (Équipe A)", value=float(st.session_state.data['buts_par_match_A']), step=0.1)  
    st.session_state.data['buts_concedes_par_match_A'] = st.number_input("Buts Concedes par Match (Équipe A)", value=float(st.session_state.data['buts_concedes_par_match_A']), step=0.1)  
    st.session_state.data['possession_moyenne_A'] = st.number_input("Possession Moyenne (%) (Équipe A)", value=float(st.session_state.data['possession_moyenne_A']), step=0.1)  
    st.session_state.data['expected_but_A'] = st.number_input("Expected Goals (Équipe A)", value=float(st.session_state.data['expected_but_A']), step=0.1)  
    st.session_state.data['tirs_cadres_A'] = st.number_input("Tirs Cadres (Équipe A)", value=int(st.session_state.data['tirs_cadres_A']), step=1)  
    st.session_state.data['grandes_chances_A'] = st.number_input("Grandes Chances (Équipe A)", value=int(st.session_state.data['grandes_chances_A']), step=1)  
    st.session_state.data['victoires_domicile_A'] = st.number_input("Victoires à Domicile (Équipe A)", value=int(st.session_state.data['victoires_domicile_A']), step=1)  
    st.session_state.data['victoires_exterieur_A'] = st.number_input("Victoires à l'Extérieur (Équipe A)", value=int(st.session_state.data['victoires_exterieur_A']), step=1)  
    st.session_state.data['joueurs_absents_A'] = st.number_input("Joueurs Absents (Équipe A)", value=int(st.session_state.data['joueurs_absents_A']), step=1)  

    # Champs pour les données de l'équipe B  
    st.subheader("Équipe B")  
    st.session_state.data['score_rating_B'] = st.number_input("Score de Performance (Équipe B)", value=float(st.session_state.data['score_rating_B']), step=0.1)  
    st.session_state.data['buts_par_match_B'] = st.number_input("Buts par Match (Équipe B)", value=float(st.session_state.data['buts_par_match_B']), step=0.1)  
    st.session_state.data['buts_concedes_par_match_B'] = st.number_input("Buts Concedes par Match (Équipe B)", value=float(st.session_state.data['buts_concedes_par_match_B']), step=0.1)  
    st.session_state.data['possession_moyenne_B'] = st.number_input("Possession Moyenne (%) (Équipe B)", value=float(st.session_state.data['possession_moyenne_B']), step=0.1)  
    st.session_state.data['expected_but_B'] = st.number_input("Expected Goals (Équipe B)", value=float(st.session_state.data['expected_but_B']), step=0.1)  
    st.session_state.data['tirs_cadres_B'] = st.number_input("Tirs Cadres (Équipe B)", value=int(st.session_state.data['tirs_cadres_B']), step=1)  
    st.session_state.data['grandes_chances_B'] = st.number_input("Grandes Chances (Équipe B)", value=int(st.session_state.data['grandes_chances_B']), step=1)  
    st.session_state.data['victoires_domicile_B'] = st.number_input("Victoires à Domicile (Équipe B)", value=int(st.session_state.data['victoires_domicile_B']), step=1)  
    st.session_state.data['victoires_exterieur_B'] = st.number_input("Victoires à l'Extérieur (Équipe B)", value=int(st.session_state.data['victoires_exterieur_B']), step=1)  
    st.session_state.data['joueurs_absents_B'] = st.number_input("Joueurs Absents (Équipe B)", value=int(st.session_state.data['joueurs_absents_B']), step=1)  

    # Bouton pour soumettre le formulaire  
    submitted = st.form_submit_button("Mettre à jour les données")  

# Création du DataFrame  
df = pd.DataFrame(st.session_state.data, index=[0])  

# Affichage des données  
st.write("Données d'analyse :")  
st.dataframe(df)  

# Validation croisée avec KFold (n_splits=3)  
kf = KFold(n_splits=3, shuffle=True, random_state=42)  

# Modèles de classification  
log_reg = LogisticRegression(max_iter=1000)  
rf_clf = RandomForestClassifier()  

# Création de la variable cible (1 si l'équipe A gagne, 0 sinon)  
y = (df['buts_par_match_A'] > df['buts_par_match_B']).astype(int)  

# Évaluation des modèles  
log_reg_scores = cross_val_score(log_reg, df.drop(columns=['score_rating_A', 'score_rating_B']), y, cv=kf)  
rf_scores = cross_val_score(rf_clf, df.drop(columns=['score_rating_A', 'score_rating_B']), y, cv=kf)  

# Calcul des scores moyens  
log_reg_score = np.mean(log_reg_scores)  
rf_score = np.mean(rf_scores)  

# Affichage des résultats  
st.subheader("🤖 Performance des Modèles")  
col_log_reg, col_rf = st.columns(2)  
with col_log_reg:  
    st.metric("📈 Régression Logistique (Précision)", f"{log_reg_score:.2%}")  
with col_rf:  
    st.metric("🌲 Forêt Aléatoire (Précision)", f"{rf_score:.2%}")  

# Entraînement des modèles sur l'ensemble des données  
log_reg.fit(df.drop(columns=['score_rating_A', 'score_rating_B']), y)  
rf_clf.fit(df.drop(columns=['score_rating_A', 'score_rating_B']), y)  

# Entraînement du modèle Random Forest pour obtenir les poids des critères  
poids_criteres = rf_clf.feature_importances_  

# Affichage des poids des critères  
st.subheader("📊 Importance des Critères")  
poids_criteres_df = pd.DataFrame({  
    "Critère": df.drop(columns=['score_rating_A', 'score_rating_B']).columns,  
    "Importance": poids_criteres  
})  
st.dataframe(poids_criteres_df.sort_values(by="Importance", ascending=False))  

# Graphique des poids des critères  
fig = px.bar(poids_criteres_df.sort_values(by="Importance", ascending=False), x='Critère', y='Importance', title='Importance des Critères', labels={'Importance': 'Importance (%)'})  
st.plotly_chart(fig)  

# Régression Poisson  
st.subheader("📈 Régression Poisson")  
# Préparation des données pour la régression Poisson  
X = df.drop(columns=['score_rating_A', 'score_rating_B'])  
y_A = df['buts_par_match_A']  
y_B = df['buts_par_match_B']  

# Entraînement du modèle de régression Poisson  
poisson_model_A = PoissonRegressor()  
poisson_model_B = PoissonRegressor()  

poisson_model_A.fit(X, y_A)  
poisson_model_B.fit(X, y_B)  

# Prédictions  
predictions_A = poisson_model_A.predict(X)  
predictions_B = poisson_model_B.predict(X)  

# Affichage des prédictions  
st.write("Prédictions de Buts (Régression Poisson) :")  
st.write(f"Buts Prévus pour Équipe A : {predictions_A[0]:.2f}")  
st.write(f"Buts Prévus pour Équipe B : {predictions_B[0]:.2f}")  

# Convertisseur de cotes  
st.subheader("🔄 Convertisseur de Cotes")  
cote_A = st.number_input("Cote proposée pour Équipe A", value=2.0, step=0.1)  
cote_B = st.number_input("Cote proposée pour Équipe B", value=2.0, step=0.1)  

# Calcul des probabilités implicites  
prob_implicite_A = 1 / cote_A  
prob_implicite_B = 1 / cote_B  

# Calcul des probabilités prédites  
prob_predite_A = predictions_A[0] / (predictions_A[0] + predictions_B[0])  
prob_predite_B = predictions_B[0] / (predictions_A[0] + predictions_B[0])  

# Affichage des résultats de comparaison  
st.write(f"Probabilité implicite pour Équipe A : {prob_implicite_A:.2%}")  
st.write(f"Probabilité implicite pour Équipe B : {prob_implicite_B:.2%}")  
st.write(f"Probabilité prédite pour Équipe A : {prob_predite_A:.2%}")  
st.write(f"Probabilité prédite pour Équipe B : {prob_predite_B:.2%}")  

# Comparaison des probabilités  
st.subheader("📊 Comparaison des Probabilités")  
comparison_df = pd.DataFrame({  
    "Équipe": ["A", "B"],  
    "Probabilité Implicite": [prob_implicite_A, prob_implicite_B],  
    "Probabilité Prédite": [prob_predite_A, prob_predite_B]  
})  

fig_comparison = px.bar(comparison_df, x='Équipe', y=['Probabilité Implicite', 'Probabilité Prédite'], title='Comparaison des Probabilités', labels={'value': 'Probabilité (%)'})  
st.plotly_chart(fig_comparison)
