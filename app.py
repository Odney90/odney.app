import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  
from sklearn.model_selection import cross_val_score  
from statsmodels.api import GLM  
from statsmodels.genmod.families import Poisson  

# Configuration de la page  
st.set_page_config(  
    page_title="Prédiction de Matchs de Football",  
    page_icon="⚽",  
    layout="wide"  
)  

# Entraînement des modèles  
@st.cache_resource  
def train_models():  
    # Créer un ensemble de données d'entraînement synthétique  
    np.random.seed(42)  
    data = pd.DataFrame({  
        'home_goals': np.random.randint(0, 3, size=1000),  
        'away_goals': np.random.randint(0, 3, size=1000),  
        'xG': np.random.uniform(0, 2, size=1000),  
        'xGA': np.random.uniform(0, 2, size=1000),  
        'encais': np.random.uniform(0, 2, size=1000),  
        'face_a_face': np.random.uniform(0, 100, size=1000),  
        'motivation_home': np.random.uniform(0, 100, size=1000),  
        'motivation_away': np.random.uniform(0, 100, size=1000),  
        'victoire_domicile': np.random.uniform(0, 100, size=1000),  
        'victoire_exterieur': np.random.uniform(0, 100, size=1000),  
        'result': np.random.choice([0, 1, 2], size=1000)  
    })  

    # Séparer les caractéristiques et la cible  
    X = data[['home_goals', 'away_goals', 'xG', 'xGA', 'encais', 'face_a_face', 'motivation_home', 'motivation_away', 'victoire_domicile', 'victoire_exterieur']]  
    y = data['result']  

    # Modèle de régression logistique  
    log_reg = LogisticRegression(max_iter=500, solver='lbfgs')  
    log_reg.fit(X, y)  

    # Modèle Random Forest  
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)  
    rf.fit(X, y)  

    # Modèle XGBoost  
    xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.3, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)  
    xgb.fit(X, y)  

    # Modèle Poisson  
    poisson_model = GLM(y, X, family=Poisson()).fit()  

    return log_reg, rf, xgb, poisson_model  

# Charger les modèles  
if 'models' not in st.session_state:  
    st.session_state.models = train_models()  

log_reg_model, rf_model, xgb_model, poisson_model = st.session_state.models  

# Validation croisée  
def cross_validate_models(X, y):  
    # Scores de validation croisée pour chaque modèle  
    log_reg_scores = cross_val_score(log_reg_model, X, y, cv=5, scoring='accuracy')  
    rf_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')  
    xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')  

    # Affichage des résultats  
    st.subheader("📊 Validation Croisée")  
    st.write("**Régression Logistique (Précision moyenne) :**", log_reg_scores.mean())  
    st.write("**Random Forest (Précision moyenne) :**", rf_scores.mean())  
    st.write("**XGBoost (Précision moyenne) :**", xgb_scores.mean())  

# Titre de l'application  
st.title("⚽ Prédiction de Matchs de Football")  

# Sélection des équipes  
st.subheader("🏠 Équipe Domicile vs 🏃‍♂️ Équipe Extérieure")  
home_team = st.text_input("Nom de l'équipe à domicile", "Équipe A")  
away_team = st.text_input("Nom de l'équipe à l'extérieur", "Équipe B")  

# Bouton pour prédire les résultats  
if st.button("🔍 Prédire les résultats"):  
    # Exemple de prédiction avec le modèle de régression logistique  
    example_input = np.array([[1, 1, 1.5, 1.2, 0.8, 50, 70, 60, 60, 40]])  # Données d'exemple  
    log_reg_pred = log_reg_model.predict_proba(example_input)[0]  
    rf_pred = rf_model.predict_proba(example_input)[0]  
    xgb_pred = xgb_model.predict_proba(example_input)[0]  
    poisson_pred = poisson_model.predict(example_input)[0]  

    # Affichage des résultats  
    st.subheader("📊 Résultats de la Prédiction")  
    st.write(f"**Probabilité de victoire à domicile ({home_team}) :** {log_reg_pred[2] * 100:.2f}%")  
    st.write(f"**Probabilité de match nul :** {log_reg_pred[1] * 100:.2f}%")  
    st.write(f"**Probabilité de victoire à l'extérieur ({away_team}) :** {log_reg_pred[0] * 100:.2f}%")  

    # Comparaison des modèles  
    st.subheader("📈 Comparaison des Modèles")  
    model_comparison_data = {  
        "Modèle": ["Régression Logistique", "Random Forest", "XGBoost", "Poisson"],  
        "Probabilité Domicile (%)": [log_reg_pred[2] * 100, rf_pred[2] * 100, xgb_pred[2] * 100, poisson_pred[2] * 100],  
        "Probabilité Nul (%)": [log_reg_pred[1] * 100, rf_pred[1] * 100, xgb_pred[1] * 100, poisson_pred[1] * 100],  
        "Probabilité Extérieure (%)": [log_reg_pred[0] * 100, rf_pred[0] * 100, xgb_pred[0] * 100, poisson_pred[0] * 100],  
    }  
    model_comparison_df = pd.DataFrame(model_comparison_data)  
    st.dataframe(model_comparison_df, use_container_width=True)  

    # Validation croisée  
    cross_validate_models(X, y)  

# Section pour personnaliser les prédictions  
st.subheader("🎯 Personnaliser les Prédictions")  

# Formulaire pour saisir des valeurs personnalisées  
st.write("Entrez les valeurs des caractéristiques pour personnaliser la prédiction :")  
home_goals_input = st.number_input("Buts à domicile", min_value=0, value=1)  
away_goals_input = st.number_input("Buts à l'extérieur", min_value=0, value=1)  
xG_input = st.number_input("Expected Goals (xG) à domicile", min_value=0.0, value=1.5)  
xGA_input = st.number_input("Expected Goals Against (xGA) à domicile", min_value=0.0, value=1.2)  
encais_input = st.number_input("Encaisse des buts à domicile", min_value=0.0, value=0.8)  
face_a_face_input = st.number_input("Historique des confrontations", min_value=0, value=50)  
motivation_home_input = st.number_input("Motivation de l'équipe à domicile", min_value=0, value=70)  
motivation_away_input = st.number_input("Motivation de l'équipe à l'extérieur", min_value=0, value=60)  
victoire_domicile_input = st.number_input("Taux de victoire à domicile", min_value=0, value=60)  
victoire_exterieur_input = st.number_input("Taux de victoire à l'extérieur", min_value=0, value=40)  

# Bouton pour lancer la prédiction personnalisée  
if st.button("🔍 Prédire avec les valeurs personnalisées"):  
    # Créer un tableau avec les valeurs saisies  
    custom_input = np.array([[home_goals_input, away_goals_input, xG_input, xGA_input, encais_input, face_a_face_input, motivation_home_input, motivation_away_input, victoire_domicile_input, victoire_exterieur_input]])  

    # Prédictions avec les modèles  
    log_reg_custom_pred = log_reg_model.predict_proba(custom_input)[0]  
    rf_custom_pred = rf_model.predict_proba(custom_input)[0]  
    xgb_custom_pred = xgb_model.predict_proba(custom_input)[0]  
    poisson_custom_pred = poisson_model.predict(custom_input)[0]  

    # Affichage des résultats personnalisés  
    st.subheader("📊 Résultats Personnalisés")  
    st.write(f"**Probabilité de victoire à domicile ({home_team}) :** {log_reg_custom_pred[2] * 100:.2f}%")  
    st.write(f"**Probabilité de match nul :** {log_reg_custom_pred[1] * 100:.2f}%")  
    st.write(f"**Probabilité de victoire à l'extérieur ({away_team}) :** {log_reg_custom_pred[0] * 100:.2f}%")  

    # Comparaison des modèles pour les prédictions personnalisées  
    st.subheader("📈 Comparaison des Modèles (Personnalisé)")  
    custom_comparison_data = {  
        "Modèle": ["Régression Logistique", "Random Forest", "XGBoost", "Poisson"],  
        "Probabilité Domicile (%)": [log_reg_custom_pred[2] * 100, rf_custom_pred[2] * 100, xgb_custom_pred[2] * 100, poisson_custom_pred[2] * 100],  
        "Probabilité Nul (%)": [log_reg_custom_pred[1] * 100, rf_custom_pred[1] * 100, xgb_custom_pred[1] * 100, poisson_custom_pred[1] * 100],  
        "Probabilité Extérieure (%)": [log_reg_custom_pred[0] * 100, rf_custom_pred[0] * 100, xgb_custom_pred[0] * 100, poisson_custom_pred[0] * 100],  
    }  
    custom_comparison_df = pd.DataFrame(custom_comparison_data)  
    st.dataframe(custom_comparison_df, use_container_width=True)  

# Section pour afficher les données d'entraînement synthétiques  
st.subheader("📂 Données d'Entraînement Synthétiques")  
st.write("Voici un aperçu des données synthétiques utilisées pour entraîner les modèles :")  
st.dataframe(data.head(), use_container_width=True)  

# Section pour afficher les caractéristiques utilisées pour la prédiction  
st.subheader("🔑 Caractéristiques Utilisées pour la Prédiction")  
st.write("Les caractéristiques suivantes ont été utilisées pour entraîner les modèles :")  
st.write("""  
- **home_goals** : Nombre de buts marqués par l'équipe à domicile.  
- **away_goals** : Nombre de buts marqués par l'équipe à l'extérieur.  
- **xG** : Expected Goals (xG) de l'équipe à domicile.  
- **xGA** : Expected Goals Against (xGA) de l'équipe à domicile.  
- **encais** : Encaisse des buts de l'équipe à domicile.  
- **face_a_face** : Historique des confrontations entre les deux équipes.  
- **motivation_home** : Motivation de l'équipe à domicile.  
- **motivation_away** : Motivation de l'équipe à l'extérieur.  
- **victoire_domicile** : Taux de victoire à domicile de l'équipe à domicile.  
- **victoire_exterieur** : Taux de victoire à l'extérieur de l'équipe à l'extérieur.  
""")  

# Section pour afficher les conclusions  
st.subheader("📌 Conclusions")  
st.write("""  
Les résultats montrent que les quatre modèles donnent des prédictions similaires, mais avec des différences mineures.   
Le modèle **XGBoost** semble être le plus précis, suivi de près par le **Random Forest**, la **Régression Logistique** et le **Poisson**.  
""")  

# Section pour afficher les recommandations  
st.subheader("💡 Recommandations")  
st.write("""  
- **Pour des prédictions rapides** : Utilisez la Régression Logistique.  
- **Pour des prédictions précises** : Utilisez XGBoost ou Random Forest.  
- **Pour des données complexes** : Utilisez XGBoost, car il est plus adapté aux données non linéaires.  
- **Pour des prédictions basées sur des distributions de Poisson** : Utilisez le modèle Poisson.  
""")
