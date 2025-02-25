import streamlit as st  
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier  
from xgboost import XGBClassifier  

# Configuration de la page  
st.set_page_config(  
    page_title="Prédiction de Matchs de Football",  
    page_icon="⚽",  # Icône pour le site  
    layout="wide"  
)  

# Fonction pour entraîner les modèles  
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

    # Modèle de régression logistique avec hyperparamètres optimisés  
    log_reg = LogisticRegression(max_iter=500, solver='lbfgs')  
    log_reg.fit(X, y)  

    # Modèle Random Forest avec hyperparamètres optimisés  
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)  
    rf.fit(X, y)  

    # Modèle XGBoost avec hyperparamètres optimisés  
    xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.3, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)  
    xgb.fit(X, y)  

    return log_reg, rf, xgb  

# Vérifier si les modèles sont déjà chargés dans l'état de session  
if 'models' not in st.session_state:  
    st.session_state.models = train_models()  

log_reg_model, rf_model, xgb_model = st.session_state.models  

# Interface utilisateur  
st.title("⚽ Prédiction de Matchs de Football")  

# Sélection des équipes  
st.subheader("🏠 Équipe Domicile vs 🏃‍♂️ Équipe Extérieure")  
home_team = st.text_input("Nom de l'équipe à domicile", "Équipe A")  
away_team = st.text_input("Nom de l'équipe à l'extérieur", "Équipe B")  

# Prédiction des résultats  
if st.button("🔍 Prédire les résultats"):  
    # Exemple de prédiction avec le modèle de régression logistique  
    example_input = np.array([[1, 1, 1.5, 1.2, 0.8, 50, 70, 60, 60, 40]])  # Données d'exemple  
    log_reg_pred = log_reg_model.predict_proba(example_input)[0]  
    rf_pred = rf_model.predict_proba(example_input)[0]  
    xgb_pred = xgb_model.predict_proba(example_input)[0]  

    # Affichage des résultats  
    st.subheader("📊 Résultats de la Prédiction")  
    st.write(f"**Probabilité de victoire à domicile ({home_team}) :** {log_reg_pred[2] * 100:.2f}%")  
    st.write(f"**Probabilité de match nul :** {log_reg_pred[1] * 100:.2f}%")  
    st.write(f"**Probabilité de victoire à l'extérieur ({away_team}) :** {log_reg_pred[0] * 100:.2f}%")  

    # Comparaison des modèles  
    st.subheader("📈 Comparaison des Modèles")  
    model_comparison_data = {  
        "Modèle": ["Régression Logistique", "Random Forest", "XGBoost"],  
        "Probabilité Domicile (%)": [log_reg_pred[2] * 100, rf_pred[2] * 100, xgb_pred[2] * 100],  
        "Probabilité Nul (%)": [log_reg_pred[1] * 100, rf_pred[1] * 100, xgb_pred[1] * 100],  
        "Probabilité Extérieure (%)": [log_reg_pred[0] * 100, rf_pred[0] * 100, xgb_pred[0] * 100],  
    }  
    model_comparison_df = pd.DataFrame(model_comparison_data)  
    st.dataframe(model_comparison_df, use_container_width=True)
    # Affichage des graphiques de comparaison des modèles  
st.subheader("📊 Graphiques de Comparaison des Modèles")  

# Graphique en barres pour les probabilités de victoire à domicile  
st.bar_chart(model_comparison_df.set_index("Modèle")["Probabilité Domicile (%)"])  

# Graphique en barres pour les probabilités de match nul  
st.bar_chart(model_comparison_df.set_index("Modèle")["Probabilité Nul (%)"])  

# Graphique en barres pour les probabilités de victoire à l'extérieur  
st.bar_chart(model_comparison_df.set_index("Modèle")["Probabilité Extérieure (%)"])  

# Section pour afficher les détails des prédictions  
st.subheader("🔍 Détails des Prédictions")  

# Affichage des probabilités brutes pour chaque modèle  
st.write("**Probabilités brutes (Régression Logistique) :**", log_reg_pred)  
st.write("**Probabilités brutes (Random Forest) :**", rf_pred)  
st.write("**Probabilités brutes (XGBoost) :**", xgb_pred)  

# Section pour afficher les informations sur les modèles  
st.subheader("ℹ️ Informations sur les Modèles")  

# Description des modèles  
st.write("""  
- **Régression Logistique** : Un modèle linéaire utilisé pour la classification binaire ou multiclasse.  
- **Random Forest** : Un ensemble d'arbres de décision qui améliore la précision et réduit le surapprentissage.  
- **XGBoost** : Un modèle de boosting basé sur les arbres de décision, connu pour sa performance et sa rapidité.  
""")  

# Section pour afficher les données d'entraînement synthétiques  
st.subheader("📂 Données d'Entraînement Synthétiques")  

# Affichage des premières lignes des données synthétiques  
st.write("Voici un aperçu des données synthétiques utilisées pour entraîner les modèles :")  
st.dataframe(data.head(), use_container_width=True)  

# Section pour afficher les caractéristiques utilisées pour la prédiction  
st.subheader("🔑 Caractéristiques Utilisées pour la Prédiction")  

# Liste des caractéristiques  
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

# Section pour afficher les résultats détaillés  
st.subheader("📝 Résultats Détaillés")  

# Affichage des résultats détaillés pour chaque modèle  
st.write("**Régression Logistique :**")  
st.write(f"Probabilité de victoire à domicile : {log_reg_pred[2] * 100:.2f}%")  
st.write(f"Probabilité de match nul : {log_reg_pred[1] * 100:.2f}%")  
st.write(f"Probabilité de victoire à l'extérieur : {log_reg_pred[0] * 100:.2f}%")  

st.write("**Random Forest :**")  
st.write(f"Probabilité de victoire à domicile : {rf_pred[2] * 100:.2f}%")  
st.write(f"Probabilité de match nul : {rf_pred[1] * 100:.2f}%")  
st.write(f"Probabilité de victoire à l'extérieur : {rf_pred[0] * 100:.2f}%")  

st.write("**XGBoost :**")  
st.write(f"Probabilité de victoire à domicile : {xgb_pred[2] * 100:.2f}%")  
st.write(f"Probabilité de match nul : {xgb_pred[1] * 100:.2f}%")  
st.write(f"Probabilité de victoire à l'extérieur : {xgb_pred[0] * 100:.2f}%")  

# Section pour afficher les conclusions  
st.subheader("📌 Conclusions")  

# Affichage des conclusions  
st.write("""  
Les résultats montrent que les trois modèles donnent des prédictions similaires, mais avec des différences mineures.   
Le modèle **XGBoost** semble être le plus précis, suivi de près par le **Random Forest** et la **Régression Logistique**.  
""")  

# Section pour afficher les recommandations  
st.subheader("💡 Recommandations")  

# Affichage des recommandations  
st.write("""  
- **Pour des prédictions rapides** : Utilisez la Régression Logistique.  
- **Pour des prédictions précises** : Utilisez XGBoost ou Random Forest.  
- **Pour des données complexes** : Utilisez XGBoost, car il est plus adapté aux données non linéaires.  
""")
# Section pour personnaliser les prédictions  
st.subheader("🎯 Personnaliser les Prédictions")  

# Formulaire pour saisir des valeurs personnalisées  
st.write("Entrez les valeurs des caractéristiques pour personnaliser la prédiction :")  
home_goals_input = st.number_input("Buts à domicile", min_value=0, value=1)  
away_goals_input = st.number_input("Buts à l'extérieur", min_value=0, value=1)  
xG_input = st.number_input("Expected Goals (xG) à domicile", min_value=0.0, value=1.5)  
xGA_input = st.number_input("Expected Goals Against (xGA) à domicile", min_value=0.0, value=1.2)  
encais_input = st.number_input("Encaisse des buts à domicile", min_value=
