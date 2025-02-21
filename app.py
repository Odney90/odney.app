import streamlit as st  
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.metrics import accuracy_score, classification_report  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
import seaborn as sns  
from scipy.stats import poisson
# Création de données fictives  
def create_fake_data(num_samples=100):  
    np.random.seed(42)  
    data = {  
        'buts_produits': np.random.randint(0, 5, num_samples),  
        'buts_encaisses': np.random.randint(0, 5, num_samples),  
        'possession': np.random.uniform(40, 70, num_samples),  
        'motivation': np.random.uniform(1, 10, num_samples),  
        'absents': np.random.randint(0, 5, num_samples),  
        'rating_joueurs': np.random.uniform(0, 10, num_samples),  
        'forme_recente': np.random.uniform(0, 10, num_samples),  
        'tirs_cadres': np.random.randint(0, 10, num_samples),  
        'corners': np.random.randint(0, 10, num_samples),  
        'xG': np.random.uniform(0, 3, num_samples),  
        'grosses_occasions': np.random.randint(0, 5, num_samples),  
        'grosses_occasions_ratees': np.random.randint(0, 5, num_samples),  
        'passes_reussies': np.random.randint(0, 1000, num_samples),  
        'passes_longues_precises': np.random.randint(0, 20, num_samples),  
        'centres_reussis': np.random.randint(0, 10, num_samples),  
        'penalties_obtenus': np.random.randint(0, 5, num_samples),  
        'touches_surface_adverse': np.random.randint(0, 10, num_samples),  
        'corners_concedes': np.random.randint(0, 10, num_samples),  
        'xG_concedes': np.random.uniform(0, 3, num_samples),  
        'interceptions': np.random.randint(0, 10, num_samples),  
        'tacles_reussis': np.random.randint(0, 10, num_samples),  
        'degagements': np.random.randint(0, 10, num_samples),  
        'possessions_recuperees': np.random.randint(0, 10, num_samples),  
        'penalties_concedes': np.random.randint(0, 5, num_samples),  
        'arrets': np.random.randint(0, 10, num_samples),  
        'fautes': np.random.randint(0, 20, num_samples),  
        'cartons_jaunes': np.random.randint(0, 5, num_samples),  
        'cartons_rouges': np.random.randint(0, 2, num_samples),  
        'tactique': np.random.choice(['4-4-2', '4-3-3', '3-5-2'], num_samples),  
        'resultat': np.random.choice([0, 1], num_samples)  # 0 pour victoire B, 1 pour victoire A  
    }  
    return pd.DataFrame(data)  

data = create_fake_data(100)
# Préparation des données  
def prepare_data(data):  
    features = data[['buts_produits', 'buts_encaisses', 'possession', 'motivation',   
                     'absents', 'rating_joueurs', 'forme_recente', 'tirs_cadres',   
                     'corners', 'xG', 'grosses_occasions', 'grosses_occasions_ratees',   
                     'passes_reussies', 'passes_longues_precises', 'centres_reussis',   
                     'penalties_obtenus', 'touches_surface_adverse', 'corners_concedes',   
                     'xG_concedes', 'interceptions', 'tacles_reussis', 'degagements',   
                     'possessions_recuperees', 'penalties_concedes', 'arrets',   
                     'fautes', 'cartons_jaunes', 'cartons_rouges']]  
    target = data['resultat']  # 1 pour victoire A, 0 pour victoire B  
    return features, target  

features, target = prepare_data(data)
# Séparation des données  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)  

# Normalisation des données  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)
# Entraînement du modèle Random Forest  
rf_model = RandomForestClassifier(random_state=42)  
rf_model.fit(X_train_scaled, y_train)  

# Prédictions  
y_pred_rf = rf_model.predict(X_test_scaled)  

# Évaluation du modèle  
rf_accuracy = accuracy_score(y_test, y_pred_rf)  
st.write(f"Précision du modèle Random Forest : {rf_accuracy:.2%}")  
st.write(classification_report(y_test, y_pred_rf))
# Optimisation des hyperparamètres pour Random Forest  
param_grid = {  
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10]  
}  

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)  
grid_search.fit(X_train_scaled, y_train)  

# Meilleurs paramètres  
st.write("Meilleurs paramètres trouvés pour Random Forest :")  
st.write(grid_search.best_params_)
# Visualisation de l'importance des caractéristiques  
importances = rf_model.feature_importances_  
feature_names = features.columns  
indices = np.argsort(importances)[::-1]  

plt.figure(figsize=(10, 6))  
sns.barplot(x=importances[indices], y=feature_names[indices])  
plt.title("Importance des caractéristiques (Random Forest)")  
plt.xlabel("Importance")  
plt.ylabel("Caractéristiques")  
st.pyplot(plt)
# Prédictions avec le modèle optimisé  
best_rf_model = grid_search.best_estimator_  
y_pred_optimized_rf = best_rf_model.predict(X_test_scaled)  

# Évaluation du modèle optimisé  
optimized_rf_accuracy = accuracy_score(y_test, y_pred_optimized_rf)  
st.write(f"Précision du modèle Random Forest optimisé : {optimized_rf_accuracy:.2%}")
# Prédiction des buts avec la méthode de Poisson  
def prediction_buts_poisson(xG_A, xG_B):  
    # Calcul des probabilités de marquer 0 à 5 buts  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  

    # Calcul des buts attendus  
    buts_attendus_A = sum(i * prob for i, prob in enumerate(buts_A))  
    buts_attendus_B = sum(i * prob for i, prob in enumerate(buts_B))  

    return buts_attendus_A, buts_attendus_B  

# Exemple d'utilisation avec des valeurs fictives  
xG_A = np.random.uniform(1, 3)  
xG_B = np.random.uniform(1, 3)  
buts_moyens_A, buts_moyens_B = prediction_buts_poisson(xG_A, xG_B)  

# Affichage des résultats de la prédiction  
st.header("⚽ Prédiction des Buts (Méthode de Poisson)")  
st.write(f"Buts attendus pour l'équipe A : **{buts_moyens_A:.2f}**")  
st.write(f"Buts attendus pour l'équipe B : **{buts_moyens_B:.2f}**")
from sklearn.linear_model import LogisticRegression  

# Entraînement du modèle de régression logistique  
log_model = LogisticRegression(max_iter=1000)  
log_model.fit(X_train_scaled, y_train)  

# Prédictions  
y_pred_log = log_model.predict(X_test_scaled)  

# Évaluation du modèle  
log_accuracy = accuracy_score(y_test, y_pred_log)  
st.write(f"Précision du modèle de régression logistique : {log_accuracy:.2%}")  
st.write(classification_report(y_test, y_pred_log))
# Conclusion  
st.header("Conclusion")  
st.write("Cette application permet de prédire les résultats des matchs de football en utilisant plusieurs modèles de machine learning.")  
st.write("Les modèles incluent la régression logistique, la méthode de Poisson et la forêt aléatoire.")  
st.write("Les résultats montrent la précision de chaque modèle et l'importance des caractéristiques utilisées.")
