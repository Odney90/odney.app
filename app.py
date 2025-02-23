import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import poisson

# Génération de données fictives
data = {
    'Equipe_Domicile': np.random.choice(['Equipe A', 'Equipe B', 'Equipe C'], 100),
    'Equipe_Exterieur': np.random.choice(['Equipe X', 'Equipe Y', 'Equipe Z'], 100),
    'Buts_Domicile': np.random.randint(0, 5, 100),
    'Buts_Exterieur': np.random.randint(0, 5, 100),
    'Possession': np.random.uniform(40, 60, 100),
    'Tirs_Cadrés': np.random.randint(1, 10, 100),
    'Passes_Réussies': np.random.uniform(70, 95, 100),
    'Corners': np.random.randint(0, 10, 100),
    'Fautes': np.random.randint(5, 20, 100),
    'XG_Domicile': np.random.uniform(0, 3, 100),
    'XG_Exterieur': np.random.uniform(0, 3, 100)
}
df = pd.DataFrame(data)

df['Résultat'] = df.apply(lambda row: 1 if row['Buts_Domicile'] > row['Buts_Exterieur'] else (0 if row['Buts_Domicile'] == row['Buts_Exterieur'] else -1), axis=1)

# Séparation des données
X = df[['Possession', 'Tirs_Cadrés', 'Passes_Réussies', 'Corners', 'Fautes', 'XG_Domicile', 'XG_Exterieur']]
y = df['Résultat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Modèles
log_reg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Validation croisée
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_reg_cv = cross_val_score(log_reg, X, y, cv=kf, scoring='accuracy').mean()
rf_cv = cross_val_score(rf, X, y, cv=kf, scoring='accuracy').mean()

# Modèle de Poisson pour prédiction de buts
def poisson_prediction(avg_goals):
    return [poisson.pmf(i, avg_goals) for i in range(6)]

# Interface Streamlit
st.set_page_config(page_title="Prédictions Football", layout="wide")
st.title("Prédictions Football - Machine Learning")
st.write("Modèle de Poisson pour les buts et Machine Learning pour le résultat.")

# Persistance des données avec session state
if 'selected_teams' not in st.session_state:
    st.session_state.selected_teams = {'Equipe_Domicile': None, 'Equipe_Exterieur': None}

equipe_dom = st.selectbox("Sélectionne l'équipe à domicile", df['Equipe_Domicile'].unique(), index=0, key='dom')
equipe_ext = st.selectbox("Sélectionne l'équipe extérieure", df['Equipe_Exterieur'].unique(), index=0, key='ext')
st.session_state.selected_teams = {'Equipe_Domicile': equipe_dom, 'Equipe_Exterieur': equipe_ext}

col1, col2 = st.columns(2)

with col1:
    if st.button("Prédire"):
        avg_goals_dom = df[df['Equipe_Domicile'] == equipe_dom]['Buts_Domicile'].mean()
        avg_goals_ext = df[df['Equipe_Exterieur'] == equipe_ext]['Buts_Exterieur'].mean()
        
        prob_buts_dom = poisson_prediction(avg_goals_dom)
        prob_buts_ext = poisson_prediction(avg_goals_ext)
        
        st.write(f"Probabilité des buts pour {equipe_dom} : {prob_buts_dom}")
        st.write(f"Probabilité des buts pour {equipe_ext} : {prob_buts_ext}")

with col2:
    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    st.write(f"Score validation croisée Logistic Regression : {log_reg_cv:.2f}")
    st.write(f"Score validation croisée Random Forest : {rf_cv:.2f}")
