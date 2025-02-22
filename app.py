import streamlit as st  
import pandas as pd  
import numpy as np  
import plotly.express as px  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split, GridSearchCV  

# Fonction pour générer des données factices (à remplacer par des données réelles)  
def generate_fake_data():  
    data = {  
        'teamA_goals': np.random.randint(0, 5, 100),  
        'teamB_goals': np.random.randint(0, 5, 100),  
        'teamA_possession': np.random.randint(40, 80, 100),  
        'teamB_possession': np.random.randint(40, 80, 100),  
        'teamA_xg': np.random.uniform(0.5, 3.5, 100),  
        'teamB_xg': np.random.uniform(0.5, 3.5, 100),  
        'teamA_shots': np.random.randint(5, 20, 100),  
        'teamB_shots': np.random.randint(5, 20, 100),  
        'teamA_corners': np.random.randint(1, 10, 100),  
        'teamB_corners': np.random.randint(1, 10, 100),  
        'teamA_interceptions': np.random.randint(5, 15, 100),  
        'teamB_interceptions': np.random.randint(5, 15, 100),  
        'teamA_saves': np.random.randint(1, 10, 100),  
        'teamB_saves': np.random.randint(1, 10, 100),  
        'teamA_clean_sheet': np.random.randint(0, 2, 100),  
        'teamB_clean_sheet': np.random.randint(0, 2, 100),  
        'teamA_form': np.random.randint(0, 15, 100),  # Score de forme (5 derniers matchs)  
        'teamB_form': np.random.randint(0, 15, 100),  
        'head_to_head': np.random.randint(0, 10, 100),  # Nombre de victoires de l'équipe A face à B  
        'result': np.random.choice(['A', 'B', 'D'], 100)  # A: Équipe A gagne, B: Équipe B gagne, D: Match nul  
    }  
    df = pd.DataFrame(data)  
    df['result'] = df['result'].map({'A': 0, 'B': 1, 'D': 2})  
    return df  

# Entraînement du modèle avec GridSearchCV  
def train_model(df):  
    X = df.drop('result', axis=1)  
    y = df['result']  
    
    # Hyperparamètres à tester  
    param_grid = {  
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 10, 20],  
        'min_samples_split': [2, 5, 10]  
    }  
    
    # Recherche des meilleurs hyperparamètres  
    model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)  
    model.fit(X, y)  
    
    return model  

# Fonction pour prédire le résultat  
def predict_match(model, teamA_stats, teamB_stats):  
    data = {**teamA_stats, **teamB_stats}  
    df = pd.DataFrame([data])  
    prediction = model.predict(df)[0]  
    return prediction  

# Interface Streamlit  
st.title("⚽ Prédiction de Matchs de Football")  

# Onglets pour la navigation  
tab1, tab2 = st.tabs(["Formulaire", "Performances des Équipes"])  

# Générer des données factices et entraîner le modèle  
df = generate_fake_data()  
model = train_model(df)  

# Onglet Formulaire  
with tab1:  
    st.header("Statistiques des Équipes")  

    # Forme récente (5 derniers matchs)  
    st.subheader("Forme récente (5 derniers matchs)")  
    col1, col2 = st.columns(2)  
    with col1:  
        teamA_form = [st.selectbox(f"Match {i+1} - Équipe A", ["Victoire", "Défaite", "Match nul"]) for i in range(5)]  
    with col2:  
        teamB_form = [st.selectbox(f"Match {i+1} - Équipe B", ["Victoire", "Défaite", "Match nul"]) for i in range(5)]  

    # Conditions face à face  
    st.subheader("Conditions face à face")  
    head_to_head = st.number_input("Nombre de victoires de l'Équipe A face à l'Équipe B", min_value=0)  

    # Statistiques générales  
    st.subheader("Statistiques générales")  
    col1, col2 = st.columns(2)  
    with col1:  
        teamA_possession = st.number_input("Possession de l'Équipe A (%)", min_value=0, max_value=100)  
        teamA_passes = st.number_input("Passes réussies de l'Équipe A", min_value=0)  
        teamA_tacles = st.number_input("Tacles réussis de l'Équipe A", min_value=0)  
        teamA_fautes = st.number_input("Fautes de l'Équipe A", min_value=0)  
        teamA_yellow_cards = st.number_input("Cartons jaunes de l'Équipe A", min_value=0)  
        teamA_red_cards = st.number_input("Cartons rouges de l'Équipe A", min_value=0)  
    with col2:  
        teamB_possession = st.number_input("Possession de l'Équipe B (%)", min_value=0, max_value=100)  
        teamB_passes = st.number_input("Passes réussies de l'Équipe B", min_value=0)  
        teamB_tacles = st.number_input("Tacles réussis de l'Équipe B", min_value=0)  
        teamB_fautes = st.number_input("Fautes de l'Équipe B", min_value=0)  
        teamB_yellow_cards = st.number_input("Cartons jaunes de l'Équipe B", min_value=0)  
        teamB_red_cards = st.number_input("Cartons rouges de l'Équipe B", min_value=0)  

    # Statistiques d'attaque  
    st.subheader("Statistiques d'attaque")  
    col1, col2 = st.columns(2)  
    with col1:  
        teamA_goals = st.number_input("Buts de l'Équipe A", min_value=0)  
        teamA_xg = st.number_input("xG de l'Équipe A", min_value=0.0)  
        teamA_shots = st.number_input("Tirs cadrés de l'Équipe A", min_value=0)  
        teamA_corners = st.number_input("Corners de l'Équipe A", min_value=0)  
        teamA_touches = st.number_input("Touches dans la surface adverse de l'Équipe A", min_value=0)  
        teamA_penalties = st.number_input("Pénalités obtenues de l'Équipe A", min_value=0)  
    with col2:  
        teamB_goals = st.number_input("Buts de l'Équipe B", min_value=0)  
        teamB_xg = st.number_input("xG de l'Équipe B", min_value=0.0)  
        teamB_shots = st.number_input("Tirs cadrés de l'Équipe B", min_value=0)  
        teamB_corners = st.number_input("Corners de l'Équipe B", min_value=0)  
        teamB_touches = st.number_input("Touches dans la surface adverse de l'Équipe B", min_value=0)  
        teamB_penalties = st.number_input("Pénalités obtenues de l'Équipe B", min_value=0)  

    # Statistiques de défense  
    st.subheader("Statistiques de défense")  
    col1, col2 = st.columns(2)  
    with col1:  
        teamA_xg_conceded = st.number_input("xG concédés de l'Équipe A", min_value=0.0)  
        teamA_interceptions = st.number_input("Interceptions de l'Équipe A", min_value=0)  
        teamA_clearances = st.number_input("Dégagements de l'Équipe A", min_value=0)  
        teamA_saves = st.number_input("Arrêts de l'Équipe A", min_value=0)  
        teamA_goals_conceded = st.number_input("Buts concédés de l'Équipe A", min_value=0)  
        teamA_clean_sheet = st.selectbox("Aucun but encaissé de l'Équipe A", ["Oui", "Non"])  
    with col2:  
        teamB_xg_conceded = st.number_input("xG concédés de l'Équipe B", min_value=0.0)  
        teamB_interceptions = st.number_input("Interceptions de l'Équipe B", min_value=0)  
        teamB_clearances = st.number_input("Dégagements de l'Équipe B", min_value=0)  
        teamB_saves = st.number_input("Arrêts de l'Équipe B", min_value=0)  
        teamB_goals_conceded = st.number_input("Buts concédés de l'Équipe B", min_value=0)  
        teamB_clean_sheet = st.selectbox("Aucun but encaissé de l'Équipe B", ["Oui", "Non"])  

    # Bouton pour prédire  
    if st.button("Prédire le Résultat"):  
        # Calculer le score de forme  
        teamA_form_score = teamA_form.count("Victoire") * 3 + teamA_form.count("Match nul")  
        teamB_form_score = teamB_form.count("Victoire") * 3 + teamB_form.count("Match nul")  

        # Préparer les données pour la prédiction  
        teamA_stats = {  
            'teamA_goals': teamA_goals,  
            'teamA_possession': teamA_possession,  
            'teamA_xg': teamA_xg,  
            'teamA_shots': teamA_shots,  
            'teamA_corners': teamA_corners,  
            'teamA_interceptions': teamA_interceptions,  
            'teamA_saves': teamA_saves,  
            'teamA_clean_sheet': 1 if teamA_clean_sheet == "Oui" else 0,  
            'teamA_form': teamA_form_score,  
            'head_to_head': head_to_head  
        }  
        teamB_stats = {  
            'teamB_goals': teamB_goals,  
            'teamB_possession': teamB_possession,  
            'teamB_xg': teamB_xg,  
            'teamB_shots': teamB_shots,  
            'teamB_corners': teamB_corners,  
            'teamB_interceptions': teamB_interceptions,  
            'teamB_saves': teamB_saves,  
            'teamB_clean_sheet': 1 if teamB_clean_sheet == "Oui" else 0,  
            'teamB_form': teamB_form_score  
        }  
        
        # Faire la prédiction  
        prediction = predict_match(model, teamA_stats, teamB_stats)  
        result = {0: "Victoire de l'Équipe A", 1: "Victoire de l'Équipe B", 2: "Match nul"}[prediction]  
        
        # Afficher le résultat  
        st.success(f"Résultat prédit : {result}")  

# Onglet Performances des Équipes  
with tab2:  
    st.header("Performances des Équipes")  

    # Tableau récapitulatif  
    st.subheader("Tableau des Performances")  
    performance_data = {  
        'Équipe': ['Équipe A', 'Équipe B'],  
        'Buts': [teamA_goals, teamB_goals],  
        'Possession (%)': [teamA_possession, teamB_possession],  
        'xG': [teamA_xg, teamB_xg],  
        'Tirs cadrés': [teamA_shots, teamB_shots],  
        'Corners': [teamA_corners, teamB_corners],  
        'Interceptions': [teamA_interceptions, teamB_interceptions],  
        'Arrêts': [teamA_saves, teamB_saves],  
        'Aucun but encaissé': [teamA_clean_sheet, teamB_clean_sheet]  
    }  
    performance_df = pd.DataFrame(performance_data)  
    st.dataframe(performance_df)  

    # Graphique des scores de forme  
    st.subheader("Scores de Forme")  
    teamA_form_score = teamA_form.count("Victoire") * 3 + teamA_form.count("Match nul")  
    teamB_form_score = teamB_form.count("Victoire") * 3 + teamB_form.count("Match nul")  
    fig = px.bar(x=['Équipe A', 'Équipe B'], y=[teamA_form_score, teamB_form_score], labels={'x': 'Équipe', 'y': 'Score de Forme'})  
    st.plotly_chart(fig)
