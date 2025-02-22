import streamlit as st  
import pandas as pd  
import numpy as np  
import plotly.express as px  
import plotly.graph_objs as go  
from scipy.stats import poisson  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import StandardScaler  

# Configuration de la page Streamlit  
st.set_page_config(  
    page_title="⚽ Prédicteur de Matchs de Football",  
    page_icon="⚽",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)  

# Style personnalisé  
st.markdown("""  
<style>  
.big-font {  
    font-size:20px !important;  
    font-weight:bold;  
    color:#2C3E50;  
}  
.highlight {  
    background-color:#F1C40F;  
    padding:10px;  
    border-radius:10px;  
}  
.stButton>button {  
    background-color:#3498DB;  
    color:white;  
    font-weight:bold;  
}  
.stButton>button:hover {  
    background-color:#2980B9;  
}  
</style>  
""", unsafe_allow_html=True)  

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
        'teamA_form': np.random.randint(0, 15, 100),  
        'teamB_form': np.random.randint(0, 15, 100),  
        'head_to_head': np.random.randint(0, 10, 100),  
        'result': np.random.choice(['A', 'B', 'D'], 100)  
    }  
    df = pd.DataFrame(data)  
    df['result'] = df['result'].map({'A': 0, 'B': 1, 'D': 2})  
    return df  

# Entraînement du modèle avec GridSearchCV  
def train_model(df):  
    # Séparer les features et la cible  
    X = df.drop('result', axis=1)  
    y = df['result']  
    
    # Normaliser les features  
    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(X)  
    
    # Hyperparamètres à tester  
    param_grid = {  
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 10, 20],  
        'min_samples_split': [2, 5, 10]  
    }  
    
    # Recherche des meilleurs hyperparamètres  
    model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)  
    model.fit(X_scaled, y)  
    
    return model, scaler  

# Fonction de prédiction de Poisson pour les buts  
def predict_poisson_goals(team_avg_goals, opponent_avg_defense, max_goals=6):  
    """  
    Calcule la probabilité de marquer un nombre de buts selon la distribution de Poisson  
    """  
    # Ajuster l'espérance de buts en fonction de la défense de l'adversaire  
    adjusted_goals = team_avg_goals * (1 - opponent_avg_defense/10)  
    
    probabilities = [poisson.pmf(k, adjusted_goals) for k in range(max_goals)]  
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(max_goals)]  
    
    return {  
        'probabilities': probabilities,  
        'cumulative_probabilities': cumulative_probabilities  
    }  

# Visualisation des probabilités de buts  
def plot_goal_probabilities(team_name, probabilities):  
    """  
    Crée un graphique des probabilités de buts  
    """  
    fig = go.Figure(data=[  
        go.Bar(  
            x=list(range(len(probabilities))),   
            y=probabilities,   
            text=[f'{p:.2%}' for p in probabilities],  
            textposition='auto',  
            name=team_name  
        )  
    ])  
    fig.update_layout(  
        title=f'Probabilités de Buts pour {team_name}',  
        xaxis_title='Nombre de Buts',  
        yaxis_title='Probabilité',  
        template='plotly_white'  
    )  
    return fig  

# Fonction pour prédire le résultat du modèle de machine learning  
def predict_match_ml(model, scaler, teamA_stats, teamB_stats):  
    # Créer un DataFrame avec toutes les colonnes nécessaires  
    columns = [  
        'teamA_goals', 'teamB_goals',   
        'teamA_possession', 'teamB_possession',   
        'teamA_xg', 'teamB_xg',   
        'teamA_shots', 'teamB_shots',   
        'teamA_corners', 'teamB_corners',   
        'teamA_interceptions', 'teamB_interceptions',   
        'teamA_saves', 'teamB_saves',   
        'teamA_clean_sheet', 'teamB_clean_sheet',   
        'teamA_form', 'teamB_form',   
        'head_to_head'  
    ]  
    
    # Combiner les statistiques des deux équipes  
    data = {col: teamA_stats.get(col, teamB_stats.get(col, 0)) for col in columns}  
    
    # Créer le DataFrame  
    df = pd.DataFrame([data])  
    
    # Normaliser les données  
    df_scaled = scaler.transform(df)  
    
    # Faire la prédiction  
    prediction = model.predict(df_scaled)[0]  
    return prediction  

# Interface principale  
def main():  
    st.title("🏆 Prédicteur Avancé de Matchs de Football")  
    
    # Sidebar pour les paramètres  
    st.sidebar.header("🛠️ Paramètres du Match")  
    
    # Sélection des équipes (à remplacer par une vraie base de données)  
    equipes = ["Real Madrid", "Barcelona", "Manchester City", "Liverpool", "Bayern Munich", "PSG"]  
    col1, col2 = st.sidebar.columns(2)  
    
    with col1:  
        equipe_domicile = st.selectbox("Équipe Domicile", equipes)  
    with col2:  
        equipe_exterieur = st.selectbox("Équipe Extérieure", [e for e in equipes if e != equipe_domicile])  
    
    # Statistiques de base  
    st.sidebar.subheader("📊 Statistiques Clés")  
    
    # Colonnes pour les statistiques  
    col1, col2 = st.sidebar.columns(2)  
    
    with col1:  
        forme_domicile = st.slider("Forme Domicile", 0, 10, 7)  
        attaque_domicile = st.slider("Force Attaque Domicile", 0, 10, 8)  
        defense_domicile = st.slider("Force Défense Domicile", 0, 10, 7)  
    
    with col2:  
        forme_exterieur = st.slider("Forme Extérieur", 0, 10, 5)  
        attaque_exterieur = st.slider("Force Attaque Extérieur", 0, 10, 6)  
        defense_exterieur = st.slider("Force Défense Extérieur", 0, 10, 6)  
    
    # Bouton de prédiction  
    if st.sidebar.button("🎲 Prédire le Match", use_container_width=True):  
        # Générer des données factices et entraîner le modèle  
        df = generate_fake_data()  
        model, scaler = train_model(df)  
        
        # Calcul des probabilités de buts  
        domicile_goals = predict_poisson_goals(attaque_domicile, defense_exterieur)  
        exterieur_goals = predict_poisson_goals(attaque_exterieur, defense_domicile)  
        
        # Affichage des résultats  
        st.header(f"🏟️ {equipe_domicile} vs {equipe_exterieur}")  
        
        # Colonnes pour les graphiques  
        col1, col2 = st.columns(2)  
        
        with col1:  
            st.plotly_chart(plot_goal_probabilities(equipe_domicile, domicile_goals['probabilities']))  
        
        with col2:  
            st.plotly_chart(plot_goal_probabilities(equipe_exterieur, exterieur_goals['probabilities']))  
        
        # Calcul du résultat le plus probable  
        max_domicile = np.argmax(domicile_goals['probabilities'])  
        max_exterieur = np.argmax(exterieur_goals['probabilities'])  
        
        # Préparation des statistiques pour le modèle ML  
        teamA_stats = {  
            'teamA_goals': max_domicile,  
            'teamA_possession': forme_domicile * 10,  
            'teamA_xg': attaque_domicile,  
            'teamA_shots': attaque_domicile * 2,  
            'teamA_corners': attaque_domicile,  
            'teamA_interceptions': defense_domicile,  
            'teamA_saves': defense_domicile,  
            'teamA_clean_sheet': 1 if defense_domicile > 7 else 0,  
            'teamA_form': forme_domicile,  
            'head_to_head': 5  # Valeur arbitraire à remplacer par des données réelles  
        }  
        
        teamB_stats = {  
            'teamB_goals': max_exterieur,  
            'teamB_possession': forme_exterieur * 10,  
            'teamB_xg': attaque_exterieur,  
            'teamB_shots': attaque_exterieur * 2,  
            'teamB_corners': attaque_exterieur,  
            'teamB_interceptions': defense_exterieur,  
            'teamB_saves': defense_exterieur,  
            'teamB_clean_sheet': 1 if defense_exterieur > 7 else 0,  
            'teamB_form': forme_exterieur  
        }  
        
        # Prédiction du résultat par machine learning  
        ml_prediction = predict_match_ml(model, scaler, teamA_stats, teamB_stats)  
        
        # Résultat du match  
        st.subheader("🏆 Résultat Probable")  
        
        if ml_prediction == 0:  
            resultat = f"🏆 Victoire de {equipe_domicile}"  
            emoji = "🥇"  
        elif ml_prediction == 1:  
            resultat = f"🏆 Victoire de {equipe_exterieur}"  
            emoji = "🥇"  
        else:  
            resultat = "🤝 Match Nul"  
            emoji = "🤝"  
        
        st.markdown(f"<div class='highlight'>{emoji} {resultat} {emoji}</div>", unsafe_allow_html=True)  
        
        # Probabilités détaillées  
        st.subheader("📊 Probabilités Détaillées")  
        
        col1, col2 = st.columns(2)  
        
        with col1:  
            st.metric(f"Probabilité Victoire {equipe_domicile}", f"{max(domicile_goals['probabilities']):.2%}")  
        
        with col2:  
            st.metric(f"Probabilité Victoire {equipe_exterieur}", f"{max(exterieur_goals['probabilities']):.2%}")  

# Exécution de l'application  
if __name__ == "__main__":  
    main()
