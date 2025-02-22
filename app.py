import streamlit as st  
import pandas as pd  
import numpy as np  
import plotly.graph_objs as go  
from scipy.stats import poisson  
import random  

# Configuration de la page  
st.set_page_config(  
    page_title="⚽ Prédicteur de Matchs de Football",  
    page_icon="⚽",  
    layout="wide",  
    initial_sidebar_state="expanded"  
)  

# Style personnalisé  
st.markdown("""  
<style>  
.stApp {  
    background-color: #f0f2f6;  
}  
.big-font {  
    font-size:20px !important;  
    font-weight:bold;  
    color:#2C3E50;  
}  
.highlight {  
    background-color:#F1C40F;  
    padding:10px;  
    border-radius:10px;  
    text-align:center;  
}  
.stButton>button {  
    background-color:#3498DB;  
    color:white;  
    font-weight:bold;  
    width:100%;  
}  
.stButton>button:hover {  
    background-color:#2980B9;  
}  
</style>  
""", unsafe_allow_html=True)  

# Base de données des équipes (fictive mais structurée)  
EQUIPES = {  
    "Real Madrid": {  
        "attaque": 8.5,  
        "defense": 8.0,  
        "forme": 7.5,  
        "domicile": 0.3,  # Bonus domicile  
    },  
    "Barcelona": {  
        "attaque": 8.0,  
        "defense": 7.5,  
        "forme": 7.0,  
        "domicile": 0.2,  
    },  
    "Manchester City": {  
        "attaque": 9.0,  
        "defense": 8.5,  
        "forme": 8.0,  
        "domicile": 0.25,  
    },  
    "Liverpool": {  
        "attaque": 8.5,  
        "defense": 8.0,  
        "forme": 7.5,  
        "domicile": 0.3,  
    },  
    "Bayern Munich": {  
        "attaque": 8.7,  
        "defense": 8.2,  
        "forme": 7.8,  
        "domicile": 0.28,  
    },  
    "PSG": {  
        "attaque": 8.3,  
        "defense": 7.7,  
        "forme": 7.2,  
        "domicile": 0.2,  
    }  
}  

# Fonction de prédiction de Poisson  
def predire_buts_poisson(force_attaque, force_defense):  
    """  
    Calcule la probabilité de buts selon la distribution de Poisson  
    """  
    # Calcul de l'espérance de buts  
    esperance_buts = force_attaque * (1 - force_defense/10)  
    
    # Limiter l'espérance entre 0.5 et 3.5  
    esperance_buts = max(0.5, min(esperance_buts, 3.5))  
    
    # Calculer les probabilités pour 0 à 5 buts  
    probabilities = [poisson.pmf(k, esperance_buts) for k in range(6)]  
    
    return {  
        'esperance': esperance_buts,  
        'probabilities': probabilities  
    }  

# Visualisation des probabilités de buts  
def visualiser_probabilites_buts(equipe, probabilities):  
    """  
    Crée un graphique des probabilités de buts  
    """  
    fig = go.Figure(data=[  
        go.Bar(  
            x=list(range(len(probabilities))),   
            y=probabilities,   
            text=[f'{p:.2%}' for p in probabilities],  
            textposition='auto',  
            name=equipe  
        )  
    ])  
    fig.update_layout(  
        title=f'Probabilités de Buts pour {equipe}',  
        xaxis_title='Nombre de Buts',  
        yaxis_title='Probabilité',  
        template='plotly_white'  
    )  
    return fig  

# Fonction de prédiction du résultat  
def predire_resultat(stats_domicile, stats_exterieur):  
    """  
    Prédit le résultat du match basé sur les statistiques  
    """  
    # Calcul des forces relatives  
    force_domicile = (  
        stats_domicile['attaque'] +   
        stats_domicile['forme'] +   
        stats_domicile['domicile']  
    )  
    
    force_exterieur = (  
        stats_exterieur['attaque'] +   
        stats_exterieur['forme']  
    )  
    
    # Calcul de la probabilité de victoire  
    prob_victoire_domicile = force_domicile / (force_domicile + force_exterieur)  
    
    # Générer un résultat aléatoire pondéré  
    resultat = random.choices(  
        ['Victoire Domicile', 'Victoire Extérieur', 'Nul'],   
        weights=[prob_victoire_domicile, 1-prob_victoire_domicile, 0.2]  
    )[0]  
    
    return resultat, prob_victoire_domicile  

# Interface principale  
def main():  
    st.title("🏆 Prédicteur Avancé de Matchs de Football")  
    
    # Sidebar pour les paramètres  
    st.sidebar.header("🛠️ Paramètres du Match")  
    
    # Sélection des équipes  
    equipes = list(EQUIPES.keys())  
    col1, col2 = st.sidebar.columns(2)  
    
    with col1:  
        equipe_domicile = st.selectbox("Équipe Domicile", equipes)  
    with col2:  
        equipe_exterieur = st.selectbox("Équipe Extérieure", [e for e in equipes if e != equipe_domicile])  
    
    # Bouton de prédiction  
    if st.sidebar.button("🎲 Prédire le Match", use_container_width=True):  
        # Récupérer les statistiques des équipes  
        stats_domicile = EQUIPES[equipe_domicile]  
        stats_exterieur = EQUIPES[equipe_exterieur]  
        
        # Prédiction des buts  
        buts_domicile = predire_buts_poisson(  
            stats_domicile['attaque'],   
            stats_exterieur['defense']  
        )  
        
        buts_exterieur = predire_buts_poisson(  
            stats_exterieur['attaque'],   
            stats_domicile['defense']  
        )  
        
        # Prédiction du résultat  
        resultat, prob_victoire = predire_resultat(stats_domicile, stats_exterieur)  
        
        # Affichage des résultats  
        st.header(f"🏟️ {equipe_domicile} vs {equipe_exterieur}")  
        
        # Colonnes pour les graphiques  
        col1, col2 = st.columns(2)  
        
        with col1:  
            st.plotly_chart(visualiser_probabilites_buts(equipe_domicile, buts_domicile['probabilities']))  
        
        with col2:  
            st.plotly_chart(visualiser_probabilites_buts(equipe_exterieur, buts_exterieur['probabilities']))  
        
        # Résultat du match  
        st.subheader("🏆 Résultat Probable")  
        
        # Déterminer l'emoji et le style du résultat  
        if resultat == 'Victoire Domicile':  
            emoji = "🥇"  
            message_resultat = f"🏆 Victoire de {equipe_domicile}"  
        elif resultat == 'Victoire Extérieur':  
            emoji = "🥇"  
            message_resultat = f"🏆 Victoire de {equipe_exterieur}"  
        else:  
            emoji = "🤝"  
            message_resultat = "🤝 Match Nul"  
        
        # Affichage du résultat  
        st.markdown(f"<div class='highlight'>{emoji} {message_resultat} {emoji}</div>", unsafe_allow_html=True)  
        
        # Probabilités détaillées  
        st.subheader("📊 Probabilités Détaillées")  
        
        col1, col2 = st.columns(2)  
        
        with col1:  
            st.metric(f"Probabilité Victoire {equipe_domicile}", f"{prob_victoire:.2%}")  
            st.metric("Buts Probables", f"{np.argmax(buts_domicile['probabilities'])}")  
        
        with col2:  
            st.metric(f"Probabilité Victoire {equipe_exterieur}", f"{1-prob_victoire:.2%}")  
            st.metric("Buts Probables", f"{np.argmax(buts_exterieur['probabilities'])}")  

# Exécution de l'application  
if __name__ == "__main__":  
    main()
