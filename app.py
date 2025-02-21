import streamlit as st  
import requests  
import os  
from dotenv import load_dotenv  

# Charger les variables d'environnement depuis le fichier .env  
load_dotenv()  

# Récupération de la clé API depuis les variables d'environnement  
API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY")  

# Vérifier si la clé API est présente  
if not API_KEY:  
    st.error("Clé API non trouvée. Veuillez vérifier votre fichier .env.")  
    st.stop()  # Arrêter l'exécution de l'application si la clé API est manquante  

BASE_URL = "https://api.football-data.org/v4/"  # Utilisation de la v4 de l'API  

def get_matches(competition_code):  
    """Récupère les matchs d'une compétition spécifique."""  
    headers = {'X-Auth-Token': API_KEY}  
    url = f"{BASE_URL}/competitions/{competition_code}/matches"  
    try:  
        response = requests.get(url, headers=headers)  
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP  
        data = response.json()  
        return data['matches']  
    except requests.exceptions.RequestException as e:  
        st.error(f"Erreur lors de la récupération des données : {e}")  
        return []  

def display_matches(matches):  
    """Affiche les matchs dans Streamlit."""  
    if not matches:  
        st.warning("Aucun match trouvé.")  
        return  

    st.subheader("Matchs en direct :")  
    for match in matches:  
        home_team = match['homeTeam']['name']  
        away_team = match['awayTeam']['name']  
        score = match['score']['fullTime']  
        status = match['status']  
        match_date = match['utcDate']  

        st.write(f"**{home_team} vs {away_team}**")  
        st.write(f"Date: {match_date}")  
        st.write(f"Status: {status}")  
        if score:  
            st.write(f"Score: {score.get('homeTeam', '?')} - {score.get('awayTeam', '?')}")  
        else:  
            st.write("Score: N/A")  
        st.write("---")  

# Sélection de la compétition  
competition_code = st.selectbox(  
    "Sélectionnez la compétition :",  
    options=["PL (Premier League)", "SA (Serie A)", "BL1 (Bundesliga)", "FL1 (Ligue 1)", "PD (La Liga)"]  
)  

# Extraction du code de la compétition  
competition_code = competition_code.split(" ")[0]  

# Récupération et affichage des matchs  
matches = get_matches(competition_code)  
display_matches(matches)
