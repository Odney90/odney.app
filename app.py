import streamlit as st

# Titre de l'application
st.title("Analyse de Matchs de Football et Prédictions de Paris Sportifs")

# Saisie des données des équipes
st.header("Saisie des données des équipes")

# Variables pour l'équipe à domicile
st.subheader("Équipe à domicile")
home_team = st.text_input("Nom de l'équipe à domicile")
home_goals_scored = st.number_input("Buts marqués (total)", min_value=0, key="home_goals_scored")
home_goals_conceded = st.number_input("Buts encaissés (total)", min_value=0, key="home_goals_conceded")
home_shots = st.number_input("Tirs (total)", min_value=0, key="home_shots")
home_shots_on_target = st.number_input("Tirs cadrés", min_value=0, key="home_shots_on_target")
home_possession = st.slider("Possession de balle (%)", 0, 100, 50, key="home_possession")
home_pass_accuracy = st.slider("Précision des passes (%)", 0, 100, 75, key="home_pass_accuracy")
home_fouls_committed = st.number_input("Fautes commises", min_value=0, key="home_fouls_committed")
home_corners = st.number_input("Corners obtenus", min_value=0, key="home_corners")
home_yellow_cards = st.number_input("Cartons jaunes", min_value=0, key="home_yellow_cards")
home_red_cards = st.number_input("Cartons rouges", min_value=0, key="home_red_cards")
home_injuries = st.number_input("Joueurs blessés", min_value=0, key="home_injuries")
home_recent_form = st.number_input("Forme récente (points)", min_value=0, max_value=15, key="home_recent_form")
home_defensive_strength = st.number_input("Force défensive (1-10)", 1, 10, key="home_defensive_strength")
home_offensive_strength = st.number_input("Force offensive (1-10)", 1, 10, key="home_offensive_strength")
home_head_to_head = st.number_input("Historique des confrontations (points)", min_value=0, key="home_head_to_head")
home_manager_experience = st.number_input("Expérience du manager (années)", min_value=0, key="home_manager_experience")

# Variables pour l'équipe à l'extérieur
st.subheader("Équipe à l'extérieur")
away_team = st.text_input("Nom de l'équipe à l'extérieur")
away_goals_scored = st.number_input("Buts marqués (total)", min_value=0, key="away_goals_scored")
away_goals_conceded = st.number_input("Buts encaissés (total)", min_value=0, key="away_goals_conceded")
away_shots = st.number_input("Tirs (total)", min_value=0, key="away_shots")
away_shots_on_target = st.number_input("Tirs cadrés", min_value=0, key="away_shots_on_target")
away_possession = st.slider("Possession de balle (%)", 0, 100, 50, key="away_possession")
away_pass_accuracy = st.slider("Précision des passes (%)", 0, 100, 75, key="away_pass_accuracy")
away_fouls_committed = st.number_input("Fautes commises", min_value=0, key="away_fouls_committed")
away_corners = st.number_input("Corners obtenus", min_value=0, key="away_corners")
away_yellow_cards = st.number_input("Cartons jaunes", min_value=0, key="away_yellow_cards")
away_red_cards = st.number_input("Cartons rouges", min_value=0, key="away_red_cards")
away_injuries = st.number_input("Joueurs blessés", min_value=0, key="away_injuries")
away_recent_form = st.number_input("Forme récente (points)", min_value=0, max_value=15, key="away_recent_form")
away_defensive_strength = st.number_input("Force défensive (1-10)", 1, 10, key="away_defensive_strength")
away_offensive_strength = st.number_input("Force offensive (1-10)", 1, 10, key="away_offensive_strength")
away_head_to_head = st.number_input("Historique des confrontations (points)", min_value=0, key="away_head_to_head")
away_manager_experience = st.number_input("Expérience du manager (années)", min_value=0, key="away_manager_experience")

# Continuez avec le reste de votre logique...
