# Fonction pour calculer la cible avec Double Chance
def generate_target_with_double_chance(match):
    # Si l'équipe 1 gagne ou il y a match nul, c'est un Double Chance pour l'équipe 1
    if match['score1'] > match['score2']:  # L'équipe à domicile gagne
        return 1  # Double Chance équipe 1
    elif match['score1'] < match['score2']:  # L'équipe à l'extérieur gagne
        return 2  # Double Chance équipe 2
    else:  # Match nul
        return 'X'  # Double Chance match nul

# Fonction pour générer les prédictions avec Double Chance
def generate_predictions_with_double_chance(team1_data, team2_data):
    # Préparer les données d'entrée (variables des deux équipes)
    X = pd.DataFrame([team1_data + team2_data], columns=team1_data.keys() + team2_data.keys())
    
    # Générer la cible en fonction des scores simulés (à remplacer par des données réelles)
    y = np.array([generate_target_with_double_chance({'score1': 2, 'score2': 1})])  # Exemple avec une cible fictive pour test
    
    # Appliquer la régression logistique et Random Forest comme précédemment...
    logreg = LogisticRegression()
    logreg.fit(X, y)
    logreg_prediction = logreg.predict(X)

    # Random Forest
    rf = RandomForestClassifier()
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)

    # Calcul des probabilités Poisson (à ajuster selon le cas)
    poisson_team1_prob = poisson.pmf(2, team1_data['⚽xG'])
    poisson_team2_prob = poisson.pmf(2, team2_data['⚽xG'])

    return poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf.mean()

# Données fictives des équipes avec Double Chance
team1_data = {
    '🧑‍💼 Nom de l\'équipe': 'Équipe 1',
    '⚽ Attaque': 0.8,
    '🛡️ Défense': 0.6,
    '🔥 Forme récente': 0.7,
    '💔 Blessures': 0.3,
    '💪 Motivation': 0.9,
    '🔄 Tactique': 0.75,
    '📊 Historique face à face': 0.5,
    '⚽xG': 1.4,
    '🔝 Nombre de corners': 5,
    '🏠 Buts à domicile': 1.5,
    '✈️ Buts à l\'extérieur': 1.2
}

team2_data = {
    '🧑‍💼 Nom de l\'équipe': 'Équipe 2',
    '⚽ Attaque': 0.6,
    '🛡️ Défense': 0.8,
    '🔥 Forme récente': 0.7,
    '💔 Blessures': 0.4,
    '💪 Motivation': 0.8,
    '🔄 Tactique': 0.7,
    '📊 Historique face à face': 0.6,
    '⚽xG': 1.2,
    '🔝 Nombre de corners': 6,
    '🏠 Buts à domicile': 1.1,
    '✈️ Buts à l\'extérieur': 1.3
}

# Calcul des forces
attack_strength_1, defense_strength_1 = calculate_forces(team1_data)
attack_strength_2, defense_strength_2 = calculate_forces(team2_data)

# Affichage des résultats
st.title("Analyse du Match ⚽")
st.write("### Team 1 vs Team 2")

st.write("### Variables pour Équipe 1:")
st.write(f"🧑‍💼 Nom de l'équipe : {team1_data['🧑‍💼 Nom de l\'équipe']}")
st.write(f"⚽ Force d'attaque : {team1_data['⚽ Attaque']}")
st.write(f"🛡️ Force de défense : {team1_data['🛡️ Défense']}")
st.write(f"🔥 Forme récente : {team1_data['🔥 Forme récente']}")
st.write(f"💔 Blessures : {team1_data['💔 Blessures']}")
st.write(f"💪 Motivation : {team1_data['💪 Motivation']}")
st.write(f"🔄 Tactique : {team1_data['🔄 Tactique']}")
st.write(f"📊 Historique face à face : {team1_data['📊 Historique face à face']}")
st.write(f"⚽xG : {team1_data['⚽xG']}")
st.write(f"🔝 Nombre de corners : {team1_data['🔝 Nombre de corners']}")
st.write(f"🏠 Buts à domicile : {team1_data['🏠 Buts à domicile']}")
st.write(f"✈️ Buts à l'extérieur : {team1_data['✈️ Buts à l\'extérieur']}")

st.write("### Variables pour Équipe 2:")
st.write(f"🧑‍💼 Nom de l'équipe : {team2_data['🧑‍💼 Nom de l\'équipe']}")
st.write(f"⚽ Force d'attaque : {team2_data['⚽ Attaque']}")
st.write(f"🛡️ Force de défense : {team2_data['🛡️ Défense']}")
st.write(f"🔥 Forme récente : {team2_data['🔥 Forme récente']}")
st.write(f"💔 Blessures : {team2_data['💔 Blessures']}")
st.write(f"💪 Motivation : {team2_data['💪 Motivation']}")
st.write(f"🔄 Tactique : {team2_data['🔄 Tactique']}")
st.write(f"📊 Historique face à face : {team2_data['📊 Historique face à face']}")
st.write(f"⚽xG : {team2_data['⚽xG']}")
st.write(f"🔝 Nombre de corners : {team2_data['🔝 Nombre de corners']}")
st.write(f"🏠 Buts à domicile : {team2_data['🏠 Buts à domicile']}")
st.write(f"✈️ Buts à l'extérieur : {team2_data['✈️ Buts à l\'extérieur']}")

# Prédictions avec Double Chance
poisson_team1_prob, poisson_team2_prob, logreg_prediction, cv_scores_rf_mean = generate_predictions_with_double_chance(team1_data, team2_data)

st.write(f"⚽ Probabilité Poisson de l'Équipe 1 : {poisson_team1_prob}")
st.write(f"⚽ Probabilité Poisson de l'Équipe 2 : {poisson_team2_prob}")
st.write(f"📊 Prédiction de la régression logistique : {logreg_prediction}")
st.write(f"📊 Moyenne des scores de validation croisée (Random Forest) : {cv_scores_rf_mean:.2f}")
