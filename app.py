0, key="poss_moyenne_B", step=0.1)  
    xG_B = st.number_input("⭐ Buts attendus (xG)", min_value=0.0, value=1.0, step=0.1, key="xG_B")  
    tirs_cadres_B = st.number_input("🎯 Tirs cadrés par match", min_value=0.0, value=8.0, key="tirs_cadres_B", step=0.1)  
    pourcentage_tirs_convertis_B = st.number_input("Convert Pourcentage de tirs convertis (%)", min_value=0.0, max_value=100.0, value=15.0, key="pourcentage_tirs_convertis_B", step=0.1)  

with col2:  
    grosses_occasions_B = st.number_input("✨ Grosses occasions", min_value=0.0, value=3.0, key="grosses_occasions_B", step=0.1)  
    grosses_occasions_ratees_B = st.number_input("💨 Grosses occasions ratées", min_value=0.0, value=1.0, key="grosses_occasions_ratees_B", step=0.1)  
    passes_reussies_B = st.number_input("✅ Passes réussies/match (sur 1000)", min_value=0.0, value=250.0, key="passes_reussies_B", step=0.1)  
    passes_longues_precises_B = st.number_input("➡️ Passes longues précises/match", min_value=0.0, value=10.0, key="passes_longues_precises_B", step=0.1)  
    centres_reussis_B = st.number_input("↗️ Centres réussis/match", min_value=0.0, value=3.0, key="centres_reussis_B", step=0.1)  
    penalties_obtenus_B = st.number_input("🎁 Pénalties obtenus", min_value=0.0, value=0.0, key="penalties_obtenus_B", step=0.1)  
    touches_surface_adverse_B = st.number_input("Entree Touches dans la surface adverse", min_value=0.0, value=8.0, key="touches_surface_adverse_B", step=0.1)  
    corners_B = st.number_input("Corner Nombre de corners", min_value=0.0, value=3.0, key="corners_B", step=0.1)  

col3, col4 = st.columns(2)  

with col3:  
    corners_par_match_B = st.number_input("Corner/Match Corners par match", min_value=0.0, value=1.0, key="corners_par_match_B", step=0.1)  
    corners_concedes_B = st.number_input("Corner Concédés Corners concédés/match", min_value=0.0, value=2.0, key="corners_concedes_B", step=0.1)  
    xG_concedes_B = st.number_input("⭐ xG concédés", min_value=0.0, value=1.5, step=0.1, key="xG_concedes_B")  
    interceptions_B = st.number_input("✋ Interceptions par match", min_value=0.0, value=8.0, key="interceptions_B", step=0.1)  
    tacles_reussis_B = st.number_input("Tacles Réussis Tacles réussis par match", min_value=0.0, value=4.0, key="tacles_reussis_B", step=0.1)  
    degagements_B = st.number_input("Dégagement Dégagements par match", min_value=0.0, value=6.0, key="degagements_B", step=0.1)  
    possessions_recuperees_B = st.number_input("Milieu Possessions récupérées au milieu/match", min_value=0.0, value=4.0, key="possessions_recuperees_B", step=0.1)  

with col4:  
    penalties_concedes_B = st.number_input("🎁 Pénalties concédés", min_value=0.0, value=1.0, key="penalties_concedes_B", step=0.1)  
    arrets_B = st.number_input("🧤 Arrêts par match", min_value=0.0, value=2.0, key="arrets_B", step=0.1)  
    fautes_B = st.number_input("Fautes Faute par match", min_value=0.0, value=12.0, key="fautes_B", step=0.1)  
    cartons_jaunes_B = st.number_input("🟨 Cartons jaunes", min_value=0.0, value=1.0, key="cartons_jaunes_B", step=0.1)  
    cartons_rouges_B = st.number_input("🟥 Cartons rouges", min_value=0.0, value=0.0, key="cartons_rouges_B", step=0.1)  
    tactique_B = st.text_input("Tactique de l'équipe B", value="4-2-3-1", key="tactique_B")  

# --- Section pour les joueurs clés de l'équipe B ---  
st.header("⭐ Joueurs Clés de l'équipe B")  
absents_B = st.number_input("Nombre de joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_B")  
ratings_B = []  
for i in range(5):  
    rating = st.number_input(f"Rating du joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_B_{i}", step=0.1)  
    ratings_B.append(rating)  

# --- Motivation des Équipes ---  
st.header("🔥 Motivation des Équipes")  
col1, col2 = st.columns(2)  

with col1:  
    motivation_A = st.number_input("Motivation de l'équipe A (1 à 10)", min_value=1.0, max_value=10.0, value=5.0, key="motivation_A", step=0.1)  
    st.write(f"Niveau de motivation de l'équipe A : **{motivation_A}**")  

with col2:  
    motivation_B = st.number_input("Motivation de l'équipe B (1 à 10)", min_value=1.0, max_value=10.0, value=5.0, key="motivation_B", step=0.1)  
    st.write(f"Niveau de motivation de l'équipe B : **{motivation_B}**")  

# --- Prédiction des buts avec la méthode de Poisson ---  
if 'xG_A' in locals() and 'xG_B' in locals():  
    buts_moyens_A, buts_moyens_B = prediction_buts_poisson(xG_A, xG_B)  
    afficher_resultats_poisson(buts_moyens_A, buts_moyens_B)  
else:  
    st.warning("Veuillez remplir les critères des équipes A et B pour activer la prédiction de Poisson.")  

# --- Prédiction multivariable avec régression logistique (version améliorée) ---  
st.header("🔮 Prédiction Multivariable (Régression Logistique)")  

# Poids pour chaque critère (ajuster selon l'importance)  
poids = {  
    'buts_produits': 0.15,  
    'buts_encaisses': -0.10,  
    'possession': 0.08,  
    'motivation': 0.12,  
    'absents': -0.15,  
    'rating_joueurs': 0.20,  
    'forme_recente': 0.20,  
    'historique_confrontations': 0.10,  # Nouveau critère  
    'victoires': 0.05,  # Poids pour l'historique général des victoires  
    'confrontations_directes': 0.10  # Poids pour les confrontations directes  
}  

# Historique des confrontations directes (1 pour victoire de A, -1 pour victoire de B, 0 pour nul)  
historique_confrontations = (victoires_A_contre_B - victoires_B_contre_A) / (victoires_A_contre_B + victoires_B_contre_A + nuls_AB) if (victoires_A_contre_B + victoires_B_contre_A + nuls_AB) > 0 else 0  

# Préparation des données pour le modèle  
X = np.array([  
    [total_buts_produits_A * poids['buts_produits'],  
     total_buts_encaisses_A * poids['buts_encaisses'],  
     poss_moyenne_A * poids['possession'],  
     motivation_A * poids['motivation'],  
     absents_A * poids['absents'],  
     np.mean(ratings_A) * poids['rating_joueurs'],  
     forme_recente_A * poids['forme_recente'],  
     historique_confrontations * poids['historique_confrontations'],  
     victoires_A * poids['victoires'],  # Ajout de l'historique des victoires  
     victoires_A_contre_B * poids['confrontations_directes']],  # Ajout des confrontations directes  

    [total_buts_produits_B * poids['buts_produits'],  
     total_buts_encaisses_B * poids['buts_encaisses'],  
     poss_moyenne_B * poids['possession'],  
     motivation_B * poids['motivation'],  
     absents_B * poids['absents'],  
     np.mean(ratings_B) * poids['rating_joueurs'],  
     forme_recente_B * poids['forme_recente'],  
     historique_confrontations * poids['historique_confrontations'],  
     victoires_B * poids['victoires'],  # Ajout de l'historique des victoires  
     victoires_B_contre_A * poids['confrontations_directes']]  # Ajout des confrontations directes  
])  

y = np.array([1, 0])  # 1 pour l'équipe A, 0 pour l'équipe B  

# Normalisation des données (important pour la régression logistique)  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Entraînement du modèle avec régularisation pour éviter le surapprentissage  
model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')  # L2 régularisation  
model.fit(X_scaled, y)  

# Prédiction des résultats  
prediction = model.predict_proba(X_scaled)  

# Affichage des probabilités de victoire  
st.write(f"Probabilité de victoire pour l'équipe A : **{prediction[0][1]:.2%}**")  
st.write(f"Probabilité de victoire pour l'équipe B : **{prediction[1][1]:.2%}**")  

# --- Interprétation des coefficients du modèle ---  
st.header("🧐 Interprétation du Modèle")  
feature_names = ['Buts produits totaux', 'Buts encaissés totaux', 'Possession moyenne',  
                 'Motivation', 'Absents', 'Rating joueurs', 'Forme récente',  
                 'Historique confrontations', 'Historique victoires', 'Confrontations directes']  
coefficients = model.coef_[0]  

for feature, coef in zip(feature_names, coefficients):  
    st.write(f"{feature}: Coefficient = {coef:.2f}")  

# --- Calcul des paris alternatifs ---  
double_chance_A = prediction[0][1] + (1 - prediction[1][1])  # Équipe A gagne ou match nul  
double_chance_B = prediction[1][1] + (1 - prediction[0][1])  # Équipe B gagne ou match nul  

# --- Affichage des paris alternatifs ---  
st.header("💰 Paris Alternatifs")  
st.write(f"Double chance pour l'équipe A (gagner ou match nul) : **{double_chance_A:.2%}**")  
st.write(f"Double chance pour l'équipe B (gagner ou match nul) : **{double_chance_B:.2%}**")
