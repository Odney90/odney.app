import streamlit as st  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import poisson  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
import pandas as pd  
import streamlit as st  
import tools  # Importe le module tools.py  
# ... (autres imports)

# Configuration de la page Streamlit  
st.set_page_config(  
    page_title="Prédiction de Match de Football",  
    page_icon=":soccer:",  
    layout="wide",  
    initial_sidebar_state="expanded",  
)  

# Ajout de CSS personnalisé pour améliorer l'apparence  
st.markdown(  
    """  
    <style>  
        body {  
            color: #333;  
            background-color: #f4f4f4;  
        }  
        .stApp {  
            max-width: 1600px;  
            margin: 0 auto;  
            padding: 20px;  
        }  
        h1 {  
            color: #2E9AFE;  
            text-align: center;  
        }  
        h2 {  
            color: #2E9AFE;  
            border-bottom: 2px solid #2E9AFE;  
            padding-bottom: 5px;  
            margin-top: 20px;  
        }  
        h3 {  
            color: #3498DB;  
            margin-top: 15px;  
        }  
        .stNumberInput > label {  
            color: #3498DB;  
        }  
        .stSlider > label {  
            color: #3498DB;  
        }  
        .stSelectbox > label {  
            color: #3498DB;  
        }  
        .sidebar .sidebar-content {  
            background-color: #ECF0F1;  
            padding: 20px;  
        }  
        .stButton > button {  
            background-color: #2E9AFE;  
            color: white;  
            border: none;  
            padding: 10px 20px;  
            border-radius: 5px;  
            cursor: pointer;  
        }  
        .stButton > button:hover {  
            background-color: #3498DB;  
        }  
        .reportview-container .main .block-container {  
            padding-top: 20px;  
        }  
    </style>  
    """,  
    unsafe_allow_html=True,  
)  

# Titre principal  
st.title("⚽ Prédiction de Match de Football ⚽")  

# Introduction et guide d'utilisation  
st.sidebar.header("Guide d'Utilisation")  
st.sidebar.markdown(  
    """  
    Bienvenue dans l'application de prédiction de match de football !   
    
    **Comment ça marche :**  
    1.  Renseignez les données des équipes dans les différentes sections.  
    2.  Les résultats et analyses sont mis à jour automatiquement.  
    3.  Utilisez les informations fournies pour prendre des décisions éclairées.  
    
    **Conseils :**  
    *   Soyez précis dans les données entrées pour une meilleure prédiction.  
    *   Consultez l'interprétation des coefficients pour comprendre l'influence de chaque facteur.  
    """  
)
# Historique des équipes  
st.header("📊 Historique des Équipes")  
st.markdown("Entrez les statistiques de la saison pour chaque équipe.")  

col1, col2 = st.columns(2)  

with col1:  
    st.subheader("Équipe A")  
    victoires_A = st.number_input("Victoires", min_value=0, value=10, key="victoires_A")  
    nuls_A = st.number_input("Nuls", min_value=0, value=5, key="nuls_A")  
    defaites_A = st.number_input("Défaites", min_value=0, value=5, key="defaites_A")  

with col2:  
    st.subheader("Équipe B")  
    victoires_B = st.number_input("Victoires", min_value=0, value=8, key="victoires_B")  
    nuls_B = st.number_input("Nuls", min_value=0, value=7, key="nuls_B")  
    defaites_B = st.number_input("Défaites", min_value=0, value=5, key="defaites_B")  

# Historique des 5 derniers matchs  
st.header("⚽ Historique des 5 Derniers Matchs")  
st.markdown("Entrez les résultats des 5 derniers matchs (1=Victoire, 0=Nul, -1=Défaite).")  

col3, col4 = st.columns(2)  

with col3:  
    st.subheader("Équipe A")  
    historique_A = []  
    for i in range(5):  
        resultat_A = st.selectbox(  
            f"Match {i + 1}",  
            options=["Victoire (1)", "Nul (0)", "Défaite (-1)"],  
            key=f"match_A_{i}",  
        )  
        historique_A.append(1 if resultat_A == "Victoire (1)" else (0 if resultat_A == "Nul (0)" else -1))  

with col4:  
    st.subheader("Équipe B")  
    historique_B = []  
    for i in range(5):  
        resultat_B = st.selectbox(  
            f"Match {i + 1}",  
            options=["Victoire (1)", "Nul (0)", "Défaite (-1)"],  
            key=f"match_B_{i}",  
        )  
        historique_B.append(1 if resultat_B == "Victoire (1)" else (0 if resultat_B == "Nul (0)" else -1))  

# Calcul de la forme récente  
def calculer_forme(historique):  
    return sum(historique) / len(historique)  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  

st.write(f"Forme récente de l'équipe A (moyenne sur 5 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 5 matchs) : **{forme_recente_B:.2f}**")  

# Visualisation de la forme des équipes  
st.header("📈 Visualisation de la Forme des Équipes")  
fig, ax = plt.subplots()  
ax.plot(range(1, 6), historique_A, marker='o', label='Équipe A', color='blue')  
ax.plot(range(1, 6), historique_B, marker='o', label='Équipe B', color='red')  
ax.set_xticks(range(1, 6))  
ax.set_xticklabels([f"Match {i}" for i in range(1, 6)], rotation=45)  
ax.set_ylim(-1.5, 1.5)  
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')  
ax.set_ylabel('Résultat')  
ax.set_title('Historique des Performances des Équipes (5 Derniers Matchs)')  
ax.legend()  
st.pyplot(fig)
# Critères détaillés pour les équipes  
st.header("🔍 Critères Détaillés des Équipes")  
st.markdown("Entrez les statistiques détaillées pour chaque équipe.")  

col5, col6 = st.columns(2)  

with col5:  
    st.subheader("Équipe A")  
    buts_produits_A = st.slider("Buts produits (moyenne/saison)", min_value=0, max_value=10, value=2, key="buts_produits_A")  
    buts_encaisse_A = st.slider("Buts encaissés (moyenne/saison)", min_value=0, max_value=10, value=1, key="buts_encaisse_A")  
    poss_moyenne_A = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
    xG_A = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.5, key="xG_A")  
    tirs_cadres_A = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=10, key="tirs_cadres_A")  
    pourcentage_tirs_convertis_A = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=20, key="pourcentage_tirs_convertis_A")  
    grosses_occasions_A = st.number_input("Grosses occasions", min_value=0, value=5, key="grosses_occasions_A")  
    grosses_occasions_ratees_A = st.number_input("Grosses occasions ratées", min_value=0, value=2, key="grosses_occasions_ratees_A")  
    passes_reussies_A = st.slider("Passes réussies/match (sur 1000)", min_value=0, max_value=1000, value=300, key="passes_reussies_A")  
    passes_longues_precises_A = st.slider("Passes longues précises/match", min_value=0, max_value=100, value=15, key="passes_longues_precises_A")  
    centres_reussis_A = st.slider("Centres réussis/match", min_value=0, max_value=50, value=5, key="centres_reussis_A")  
    penalties_obtenus_A = st.number_input("Pénalties obtenus", min_value=0, value=1, key="penalties_obtenus_A")  
    touches_surface_adverse_A = st.number_input("Touches dans la surface adverse", min_value=0, value=10, key="touches_surface_adverse_A")  
    corners_A = st.number_input("Nombre de corners", min_value=0, value=4, key="corners_A")  
    corners_par_match_A = st.number_input("Corners par match", min_value=0, value=2, key="corners_par_match_A")  
    corners_concedes_A = st.number_input("Corners concédés par match", min_value=0, value=3, key="corners_concedes_A")  
    xG_concedes_A = st.slider("xG concédés", min_value=0.0, max_value=5.0, value=1.0, key="xG_concedes_A")  
    interceptions_A = st.slider("Interceptions par match", min_value=0, max_value=50, value=10, key="interceptions_A")  
    tacles_reussis_A = st.slider("Tacles réussis par match", min_value=0, max_value=50, value=5, key="tacles_reussis_A")  
    degagements_A = st.slider("Dégagements par match", min_value=0, max_value=50, value=8, key="degagements_A")  
    possessions_recuperees_A = st.slider("Possessions récupérées au milieu/match", min_value=0, max_value=50, value=5, key="possessions_recuperees_A")  
    penalties_concedes_A = st.number_input("Pénalties concédés", min_value=0, value=0, key="penalties_concedes_A")  
    arrets_A = st.slider("Arrêts par match", min_value=0, max_value=20, value=3, key="arrets_A")  
    fautes_A = st.slider("Fautes par match", min_value=0, max_value=50, value=10, key="fautes_A")  
    cartons_jaunes_A = st.number_input("Cartons jaunes", min_value=0, value=2, key="cartons_jaunes_A")  
    cartons_rouges_A = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_A")  
    tactique_A = st.text_input("Tactique de l'équipe A", value="4-3-3", key="tactique_A")  

    st.subheader("Joueurs Clés de l'équipe A")  
    absents_A = st.number_input("Nb joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_A")  
    ratings_A = []  
    for i in range(5):  
        rating = st.number_input(f"Rating joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_A_{i}")  
        ratings_A.append(rating)  

with col6:  
    st.subheader("Équipe B")  
    buts_produits_B = st.slider("Buts produits (moyenne/saison)", min_value=0, max_value=10, value=1, key="buts_produits_B")  
    buts_encaisse_B = st.slider("Buts encaissés (moyenne/saison)", min_value=0, max_value=10, value=2, key="buts_encaisse_B")  
    poss_moyenne_B = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
    xG_B = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.0, key="xG_B")  
    tirs_cadres_B = st.slider("Tirs cadrés par match", min_value=0, max_value=50, value=8, key="tirs_cadres_B")  
    pourcentage_tirs_convertis_B = st.slider("Pourcentage de tirs convertis (%)", min_value=0, max_value=100, value=15, key="pourcentage_tirs_convertis_B")  
    grosses_occasions_B = st.number_input("Grosses occasions", min_value=0, value=3, key="grosses_occasions_B")  
    grosses_occasions_ratees_B = st.number_input("Grosses occasions ratées", min_value=0, value=1, key="grosses_occasions_ratees_B")  
    passes_reussies_B = st.slider("Passes réussies/match (sur 1000)", min_value=0, max_value=1000, value=250, key="passes_reussies_B")  
    passes_longues_precises_B = st.slider("Passes longues précises/match", min_value=0, max_value=100, value=10, key="passes_longues_precises_B")  
    centres_reussis_B = st.slider("Centres réussis/match", min_value=0, max_value=50, value=3, key="centres_reussis_B")  
    penalties_obtenus_B = st.number_input("Pénalties obtenus", min_value=0, value=0, key="penalties_obtenus_B")  
    touches_surface_adverse_B = st.number_input("Touches dans la surface adverse", min_value=0, value=8, key="touches_surface_adverse_B")  
    corners_B = st.number_input("Nombre de corners", min_value=0, value=3, key="corners_B")  
    corners_par_match_B = st.number_input("Corners par match", min_value=0, value=1, key="corners_par_match_B")  
    corners_concedes_B = st.number_input("Corners concédés par match", min_value=0, value=2, key="corners_concedes_B")  
    xG_concedes_B = st.slider("xG concédés", min_value=0.0, max_value=5.0, value=1.5, key="xG_concedes_B")  
    interceptions_B = st.slider("Interceptions par match", min_value=0, max_value=50, value=8, key="interceptions_B")  
    tacles_reussis_B = st.slider("Tacles réussis par match", min_value=0, max_value=50, value=4, key="tacles_reussis_B")  
    degagements_B = st.slider("Dégagements par match", min_value=0, max_value=50, value=6, key="degagements_B")  
    possessions_recuperees_B = st.slider("Possessions récupérées au milieu/match", min_value=0, max_value=50, value=4, key="possessions_recuperees_B")  
    penalties_concedes_B = st.number_input("Pénalties concédés", min_value=0, value=1, key="penalties_concedes_B")  
    arrets_B = st.slider("Arrêts par match", min_value=0, max_value=20, value=2, key="arrets_B")  
    fautes_B = st.slider("Fautes par match", min_value=0, max_value=50, value=12, key="fautes_B")  
    cartons_jaunes_B = st.number_input("Cartons jaunes", min_value=0, value=1, key="cartons_jaunes_B")  
    cartons_rouges_B = st.number_input("Cartons rouges", min_value=0, value=0, key="cartons_rouges_B")  
    tactique_B = st.text_input("Tactique de l'équipe B", value="4-2-3-1", key="tactique_B")  

    st.subheader("Joueurs Clés de l'équipe B")  
    absents_B = st.number_input("Nb joueurs clés absents (sur 5)", min_value=0, max_value=5, value=0, key="absents_B")  
    ratings_B = []  
    for i in range(5):  
        rating = st.number_input(f"Rating joueur clé {i + 1} (0-10)", min_value=0.0, max_value=10.0, value=5.0, key=f"rating_B_{i}")  
        ratings_B.append(rating)  

# Ajout de la variable "face à face"  
st.header("⚔️ Face à Face")  
st.markdown("Entrez l'historique des confrontations directes entre les équipes.")  

col9, col10 = st.columns(2)  

with col9:  
    st.subheader("Équipe A")  
    victoires_face_a_face_A = st.number_input("Victoires contre l'équipe B", min_value=0, value=0, key="victoires_face_a_face_A")  

with col10:  
    st.subheader("Équipe B")  
    victoires_face_a_face_B = st.number_input("Victoires contre l'équipe A", min_value=0, value=0, key="victoires_face_a_face_B")  

# Calcul du ratio de victoires en face à face  
total_face_a_face = victoires_face_a_face_A + victoires_face_a_face_B  
if total_face_a_face > 0:  
    ratio_face_a_face_A = victoires_face_a_face_A / total_face_a_face  
    ratio_face_a_face_B = victoires_face_a_face_B / total_face_a_face  
else:  
    ratio_face_a_face_A = 0.5  # Valeur par défaut si aucune confrontation  
    ratio_face_a_face_B = 0.5  

st.write(f"Ratio de victoires en face à face pour l'équipe A : **{ratio_face_a_face_A:.2f}**")  
st.write(f"Ratio de victoires en face à face pour l'équipe B : **{ratio_face_a_face_B:.2f}**")
# Motivation des équipes  
st.header("🔥 Motivation des Équipes")  
st.markdown("Évaluez le niveau de motivation de chaque équipe.")  

col7, col8 = st.columns(2)  

with col7:  
    st.subheader("Équipe A")  
    motivation_A = st.slider("Niveau de motivation (1 à 10)", min_value=1, max_value=10, value=5, key="motivation_A")  

with col8:  
    st.subheader("Équipe B")  
    motivation_B = st.slider("Niveau de motivation (1 à 10)", min_value=1, max_value=10, value=5, key="motivation_B")  

# Prédiction des buts avec la méthode de Poisson  
def prediction_buts_poisson(xG_A, xG_B):  
    buts_A = [poisson.pmf(i, xG_A) for i in range(6)]  
    buts_B = [poisson.pmf(i, xG_B) for i in range(6)]  
    buts_attendus_A = sum(i * prob for i, prob in enumerate(buts_A))  
    buts_attendus_B = sum(i * prob for i, prob in enumerate(buts_B))  
    return buts_attendus_A, buts_attendus_B  

buts_moyens_A, buts_moyens_B = prediction_buts_poisson(xG_A, xG_B)  

st.header("🎯 Prédiction des Buts (Poisson)")  
st.write(f"Buts attendus pour l'équipe A : **{buts_moyens_A:.2f}**")  
st.write(f"Buts attendus pour l'équipe B : **{buts_moyens_B:.2f}**")
# Prédiction multivariable avec régression logistique (version améliorée)  
st.header("🔮 Prédiction Multivariable (Régression Logistique)")  

# Préparation des données pour le modèle  
X = np.array([  
    [buts_produits_A, buts_encaisse_A, poss_moyenne_A, motivation_A, absents_A, np.mean(ratings_A), forme_recente_A,  
     tirs_cadres_A, pourcentage_tirs_convertis_A, grosses_occasions_A, passes_reussies_A, passes_longues_precises_A,  
     centres_reussis_A, penalties_obtenus_A, touches_surface_adverse_A, corners_A, corners_par_match_A, corners_concedes_A,  
     xG_concedes_A, interceptions_A, tacles_reussis_A, degagements_A, possessions_recuperees_A, penalties_concedes_A,  
     arrets_A, fautes_A, cartons_jaunes_A, cartons_rouges_A, ratio_face_a_face_A],  # Ajout de ratio_face_a_face_A  

    [buts_produits_B, buts_encaisse_B, poss_moyenne_B, motivation_B, absents_B, np.mean(ratings_B), forme_recente_B,  
     tirs_cadres_B, pourcentage_tirs_convertis_B, grosses_occasions_B, passes_reussies_B, passes_longues_precises_B,  
     centres_reussis_B, penalties_obtenus_B, touches_surface_adverse_B, corners_B, corners_par_match_B, corners_concedes_B,  
     xG_concedes_B, interceptions_A, tacles_reussis_B, degagements_B, possessions_recuperees_B, penalties_concedes_B,  
     arrets_B, fautes_B, cartons_jaunes_B, cartons_rouges_B, ratio_face_a_face_B]   # Ajout de ratio_face_a_face_B  
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
st.subheader("Probabilités de Victoire")  
st.write(f"Probabilité de victoire pour l'équipe A : **{prediction[0][1]:.2%}**")  
st.write(f"Probabilité de victoire pour l'équipe B : **{prediction[1][1]:.2%}**")  

# Interprétation des coefficients du modèle  
st.subheader("Interprétation du Modèle")  
feature_names = ['Buts produits', 'Buts encaissés', 'Possession moyenne', 'Motivation', 'Absents', 'Rating joueurs', 'Forme récente',  
                 'Tirs cadrés', 'Pourcentage tirs convertis', 'Grosses occasions', 'Passes réussies', 'Passes longues précises',  
                 'Centres réussis', 'Pénalties obtenus', 'Touches surface adverse', 'Corners', 'Corners par match', 'Corners concédés',  
                 'xG concédés', 'Interceptions', 'Tacles réussis', 'Dégagements', 'Possessions récupérées', 'Pénalties concédés',  
                 'Arrêts', 'Fautes', 'Cartons jaunes', 'Cartons rouges', 'Ratio Face à Face']  # Ajout de 'Ratio Face à Face'  
coefficients = model.coef_[0]  

# Créer un DataFrame pour afficher les coefficients de manière plus lisible  
df_coefficients = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})  
st.dataframe(df_coefficients)  

# Informations supplémentaires pour les paris  
st.header("💡 Informations Utiles pour les Paris")  

# Analyse des forces et faiblesses  
st.subheader("Forces et Faiblesses")  
if prediction[0][1] > prediction[1][1]:  
    st.write("L'équipe A a une plus grande probabilité de gagner en raison de ses forces en attaque, de sa solidité défensive et de son avantage en face à face.")  
else:  
    st.write("L'équipe B a une plus grande probabilité de gagner en raison de ses forces en attaque, de sa solidité défensive et de son avantage en face à face.")  

# Facteurs clés à considérer  
st.subheader("Facteurs Clés à Considérer")  
st.write("Les facteurs clés incluent la forme récente, la motivation, les joueurs absents, les statistiques détaillées des équipes et l'historique des confrontations directes.")  

# Conseils de paris (Disclaimer : Les paris comportent des risques)  
st.subheader("Conseils de Paris (Disclaimer : Les paris comportent des risques)")  
st.write("Considérez les paris suivants :")  
st.write("- Parier sur la victoire de l'équipe avec la plus grande probabilité, en tenant compte de son avantage en face à face.")  
st.write("- Parier sur le nombre total de buts (plus ou moins) en fonction des buts attendus (xG).")  
st.write("- Parier sur les corners ou les cartons en fonction des statistiques des équipes.")
# Calcul des paris alternatifs  
double_chance_A = prediction[0][1] + (1 - prediction[1][1])  # Équipe A gagne ou match nul  
double_chance_B = prediction[1][1] + (1 - prediction[0][1])  # Équipe B gagne ou match nul  

# Affichage des paris alternatifs  
st.header("🤝 Paris Alternatifs (Double Chance)")  
st.write(f"Probabilité de double chance pour l'équipe A (gagner ou match nul) : **{double_chance_A:.2%}**")  
st.write(f"Probabilité de double chance pour l'équipe B (gagner ou match nul) : **{double_chance_B:.2%}**")  

# Seuil de probabilité pour le résultat nul  
st.header("⚖️ Analyse du Résultat Nul")  
seuil_nul = st.slider("Seuil de probabilité pour le résultat nul", min_value=0.0, max_value=1.0, value=0.3, step=0.05)  

# Analyse du résultat nul  
proba_nul = 1 - (prediction[0][1] + prediction[1][1])  
st.write(f"Probabilité estimée d'un match nul : **{proba_nul:.2%}**")  

if proba_nul >= seuil_nul:  
    st.warning("Attention : La probabilité d'un match nul est élevée selon le seuil défini.")  
else:  
    st.success("La probabilité d'un match nul est considérée comme faible.")  

# Conclusion  
st.header("✅ Conclusion")  
st.write("Merci d'avoir utilisé l'outil d'analyse de match ! Cette application vous permet d'évaluer les performances des équipes et de prendre des décisions éclairées sur les paris.")  
st.write("N'oubliez pas que les prédictions sont basées sur les données que vous avez fournies et que le football reste imprévisible.")
