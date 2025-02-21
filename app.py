import streamlit as st  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt  

# Configuration de l'application  
st.set_page_config(page_title="Prédiction de Match", layout="wide")  

# Titre de la page d'analyse  
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Analyse des Équipes</h1>", unsafe_allow_html=True)  

# Historique des équipes  
st.subheader("Historique des équipes")  
st.markdown("Veuillez entrer les résultats des 10 derniers matchs pour chaque équipe (1 pour victoire, 0 pour match nul, -1 pour défaite).")  

# Historique de l'équipe A  
historique_A = []  
for i in range(10):  
    resultat = st.selectbox(f"Match {i + 1} de l'équipe A", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_A_{i}")  
    historique_A.append(1 if resultat == "Victoire (1)" else (0 if resultat == "Match Nul (0)" else -1))  

# Historique de l'équipe B  
historique_B = []  
for i in range(10):  
    resultat = st.selectbox(f"Match {i + 1} de l'équipe B", options=["Victoire (1)", "Match Nul (0)", "Défaite (-1)"], key=f"match_B_{i}")  
    historique_B.append(1 if resultat == "Victoire (1)" else (0 if resultat == "Match Nul (0)" else -1))  

# Calcul de la forme récente sur 10 matchs  
def calculer_forme(historique):  
    return sum(historique) / len(historique)  # Moyenne des résultats  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  

st.write(f"Forme récente de l'équipe A (moyenne sur 10 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 10 matchs) : **{forme_recente_B:.2f}**")  

# Visualisation de la forme des équipes  
st.subheader("Visualisation de la Forme des Équipes")  
fig, ax = plt.subplots()  
ax.plot(range(1, 11), historique_A, marker='o', label='Équipe A', color='blue')  
ax.plot(range(1, 11), historique_B, marker='o', label='Équipe B', color='red')  
ax.set_xticks(range(1, 11))  
ax.set_xticklabels([f"Match {i}" for i in range(1, 11)])  
ax.set_ylim(-1.5, 1.5)  
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')  
ax.set_ylabel('Résultat')  
ax.set_title('Historique des Performances des Équipes')  
ax.legend()  
st.pyplot(fig)  

# Sélection des systèmes tactiques  
st.subheader("Systèmes Tactiques")  
tactiques = ["4-4-2", "4-3-3", "3-5-2", "4-2-3-1", "5-3-2", "4-1-4-1"]  
tactique_A = st.selectbox("Choisissez le système tactique de l'équipe A", tactiques)  
tactique_B = st.selectbox("Choisissez le système tactique de l'équipe B", tactiques)  

# Critères pour l'équipe A  
st.subheader("Critères pour l'équipe A")  
buts_produits_A = st.slider("Nombre de buts produits", min_value=0, max_value=100, value=80, key="buts_produits_A")  
buts_encaisse_A = st.slider("Nombre de buts encaissés", min_value=0, max_value=100, value=30, key="buts_encaisse_A")  
buts_par_match_A = st.slider("Buts par match", min_value=0, max_value=10, value=2, key="buts_par_match_A")  
poss_moyenne_A = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=55, key="poss_moyenne_A")  
xG_A = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=2.5, key="xG_A")  
grosses_occasions_A = st.slider("Grosses occasions créées", min_value=0, max_value=100, value=10, key="grosses_occasions_A")  
passes_longues_A = st.slider("Passes longues précises par match", min_value=0, max_value=50, value=10, key="passes_longues_A")  
touches_surface_A = st.slider("Touches dans la surface adverse", min_value=0, max_value=50, value=15, key="touches_surface_A")  
corners_A = st.slider("Nombre de corners", min_value=0, max_value=20, value=5, key="corners_A")  
interceptions_A = st.slider("Interceptions par match", min_value=0, max_value=20, value=5, key="interceptions_A")  
arrets_A = st.slider("Arrêts par match", min_value=0, max_value=20, value=3, key="arrets_A")  
fautes_A = st.slider("Fautes par match", min_value=0, max_value=20, value=10, key="fautes_A")  

# Critères pour l'équipe B  
st.subheader("Critères pour l'équipe B")  
buts_produits_B = st.slider("Nombre de buts produits", min_value=0, max_value=100, value=60, key="buts_produits_B")  
buts_encaisse_B = st.slider("Nombre de buts encaissés", min_value=0, max_value=100, value=40, key="buts_encaisse_B")  
buts_par_match_B = st.slider("Buts par match", min_value=0, max_value=10, value=1, key="buts_par_match_B")  
poss_moyenne_B = st.slider("Possession moyenne (%)", min_value=0, max_value=100, value=45, key="poss_moyenne_B")  
xG_B = st.slider("Buts attendus (xG)", min_value=0.0, max_value=5.0, value=1.5, key="xG_B")  
grosses_occasions_B = st.slider("Grosses occasions créées", min_value=0, max_value=100, value=8, key="grosses_occasions_B")  
passes_longues_B = st.slider("Passes longues précises par match", min_value=0, max_value=50, value=8, key="passes_longues_B")  
touches_surface_B = st.slider("Touches dans la surface adverse", min_value=0, max_value=50, value=12, key="touches_surface_B")  
corners_B = st.slider("Nombre de corners", min_value=0, max_value=20, value=4, key="corners_B")  
interceptions_B = st.slider("Interceptions par match", min_value=0, max_value=20, value=6, key="interceptions_B")  
arrets_B = st.slider("Arrêts par match", min_value=0, max_value=20, value=4, key="arrets_B")  
fautes_B = st.slider("Fautes par match", min_value=0, max_value=20, value=8, key="fautes_B")  

# Modèle de prédiction des buts  
if st.button("Analyser le Match"):  
    # Données d'entraînement fictives  
    data = {  
        "buts_produits_A": [80, 60, 70, 90, 50],  
        "buts_encaisse_A": [30, 40, 20, 50, 35],  
        "buts_par_match_A": [2, 1, 2, 3, 1],  
        "poss_moyenne_A": [55, 45, 60, 40, 50],  
        "xG_A": [2.5, 1.5, 3.0, 0.5, 2.0],  
        "grosses_occasions_A": [10, 8, 12, 5, 9],  
        "passes_longues_A": [10, 12, 15, 8, 11],  
        "touches_surface_A": [15, 20, 18, 10, 14],  
        "corners_A": [5, 3, 6, 2, 4],  
        "interceptions_A": [5, 6, 4, 7, 5],  
        "arrets_A": [3, 2, 4, 1, 3],  
        "fautes_A": [10, 12, 8, 9, 11],  
        "buts_produits_B": [60, 70, 50, 20, 40],  
        "buts_encaisse_B": [40, 30, 20, 50, 35],  
        "buts_par_match_B": [1, 2, 1, 0, 1],  
        "poss_moyenne_B": [45, 55, 40, 50, 45],  
        "xG_B": [1.5, 2.0, 1.0, 0.5, 1.2],  
        "grosses_occasions_B": [8, 10, 6, 4, 7],  
        "passes_longues_B": [8, 10, 12, 6, 9],  
        "touches_surface_B": [12, 15, 10, 8, 11],  
        "corners_B": [4, 5, 3, 2, 4],  
        "interceptions_B": [6, 5, 7, 4, 5],  
        "arrets_B": [4, 3, 5, 2, 4],  
        "fautes_B": [8, 9, 7, 10, 8],  
    }  
    
    df = pd.DataFrame(data)  

    # Entraînement du modèle  
    features = [  
        "buts_produits_A", "buts_encaisse_A", "poss_moyenne_A", "xG_A",  
        "grosses_occasions_A", "passes_longues_A", "touches_surface_A",  
        "corners_A", "interceptions_A", "arrets_A", "fautes_A",  
        "buts_produits_B", "buts_encaisse_B", "poss_moyenne_B", "xG_B",  
        "grosses_occasions_B", "passes_longues_B", "touches_surface_B",  
        "corners_B", "interceptions_B", "arrets_B", "fautes_B"  
    ]  

    X = df[features]  
    y_A = df["buts_par_match_A"]  
    y_B = df["buts_par_match_B"]  

    # Entraînement des modèles  
    X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X, y_A, test_size=0.2, random_state=42)  
    X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y_B, test_size=0.2, random_state=42)  

    model_A = LinearRegression()  
    model_A.fit(X_train_A, y_train_A)  

    model_B = LinearRegression()  
    model_B.fit(X_train_B, y_train_B)  

    # Prédictions pour les équipes A et B  
    nouvelle_donnee = pd.DataFrame({  
        "buts_produits_A": [buts_produits_A],  
        "buts_encaisse_A": [buts_encaisse_A],  
        "poss_moyenne_A": [poss_moyenne_A],  
        "xG_A": [xG_A],  
        "grosses_occasions_A": [grosses_occasions_A],  
        "passes_longues_A": [passes_longues_A],  
        "touches_surface_A": [touches_surface_A],  
        "corners_A": [corners_A],  
        "interceptions_A": [interceptions_A],  
        "arrets_A": [arrets_A],  
        "fautes_A": [fautes_A],  
        "buts_produits_B": [buts_produits_B],  
        "buts_encaisse_B": [buts_encaisse_B],  
        "poss_moyenne_B": [poss_moyenne_B],  
        "xG_B": [xG_B],  
        "grosses_occasions_B": [grosses_occasions_B],  
        "passes_longues_B": [passes_longues_B],  
        "touches_surface_B": [touches_surface_B],  
        "corners_B": [corners_B],  
        "interceptions_B": [interceptions_B],  
        "arrets_B": [arrets_B],  
        "fautes_B": [fautes_B]  
    })  

    prediction_A = model_A.predict(nouvelle_donnee)[0]  
    prediction_B = model_B.predict(nouvelle_donnee)[0]  

    # Calcul des pourcentages de victoire  
    total_buts = prediction_A + prediction_B  
    if total_buts > 0:  
        pourcentage_A = (prediction_A / total_buts) * 100  
        pourcentage_B = (prediction_B / total_buts) * 100  
    else:  
        pourcentage_A = 0  
        pourcentage_B = 0  

    # Affichage des résultats  
    st.markdown("<h2 style='color: #FF5722;'>Prédictions des Buts</h2>", unsafe_allow_html=True)  
    st.write(f"Prédiction des buts pour l'équipe A : **{prediction_A:.2f}**")  
    st.write(f"Prédiction des buts pour l'équipe B : **{prediction_B:.2f}**")  

    st.markdown("<h2 style='color: #FF5722;'>Pourcentages de Victoire</h2>", unsafe_allow_html=True)  
    st.write(f"Pourcentage de victoire pour l'équipe A : **{pourcentage_A:.2f}%**")  
    st.write(f"Pourcentage de victoire pour l'équipe B : **{pourcentage_B:.2f}%**")  

    # Détermination du résultat  
    if prediction_A > prediction_B:  
        st.write("Résultat : Victoire de l'équipe A")  
    elif prediction_A < prediction_B:  
        st.write("Résultat : Victoire de l'équipe B")  
    else:  
        st.write("Résultat : Match nul")  

    # Visualisation des résultats  
    st.subheader("Visualisation des Prédictions")  
    fig, ax = plt.subplots()  
    ax.bar(['Équipe A', 'Équipe B'], [prediction_A, prediction_B], color=['blue', 'red'])  
    ax.set_ylabel('Nombre de buts prédit')  
    ax.set_title('Prédiction des buts par équipe')  
    st.pyplot(fig)  

    # Visualisation des pourcentages de victoire  
    fig2, ax2 = plt.subplots()  
    ax2.bar(['Équipe A', 'Équipe B'], [pourcentage_A, pourcentage_B], color=['blue', 'red'])  
    ax2.set_ylabel('Pourcentage de victoire (%)')  
    ax2.set_title('Pourcentage de victoire par équipe')  
    st.pyplot(fig2)  

    # Affichage des systèmes tactiques  
    st.markdown("<h2 style='color: #FF5722;'>Systèmes Tactiques Choisis</h2>", unsafe_allow_html=True)  
    st.write(f"Système tactique de l'équipe A : **{tactique_A}**")  
    st.write(f"Système tactique de l'équipe B : **{tactique_B}**")  

    # Analyse des systèmes tactiques  
    st.markdown("<h2 style='color: #FF5722;'>Analyse des Systèmes Tactiques</h2>", unsafe_allow_html=True)  
    if tactique_A == "4-4-2" and tactique_B == "4-3-3":  
        st.write("L'équipe A joue en 4-4-2, ce qui lui donne une bonne stabilité défensive, mais peut être vulnérable au milieu de terrain contre un 4-3-3.")  
    elif tactique_A == "4-3-3" and tactique_B == "4-4-2":  
        st.write("L'équipe A en 4-3-3 peut dominer le milieu de terrain, mais doit faire attention aux contre-attaques de l'équipe B.")  
    elif tactique_A == "3-5-2" and tactique_B == "4-2-3-1":  
        st.write("L'équipe A en 3-5-2 peut avoir un avantage au milieu, mais doit gérer les espaces laissés derrière les arrières.")  
    elif tactique_A == "4-2-3-1" and tactique_B == "3-5-2":  
        st.write("L'équipe A en 4-2-3-1 peut exploiter les ailes, mais doit être vigilante aux attaques rapides de l'équipe B.")  
    else:  
        st.write("Les systèmes tactiques choisis peuvent avoir des impacts variés sur le match. Une analyse plus approfondie est nécessaire pour évaluer les forces et faiblesses.")
