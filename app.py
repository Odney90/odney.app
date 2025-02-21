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
historique_A = st.text_area("Historique des performances de l'équipe A (10 derniers matchs)", height=100)  
historique_B = st.text_area("Historique des performances de l'équipe B (10 derniers matchs)", height=100)  

# Calcul de la forme récente sur 10 matchs  
def calculer_forme(historique):  
    if historique:  
        matchs = historique.split(',')  
        if len(matchs) > 10:  
            matchs = matchs[-10:]  # Prendre les 10 derniers matchs  
        return sum(int(resultat) for resultat in matchs) / len(matchs)  # Moyenne des résultats  
    return 0.0  

forme_recente_A = calculer_forme(historique_A)  
forme_recente_B = calculer_forme(historique_B)  

st.write(f"Forme récente de l'équipe A (moyenne sur 10 matchs) : **{forme_recente_A:.2f}**")  
st.write(f"Forme récente de l'équipe B (moyenne sur 10 matchs) : **{forme_recente_B:.2f}**")  

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
tactique_A = st.text_input("Tactique de l'équipe A", value="Offensive")  

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
tactique_B = st.text_input("Tactique de l'équipe B", value="Défensive")  

# Modèle de prédiction des buts  
if st.button("Prédire le nombre de buts"):  
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
