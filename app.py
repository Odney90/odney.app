import streamlit as st  
import numpy as np  
from io import BytesIO  
from docx import Document  
from scipy.special import factorial  

# Fonction de prédiction Poisson  
def poisson_prediction(goals):  
    return [np.exp(-goals) * (goals ** k) / factorial(k) for k in range(6)]  

# Fonction pour générer un fichier Word avec les données des équipes et prédictions  
def create_doc(team_a, team_b, teams_data, predictions):  
    doc = Document()  
    doc.add_heading('🏆 Analyse de Matchs de Football et Prédictions', level=1)  

    doc.add_heading("📋 Données des Équipes", level=2)  
    doc.add_paragraph(f"🏠 Équipe Domicile : {team_a}")  
    doc.add_paragraph(f"🚀 Équipe Extérieure : {team_b}")  

    for key, value in teams_data.items():  
        doc.add_paragraph(f"{key}: {value}")  

    doc.add_heading("🔮 Prédictions", level=2)  
    for key, value in predictions.items():  
        doc.add_paragraph(f"{key}: {value:.2%}")  

    buffer = BytesIO()  
    doc.save(buffer)  
    buffer.seek(0)  
    return buffer  

# Interface utilisateur  
st.title("🏆 Analyse et Prédictions de Matchs de Football")  

# **Saisie des équipes et des statistiques**  
st.header("📋 Informations des Équipes")  

col1, col2 = st.columns(2)  

with col1:  
    team_a = st.text_input("🏠 Nom de l'Équipe Domicile", "Équipe A")  
    home_goals = st.number_input("⚽ Moyenne de buts marqués à domicile", min_value=0.0, max_value=5.0, value=1.5)  
    home_xG = st.number_input("🎯 Moyenne de xG à domicile", min_value=0.0, max_value=5.0, value=1.6)  
    home_encais = st.number_input("🛑 Moyenne de buts encaissés à domicile", min_value=0.0, max_value=5.0, value=1.2)  

with col2:  
    team_b = st.text_input("🚀 Nom de l'Équipe Extérieure", "Équipe B")  
    away_goals = st.number_input("⚽ Moyenne de buts marqués à l'extérieur", min_value=0.0, max_value=5.0, value=1.2)  
    away_xG = st.number_input("🎯 Moyenne de xG à l'extérieur", min_value=0.0, max_value=5.0, value=1.3)  
    away_encais = st.number_input("🛑 Moyenne de buts encaissés à l'extérieur", min_value=0.0, max_value=5.0, value=1.4)  

# **Bouton de prédiction**  
if st.button("🔮 Générer la prédiction"):  
    # Stockage des données des équipes  
    teams_data = {  
        "Buts à domicile": home_goals,  
        "xG à domicile": home_xG,  
        "Buts encaissés à domicile": home_encais,  
        "Buts à l'extérieur": away_goals,  
        "xG à l'extérieur": away_xG,  
        "Buts encaissés à l'extérieur": away_encais  
    }  

    # Calcul des probabilités avec Poisson  
    home_poisson = poisson_prediction(home_goals)  
    away_poisson = poisson_prediction(away_goals)  

    # Probabilités simplifiées  
    prob_victoire_domicile = sum(home_poisson[i] * away_poisson[j] for i in range(6) for j in range(6) if i > j)  
    prob_nul = sum(home_poisson[i] * away_poisson[j] for i in range(6) for j in range(6) if i == j)  
    prob_victoire_exterieur = sum(home_poisson[i] * away_poisson[j] for i in range(6) for j in range(6) if i < j)  

    # Stockage des prédictions  
    predictions = {  
        "Victoire domicile": prob_victoire_domicile,  
        "Match nul": prob_nul,  
        "Victoire extérieur": prob_victoire_exterieur  
    }  

    # Affichage des prédictions  
    st.subheader("🔮 Prédictions")  
    st.write(f"🏠 Victoire domicile: {prob_victoire_domicile:.2%}")  
    st.write(f"⚖️ Match nul: {prob_nul:.2%}")  
    st.write(f"🚀 Victoire extérieur: {prob_victoire_exterieur:.2%}")  

    # Génération du fichier Word  
    doc_buffer = create_doc(team_a, team_b, teams_data, predictions)  

    # Bouton de téléchargement  
    st.download_button(  
        label="📂 Télécharger l'analyse en fichier Word",  
        data=doc_buffer,  
        file_name=f"analyse_{team_a}_vs_{team_b}.docx",  
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"  
    )  
