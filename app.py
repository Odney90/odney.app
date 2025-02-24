from fpdf import FPDF
import streamlit as st
import numpy as np

# Fonction pour la prédiction des résultats (exemple simplifié)
def generate_predictions(team1_data, team2_data):
    poisson_prediction = {'team1_goals': np.random.poisson(lam=team1_data['attack'] * 2), 
                          'team2_goals': np.random.poisson(lam=team2_data['attack'] * 2)}
    rf_prediction = {'team1_win_prob': np.random.random()}
    return poisson_prediction, rf_prediction

# Fonction pour générer un fichier PDF
def generate_pdf(team1_data, team2_data, poisson_prediction, rf_prediction):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Analyse de Match de Football ⚽", ln=True, align='C')
    pdf.ln(10)  # Line break
    
    # Section for Team 1 and Team 2 data
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Données des Équipes", ln=True)
    
    # Team 1
    pdf.ln(5)  # Line break
    pdf.cell(95, 10, txt=f"Équipe 1 : {team1_data['name']}")
    pdf.ln(5)
    for key, value in team1_data.items():
        if key != 'name':
            pdf.cell(95, 10, txt=f"{key.capitalize()} : {value}")
            pdf.ln(5)
    
    # Team 2
    pdf.ln(10)  # Line break
    pdf.cell(95, 10, txt=f"Équipe 2 : {team2_data['name']}")
    pdf.ln(5)
    for key, value in team2_data.items():
        if key != 'name':
            pdf.cell(95, 10, txt=f"{key.capitalize()} : {value}")
            pdf.ln(5)
    
    # Predictions
    pdf.ln(10)  # Line break
    pdf.cell(200, 10, txt="Prédictions Poisson", ln=True)
    pdf.cell(95, 10, txt=f"Team 1 Goals: {poisson_prediction['team1_goals']}")
    pdf.ln(5)
    pdf.cell(95, 10, txt=f"Team 2 Goals: {poisson_prediction['team2_goals']}")
    
    pdf.ln(10)  # Line break
    pdf.cell(200, 10, txt="Prédictions Random Forest", ln=True)
    pdf.cell(95, 10, txt=f"Team 1 Win Probability: {rf_prediction['team1_win_prob']*100:.2f}%")
    
    # Save the PDF to a file
    pdf.output("predictions_football_match.pdf")

# Interface Streamlit
st.title('Analyse de Match de Football ⚽')

# Entrée des noms des équipes
team1_name = st.text_input("Nom de l'Équipe 1")
team2_name = st.text_input("Nom de l'Équipe 2")

# Variables des équipes
team1_data = {
    'name': team1_name,
    'attack': st.slider("Force d'attaque de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'defense': st.slider("Force de défense de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'form': st.slider("Forme récente de l'équipe 1 (0-1)", 0.0, 1.0, 0.5),
    'injuries': st.slider("Blessures dans l'équipe 1 (0-1)", 0.0, 1.0, 0.0)
}

team2_data = {
    'name': team2_name,
    'attack': st.slider("Force d'attaque de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'defense': st.slider("Force de défense de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'form': st.slider("Forme récente de l'équipe 2 (0-1)", 0.0, 1.0, 0.5),
    'injuries': st.slider("Blessures dans l'équipe 2 (0-1)", 0.0, 1.0, 0.0)
}

# Option pour générer les prédictions et PDF
if st.button('Générer les Prédictions et Télécharger le PDF'):
    poisson_prediction, rf_prediction = generate_predictions(team1_data, team2_data)
    
    # Affichage des prédictions
    st.write("### Prédictions Poisson")
    st.write(f"Team 1 Goals: {poisson_prediction['team1_goals']}")
    st.write(f"Team 2 Goals: {poisson_prediction['team2_goals']}")
    
    st.write("### Prédictions Random Forest")
    st.write(f"Team 1 Win Probability: {rf_prediction['team1_win_prob']*100:.2f}%")
    
    # Générer le PDF
    generate_pdf(team1_data, team2_data, poisson_prediction, rf_prediction)
    
    # Option de téléchargement
    st.download_button(
        label="Télécharger le PDF",
        data=open("predictions_football_match.pdf", "rb").read(),
        file_name="predictions_football_match.pdf"
    )
