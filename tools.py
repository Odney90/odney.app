import streamlit as st  
import numpy as np  

def cotes_vers_probabilite(cote):  
    """Convertit une cote décimale en probabilité implicite."""  
    return 1 / cote  

def enlever_marge(probabilites, marge):  
    """Enlève la marge du bookmaker des probabilités."""  
    total_probabilite = sum(probabilites)  
    facteur_correction = (1 - marge) / total_probabilite  
    return [p * facteur_correction for p in probabilites]  

def kelly_criterion(cote, probabilite, bankroll, kelly_fraction):  
    """Calcule la mise de Kelly."""  
    probabilite_avantage = probabilite * cote - 1  
    fraction = kelly_fraction * probabilite_avantage / (cote - 1)  
    return bankroll * fraction  

def tools_page():  
    st.title("🛠️ Outils de Paris")  

    # Convertisseur de cotes  
    st.header("🧮 Convertisseur de Cotes")  
    cote_decimale = st.number_input("Cote décimale", min_value=1.01, value=2.0)  
    probabilite_implicite = cotes_vers_probabilite(cote_decimale)  
    st.write(f"Probabilité implicite : **{probabilite_implicite:.2%}**")  

    # Calcul de value bets  
    st.header("💰 Calcul de Value Bets")  
    cote_value_bet = st.number_input("Cote (value bet)", min_value=1.01, value=2.0, key="cote_value_bet")  
    probabilite_estimee = st.slider("Probabilité estimée (%)", min_value=1, max_value=100, value=50) / 100  
    probabilite_implicite_value_bet = cotes_vers_probabilite(cote_value_bet)  
    if probabilite_estimee > probabilite_implicite_value_bet:  
        st.success("Value bet détecté !")  
    else:  
        st.warning("Pas de value bet.")  

    # Simulateur de paris combinés  
    st.header("➕ Simulateur de Paris Combinés")  
    nombre_equipes = st.slider("Nombre d'équipes (jusqu'à 3)", min_value=1, max_value=3, value=2)  
    probabilites = []  
    for i in range(nombre_equipes):  
        probabilite = st.slider(f"Probabilité implicite équipe {i + 1} (%)", min_value=1, max_value=100, value=50, key=f"probabilite_{i}") / 100  
        probabilites.append(probabilite)  
    probabilite_combinee = np.prod(probabilites)  
    st.write(f"Probabilité combinée : **{probabilite_combinee:.2%}**")  

    # Mise de Kelly  
    st.header("🏦 Mise de Kelly")  
    cote_kelly = st.number_input("Cote (Kelly)", min_value=1.01, value=2.0, key="cote_kelly")  
    probabilite_kelly = st.slider("Probabilité estimée (Kelly) (%)", min_value=1, max_value=100, value=50, key="probabilite_kelly") / 100  
    bankroll = st.number_input("Bankroll", min_value=1, value=1000)  
    kelly_fraction = st.slider("Fraction de Kelly (1 à 5)", min_value=1, max_value=5, value=1) / 5  
    mise_kelly = kelly_criterion(cote_kelly, probabilite_kelly, bankroll, kelly_fraction)  
    st.write(f"Mise de Kelly recommandée : **{mise_kelly:.2f}**")
