Je vous fournis le code complet en français avec tous les commentaires et libellés traduits :

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from docx import Document
from io import BytesIO

# Configuration de la page
st.set_page_config(page_title="Prédicteur Football", layout="wide")

# Fonctions utilitaires
def calcul_probabilites_poisson(attaque_domicile, defense_exterieur, attaque_exterieur, defense_domicile):
    """Calcule les probabilités de résultats avec le modèle Poisson"""
    lambda_domicile = attaque_domicile * defense_exterieur * 0.1
    lambda_exterieur = attaque_exterieur * defense_domicile * 0.1
    
    max_buts = 10
    matrice_prob = np.outer(
        [poisson.pmf(i, lambda_domicile) for i in range(max_buts)],
        [poisson.pmf(i, lambda_exterieur) for i in range(max_buts)]
    )
    
    return {
        '1': np.sum(np.tril(matrice_prob, -1)),  # Victoire domicile
        'X': np.sum(np.diag(matrice_prob)),      # Match nul
        '2': np.sum(np.triu(matrice_prob, 1))    # Victoire extérieur
    }

def critere_kelly(probabilite, cote):
    """Calcule la mise optimale avec le critère de Kelly"""
    if cote <= 1:
        return 0.0
    b = cote - 1
    return max((b * probabilite - (1 - probabilite)) / b, 0)

# Interface utilisateur
st.title("⚽ Analyseur de Matchs Football")
st.markdown("---")

with st.form("formulaire_match"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Équipe Domicile")
        attaque_dom = st.slider("Niveau d'attaque", 1, 10, 7, key='att_dom')
        defense_dom = st.slider("Niveau de défense", 1, 10, 6, key='def_dom')
        possession_dom = st.slider("Possession (%)", 0, 100, 55, key='pos_dom')
        
    with col2:
        st.header("Équipe Extérieur")
        attaque_ext = st.slider("Niveau d'attaque", 1, 10, 6, key='att_ext')
        defense_ext = st.slider("Niveau de défense", 1, 10, 5, key='def_ext')
        possession_ext = st.slider("Possession (%)", 0, 100, 45, key='pos_ext')
    
    soumis = st.form_submit_button("🔍 Analyser le match")

if soumis:
    # Génération de données synthétiques pour l'entraînement
    np.random.seed(42)
    n_echantillons = 1000
    
    X = pd.DataFrame({
        'attaque_dom': np.clip(np.random.normal(attaque_dom, 1.5, n_echantillons), 1, 10),
        'defense_dom': np.clip(np.random.normal(defense_dom, 1.5, n_echantillons), 1, 10),
        'attaque_ext': np.clip(np.random.normal(attaque_ext, 1.5, n_echantillons), 1, 10),
        'defense_ext': np.clip(np.random.normal(defense_ext, 1.5, n_echantillons), 1, 10),
    })
    
    # Création de la variable cible
    y = np.where(
        X['attaque_dom']/X['defense_ext'] > X['attaque_ext']/X['defense_dom'], 
        1,  # Victoire domicile
        np.where(
            X['attaque_dom']/X['defense_ext'] < X['attaque_ext']/X['defense_dom'], 
            2,  # Victoire extérieur
            0   # Match nul
        )
    )
    
    match_actuel = [[attaque_dom, defense_dom, attaque_ext, defense_ext]]
    
    # Initialisation des modèles
    modeles = {
        "Poisson": calcul_probabilites_poisson(attaque_dom, defense_ext, attaque_ext, defense_dom),
        "Régression Logistique": LogisticRegression(multi_class='multinomial', max_iter=1000),
        "Forêt Aléatoire": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False)
    }
    
    resultats = {}
    for nom, modele in modeles.items():
        if nom != "Poisson":
            predictions = cross_val_predict(modele, X, y, cv=10, method='predict_proba')
            modele.fit(X, y)
            resultats[nom] = modele.predict_proba(match_actuel)[0]
        else:
            resultats[nom] = modele
    
    # Affichage des résultats
    st.markdown("---")
    st.header("📊 Probabilités Prédites")
    
    colonnes = st.columns(4)
    noms_modeles = list(modeles.keys())
    
    for idx, col in enumerate(colonnes):
        with col:
            st.subheader(noms_modeles[idx])
            if noms_modeles[idx] == "Poisson":
                probs = resultats[noms_modeles[idx]]
                col.metric("Victoire Domicile", f"{probs['1']:.1%}")
                col.metric("Match Nul", f"{probs['X']:.1%}")
                col.metric("Victoire Extérieur", f"{probs['2']:.1%}")
            else:
                probs = resultats[noms_modeles[idx]]
                col.metric("Victoire Domicile", f"{probs[1]:.1%}")
                col.metric("Match Nul", f"{probs[0]:.1%}")
                col.metric("Victoire Extérieur", f"{probs[2]:.1%}")
    
    # Visualisation des variables importantes
    st.markdown("---")
    st.header("📈 Importance des Variables")
    
    fig, ax = plt.subplots()
    importance = pd.Series(modeles["Forêt Aléatoire"].feature_importances_, index=X.columns)
    importance.sort_values().plot.barh(ax=ax)
    st.pyplot(fig)
    
    # Gestion de la bankroll
    st.markdown("---")
    st.header("💶 Gestion des Mises")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cote_1 = st.number_input("Cote Victoire Domicile", min_value=1.0, value=2.0)
    with col2:
        cote_X = st.number_input("Cote Match Nul", min_value=1.0, value=3.5)
    with col3:
        cote_2 = st.number_input("Cote Victoire Extérieur", min_value=1.0, value=2.8)
    
    bankroll = st.number_input("Bankroll totale (€)", min_value=0, value=1000)
    
    mises_kelly = {
        '1': critere_kelly(resultats['Poisson']['1'], cote_1) * bankroll,
        'X': critere_kelly(resultats['Poisson']['X'], cote_X) * bankroll,
        '2': critere_kelly(resultats['Poisson']['2'], cote_2) * bankroll
    }
    
    st.subheader("Mises Recommandées (Critère de Kelly)")
    colonnes = st.columns(3)
    colonnes[0].metric("Mise 1", f"{mises_kelly['1']:.2f} €")
    colonnes[1].metric("Mise X", f"{mises_kelly['X']:.2f} €")
    colonnes[2].metric("Mise 2", f"{mises_kelly['2']:.2f} €")
    
    # Génération du rapport
    doc = Document()
    doc.add_heading('Rapport d\'Analyse', 0)
    doc.add_paragraph(
        f"Configuration du match :\n"
        f"Domicile: Attaque {attaque_dom}/10, Défense {defense_dom}/10\n"
        f"Extérieur: Attaque {attaque_ext}/10, Défense {defense_ext}/10"
    )
    
    tableau = doc.add_table(rows=1, cols=4)
    tableau.style = 'Table Grid'
    cellules_entete = tableau.rows[0].cells
    cellules_entete[0].text = 'Modèle'
    cellules_entete[1].text = '1'
    cellules_entete[2].text = 'X'
    cellules_entete[3].text = '2'
    
    for modele in modeles:
        cellules = tableau.add_row().cells
        cellules[0].text = modele
        if modele == 'Poisson':
            probs = resultats[modele]
            cellules[1].text = f"{probs['1']:.1%}"
            cellules[2].text = f"{probs['X']:.1%}"
            cellules[3].text = f"{probs['2']:.1%}"
        else:
            probs = resultats[modele]
            cellules[1].text = f"{probs[1]:.1%}"
            cellules[2].text = f"{probs[0]:.1%}"
            cellules[3].text = f"{probs[2]:.1%}"
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    st.download_button(
        label="📥 Télécharger le Rapport",
        data=b
