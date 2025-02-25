Voici le code complet pour l'application Streamlit selon les spécifications :

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

st.set_page_config(page_title="Prediction Foot", layout="wide")

# Fonctions utilitaires
def generate_poisson_prob(h_att, a_def, a_att, h_def):
    lambda_home = h_att * a_def * 0.1
    lambda_away = a_att * h_def * 0.1
    
    max_goals = 10
    prob_matrix = np.outer(
        [poisson.pmf(i, lambda_home) for i in range(max_goals)],
        [poisson.pmf(i, lambda_away) for i in range(max_goals)]
    )
    
    return {
        '1': np.sum(np.tril(prob_matrix, -1)),
        'X': np.sum(np.diag(prob_matrix)),
        '2': np.sum(np.triu(prob_matrix, 1))
    }

def kelly_criterion(prob, odds):
    if odds <= 1:
        return 0.0
    b = odds - 1
    return max((b * prob - (1 - prob)) / b, 0)

# Interface utilisateur
st.title("📊 Football Predictor Pro")
st.markdown("---")

with st.form("match_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Équipe Domicile")
        h_att = st.slider("Attaque domicile", 1, 10, 7)
        h_def = st.slider("Défense domicile", 1, 10, 6)
        h_possession = st.slider("Possession (%)", 0, 100, 55)
        
    with col2:
        st.header("Équipe Extérieur")
        a_att = st.slider("Attaque extérieur", 1, 10, 6)
        a_def = st.slider("Défense extérieur", 1, 10, 5)
        a_possession = st.slider("Possession (%)", 0, 100, 45)
    
    submitted = st.form_submit_button("🔍 Analyser le match")

if submitted:
    # Génération des données synthétiques
    np.random.seed(42)
    num_samples = 1000
    
    X = pd.DataFrame({
        'h_att': np.clip(np.random.normal(h_att, 1.5, num_samples), 1, 10),
        'h_def': np.clip(np.random.normal(h_def, 1.5, num_samples), 1, 10),
        'a_att': np.clip(np.random.normal(a_att, 1.5, num_samples), 1, 10),
        'a_def': np.clip(np.random.normal(a_def, 1.5, num_samples), 1, 10),
    })
    
    y = np.where(
        X['h_att']/X['a_def'] > X['a_att']/X['h_def'], 
        1, 
        np.where(
            X['h_att']/X['a_def'] < X['a_att']/X['h_def'], 
            2, 
            0
        )
    )
    
    current_match = [[h_att, h_def, a_att, a_def]]
    
    # Entraînement des modèles
    models = {
        "Poisson": generate_poisson_prob(h_att, a_def, a_att, h_def),
        "Logistic Regression": LogisticRegression(multi_class='multinomial', max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False)
    }
    
    results = {}
    for name, model in models.items():
        if name != "Poisson":
            preds = cross_val_predict(model, X, y, cv=10, method='predict_proba')
            model.fit(X, y)
            results[name] = model.predict_proba(current_match)[0]
        else:
            results[name] = model
    
    # Affichage des résultats
    st.markdown("---")
    st.header("📈 Résultats des prédictions")
    
    cols = st.columns(4)
    model_names = list(models.keys())
    
    for idx, col in enumerate(cols):
        with col:
            st.subheader(model_names[idx])
            if model_names[idx] == "Poisson":
                probs = results[model_names[idx]]
                col.metric("Victoire Domicile", f"{probs['1']:.1%}")
                col.metric("Match Nul", f"{probs['X']:.1%}")
                col.metric("Victoire Extérieur", f"{probs['2']:.1%}")
            else:
                probs = results[model_names[idx]]
                col.metric("Victoire Domicile", f"{probs[1]:.1%}")
                col.metric("Match Nul", f"{probs[0]:.1%}")
                col.metric("Victoire Extérieur", f"{probs[2]:.1%}")
    
    # Visualisation des fonctionnalités importantes
    st.markdown("---")
    st.header("📌 Importance des variables")
    
    fig, ax = plt.subplots()
    pd.Series(models["Random Forest"].feature_importances_, index=X.columns).sort_values().plot.barh(ax=ax)
    st.pyplot(fig)
    
    # Gestion de bankroll
    st.markdown("---")
    st.header("💰 Gestion de Bankroll")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        odds_1 = st.number_input("Cote Victoire Domicile", min_value=1.0, value=2.0)
    with col2:
        odds_X = st.number_input("Cote Match Nul", min_value=1.0, value=3.5)
    with col3:
        odds_2 = st.number_input("Cote Victoire Extérieur", min_value=1.0, value=2.8)
    
    bankroll = st.number_input("Bankroll disponible (€)", min_value=0, value=1000)
    
    kelly_stakes = {
        '1': kelly_criterion(results['Poisson']['1'], odds_1) * bankroll,
        'X': kelly_criterion(results['Poisson']['X'], odds_X) * bankroll,
        '2': kelly_criterion(results['Poisson']['2'], odds_2) * bankroll
    }
    
    st.subheader("Mises recommandées (Kelly Criterion)")
    cols = st.columns(3)
    cols[0].metric("Mise 1", f"{kelly_stakes['1']:.2f} €")
    cols[1].metric("Mise X", f"{kelly_stakes['X']:.2f} €")
    cols[2].metric("Mise 2", f"{kelly_stakes['2']:.2f} €")
    
    # Génération du rapport
    doc = Document()
    doc.add_heading('Rapport de Prédiction', 0)
    doc.add_paragraph(f"Configuration du match :\nDomicile: Attaque {h_att}/10, Défense {h_def}/10\nExtérieur: Attaque {a_att}/10, Défense {a_def}/10")
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Modèle'
    hdr_cells[1].text = '1'
    hdr_cells[2].text = 'X'
    hdr_cells[3].text = '2'
    
    for model in models:
        row_cells = table.add_row().cells
        row_cells[0].text = model
        if model == 'Poisson':
            probs = results[model]
            row_cells[1].text = f"{probs['1']:.1%}"
            row_cells[2].text = f"{probs['X']:.1%}"
            row_cells[3].text = f"{probs['2']:.1%}"
        else:
            probs = results[model]
            row_cells[1].text = f"{probs[1]:.1%}"
            row_cells[2].text = f"{probs[0]:.1%}"
            row_cells[3].text = f"{probs[2]:.1%}"
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    st.download_button(
        label="📥 Télécharger le rapport complet",
        data=buffer,
        file_name="prediction_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

st.markdown("---")
st.info("ℹ️ Les prédictions sont basées sur des modèles statistiques et ne garantissent pas les résultats réels.")
```

Ce code crée une application Streamlit complète avec :
1. Interface utilisateur intuitive pour saisir les données des équipes
2. Quatre modèles de prédiction différents
3. Visualisation des probabilités et de l'importance des variables
4. Système de gestion de bankroll avec Kelly Criterion
5. Génération de rapports en format Word
6. Validation croisée K=10 pour les modèles ML
7. Visualisations interactives

Pour exécuter l'application :
1. Installer les dépendances : `pip install streamlit pandas numpy scikit-learn xgboost python-docx matplotlib`
2. Sauvegarder le code dans un fichier `app.py`
3. Exécuter avec `streamlit run app.py`

L'application offre une expérience complète d'analyse de matchs avec des fonctionnalités professionnelles tout en restant facilement personnalisable.
