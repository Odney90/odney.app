import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
import statsmodels.api as sm  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix  

# --- CONFIG INTERFACE ---  
st.set_page_config(page_title="Analyse Foot & Value Bets", layout="wide", page_icon="⚽")  

# --- STYLE PERSONNALISÉ ---  
st.markdown("""  
    <style>  
    .main {  
        background-color: #f0f2f6;  
        padding: 2rem;  
    }  
    .stButton>button {  
        background-color: #4CAF50;  
        color: white;  
        border-radius: 5px;  
        padding: 10px 20px;  
        font-size: 16px;  
    }  
    .stButton>button:hover {  
        background-color: #45a049;  
    }  
    .stHeader {  
        color: #2c3e50;  
        font-size: 2.5rem;  
        font-weight: bold;  
    }  
    .stSubheader {  
        color: #34495e;  
        font-size: 1.8rem;  
        font-weight: bold;  
    }  
    .stDataFrame {  
        border-radius: 10px;  
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  
    }  
    .stMetric {  
        background-color: white;  
        padding: 1rem;  
        border-radius: 10px;  
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  
    }  
    </style>  
""", unsafe_allow_html=True)  

# --- PERSISTANCE DES DONNÉES ---  
if "data" not in st.session_state:  
    st.session_state.data = None  
if "historique" not in st.session_state:  
    st.session_state.historique = []  

# --- FONCTIONS UTILES ---  
def convertir_cote_en_proba(cote):  
    return 1 / cote if cote > 0 else 0  

def detecter_value_bet(prob_predite, cote_bookmaker):  
    prob_bookmaker = convertir_cote_en_proba(cote_bookmaker)  
    value_bet = prob_predite > prob_bookmaker  
    return value_bet, prob_bookmaker  

def enregistrer_historique(equipe_A, equipe_B, prediction):  
    historique = {"Équipe A": equipe_A, "Équipe B": equipe_B, "Prédiction": prediction}  
    st.session_state.historique.append(historique)  
    pd.DataFrame(st.session_state.historique).to_csv("historique_predictions.csv", index=False)  

# --- ANALYSE DES ÉQUIPES ---  
def page_analyse_equipes():  
    st.title("🏟️ Analyse des Équipes & Prédictions")  
    
    # Utilisation d'icônes pour représenter les équipes  
    col1, col2 = st.columns(2)  
    with col1:  
        st.markdown("""  
            <div style="text-align: center;">  
                <span style="font-size: 48px;">⚽</span>  
                <h3>Équipe A</h3>  
            </div>  
        """, unsafe_allow_html=True)  
        equipe_A = st.text_input("Nom de l'équipe A", "Équipe 1", key="equipe_A")  
    
    with col2:  
        st.markdown("""  
            <div style="text-align: center;">  
                <span style="font-size: 48px;">⚽</span>  
                <h3>Équipe B</h3>  
            </div>  
        """, unsafe_allow_html=True)  
        equipe_B = st.text_input("Nom de l'équipe B", "Équipe 2", key="equipe_B")  

    # Option pour téléverser un fichier CSV ou saisir manuellement les données  
    st.subheader("📊 Entrez vos données")  
    option_donnees = st.radio("Choisissez une option", ["Téléverser un fichier CSV", "Saisir manuellement les données"])  

    if option_donnees == "Téléverser un fichier CSV":  
        st.write("Téléversez un fichier CSV contenant les données des équipes.")  
        st.write("Le fichier doit contenir les colonnes suivantes : Possession (%), Tirs, Tirs cadrés, Passes réussies (%), xG, xGA, Corners, Fautes, Cartons jaunes, Cartons rouges, Domicile, Forme (pts), Classement, Buts marqués, Buts encaissés.")  
        fichier_csv = st.file_uploader("Téléversez un fichier CSV", type=["csv"])  
        if fichier_csv is not None:  
            try:  
                st.session_state.data = pd.read_csv(fichier_csv)  
                st.success("Fichier CSV téléversé avec succès !")  
            except Exception as e:  
                st.error(f"Erreur lors de la lecture du fichier CSV : {e}")  
    else:  
        # Saisie manuelle des données  
        st.write("Entrez les données pour chaque équipe :")  
        nombre_joueurs = st.number_input("Nombre de joueurs", min_value=1, max_value=30, value=22, step=1)  
        
        if st.button("Générer les données"):  
            np.random.seed(42)  
            st.session_state.data = pd.DataFrame({  
                'Possession (%)': np.random.uniform(40, 70, nombre_joueurs),  
                'Tirs': np.random.randint(5, 20, nombre_joueurs),  
                'Tirs cadrés': np.random.randint(2, 10, nombre_joueurs),  
                'Passes réussies (%)': np.random.uniform(60, 90, nombre_joueurs),  
                'xG': np.random.uniform(0.5, 3.0, nombre_joueurs),  
                'xGA': np.random.uniform(0.5, 3.0, nombre_joueurs),  
                'Corners': np.random.randint(2, 10, nombre_joueurs),  
                'Fautes': np.random.randint(5, 15, nombre_joueurs),  
                'Cartons jaunes': np.random.randint(0, 5, nombre_joueurs),  
                'Cartons rouges': np.random.randint(0, 2, nombre_joueurs),  
                'Domicile': np.random.choice([0, 1], nombre_joueurs),  
                'Forme (pts)': np.random.randint(3, 15, nombre_joueurs),  
                'Classement': np.random.randint(1, 20, nombre_joueurs),  
                'Buts marqués': np.random.randint(0, 4, nombre_joueurs),  
                'Buts encaissés': np.random.randint(0, 4, nombre_joueurs)  
            })  
            st.success("Données générées avec succès !")  

    if st.session_state.data is not None:  
        data = st.session_state.data  
        
        # Coloration pour les variables spécifiques au modèle de Poisson  
        poisson_cols = ['xG', 'xGA']  
        def color_poisson(col):  
            return ['background-color: yellow' if col.name in poisson_cols else '' for _ in col]  
        
        styled_data = data.style.apply(color_poisson)  
        st.dataframe(styled_data, height=400)  
        
        # --- MODÉLISATION ---  
        st.subheader("📈 Résultats des Modèles")  
        
        X = data.drop(columns=['Buts marqués'])  
        y = data['Buts marqués']  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
        
        # Régression Logistique  
        log_reg = LogisticRegression(max_iter=1000)  
        log_reg.fit(X_train, y_train)  
        y_pred_log = log_reg.predict(X_test)  
        acc_log = accuracy_score(y_test, y_pred_log)  
        report_log = classification_report(y_test, y_pred_log)  
        conf_matrix_log = confusion_matrix(y_test, y_pred_log)  
        
        # Random Forest  
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  
        rf.fit(X_train, y_train)  
        y_pred_rf = rf.predict(X_test)  
        acc_rf = accuracy_score(y_test, y_pred_rf)  
        report_rf = classification_report(y_test, y_pred_rf)  
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)  
        
        # Modèle de Poisson  
        poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()  
        y_pred_poisson = poisson_model.predict(X_test)  
        mse_poisson = mean_squared_error(y_test, y_pred_poisson)  
        
        # Affichage des résultats  
        col1, col2, col3 = st.columns(3)  
        with col1:  
            st.metric("Régression Logistique", f"Accuracy : {acc_log:.2f}")  
            st.write("Rapport de classification :")  
            st.text(report_log)  
            st.write("Matrice de confusion :")  
            st.write(conf_matrix_log)  
        
        with col2:  
            st.metric("Random Forest", f"Accuracy : {acc_rf:.2f}")  
            st.write("Rapport de classification :")  
            st.text(report_rf)  
            st.write("Matrice de confusion :")  
            st.write(conf_matrix_rf)  
        
        with col3:  
            st.metric("Modèle de Poisson", f"MSE : {mse_poisson:.2f}")  
            st.write("Prédictions :")  
            st.write(y_pred_poisson)  
        
        # Graphiques  
        st.subheader("📊 Visualisation des Performances")  
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))  
        sns.heatmap(conf_matrix_log, annot=True, fmt="d", ax=ax[0])  
        ax[0].set_title("Matrice de Confusion - Régression Logistique")  
        sns.heatmap(conf_matrix_rf, annot=True, fmt="d", ax=ax[1])  
        ax[1].set_title("Matrice de Confusion - Random Forest")  
        st.pyplot(fig)  
        
        enregistrer_historique(equipe_A, equipe_B, {"Logistique": acc_log, "RandomForest": acc_rf, "Poisson": mse_poisson})  

# --- COMPARATEUR DE COTES ---  
def page_comparateur_cotes():  
    st.title("💰 Comparateur de Cotes & Value Bets")  
    
    # Utilisation d'une icône pour représenter les cotes  
    st.markdown("""  
        <div style="text-align: center;">  
            <span style="font-size: 48px;">💰</span>  
        </div>  
    """, unsafe_allow_html=True)  
    
    cote_bookmaker = st.number_input("Cote du Bookmaker", min_value=1.01, max_value=10.0, value=2.5, step=0.01)  
    prob_predite = st.slider("Probabilité Prédite (%)", min_value=1, max_value=100, value=50) / 100  
    
    value_bet, prob_bookmaker = detecter_value_bet(prob_predite, cote_bookmaker)  
    st.write(f"Probabilité implicite du bookmaker : {prob_bookmaker:.2f}")  
    st.write(f"Value Bet détecté : {'✅ OUI' if value_bet else '❌ NON'}")  

# --- HISTORIQUE DES PRÉDICTIONS ---  
def page_historique_predictions():  
    st.title("📜 Historique des Prédictions")  
    
    # Utilisation d'une icône pour représenter l'historique  
    st.markdown("""  
        <div style="text-align: center;">  
            <span style="font-size: 48px;">📜</span>  
        </div>  
    """, unsafe_allow_html=True)  
    
    historique_df = pd.DataFrame(st.session_state.historique)  
    st.dataframe(historique_df, height=400)  

# --- NAVIGATION ---  
def main():  
    st.sidebar.title("🏆 Navigation")  
    page = st.sidebar.radio("Menu", ["🏟️ Analyse des Équipes", "💰 Comparateur de Cotes", "📜 Historique des Prédictions"])  
    
    if page == "🏟️ Analyse des Équipes":  
        page_analyse_equipes()  
    elif page == "💰 Comparateur de Cotes":  
        page_comparateur_cotes()  
    elif page == "📜 Historique des Prédictions":  
        page_historique_predictions()  

if __name__ == "__main__":  
    main()
