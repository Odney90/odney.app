import streamlit as st  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
import statsmodels.api as sm  
from sklearn.model_selection import train_test_split, cross_val_score, KFold  
from sklearn.metrics import accuracy_score, mean_squared_error  

# --- CONFIG INTERFACE ---  
st.set_page_config(page_title="Analyse Foot & Value Bets", layout="wide")  

# Utilisation d'une icône pour le logo dans la barre latérale  
st.sidebar.markdown("""  
    <div style="text-align: center;">  
        <span style="font-size: 48px;">🏆</span>  
    </div>  
""", unsafe_allow_html=True)  

st.sidebar.title("🏆 Navigation")  
page = st.sidebar.radio("Menu", ["🏟️ Analyse des Équipes", "💰 Comparateur de Cotes", "📜 Historique des Prédictions"])  

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
if page == "🏟️ Analyse des Équipes":  
    st.title("Analyse des Équipes & Prédictions")  
    
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
    st.subheader("Entrez vos données")  
    option_donnees = st.radio("Choisissez une option", ["Téléverser un fichier CSV", "Saisir manuellement les données"])  

    if option_donnees == "Téléverser un fichier CSV":  
        fichier_csv = st.file_uploader("Téléversez un fichier CSV", type=["csv"])  
        if fichier_csv is not None:  
            st.session_state.data = pd.read_csv(fichier_csv)  
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

    if st.session_state.data is not None:  
        data = st.session_state.data  
        
        # Coloration pour les variables spécifiques au modèle de Poisson  
        poisson_cols = ['xG', 'xGA']  
        def color_poisson(col):  
            return ['background-color: yellow' if col.name in poisson_cols else '' for _ in col]  
        
        styled_data = data.style.apply(color_poisson)  
        st.dataframe(styled_data)  
        
        # --- MODÉLISATION ---  
        X = data.drop(columns=['Buts marqués'])  
        y = data['Buts marqués']  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  
        
        log_reg = LogisticRegression(max_iter=1000)  
        log_reg.fit(X_train, y_train)  
        acc_log = np.mean(cross_val_score(log_reg, X_train, y_train, cv=kf, scoring='accuracy'))  
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  
        rf.fit(X_train, y_train)  
        acc_rf = np.mean(cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy'))  
        
        poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()  
        mse_poisson = mean_squared_error(y_test, poisson_model.predict(X_test))  
        
        st.subheader("Résultats des Modèles")  
        
        # Utilisation d'icônes pour représenter les résultats des modèles  
        col1, col2, col3 = st.columns(3)  
        with col1:  
            st.markdown("""  
                <div style="text-align: center;">  
                    <span style="font-size: 48px;">📊</span>  
                    <h4>Régression Logistique</h4>  
                    <p>Accuracy : {:.2f}</p>  
                </div>  
            """.format(acc_log), unsafe_allow_html=True)  
        
        with col2:  
            st.markdown("""  
                <div style="text-align: center;">  
                    <span style="font-size: 48px;">🌲</span>  
                    <h4>Random Forest</h4>  
                    <p>Accuracy : {:.2f}</p>  
                </div>  
            """.format(acc_rf), unsafe_allow_html=True)  
        
        with col3:  
            st.markdown("""  
                <div style="text-align: center;">  
                    <span style="font-size: 48px;">📉</span>  
                    <h4>Modèle de Poisson</h4>  
                    <p>MSE : {:.2f}</p>  
                </div>  
            """.format(mse_poisson), unsafe_allow_html=True)  
        
        enregistrer_historique(equipe_A, equipe_B, {"Logistique": acc_log, "RandomForest": acc_rf, "Poisson": mse_poisson})  

# --- COMPARATEUR DE COTES ---  
if page == "💰 Comparateur de Cotes":  
    st.title("Comparateur de Cotes & Value Bets")  
    
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
if page == "📜 Historique des Prédictions":  
    st.title("Historique des Prédictions")  
    
    # Utilisation d'une icône pour représenter l'historique  
    st.markdown("""  
        <div style="text-align: center;">  
            <span style="font-size: 48px;">📜</span>  
        </div>  
    """, unsafe_allow_html=True)  
    
    historique_df = pd.DataFrame(st.session_state.historique)  
    st.dataframe(historique_df)
