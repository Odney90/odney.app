import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def convertir_cote_en_probabilite(cote):
    return 1 / cote if cote > 0 else 0

# Données fictives modifiables
stats_equipe_A = {
    "score_rating": 85,
    "buts_par_match": 2.1,
    "buts_produits": 42,
    "buts_concédés": 20,
    "buts_concédés_par_match": 1.0,
    "possession": 58.5,
    "matchs_sans_encaisser": 8,
    "xG": 1.9,
    "tirs_cadrés": 5.2,
    "conversion_tirs": 15.4,
    "grosses_occasions": 45,
    "grosses_occasions_ratées": 22,
    "passes_réussies": 520,
    "passes_longues": 30,
    "centres_réussis": 4.3,
    "pénalties_obtenus": 3,
    "touches_surface": 22,
    "corners": 6.1,
    "corners_concédés": 4.2,
    "xG_concédés": 1.1,
    "interceptions": 9.5,
    "tacles_réussis": 12.3,
    "dégagements": 18,
    "possession_récupérée": 11,
    "pénalties_concédés": 2,
    "arrêts_par_match": 3.5,
    "fautes": 12,
    "cartons_jaunes": 3,
    "cartons_rouges": 0,
    "tactique": "4-3-3",
    "joueurs_clés": {"attaquant": 90, "milieu": 85, "défenseur": 88}
}

stats_equipe_B = {
    "score_rating": 78,
    "buts_par_match": 1.8,
    "buts_produits": 36,
    "buts_concédés": 28,
    "buts_concédés_par_match": 1.4,
    "possession": 52.3,
    "matchs_sans_encaisser": 6,
    "xG": 1.6,
    "tirs_cadrés": 4.8,
    "conversion_tirs": 12.7,
    "grosses_occasions": 38,
    "grosses_occasions_ratées": 19,
    "passes_réussies": 480,
    "passes_longues": 28,
    "centres_réussis": 3.8,
    "pénalties_obtenus": 2,
    "touches_surface": 18,
    "corners": 5.5,
    "corners_concédés": 5.0,
    "xG_concédés": 1.4,
    "interceptions": 8.9,
    "tacles_réussis": 11.8,
    "dégagements": 20,
    "possession_récupérée": 10,
    "pénalties_concédés": 3,
    "arrêts_par_match": 2.8,
    "fautes": 14,
    "cartons_jaunes": 4,
    "cartons_rouges": 1,
    "tactique": "4-4-2",
    "joueurs_clés": {"attaquant": 85, "milieu": 80, "défenseur": 82}
}

# Prédiction du match avec Logistic Regression et Random Forest
def predire_resultat(stats_A, stats_B):
    features = [stats_A["buts_par_match"], stats_A["xG"], stats_B["buts_concédés_par_match"], stats_B["xG_concédés"]]
    X = np.array([features])
    
    # Modèles fictifs entraînés ailleurs
    log_reg = LogisticRegression()
    log_reg.fit([[2.1, 1.9, 1.0, 1.1]], [1])
    rf_model = RandomForestClassifier()
    rf_model.fit([[2.1, 1.9, 1.0, 1.1]], [1])
    
    prob_log_reg = log_reg.predict_proba(X)[0][1]
    prob_rf = rf_model.predict_proba(X)[0][1]
    
    return prob_log_reg, prob_rf

# Cotes fictives
cote_A, cote_B, cote_nul = 2.10, 3.50, 3.20
prob_A = convertir_cote_en_probabilite(cote_A)
prob_B = convertir_cote_en_probabilite(cote_B)
prob_nul = convertir_cote_en_probabilite(cote_nul)

# Mise de Kelly
def mise_kelly(prob, cote, bankroll):
    edge = (prob * cote - 1) / (cote - 1)
    return max(0, edge * bankroll)

bankroll = 1000
mise_A = mise_kelly(prob_A, cote_A, bankroll)
mise_B = mise_kelly(prob_B, cote_B, bankroll)
solde_restant = bankroll - (mise_A + mise_B)

# Affichage des stats et des prédictions
def afficher_stats():
    print("Statistiques de l'équipe A:", stats_equipe_A)
    print("Statistiques de l'équipe B:", stats_equipe_B)
    pred_log, pred_rf = predire_resultat(stats_equipe_A, stats_equipe_B)
    print(f"Prédiction (Logistic Regression): {pred_log:.2%}")
    print(f"Prédiction (Random Forest): {pred_rf:.2%}")
    print(f"Mise de Kelly sur A: {mise_A:.2f}€")
    print(f"Mise de Kelly sur B: {mise_B:.2f}€")
    print(f"Solde restant: {solde_restant:.2f}€")

afficher_stats()
