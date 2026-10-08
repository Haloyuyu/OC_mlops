"""Détection de dérive entre une référence (jeu d'entraînement) et la production.

Deux indicateurs complémentaires par feature :

- **PSI** (Population Stability Index) : taille d'effet, l'indicateur de référence du
  risque crédit. Seuils d'usage : < 0,1 stable, 0,1–0,25 dérive modérée, > 0,25 dérive.
- **Kolmogorov-Smirnov** : test statistique, qui dit si l'écart est significatif. Avec
  33 features testées, un seuil à 5 % produirait ~1,7 fausse alerte par analyse : le
  seuil est donc corrigé par Bonferroni (0,05 / nombre de features).

Un indicateur ne suffit pas : KS devient significatif pour des écarts minuscules dès
que le volume monte, et le PSI seul ne dit rien de la fiabilité de l'écart. Une feature
n'est déclarée en dérive que si l'effet est net (PSI ≥ 0,25), ou s'il est modéré ET
statistiquement significatif.

Le **taux de valeurs manquantes** est suivi séparément : en production, une feature qui
cesse d'être renseignée (changement de formulaire, panne d'un fournisseur de données)
est au moins aussi fréquent qu'un vrai glissement de population, et le PSI calculé sur
les valeurs présentes ne le verrait pas.
"""

import numpy as np
import pandas as pd
from scipy import stats

PSI_MODERE = 0.1
PSI_DERIVE = 0.25
ALPHA = 0.05
N_MINIMUM = 30          # en dessous, l'échantillon de production ne prouve rien
ECART_MANQUANTS = 0.10  # 10 points d'écart sur le taux de valeurs manquantes
N_BINS = 10


def psi(reference, courant, n_bins=N_BINS):
    """Population Stability Index entre deux échantillons (valeurs manquantes exclues).

    Les bornes viennent des quantiles de la référence : chaque classe contient ~10 %
    de la référence, ce qui rend l'indicateur insensible à la forme de la distribution.
    Renvoie NaN si la référence est trop pauvre pour être découpée.
    """
    reference = pd.Series(reference).dropna().to_numpy(dtype="float64")
    courant = pd.Series(courant).dropna().to_numpy(dtype="float64")
    if len(reference) < n_bins or len(courant) == 0:
        return float("nan")

    bornes = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(bornes) < 3:            # features quasi constantes : PSI non interprétable
        return float("nan")
    bornes[0], bornes[-1] = -np.inf, np.inf

    part_ref = np.histogram(reference, bins=bornes)[0] / len(reference)
    part_cur = np.histogram(courant, bins=bornes)[0] / len(courant)
    eps = 1e-6                      # évite les divisions par zéro sur les classes vides
    part_ref, part_cur = np.clip(part_ref, eps, None), np.clip(part_cur, eps, None)
    return float(np.sum((part_cur - part_ref) * np.log(part_cur / part_ref)))


def kolmogorov_smirnov(reference, courant):
    """Renvoie (statistique, p-value) du test KS à deux échantillons."""
    reference = pd.Series(reference).dropna().to_numpy(dtype="float64")
    courant = pd.Series(courant).dropna().to_numpy(dtype="float64")
    if len(reference) < 2 or len(courant) < 2:
        return float("nan"), float("nan")
    resultat = stats.ks_2samp(reference, courant)
    return float(resultat.statistic), float(resultat.pvalue)


def _verdict(psi_valeur, p_value, alpha, n_courant, ecart_manquants):
    if n_courant < N_MINIMUM:
        return "échantillon insuffisant"
    if abs(ecart_manquants) > ECART_MANQUANTS:
        return "dérive des valeurs manquantes"
    if np.isnan(psi_valeur):
        return "non évaluable"
    significatif = not np.isnan(p_value) and p_value < alpha
    if psi_valeur >= PSI_DERIVE or (psi_valeur >= PSI_MODERE and significatif):
        return "dérive"
    if psi_valeur >= PSI_MODERE or significatif:
        return "dérive modérée"
    return "stable"


def analyser_features(reference, production, features=None):
    """Compare chaque feature suivie entre référence et production.

    Renvoie un DataFrame trié par PSI décroissant : une ligne par feature, avec
    effectifs, taux de valeurs manquantes, PSI, KS et verdict.
    """
    features = list(features or [c for c in reference.columns if c in production.columns])
    alpha = ALPHA / max(len(features), 1)       # correction de Bonferroni
    lignes = []
    for feature in features:
        ref, cur = reference[feature], production[feature]
        manquants_ref = float(ref.isna().mean())
        manquants_cur = float(cur.isna().mean())
        psi_valeur = psi(ref, cur)
        ks_stat, p_value = kolmogorov_smirnov(ref, cur)
        lignes.append({
            "feature": feature,
            "n_reference": int(ref.notna().sum()),
            "n_production": int(cur.notna().sum()),
            "moyenne_reference": float(ref.mean()) if ref.notna().any() else float("nan"),
            "moyenne_production": float(cur.mean()) if cur.notna().any() else float("nan"),
            "manquants_reference": manquants_ref,
            "manquants_production": manquants_cur,
            "psi": psi_valeur,
            "ks_statistique": ks_stat,
            "ks_p_value": p_value,
            "verdict": _verdict(psi_valeur, p_value, alpha, len(cur), manquants_cur - manquants_ref),
        })
    resultats = pd.DataFrame(lignes)
    return resultats.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def analyser_predictions(reference, production, seuil):
    """Dérive des sorties du modèle : distribution des scores et taux de refus.

    C'est le signal le plus parlant pour le métier : même sans dérive visible sur une
    feature, un taux de refus qui bouge de plusieurs points change le coût métier.
    """
    ref = pd.Series(reference).dropna()
    cur = pd.Series(production).dropna()
    psi_valeur = psi(ref, cur)
    ks_stat, p_value = kolmogorov_smirnov(ref, cur)
    taux_ref = float((ref >= seuil).mean()) if len(ref) else float("nan")
    taux_cur = float((cur >= seuil).mean()) if len(cur) else float("nan")
    return {
        "n_reference": int(len(ref)),
        "n_production": int(len(cur)),
        "score_moyen_reference": float(ref.mean()) if len(ref) else float("nan"),
        "score_moyen_production": float(cur.mean()) if len(cur) else float("nan"),
        "taux_refus_reference": taux_ref,
        "taux_refus_production": taux_cur,
        "ecart_taux_refus": taux_cur - taux_ref,
        "psi": psi_valeur,
        "ks_statistique": ks_stat,
        "ks_p_value": p_value,
        "verdict": _verdict(psi_valeur, p_value, ALPHA, int(len(cur)), 0.0),
    }


def synthese(features_resultats, predictions_resultats=None, part_alerte=0.2):
    """Verdict global : ALERTE, VIGILANCE ou STABLE, avec ce qui l'a déclenché."""
    en_derive = features_resultats[features_resultats["verdict"].str.startswith("dérive")
                                   & ~features_resultats["verdict"].str.contains("modérée")]
    moderees = features_resultats[features_resultats["verdict"] == "dérive modérée"]
    evaluables = features_resultats[~features_resultats["verdict"].isin(
        ["échantillon insuffisant", "non évaluable"])]
    part = len(en_derive) / len(evaluables) if len(evaluables) else 0.0

    motifs = []
    if len(en_derive):
        motifs.append(f"{len(en_derive)} feature(s) en dérive : "
                      + ", ".join(en_derive["feature"].head(5)))
    if predictions_resultats and predictions_resultats["verdict"].startswith("dérive"):
        motifs.append("dérive de la distribution des scores")
    if predictions_resultats and abs(predictions_resultats.get("ecart_taux_refus") or 0) > 0.05:
        motifs.append(f"taux de refus déplacé de "
                      f"{predictions_resultats['ecart_taux_refus']:+.1%}")

    if part > part_alerte or (predictions_resultats
                             and abs(predictions_resultats.get("ecart_taux_refus") or 0) > 0.05):
        niveau = "ALERTE"
    elif motifs or len(moderees):
        niveau = "VIGILANCE"
    else:
        niveau = "STABLE"
    return {
        "niveau": niveau,
        "features_suivies": int(len(features_resultats)),
        "features_evaluables": int(len(evaluables)),
        "features_en_derive": int(len(en_derive)),
        "features_derive_moderee": int(len(moderees)),
        "part_en_derive": round(part, 3),
        "motifs": motifs,
    }
