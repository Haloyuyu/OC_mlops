"""Analyse d'exploitation : taux d'erreur et latences anormales.

Deux façons de juger une latence, et il faut les deux :

- par rapport à un **objectif de service** (SLO) fixé à l'avance, sinon une API lente
  depuis toujours paraît normale ;
- par rapport à son **propre passé** : une latence qui triple reste sous le SLO
  pendant un moment, mais annonce la panne.

Les seuils ci-dessous sont dimensionnés sur la mesure réelle de l'API et se règlent
par `SEUILS`. Depuis l'optimisation de l'inférence (docs/rapport_optimisation.md), une
prédiction répond en ~5 ms (p95 ~6 ms) contre ~40 ms (p95 ~120 ms) auparavant : le SLO
a été abaissé en conséquence, sans quoi une régression vers l'ancien chemin
d'inférence passerait inaperçue.
"""

import pandas as pd

SEUILS = {
    "taux_erreur_serveur": 0.01,   # 1 % de 5xx
    "taux_erreur_client": 0.20,    # 20 % de 4xx : signale un appelant mal intégré
    "latence_p95_ms": 100.0,       # SLO : ~15× le p95 nominal (6 ms), sous l'ancien (120 ms)
    "facteur_degradation": 3.0,    # p95 récent / p95 de référence
}


def percentiles(latences):
    valeurs = pd.Series(latences).dropna()
    if valeurs.empty:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "p50": float(valeurs.quantile(0.50)),
        "p95": float(valeurs.quantile(0.95)),
        "p99": float(valeurs.quantile(0.99)),
        "max": float(valeurs.max()),
    }


def _taux(requetes, minimum, maximum=None):
    if requetes.empty:
        return 0.0
    statut = requetes["statut"]
    selection = statut >= minimum if maximum is None else (statut >= minimum) & (statut < maximum)
    return float(selection.mean())


def analyser(requetes, seuils=None, fenetre_recente=0.25):
    """Métriques d'exploitation et anomalies, à partir des événements de requête.

    `fenetre_recente` : part la plus récente des appels comparée au reste, pour
    détecter une dégradation progressive de la latence.
    """
    seuils = {**SEUILS, **(seuils or {})}
    requetes = requetes.sort_values("horodatage") if not requetes.empty else requetes
    scoring = requetes[requetes["route"] == "/predict"] if not requetes.empty else requetes

    mesures = {
        "appels": int(len(requetes)),
        "appels_predict": int(len(scoring)),
        "periode_debut": requetes["horodatage"].min() if not requetes.empty else None,
        "periode_fin": requetes["horodatage"].max() if not requetes.empty else None,
        "taux_erreur_serveur": _taux(requetes, 500),
        "taux_erreur_client": _taux(requetes, 400, 500),
        "taux_succes": _taux(requetes, 200, 300),
        "par_statut": (requetes["statut"].value_counts().sort_index().to_dict()
                       if not requetes.empty else {}),
        "latences_ms": percentiles(scoring["latence_ms"] if not scoring.empty else []),
    }

    # dégradation : derniers appels contre les précédents
    degradation = None
    if len(scoring) >= 40:
        coupure = int(len(scoring) * (1 - fenetre_recente))
        p95_passe = percentiles(scoring["latence_ms"].iloc[:coupure])["p95"]
        p95_recent = percentiles(scoring["latence_ms"].iloc[coupure:])["p95"]
        if p95_passe and p95_passe > 0:
            degradation = {"p95_precedent_ms": p95_passe, "p95_recent_ms": p95_recent,
                           "facteur": round(p95_recent / p95_passe, 2)}
    mesures["degradation_latence"] = degradation

    anomalies = []
    if mesures["taux_erreur_serveur"] > seuils["taux_erreur_serveur"]:
        anomalies.append(f"taux d'erreurs serveur {mesures['taux_erreur_serveur']:.1%} "
                         f"> {seuils['taux_erreur_serveur']:.1%}")
    if mesures["taux_erreur_client"] > seuils["taux_erreur_client"]:
        anomalies.append(f"taux d'erreurs client {mesures['taux_erreur_client']:.1%} "
                         f"> {seuils['taux_erreur_client']:.0%} : entrées refusées en masse")
    p95 = mesures["latences_ms"]["p95"]
    if p95 == p95 and p95 > seuils["latence_p95_ms"]:
        anomalies.append(f"latence p95 {p95:.0f} ms > {seuils['latence_p95_ms']:.0f} ms")
    if degradation and degradation["facteur"] > seuils["facteur_degradation"]:
        anomalies.append(f"latence p95 multipliée par {degradation['facteur']} "
                         "sur les appels les plus récents")

    mesures["anomalies"] = anomalies
    mesures["niveau"] = "ALERTE" if anomalies else "STABLE"
    return mesures


def erreurs_frequentes(predictions, limite=5):
    """Causes de refus les plus fréquentes, pour savoir quoi corriger côté appelant."""
    if predictions.empty or "erreur_type" not in predictions:
        return []
    erreurs = predictions.dropna(subset=["erreur_type"])
    if erreurs.empty:
        return []
    comptes = erreurs.groupby(["erreur_type", "erreur_message"]).size()
    return [{"type": t, "message": m[:160], "occurrences": int(n)}
            for (t, m), n in comptes.sort_values(ascending=False).head(limite).items()]


def ressources(predictions):
    """Occupation CPU et mémoire du processus de l'API, relevées à chaque prédiction.

    Les journaux antérieurs à cette mesure n'ont pas les colonnes (ou les ont vides) :
    on renvoie alors des valeurs NaN plutôt qu'une erreur.
    """
    vide = {"cpu_pct": percentiles([]), "memoire_mo_max": float("nan"), "mesures": 0}
    if predictions.empty or "cpu_pct" not in predictions:
        return vide
    mesures = predictions.dropna(subset=["cpu_pct"])
    if mesures.empty:
        return vide
    return {
        "cpu_pct": percentiles(mesures["cpu_pct"]),
        "memoire_mo_max": float(mesures["memoire_mo"].max()),
        "mesures": int(len(mesures)),
    }
