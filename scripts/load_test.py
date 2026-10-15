"""Test de charge HTTP de l'API : temps de réponse de bout en bout, vu du client.

Envoie des clients du jeu test à `/predict` à plusieurs niveaux de concurrence et
mesure, côté client, la latence (p50/p95/p99) et le débit. Si le dossier de journaux
de l'API est fourni, y ajoute la vue serveur relevée par le monitoring de
production : latence du modèle, CPU et mémoire du processus.

    # API lancée avec un dossier de journaux dédié à la mesure
    MONITORING_DIR=logs/charge-booster uvicorn api.main:app --port 8000 --env-file .env
    python scripts/load_test.py --label booster --journaux logs/charge-booster

Résultats : docs/optimisation/load_test_<label>.json.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from _env import charger_env
from simulate_traffic import charger_clients, en_features

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "docs" / "optimisation"


def percentiles(valeurs):
    v = np.asarray([x for x in valeurs if x is not None and x == x], dtype="float64")
    if v.size == 0:
        return None
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def palier(url, cle, charges, n, concurrence):
    """n appels à `concurrence` clients simultanés (une session HTTP par thread)."""
    sessions = {}

    def appel(i):
        session = sessions.setdefault(i % concurrence, requests.Session())
        debut = time.perf_counter()
        reponse = session.post(f"{url}/predict", data=charges[i % len(charges)],
                               headers={"X-API-Key": cle, "Content-Type": "application/json"},
                               timeout=60)
        return (time.perf_counter() - debut) * 1000, reponse.status_code

    with ThreadPoolExecutor(max_workers=concurrence) as pool:
        list(pool.map(appel, range(concurrence * 2)))       # échauffement
        debut = time.perf_counter()
        resultats = list(pool.map(appel, range(n)))
        duree = time.perf_counter() - debut
    latences = [ms for ms, _ in resultats]
    erreurs = sum(1 for _, statut in resultats if statut != 200)
    return {"concurrence": concurrence, "appels": n, "erreurs": erreurs,
            "debit_par_s": n / duree, "latence_client_ms": percentiles(latences)}


def vue_serveur(dossier):
    """Latences, CPU et mémoire relevés par l'API elle-même dans ses journaux."""
    evenements, illisibles = [], 0
    for fichier in sorted(Path(dossier).glob("production-*.jsonl")):
        for ligne in fichier.read_bytes().splitlines():
            if not ligne.strip():
                continue
            try:
                evenements.append(json.loads(ligne))
            except ValueError:   # ligne entrelacée par des écritures concurrentes
                illisibles += 1
    predictions = [e for e in evenements if e.get("evenement") == "prediction"
                   and e.get("statut") == 200]
    requetes = [e for e in evenements if e.get("evenement") == "requete"
                and e.get("route") == "/predict"]
    return {
        "predictions": len(predictions),
        "lignes_illisibles": illisibles,
        "latence_modele_ms": percentiles(e.get("latence_modele_ms") for e in predictions),
        "latence_requete_ms": percentiles(e.get("latence_ms") for e in requetes),
        "cpu_pct": percentiles(e.get("cpu_pct") for e in predictions),
        "memoire_mo_max": max((e.get("memoire_mo") or 0 for e in predictions), default=None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True, help="nom de la configuration mesurée")
    parser.add_argument("--n", type=int, default=400, help="appels par palier")
    parser.add_argument("--concurrence", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--journaux", help="dossier MONITORING_DIR de l'API (vue serveur)")
    arguments = parser.parse_args()

    charger_env()
    cle = os.getenv("API_KEY", "")
    if not cle:
        print("Aucune clé d'API : renseignez API_KEY dans .env.")
        return 1
    clients = charger_clients(200)
    # corps JSON préparés à l'avance : on mesure l'API, pas la sérialisation du client
    charges = [json.dumps({"features": en_features(ligne)}) for _, ligne in clients.iterrows()]

    try:
        requests.get(f"{arguments.url}/health", timeout=10).raise_for_status()
    except requests.exceptions.RequestException as erreur:
        print(f"API indisponible sur {arguments.url} : {erreur}")
        return 1

    paliers = []
    for concurrence in arguments.concurrence:
        resultat = palier(arguments.url, cle, charges, arguments.n, concurrence)
        paliers.append(resultat)
        lat = resultat["latence_client_ms"]
        print(f"{arguments.label:>10} | {concurrence:>2} clients | p50 {lat['p50']:7.1f} ms | "
              f"p95 {lat['p95']:7.1f} ms | p99 {lat['p99']:7.1f} ms | "
              f"{resultat['debit_par_s']:6.1f} appels/s | {resultat['erreurs']} erreur(s)")

    sortie = {"label": arguments.label, "date": time.strftime("%Y-%m-%d %H:%M"),
              "url": arguments.url, "paliers": paliers}
    if arguments.journaux:
        sortie["serveur"] = vue_serveur(arguments.journaux)
        serveur = sortie["serveur"]
        if serveur["latence_modele_ms"]:
            print(f"Serveur : modèle p50 {serveur['latence_modele_ms']['p50']:.2f} ms, "
                  f"p95 {serveur['latence_modele_ms']['p95']:.2f} ms ; CPU p50 "
                  f"{serveur['cpu_pct']['p50']:.0f} %, p95 {serveur['cpu_pct']['p95']:.0f} % ; "
                  f"mémoire max {serveur['memoire_mo_max']:.0f} Mo ; "
                  f"{serveur['lignes_illisibles']} ligne(s) de journal illisible(s)")

    SORTIE.mkdir(parents=True, exist_ok=True)
    chemin = SORTIE / f"load_test_{arguments.label}.json"
    chemin.write_text(json.dumps(sortie, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Résultats : {chemin.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
