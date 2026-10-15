"""Simule du trafic de production sur l'API, pour alimenter le suivi.

Sans trafic réel, rien à analyser : ce script envoie de vrais clients du jeu test à
l'API, et sait fabriquer une population dérivée pour vérifier que la détection
fonctionne (et non seulement qu'elle ne crie pas au loup).

    # trafic nominal
    python scripts/simulate_traffic.py --n 300

    # population dérivée + appels fautifs (erreurs 422 et 401)
    python scripts/simulate_traffic.py --n 200 --profil derive --erreurs 20

Prérequis : l'API doit tourner avec la même clé. Celle-ci est lue dans `.env`, ou
dans la variable d'environnement `API_KEY` si elle est déjà exportée.
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
import requests

from _env import charger_env

RACINE = Path(__file__).resolve().parent.parent
DONNEES = RACINE / "data" / "processed_data.csv"
CACHE = RACINE / "data" / "clients_simulation.csv"
N_CACHE = 1000

# Dérive simulée : une population plus risquée et plus endettée. Les valeurs restent
# dans les bornes métier de l'API, sinon les appels seraient refusés en 422 et la
# dérive ne serait jamais mesurée sur les données.
DERIVE = {
    "EXT_SOURCE_1": lambda s: (s - 0.15).clip(0.001, 1),
    "EXT_SOURCE_2": lambda s: (s - 0.15).clip(0.001, 1),
    "EXT_SOURCE_3": lambda s: (s - 0.12).clip(0.001, 1),
    "AMT_CREDIT": lambda s: s * 1.4,
    "AMT_INCOME_TOTAL": lambda s: s * 0.8,
    "DAYS_EMPLOYED": lambda s: s * 0.4,
}


def charger_clients(n):
    """Clients du jeu test, mis en cache pour ne pas relire 1,4 Go à chaque passage."""
    if CACHE.exists() and len(pd.read_csv(CACHE, nrows=1).columns) > 1:
        clients = pd.read_csv(CACHE)
        if len(clients) >= n:
            return clients.head(n)
    morceaux, total = [], 0
    for morceau in pd.read_csv(DONNEES, chunksize=50_000):
        morceau.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in morceau.columns]
        test = morceau.loc[morceau["TARGET"].isna()]
        if not test.empty:
            morceaux.append(test)
            total += len(test)
        if total >= max(n, N_CACHE):
            break
    clients = pd.concat(morceaux).head(max(n, N_CACHE))
    clients = clients.drop(columns=[c for c in ("TARGET", "SK_ID_CURR", "index") if c in clients])
    clients.astype("float64").to_csv(CACHE, index=False)
    print(f"{len(clients)} clients mis en cache dans {CACHE.relative_to(RACINE)}")
    return clients.head(n)


def appliquer_derive(clients):
    clients = clients.copy()
    for colonne, transformation in DERIVE.items():
        if colonne in clients:
            clients[colonne] = transformation(clients[colonne])
    # cohérence des features dérivées, sinon la dérive serait incohérente avec le dossier
    if {"AMT_ANNUITY", "AMT_CREDIT"} <= set(clients):
        clients["PAYMENT_RATE"] = clients["AMT_ANNUITY"] / clients["AMT_CREDIT"]
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= set(clients):
        clients["ANNUITY_INCOME_PERC"] = clients["AMT_ANNUITY"] / clients["AMT_INCOME_TOTAL"]
    return clients


def en_features(ligne):
    return {nom: (None if pd.isna(valeur) else float(valeur)) for nom, valeur in ligne.items()}


def envoyer(url, cle, features, timeout=30):
    reponse = requests.post(f"{url}/predict", json={"features": features},
                            headers={"X-API-Key": cle}, timeout=timeout)
    return reponse.status_code


def envoyer_erreurs(url, cle, clients, n):
    """Appels fautifs : entrées aberrantes, feature inconnue, et appel sans clé."""
    fautes = [
        lambda f: {**f, "AMT_INCOME_TOTAL": 0.0},          # revenu nul -> 422
        lambda f: {**f, "DAYS_BIRTH": 1825.0},             # âge négatif -> 422
        lambda f: {**f, "REVENU_IMAGINAIRE": 1.0},         # feature inconnue -> 422
    ]
    statuts = []
    for i in range(n):
        features = en_features(clients.iloc[i % len(clients)])
        if i % 4 == 3:
            reponse = requests.post(f"{url}/predict", json={"features": features},
                                    headers={"X-API-Key": "mauvaise-cle"}, timeout=30)
            statuts.append(reponse.status_code)           # 401
        else:
            statuts.append(envoyer(url, cle, fautes[i % len(fautes)](features)))
    return statuts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=300, help="nombre d'appels nominaux")
    parser.add_argument("--profil", choices=["normal", "derive"], default="normal")
    parser.add_argument("--erreurs", type=int, default=0, help="nombre d'appels fautifs")
    parser.add_argument("--graine", type=int, default=None)
    arguments = parser.parse_args()

    charger_env()                     # .env si la variable n'est pas déjà exportée
    cle = os.getenv("API_KEY", "")
    if not cle:
        print("Aucune clé d'API : renseignez API_KEY dans .env (cf. .env.example) "
              "ou exportez-la avant de lancer la simulation.")
        return 1
    if arguments.graine is not None:
        random.seed(arguments.graine)

    clients = charger_clients(arguments.n)
    if arguments.profil == "derive":
        clients = appliquer_derive(clients)

    try:
        statuts = [envoyer(arguments.url, cle, en_features(ligne))
                   for _, ligne in clients.iterrows()]
    except requests.exceptions.ConnectionError:
        print(f"Aucune API sur {arguments.url} : lancez-la avant la simulation.")
        return 1

    if arguments.erreurs:
        statuts += envoyer_erreurs(arguments.url, cle, clients, arguments.erreurs)

    repartition = pd.Series(statuts).value_counts().sort_index()
    print(f"Profil « {arguments.profil} » : {len(statuts)} appels envoyés à {arguments.url}")
    for statut, nombre in repartition.items():
        print(f"  {statut} : {nombre}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
