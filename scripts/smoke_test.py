"""Test de fumée d'une API déployée (utilisé par l'étape de déploiement du pipeline).

    python scripts/smoke_test.py http://localhost:8000

Uniquement la bibliothèque standard : s'exécute sur n'importe quel hôte cible.
"""

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CLIENTS_EXEMPLE = Path(__file__).resolve().parent.parent / "model" / "clients_exemple.csv"


def requete(url, corps=None):
    donnees = None if corps is None else json.dumps(corps).encode()
    req = urllib.request.Request(url, data=donnees, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as reponse:
        return reponse.status, reponse.read().decode()


def attendre_api(url, delai_max=120):
    debut = time.monotonic()
    while time.monotonic() - debut < delai_max:
        try:
            if requete(f"{url}/health")[0] == 200:
                return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError(f"L'API {url} ne répond pas après {delai_max} s")


def premier_client():
    with open(CLIENTS_EXEMPLE, encoding="utf-8") as f:
        ligne = next(csv.DictReader(f))
    ligne.pop("SK_ID_CURR")
    return {nom: (float(valeur) if valeur else None) for nom, valeur in ligne.items()}


def main(url):
    url = url.rstrip("/")
    attendre_api(url)

    statut, corps = requete(f"{url}/health")
    print(f"GET  /health  -> {statut} {corps}")

    statut, corps = requete(f"{url}/predict", {"features": premier_client()})
    prediction = json.loads(corps)
    print(f"POST /predict -> {statut} {prediction}")
    assert statut == 200 and 0 <= prediction["probabilite_defaut"] <= 1

    statut, _ = requete(f"{url}/ui/")
    print(f"GET  /ui/     -> {statut}")
    assert statut == 200

    print("Déploiement vérifié.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
