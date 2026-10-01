"""Test de fumée d'une API déployée (utilisé par l'étape de déploiement du pipeline).

    python scripts/smoke_test.py http://localhost:8000

Uniquement la bibliothèque standard : s'exécute sur n'importe quel hôte cible.
"""

import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

EXAMPLE_CLIENTS = Path(__file__).resolve().parent.parent / "model" / "clients_exemple.csv"
# Même clé que celle passée au conteneur (secret GitHub en CI, variable en local).
API_KEY = os.getenv("API_KEY", "")


def call_api(url, body=None, cle=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json",
               "X-API-Key": API_KEY if cle is None else cle}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read().decode()


def wait_for_api(url, timeout=120):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            if call_api(f"{url}/health")[0] == 200:
                return
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError(f"L'API {url} ne répond pas après {timeout} s")


def first_client():
    with open(EXAMPLE_CLIENTS, encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    row.pop("SK_ID_CURR")
    return {name: (float(value) if value else None) for name, value in row.items()}


def main(url):
    url = url.rstrip("/")
    wait_for_api(url)

    status, body = call_api(f"{url}/health")
    print(f"GET  /health  -> {status} {body}")

    try:
        status, body = call_api(f"{url}/predict", {"features": first_client()})
    except urllib.error.HTTPError as erreur:
        if erreur.code == 401:
            print("Clé d'API absente ou invalide : exportez la variable API_KEY utilisée "
                  "par l'API avant de lancer ce test.")
            return 1
        raise
    prediction = json.loads(body)
    print(f"POST /predict -> {status} {prediction}")
    assert status == 200 and 0 <= prediction["probabilite_defaut"] <= 1

    status, _ = call_api(f"{url}/ui/")
    print(f"GET  /ui/     -> {status}")
    assert status == 200

    # contrôle de sécurité : l'API déployée doit refuser un appel sans clé
    try:
        statut_sans_cle, _ = call_api(f"{url}/predict", {"features": {"EXT_SOURCE_2": 0.5}}, cle="")
    except urllib.error.HTTPError as refus:
        statut_sans_cle = refus.code
    print(f"POST /predict sans clé -> {statut_sans_cle} (401 attendu)")
    assert statut_sans_cle == 401, "l'API accepte un appel non authentifié"

    print("Déploiement vérifié.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
