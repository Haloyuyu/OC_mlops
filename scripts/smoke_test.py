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

EXAMPLE_CLIENTS = Path(__file__).resolve().parent.parent / "model" / "clients_exemple.csv"


def call_api(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
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

    status, body = call_api(f"{url}/predict", {"features": first_client()})
    prediction = json.loads(body)
    print(f"POST /predict -> {status} {prediction}")
    assert status == 200 and 0 <= prediction["probabilite_defaut"] <= 1

    status, _ = call_api(f"{url}/ui/")
    print(f"GET  /ui/     -> {status}")
    assert status == 200

    print("Déploiement vérifié.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
