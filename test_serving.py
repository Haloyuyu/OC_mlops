"""Test du serving MLFlow du modèle de scoring crédit.

Prérequis (deux terminaux) :

    poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
    poetry run mlflow models serve -m "models:/credit_scoring_lgbm@champion" \
        --port 5001 --env-manager local

Puis :

    poetry run python test_serving.py

Le modèle servi renvoie la probabilité de défaut ET la décision au seuil métier
(et non la décision à 0.5 de `predict`).
"""

import sys

import pandas as pd
import requests

URL_SERVING = "http://127.0.0.1:5001/invocations"
DONNEES = "data/processed_data.csv"
N_CLIENTS = 5
EXCLUES = ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]


def charger_clients(n=N_CLIENTS):
    """Renvoie (features des n premiers clients du train, leur TARGET réel)."""
    df = pd.read_csv(DONNEES, nrows=20_000)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    train = df.loc[df["TARGET"].notnull()].head(n)
    feats = [c for c in train.columns if c not in EXCLUES]
    return train[feats].astype("float64"), train["TARGET"].astype(int)


def construire_payload(clients):
    """Format `dataframe_split` attendu par MLFlow ; NaN -> null JSON valide."""
    return {"dataframe_split": {
        "columns": clients.columns.tolist(),
        "data": clients.astype(object).where(pd.notnull(clients), None).values.tolist(),
    }}


def main():
    clients, cible = charger_clients()
    try:
        reponse = requests.post(URL_SERVING, json=construire_payload(clients), timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"Aucun serveur sur {URL_SERVING} — lancez `mlflow models serve` (cf. docstring).")
        return 1

    if reponse.status_code != 200:
        print(f"Erreur {reponse.status_code} : {reponse.text[:500]}")
        return 1

    predictions = pd.DataFrame(reponse.json()["predictions"])
    predictions.insert(0, "TARGET_reel", cible.values)
    print(f"Réponse de {URL_SERVING} :\n")
    print(predictions.round(4).to_string(index=False))
    print("\ndecision = 1 -> crédit refusé, au seuil métier embarqué dans le modèle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
