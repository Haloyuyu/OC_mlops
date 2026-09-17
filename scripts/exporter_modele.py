"""Exporte le modèle champion du Model Registry vers `model/` pour l'API.

L'image Docker et la CI n'ont pas accès au serveur MLFlow local : le modèle servi
par l'API est donc figé dans le dépôt, avec sa version, son seuil métier et un
échantillon de nouveaux clients (jeu test, TARGET inconnue) pour l'interface Gradio.

Prérequis :

    poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

Puis :

    poetry run python scripts/exporter_modele.py
"""

import json
import re
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import Model

TRACKING_URI = "http://127.0.0.1:5000"
NOM_MODELE = "credit_scoring_lgbm"
ALIAS = "champion"

RACINE = Path(__file__).resolve().parent.parent
DOSSIER_SORTIE = RACINE / "model"
DOSSIER_MODELE = DOSSIER_SORTIE / NOM_MODELE
DONNEES = RACINE / "data" / "processed_data.csv"
N_CLIENTS_EXEMPLE = 20


def exporter_modele(client):
    """Télécharge les artefacts de la version @champion et écrit ses métadonnées."""
    version = client.get_model_version_by_alias(NOM_MODELE, ALIAS)
    if DOSSIER_MODELE.exists():
        shutil.rmtree(DOSSIER_MODELE)
    DOSSIER_SORTIE.mkdir(exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{NOM_MODELE}@{ALIAS}", dst_path=str(DOSSIER_MODELE)
    )
    metadonnees = {
        "nom": NOM_MODELE,
        "alias": ALIAS,
        "version": version.version,
        "run_id": version.run_id,
        "tags": version.tags,
    }
    with open(DOSSIER_SORTIE / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadonnees, f, indent=2, ensure_ascii=False)
    print(f"Modèle {NOM_MODELE} v{version.version} exporté dans {DOSSIER_MODELE}")


def exporter_clients_exemple(n=N_CLIENTS_EXEMPLE):
    """Extrait n clients du jeu test (TARGET inconnue), restreints aux features du modèle."""
    features = Model.load(str(DOSSIER_MODELE)).get_input_schema().input_names()
    morceaux = []
    for morceau in pd.read_csv(DONNEES, chunksize=50_000):
        # même nettoyage des noms de colonnes qu'à l'entraînement (modelisation.ipynb)
        morceau.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in morceau.columns]
        morceaux.append(morceau.loc[morceau["TARGET"].isna()])
        if sum(len(m) for m in morceaux) >= n:
            break
    clients = pd.concat(morceaux).head(n)
    clients["SK_ID_CURR"] = clients["SK_ID_CURR"].astype(int)
    clients = clients.set_index("SK_ID_CURR")[features].astype("float64")
    clients.to_csv(DOSSIER_SORTIE / "clients_exemple.csv")
    print(f"{len(clients)} clients d'exemple exportés ({clients.shape[1]} features)")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    exporter_modele(MlflowClient())
    exporter_clients_exemple()


if __name__ == "__main__":
    main()
