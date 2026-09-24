"""Exporte le modèle champion du Model Registry vers `model/` pour l'API.

L'image Docker et la CI n'ont pas accès au serveur MLFlow local : le modèle servi
par l'API est donc figé dans le dépôt, avec sa version, son seuil métier et un
échantillon de nouveaux clients (jeu test, TARGET inconnue) pour l'interface Gradio.

Prérequis :

    poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

Puis :

    poetry run python scripts/export_model.py
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
MODEL_NAME = "credit_scoring_lgbm"
ALIAS = "champion"

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "model"
MODEL_DIR = OUTPUT_DIR / MODEL_NAME
DATA = ROOT / "data" / "processed_data.csv"
N_EXAMPLE_CLIENTS = 20


def export_model(client):
    """Télécharge les artefacts de la version @champion et écrit ses métadonnées."""
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{ALIAS}", dst_path=str(MODEL_DIR)
    )
    metadata = {
        "nom": MODEL_NAME,
        "alias": ALIAS,
        "version": version.version,
        "run_id": version.run_id,
        "tags": version.tags,
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Modèle {MODEL_NAME} v{version.version} exporté dans {MODEL_DIR}")


def export_example_clients(n=N_EXAMPLE_CLIENTS):
    """Extrait n clients du jeu test (TARGET inconnue), restreints aux features du modèle."""
    features = Model.load(str(MODEL_DIR)).get_input_schema().input_names()
    chunks = []
    for chunk in pd.read_csv(DATA, chunksize=50_000):
        # même nettoyage des noms de colonnes qu'à l'entraînement (modelisation.ipynb)
        chunk.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in chunk.columns]
        chunks.append(chunk.loc[chunk["TARGET"].isna()])
        if sum(len(c) for c in chunks) >= n:
            break
    clients = pd.concat(chunks).head(n)
    clients["SK_ID_CURR"] = clients["SK_ID_CURR"].astype(int)
    clients = clients.set_index("SK_ID_CURR")[features].astype("float64")
    clients.to_csv(OUTPUT_DIR / "clients_exemple.csv")
    print(f"{len(clients)} clients d'exemple exportés ({clients.shape[1]} features)")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    export_model(MlflowClient())
    export_example_clients()


if __name__ == "__main__":
    main()
