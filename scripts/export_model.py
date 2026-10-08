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
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.scoring import FEATURE_BOUNDS  # noqa: E402  (après l'ajout de la racine au path)

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "credit_scoring_lgbm"
ALIAS = "champion"

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "model"
MODEL_DIR = OUTPUT_DIR / MODEL_NAME
DATA = ROOT / "data" / "processed_data.csv"
N_EXAMPLE_CLIENTS = 20

# Référence de dérive : échantillon du jeu d'ENTRAÎNEMENT. Suivre les 795 features
# serait illisible et pèserait 12 Mo ; on suit les plus influentes du modèle et les
# variables du dossier client, qui sont celles qu'un métier sait interpréter.
N_REFERENCE = 2000
N_FEATURES_SUIVIES = 30
REFERENCE_FILE = "reference_drift.csv"


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


def features_suivies(pyfunc, n=N_FEATURES_SUIVIES):
    """Features surveillées : les n plus influentes du modèle + celles du dossier client."""
    lgbm = pyfunc.unwrap_python_model().model
    importances = pd.Series(lgbm.feature_importances_, index=lgbm.feature_name_)
    top = set(importances.sort_values(ascending=False).head(n).index)
    return sorted(top | set(FEATURE_BOUNDS))


def export_reference_sample(n=N_REFERENCE):
    """Échantillonne le train et enregistre la référence de dérive (features + prédiction).

    La prédiction du modèle sur ces mêmes lignes sert de référence à la dérive des
    sorties : en production il n'y a pas de TARGET, mais la distribution des scores
    reste comparable.
    """
    pyfunc = mlflow.pyfunc.load_model(str(MODEL_DIR))
    toutes = pyfunc.metadata.get_input_schema().input_names()
    suivies = features_suivies(pyfunc)

    morceaux = []
    for chunk in pd.read_csv(DATA, chunksize=50_000):
        chunk.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in chunk.columns]
        train = chunk.loc[chunk["TARGET"].notna()]
        if not train.empty:
            morceaux.append(train.sample(n=min(len(train), n // 5), random_state=1001))
    echantillon = pd.concat(morceaux).sample(n=n, random_state=1001)

    predictions = pyfunc.predict(echantillon[toutes].astype("float64"))
    reference = echantillon[suivies].astype("float64").reset_index(drop=True)
    reference["probabilite_defaut"] = predictions["probabilite_defaut"].values
    reference["decision"] = predictions["decision"].values
    reference["TARGET"] = echantillon["TARGET"].astype(int).values
    reference.to_csv(OUTPUT_DIR / REFERENCE_FILE, index=False)
    taille_ko = (OUTPUT_DIR / REFERENCE_FILE).stat().st_size / 1024
    print(f"Référence de dérive : {len(reference)} lignes × {len(suivies)} features "
          f"suivies ({taille_ko:.0f} Ko), taux de défaut {reference.TARGET.mean():.1%}")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    export_model(MlflowClient())
    export_example_clients()
    export_reference_sample()


if __name__ == "__main__":
    main()
