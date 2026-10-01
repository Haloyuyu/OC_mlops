import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.scoring import ScoringModel, client_to_dict, load_example_clients

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Clé fixée avant tout import de `api.main` / `api.security`, qui lisent l'environnement
# au chargement. Sans cela, la clé serait générée aléatoirement et inconnue des tests.
API_KEY = "cle-de-test-pytest"
os.environ["API_KEY"] = API_KEY


@pytest.fixture(scope="session")
def model():
    return ScoringModel(MODEL_DIR)


@pytest.fixture(scope="session")
def example_clients():
    return load_example_clients(MODEL_DIR)


@pytest.fixture
def complete_client(example_clients):
    return client_to_dict(example_clients.iloc[0])


@pytest.fixture(scope="session")
def train_clients():
    """5 clients du train enregistrés avec le modèle (TARGET connue : le 1er est un défaut)."""
    with open(MODEL_DIR / "credit_scoring_lgbm" / "serving_input_example.json", encoding="utf-8") as f:
        split = json.load(f)["dataframe_split"]
    return pd.DataFrame(split["data"], columns=split["columns"])


@pytest.fixture(scope="session")
def api():
    """Client authentifié : la clé est envoyée sur toutes les requêtes."""
    from api.main import app

    with TestClient(app, headers={"X-API-Key": API_KEY}) as client:
        yield client


@pytest.fixture(scope="session")
def api_anonyme():
    """Client sans clé, pour vérifier que l'API refuse bien les appels non authentifiés."""
    from api.main import app

    with TestClient(app) as client:
        yield client
