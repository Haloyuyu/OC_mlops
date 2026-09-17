import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.scoring import ModeleScoring, charger_clients_exemple, client_en_dict

DOSSIER_MODELE = Path(__file__).resolve().parent.parent / "model"


@pytest.fixture(scope="session")
def modele():
    return ModeleScoring(DOSSIER_MODELE)


@pytest.fixture(scope="session")
def clients_exemple():
    return charger_clients_exemple(DOSSIER_MODELE)


@pytest.fixture
def client_complet(clients_exemple):
    return client_en_dict(clients_exemple.iloc[0])


@pytest.fixture(scope="session")
def clients_train():
    """5 clients du train enregistrés avec le modèle (TARGET connue : le 1er est un défaut)."""
    with open(DOSSIER_MODELE / "credit_scoring_lgbm" / "serving_input_example.json", encoding="utf-8") as f:
        split = json.load(f)["dataframe_split"]
    return pd.DataFrame(split["data"], columns=split["columns"])


@pytest.fixture(scope="session")
def api():
    from api.main import app

    with TestClient(app) as client:
        yield client
