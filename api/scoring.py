"""Chargement du modèle champion et calcul du score d'un client.

FastAPI et Gradio passent tous deux par `ScoringModel.predict` : une seule logique
de prédiction, donc une seule logique à tester.
"""

import json
import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))

DECISION_LABELS = {0: "crédit accordé", 1: "crédit refusé"}


class UnknownFeatures(ValueError):
    """Le client contient des variables que le modèle ne connaît pas."""

    def __init__(self, names):
        self.names = sorted(names)
        super().__init__(f"Features inconnues du modèle : {', '.join(self.names[:10])}")


class ScoringModel:
    """Modèle pyfunc exporté du registry (probabilité + décision au seuil métier)."""

    def __init__(self, directory=MODEL_DIR):
        directory = Path(directory)
        self.pyfunc = mlflow.pyfunc.load_model(str(directory / "credit_scoring_lgbm"))
        self.features = self.pyfunc.metadata.get_input_schema().input_names()
        self.threshold = round(float(self.pyfunc.unwrap_python_model().seuil), 6)
        with open(directory / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def predict(self, features):
        """Score un client décrit par un dict {feature: valeur}.

        Les features absentes ou nulles sont des valeurs manquantes (NaN), que
        LightGBM gère nativement comme à l'entraînement. Une feature inconnue est
        en revanche une erreur : elle signale un contrat d'interface non respecté.
        """
        unknown = set(features) - set(self.features)
        if unknown:
            raise UnknownFeatures(unknown)

        client = pd.DataFrame([features]).reindex(columns=self.features).astype("float64")
        result = self.pyfunc.predict(client).iloc[0]
        decision = int(result["decision"])
        return {
            "probabilite_defaut": float(result["probabilite_defaut"]),
            "decision": decision,
            "libelle_decision": DECISION_LABELS[decision],
            "seuil": self.threshold,
            "features_renseignees": int(client.notna().sum(axis=1).iloc[0]),
        }


def load_example_clients(directory=MODEL_DIR):
    """Clients du jeu test (index SK_ID_CURR) utilisés par l'interface et les tests."""
    return pd.read_csv(Path(directory) / "clients_exemple.csv", index_col="SK_ID_CURR")


def client_to_dict(row):
    """Ligne pandas -> dict JSON-compatible (NaN -> None)."""
    return {name: (None if pd.isna(value) else float(value)) for name, value in row.items()}
