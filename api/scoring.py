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

# Bornes métier des variables qui viennent directement du dossier client, avec une
# marge sur les valeurs observées dans les 307 511 dossiers d'entraînement. Un
# dossier hors de ces bornes est une erreur de saisie ou d'appel, pas un client
# atypique : le scorer donnerait un résultat dénué de sens. Les ~780 autres features
# sont des agrégats calculés, dont la plage n'est pas un contrat métier.
FEATURE_BOUNDS = {
    "EXT_SOURCE_1": (0.0, 1.0),  # scores externes normalisés
    "EXT_SOURCE_2": (0.0, 1.0),
    "EXT_SOURCE_3": (0.0, 1.0),
    "DAYS_BIRTH": (-27000.0, -6570.0),  # jours avant la demande : 18 à 74 ans
    "DAYS_EMPLOYED": (-20000.0, 0.0),  # négatif ; l'anomalie 365243 devient NaN au préprocessing
    "DAYS_REGISTRATION": (-27000.0, 0.0),
    "DAYS_ID_PUBLISH": (-8000.0, 0.0),
    "AMT_INCOME_TOTAL": (1.0, 1e9),  # revenu strictement positif
    "AMT_CREDIT": (1.0, 1e8),
    "AMT_ANNUITY": (1.0, 1e7),
    "AMT_GOODS_PRICE": (1.0, 1e8),
    "CNT_CHILDREN": (0.0, 20.0),
    "CNT_FAM_MEMBERS": (1.0, 25.0),
    "REGION_POPULATION_RELATIVE": (0.0, 1.0),
    "PAYMENT_RATE": (0.0, 1.0),  # annuité / crédit
}


class UnknownFeatures(ValueError):
    """Le client contient des variables que le modèle ne connaît pas."""

    def __init__(self, names):
        self.names = sorted(names)
        super().__init__(f"Features inconnues du modèle : {', '.join(self.names[:10])}")


class OutOfBoundsValues(ValueError):
    """Le client contient des valeurs hors des bornes métier (saisie aberrante)."""

    def __init__(self, faults):
        self.faults = sorted(faults)
        self.details = [
            f"{name} = {value:g} hors des bornes [{low:g}, {high:g}]"
            for name, value, low, high in self.faults
        ]
        super().__init__("Valeurs hors bornes : " + " ; ".join(self.details[:10]))


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
        LightGBM gère nativement comme à l'entraînement. Une feature inconnue ou
        une valeur hors des bornes métier est en revanche une erreur : elle signale
        un contrat d'interface non respecté ou une saisie aberrante.
        """
        unknown = set(features) - set(self.features)
        if unknown:
            raise UnknownFeatures(unknown)

        faults = [
            (name, float(value), *FEATURE_BOUNDS[name])
            for name, value in features.items()
            if value is not None and name in FEATURE_BOUNDS
            and not FEATURE_BOUNDS[name][0] <= float(value) <= FEATURE_BOUNDS[name][1]
        ]
        if faults:
            raise OutOfBoundsValues(faults)

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
