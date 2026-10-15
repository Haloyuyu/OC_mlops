"""Chargement du modèle champion et calcul du score d'un client.

FastAPI et Gradio passent tous deux par `ScoringModel.predict` : une seule logique
de prédiction, donc une seule logique à tester.

Deux chemins d'inférence, au choix par `MODEL_BACKEND` :

- `booster` (défaut) : appel direct du Booster LightGBM sur un vecteur numpy. Le
  profilage (docs/rapport_optimisation.md) montre que ~85 % du temps d'un appel
  partait dans la validation de schéma MLflow et la construction de DataFrames
  pandas de 795 colonnes, pas dans les arbres ;
- `pyfunc` : le modèle MLflow tel qu'enregistré au registry, conservé comme
  référence de non-régression.

Les deux partagent le même artefact (`model/credit_scoring_lgbm`), donc les mêmes
arbres, les mêmes features et le même seuil : seul le chemin d'appel change.
"""

import json
import os
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "booster")
BACKENDS = ("booster", "pyfunc")

# Threads OpenMP par prédiction LightGBM. Un client = une ligne : paralléliser 2 550
# arbres sur plusieurs cœurs coûte plus en synchronisation qu'il ne rapporte, et
# sous charge les requêtes concurrentes se disputent les mêmes cœurs. La
# parallélisation se fait entre requêtes (pool de threads de FastAPI), pas dedans.
LGBM_NUM_THREADS = int(os.getenv("LGBM_NUM_THREADS", "1"))

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

    def __init__(self, directory=MODEL_DIR, backend=None, num_threads=None):
        directory = Path(directory)
        self.backend = backend or MODEL_BACKEND
        if self.backend not in BACKENDS:
            raise ValueError(f"MODEL_BACKEND inconnu : {self.backend!r} (attendu : {BACKENDS})")
        self.num_threads = LGBM_NUM_THREADS if num_threads is None else num_threads
        self.pyfunc = mlflow.pyfunc.load_model(str(directory / "credit_scoring_lgbm"))
        self.features = self.pyfunc.metadata.get_input_schema().input_names()
        wrapper = self.pyfunc.unwrap_python_model()
        self._seuil = float(wrapper.seuil)  # seuil exact, celui qu'applique le pyfunc
        self.threshold = round(self._seuil, 6)
        self.booster = wrapper.model.booster_
        self._positions = {name: i for i, name in enumerate(self.features)}
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

        if self.backend == "pyfunc":
            client = pd.DataFrame([features]).reindex(columns=self.features).astype("float64")
            result = self.pyfunc.predict(client).iloc[0]
            probabilite, decision = float(result["probabilite_defaut"]), int(result["decision"])
            renseignees = int(client.notna().sum(axis=1).iloc[0])
        else:
            vecteur = self.vectorize(features)
            probabilite = float(self.predict_proba(vecteur)[0])
            decision = int(probabilite >= self._seuil)
            renseignees = int(np.count_nonzero(~np.isnan(vecteur)))
        return {
            "probabilite_defaut": probabilite,
            "decision": decision,
            "libelle_decision": DECISION_LABELS[decision],
            "seuil": self.threshold,
            "features_renseignees": renseignees,
        }

    def vectorize(self, features):
        """dict {feature: valeur} -> ligne numpy dans l'ordre du modèle (absente = NaN)."""
        vecteur = np.full((1, len(self.features)), np.nan)
        for name, value in features.items():
            if value is not None:
                vecteur[0, self._positions[name]] = value
        return vecteur

    def predict_proba(self, matrice):
        """Probabilités de défaut d'une matrice (n_clients, n_features) en float64."""
        return self.booster.predict(matrice, num_threads=self.num_threads)


def load_example_clients(directory=MODEL_DIR):
    """Clients du jeu test (index SK_ID_CURR) utilisés par l'interface et les tests."""
    return pd.read_csv(Path(directory) / "clients_exemple.csv", index_col="SK_ID_CURR")


def client_to_dict(row):
    """Ligne pandas -> dict JSON-compatible (NaN -> None)."""
    return {name: (None if pd.isna(value) else float(value)) for name, value in row.items()}
