"""Chargement du modèle champion et calcul du score d'un client.

FastAPI et Gradio passent tous deux par `ModeleScoring.predire` : une seule logique
de prédiction, donc une seule logique à tester.
"""

import json
import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

DOSSIER_MODELE = Path(os.getenv("DOSSIER_MODELE", Path(__file__).resolve().parent.parent / "model"))

LIBELLES_DECISION = {0: "crédit accordé", 1: "crédit refusé"}


class FeaturesInconnues(ValueError):
    """Le client contient des variables que le modèle ne connaît pas."""

    def __init__(self, noms):
        self.noms = sorted(noms)
        super().__init__(f"Features inconnues du modèle : {', '.join(self.noms[:10])}")


class ModeleScoring:
    """Modèle pyfunc exporté du registry (probabilité + décision au seuil métier)."""

    def __init__(self, dossier=DOSSIER_MODELE):
        dossier = Path(dossier)
        self.pyfunc = mlflow.pyfunc.load_model(str(dossier / "credit_scoring_lgbm"))
        self.features = self.pyfunc.metadata.get_input_schema().input_names()
        self.seuil = round(float(self.pyfunc.unwrap_python_model().seuil), 6)
        with open(dossier / "metadata.json", encoding="utf-8") as f:
            self.metadonnees = json.load(f)

    def predire(self, features):
        """Score un client décrit par un dict {feature: valeur}.

        Les features absentes ou nulles sont des valeurs manquantes (NaN), que
        LightGBM gère nativement comme à l'entraînement. Une feature inconnue est
        en revanche une erreur : elle signale un contrat d'interface non respecté.
        """
        inconnues = set(features) - set(self.features)
        if inconnues:
            raise FeaturesInconnues(inconnues)

        client = pd.DataFrame([features]).reindex(columns=self.features).astype("float64")
        resultat = self.pyfunc.predict(client).iloc[0]
        decision = int(resultat["decision"])
        return {
            "probabilite_defaut": float(resultat["probabilite_defaut"]),
            "decision": decision,
            "libelle_decision": LIBELLES_DECISION[decision],
            "seuil": self.seuil,
            "features_renseignees": int(client.notna().sum(axis=1).iloc[0]),
        }


def charger_clients_exemple(dossier=DOSSIER_MODELE):
    """Clients du jeu test (index SK_ID_CURR) utilisés par l'interface et les tests."""
    return pd.read_csv(Path(dossier) / "clients_exemple.csv", index_col="SK_ID_CURR")


def client_en_dict(ligne):
    """Ligne pandas -> dict JSON-compatible (NaN -> None)."""
    return {nom: (None if pd.isna(valeur) else float(valeur)) for nom, valeur in ligne.items()}

