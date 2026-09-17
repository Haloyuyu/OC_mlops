"""API de scoring crédit : FastAPI (REST) + Gradio (interface web sur /ui).

    uvicorn api.main:app --port 8000

Documentation interactive : http://localhost:8000/docs
"""

from typing import Literal

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from api.interface import creer_interface
from api.scoring import FeaturesInconnues, ModeleScoring, charger_clients_exemple

modele = ModeleScoring()
clients_exemple = charger_clients_exemple()


class Client(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    features: dict[str, float | None] = Field(
        min_length=1,
        description="Features du client {nom: valeur}. Une feature absente ou null est une valeur manquante.",
        examples=[{"EXT_SOURCE_1": 0.75, "EXT_SOURCE_2": 0.79, "EXT_SOURCE_3": 0.16,
                   "PAYMENT_RATE": 0.036, "DAYS_BIRTH": -19241}],
    )


class Prediction(BaseModel):
    probabilite_defaut: float = Field(ge=0, le=1)
    decision: Literal[0, 1] = Field(description="1 = crédit refusé")
    libelle_decision: Literal["crédit accordé", "crédit refusé"]
    seuil: float
    features_renseignees: int


class InfosModele(BaseModel):
    nom: str
    version: str
    alias: str
    seuil: float
    features: list[str]


app = FastAPI(
    title="API de scoring crédit — Prêt à dépenser",
    description="Probabilité de défaut et décision d'octroi au seuil métier du modèle champion.",
    version="1.0.0",
)


@app.get("/", include_in_schema=False)
def accueil():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    return {"status": "ok", "modele": modele.metadonnees["nom"], "version": modele.metadonnees["version"]}


@app.get("/model", response_model=InfosModele)
def infos_modele():
    return InfosModele(
        nom=modele.metadonnees["nom"],
        version=modele.metadonnees["version"],
        alias=modele.metadonnees["alias"],
        seuil=modele.seuil,
        features=modele.features,
    )


@app.post("/predict", response_model=Prediction)
def predict(client: Client):
    try:
        return modele.predire(client.features)
    except FeaturesInconnues as erreur:
        raise HTTPException(status_code=422, detail={"message": str(erreur), "features_inconnues": erreur.noms})


app = gr.mount_gradio_app(app, creer_interface(modele, clients_exemple), path="/ui")
