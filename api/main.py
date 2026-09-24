"""API de scoring crédit : FastAPI (REST) + Gradio (interface web sur /ui).

    uvicorn api.main:app --port 8000

Documentation interactive : http://localhost:8000/docs
"""

import logging
from typing import Literal

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from api.interface import create_interface
from api.scoring import OutOfBoundsValues, ScoringModel, UnknownFeatures, load_example_clients

logger = logging.getLogger("api.scoring")

# Le modèle est chargé une seule fois, au démarrage : le charger à chaque requête
# ajouterait ~2,5 s et autant de copies en mémoire.
model = ScoringModel()
example_clients = load_example_clients()


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


class ModelInfo(BaseModel):
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
def home():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    return {"status": "ok", "modele": model.metadata["nom"], "version": model.metadata["version"]}


@app.get("/model", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        nom=model.metadata["nom"],
        version=model.metadata["version"],
        alias=model.metadata["alias"],
        seuil=model.threshold,
        features=model.features,
    )


ERREURS_PREDICT = {
    422: {
        "description": "Entrée refusée : corps de requête invalide, feature inconnue du "
                       "modèle, ou valeur hors des bornes métier.",
        "content": {"application/json": {"examples": {
            "feature_inconnue": {"value": {"detail": {
                "message": "Features inconnues du modèle : REVENU_IMAGINAIRE",
                "features_inconnues": ["REVENU_IMAGINAIRE"]}}},
            "valeur_hors_bornes": {"value": {"detail": {
                "message": "Valeurs hors bornes : AMT_INCOME_TOTAL = 0 hors des bornes [1, 1e+09]",
                "features_hors_bornes": ["AMT_INCOME_TOTAL = 0 hors des bornes [1, 1e+09]"]}}},
        }}},
    },
    500: {"description": "Erreur interne du service de scoring."},
}


@app.post("/predict", response_model=Prediction, responses=ERREURS_PREDICT)
def predict(client: Client):
    try:
        return model.predict(client.features)
    except UnknownFeatures as error:
        raise HTTPException(status_code=422, detail={"message": str(error), "features_inconnues": error.names})
    except OutOfBoundsValues as error:
        raise HTTPException(status_code=422, detail={"message": str(error), "features_hors_bornes": error.details})


@app.exception_handler(Exception)
async def erreur_inattendue(request: Request, exc: Exception):
    """Toute erreur non prévue devient une 500 explicite, tracée côté serveur.

    Sans ce garde-fou, une exception du modèle remonterait en page d'erreur brute.
    """
    logger.exception("Erreur inattendue sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du service de scoring."})


app = gr.mount_gradio_app(app, create_interface(model, example_clients), path="/ui")
