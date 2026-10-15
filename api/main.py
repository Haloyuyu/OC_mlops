"""API de scoring crédit : FastAPI (REST) + Gradio (interface web sur /ui).

    uvicorn api.main:app --port 8000

Documentation interactive : http://localhost:8000/docs
"""

import logging
import time
from typing import Literal

import gradio as gr
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from api.interface import create_interface
from api.monitoring import (ProductionLog, evenement_prediction, evenement_requete,
                            nouvel_identifiant, ressources_processus)
from api.scoring import OutOfBoundsValues, ScoringModel, UnknownFeatures, load_example_clients
from api.security import API_KEY, HEADER_NAME, UI_USERNAME, log_key_status, require_api_key

logger = logging.getLogger("api.scoring")

# Journal de production : entrées, sorties, latences et erreurs de chaque appel.
production_log = ProductionLog()

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
    description=(
        "Probabilité de défaut et décision d'octroi au seuil métier du modèle champion.\n\n"
        f"Les routes de scoring exigent la clé d'API dans l'en-tête `{HEADER_NAME}` "
        "(bouton **Authorize** ci-dessus). Seul `/health` reste public."
    ),
    version="1.0.0",
)

log_key_status()


@app.middleware("http")
async def journaliser_requete(request: Request, call_next):
    """Journalise chaque requête HTTP : statut et latence, pour le suivi d'exploitation.

    Les requêtes de l'interface Gradio (`/ui`) sont exclues : ce sont surtout des
    fichiers statiques, qui noieraient le journal sans rien apprendre sur le modèle.
    """
    request.state.requete_id = nouvel_identifiant()
    debut = time.perf_counter()
    reponse = await call_next(request)
    reponse.headers["X-Request-Id"] = request.state.requete_id
    if not request.url.path.startswith("/ui"):
        production_log.record({
            "requete_id": request.state.requete_id,
            **evenement_requete(request.url.path, request.method, reponse.status_code,
                                (time.perf_counter() - debut) * 1000),
        })
    return reponse

ERREUR_AUTH = {401: {"description": f"Clé d'API absente ou invalide (en-tête {HEADER_NAME})."}}


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    """Sonde de vie : volontairement publique, car utilisée par Docker et le pipeline."""
    return {"status": "ok", "modele": model.metadata["nom"], "version": model.metadata["version"]}


@app.get("/model", response_model=ModelInfo, responses=ERREUR_AUTH,
         dependencies=[Depends(require_api_key)])
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
    **ERREUR_AUTH,
}


def _signature_modele():
    return {"nom": model.metadata["nom"], "version": model.metadata["version"], "seuil": model.threshold}


@app.post("/predict", response_model=Prediction, responses=ERREURS_PREDICT,
          dependencies=[Depends(require_api_key)])
def predict(client: Client, request: Request):
    """Score un client et journalise l'appel (entrées, sortie, latence, erreur)."""
    debut = time.perf_counter()
    journal = {"requete_id": getattr(request.state, "requete_id", nouvel_identifiant())}
    try:
        sortie = model.predict(client.features)
    except (UnknownFeatures, OutOfBoundsValues) as error:
        cle_detail = "features_inconnues" if isinstance(error, UnknownFeatures) else "features_hors_bornes"
        detail = error.names if isinstance(error, UnknownFeatures) else error.details
        production_log.record({**journal, **evenement_prediction(
            entrees=client.features, statut=422,
            erreur={"type": type(error).__name__, "message": str(error)},
            latence_ms=(time.perf_counter() - debut) * 1000,
            modele=_signature_modele(), ressources=ressources_processus(),
        )})
        raise HTTPException(status_code=422, detail={"message": str(error), cle_detail: detail})

    latence_modele_ms = (time.perf_counter() - debut) * 1000
    production_log.record({**journal, **evenement_prediction(
        entrees=client.features, sortie=sortie, statut=200,
        latence_ms=latence_modele_ms, latence_modele_ms=latence_modele_ms,
        modele=_signature_modele(), ressources=ressources_processus(),
    )})
    return sortie


@app.exception_handler(Exception)
async def erreur_inattendue(request: Request, exc: Exception):
    """Toute erreur non prévue devient une 500 explicite, tracée côté serveur.

    Sans ce garde-fou, une exception du modèle remonterait en page d'erreur brute.
    """
    logger.exception("Erreur inattendue sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du service de scoring."})


# L'interface appelle le modèle en interne, sans passer par REST : sans ce formulaire
# de connexion, /ui contournerait l'authentification des routes de scoring.
app = gr.mount_gradio_app(
    app,
    create_interface(model, example_clients),
    path="/ui",
    auth=(UI_USERNAME, API_KEY),
    auth_message=f"Utilisateur « {UI_USERNAME} », mot de passe = la clé d'API.",
)
