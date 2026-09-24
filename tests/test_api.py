"""Tests d'intégration : routes HTTP de l'API (FastAPI + Gradio monté)."""

import pytest


def test_health(api):
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(api, model):
    response = api.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["alias"] == "champion"
    assert body["seuil"] == model.threshold
    assert body["features"] == model.features


def test_predict_complete_client(api, model, complete_client):
    response = api.post("/predict", json={"features": complete_client})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probabilite_defaut", "decision", "libelle_decision", "seuil", "features_renseignees"}
    # l'API renvoie exactement ce que calcule le modèle
    assert body == pytest.approx(model.predict(complete_client))


def test_predict_partial_client(api):
    response = api.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5, "DAYS_BIRTH": -12000}})
    assert response.status_code == 200
    assert response.json()["features_renseignees"] == 2


def test_predict_unknown_feature(api):
    response = api.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5, "INCONNUE": 1}})
    assert response.status_code == 422
    assert response.json()["detail"]["features_inconnues"] == ["INCONNUE"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"features": {}},
        {"features": {"EXT_SOURCE_2": "élevé"}},
        {"features": [0.5, 0.2]},
    ],
    ids=["sans_features", "features_vides", "valeur_texte", "liste_au_lieu_de_dict"],
)
def test_predict_invalid_input(api, body):
    assert api.post("/predict", json=body).status_code == 422


def test_openapi_documentation(api):
    response = api.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


def test_gradio_interface_is_mounted(api):
    assert api.get("/ui/").status_code == 200
    response = api.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui"
