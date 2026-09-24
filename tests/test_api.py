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
    "features",
    [
        {"DAYS_BIRTH": 1825},  # âge de -5 ans
        {"AMT_INCOME_TOTAL": 0},  # revenu nul
        {"EXT_SOURCE_2": 1.5},  # score externe hors de [0, 1]
        {"CNT_CHILDREN": -1},
    ],
    ids=["age_negatif", "revenu_nul", "score_hors_plage", "enfants_negatifs"],
)
def test_predict_out_of_bounds_value(api, features):
    response = api.post("/predict", json={"features": features})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert list(features)[0] in detail["features_hors_bornes"][0]


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
    schema = response.json()
    assert "/predict" in schema["paths"]
    # les cas d'erreur sont documentés dans Swagger, pas seulement dans le code
    assert {"422", "500"} <= set(schema["paths"]["/predict"]["post"]["responses"])


def test_model_is_loaded_once_at_startup(api, monkeypatch):
    """Le modèle ne doit pas être rechargé à chaque requête (~2,5 s par chargement)."""
    import mlflow.pyfunc

    import api.main as module

    loaded = module.model.pyfunc
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda *a, **k: pytest.fail("modèle rechargé"))
    for _ in range(3):
        assert api.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5}}).status_code == 200
    assert module.model.pyfunc is loaded


def test_unexpected_error_returns_500(monkeypatch):
    """Une panne imprévue du modèle reste une réponse JSON propre, pas une trace brute."""
    from fastapi.testclient import TestClient

    import api.main as module

    def modele_en_panne(features):
        raise RuntimeError("panne simulée")

    monkeypatch.setattr(module.model, "predict", modele_en_panne)
    with TestClient(module.app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5}})
    assert response.status_code == 500
    assert response.json() == {"detail": "Erreur interne du service de scoring."}


def test_gradio_interface_is_mounted(api):
    assert api.get("/ui/").status_code == 200
    response = api.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui"
