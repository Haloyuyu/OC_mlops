"""Tests d'intégration : routes HTTP de l'API (FastAPI + Gradio monté)."""

import pytest

import conftest


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


def test_predict_requires_api_key(api_anonyme, complete_client):
    response = api_anonyme.post("/predict", json={"features": complete_client})
    assert response.status_code == 401
    assert "X-API-Key" in response.json()["detail"]


def test_model_info_requires_api_key(api_anonyme):
    assert api_anonyme.get("/model").status_code == 401


@pytest.mark.parametrize("cle", ["", "mauvaise-cle", "cle-de-test-pytes"], ids=["vide", "fausse", "tronquee"])
def test_predict_rejects_wrong_api_key(api_anonyme, cle):
    response = api_anonyme.post(
        "/predict", json={"features": {"EXT_SOURCE_2": 0.5}}, headers={"X-API-Key": cle}
    )
    assert response.status_code == 401


def test_health_stays_public(api_anonyme):
    # sonde utilisée par Docker et le pipeline : elle ne doit pas exiger de clé
    assert api_anonyme.get("/health").status_code == 200


def test_gradio_ui_requires_login(api_anonyme):
    """L'interface appelle le modèle en interne : sans session, elle le contournerait.

    Gradio sert la coquille HTML publiquement (c'est le formulaire de connexion), mais
    refuse la configuration et les appels de fonction sans session authentifiée.
    """
    assert api_anonyme.get("/ui/").status_code == 200
    assert api_anonyme.get("/ui/config").status_code == 401
    appel = api_anonyme.post(
        "/ui/gradio_api/call/score_json", json={"data": ['{"EXT_SOURCE_2": 0.1}']}
    )
    assert appel.status_code == 401


def test_gradio_login_accepts_the_api_key(api_anonyme):
    refuse = api_anonyme.post("/ui/login", data={"username": "analyste", "password": "mauvais"})
    assert refuse.status_code == 400
    accepte = api_anonyme.post(
        "/ui/login", data={"username": "analyste", "password": conftest.API_KEY}
    )
    assert accepte.status_code == 200
    # la session ouverte donne accès à l'interface
    assert api_anonyme.get("/ui/config").status_code == 200
    api_anonyme.cookies.clear()


def test_openapi_declares_the_api_key_scheme(api):
    schemes = api.get("/openapi.json").json()["components"]["securitySchemes"]
    assert any(s.get("in") == "header" and s.get("name") == "X-API-Key" for s in schemes.values())


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
    with TestClient(module.app, raise_server_exceptions=False,
                    headers={"X-API-Key": conftest.API_KEY}) as client:
        response = client.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5}})
    assert response.status_code == 500
    assert response.json() == {"detail": "Erreur interne du service de scoring."}


def test_gradio_interface_is_mounted(api):
    assert api.get("/ui/").status_code == 200
    response = api.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui"
