"""Tests d'intégration : routes HTTP de l'API (FastAPI + Gradio monté)."""

import pytest


def test_health(api):
    reponse = api.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"


def test_infos_modele(api, modele):
    reponse = api.get("/model")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["alias"] == "champion"
    assert corps["seuil"] == modele.seuil
    assert corps["features"] == modele.features


def test_predict_client_complet(api, modele, client_complet):
    reponse = api.post("/predict", json={"features": client_complet})
    assert reponse.status_code == 200
    corps = reponse.json()
    assert set(corps) == {"probabilite_defaut", "decision", "libelle_decision", "seuil", "features_renseignees"}
    # l'API renvoie exactement ce que calcule le modèle
    assert corps == pytest.approx(modele.predire(client_complet))


def test_predict_client_partiel(api):
    reponse = api.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5, "DAYS_BIRTH": -12000}})
    assert reponse.status_code == 200
    assert reponse.json()["features_renseignees"] == 2


def test_predict_feature_inconnue(api):
    reponse = api.post("/predict", json={"features": {"EXT_SOURCE_2": 0.5, "INCONNUE": 1}})
    assert reponse.status_code == 422
    assert reponse.json()["detail"]["features_inconnues"] == ["INCONNUE"]


@pytest.mark.parametrize(
    "corps",
    [
        {},
        {"features": {}},
        {"features": {"EXT_SOURCE_2": "élevé"}},
        {"features": [0.5, 0.2]},
    ],
    ids=["sans_features", "features_vides", "valeur_texte", "liste_au_lieu_de_dict"],
)
def test_predict_entree_invalide(api, corps):
    assert api.post("/predict", json=corps).status_code == 422


def test_documentation_openapi(api):
    reponse = api.get("/openapi.json")
    assert reponse.status_code == 200
    assert "/predict" in reponse.json()["paths"]


def test_interface_gradio_montee(api):
    assert api.get("/ui/").status_code == 200
    reponse = api.get("/", follow_redirects=False)
    assert reponse.status_code in (302, 307)
    assert reponse.headers["location"] == "/ui"
