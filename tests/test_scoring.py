"""Tests unitaires : chargement du modèle champion et logique de prédiction."""

import math

import pytest

from api.scoring import UnknownFeatures, client_to_dict


def test_exported_model_is_the_champion(model):
    assert model.metadata["nom"] == "credit_scoring_lgbm"
    assert model.metadata["alias"] == "champion"
    assert len(model.features) == 795


def test_embedded_threshold_is_the_business_threshold(model):
    # seuil optimisé sur le coût métier, cohérent avec le tag du registry (et non 0.5)
    assert model.threshold == pytest.approx(float(model.metadata["tags"]["seuil_metier"]))
    assert model.threshold < 0.5


def test_example_clients_have_all_features(model, example_clients):
    assert list(example_clients.columns) == model.features
    assert len(example_clients) > 0


def test_predict_complete_client(model, complete_client):
    result = model.predict(complete_client)
    assert 0 <= result["probabilite_defaut"] <= 1
    assert result["decision"] in (0, 1)
    assert result["seuil"] == model.threshold


@pytest.mark.parametrize("index", range(5))
def test_decision_matches_threshold(model, train_clients, index):
    result = model.predict(client_to_dict(train_clients.iloc[index]))
    expected = int(result["probabilite_defaut"] >= model.threshold)
    assert result["decision"] == expected
    assert result["libelle_decision"] == ("crédit refusé" if expected else "crédit accordé")


def test_known_default_is_refused(model, train_clients):
    # le 1er client d'exemple du train est un défaut réel (TARGET = 1)
    result = model.predict(client_to_dict(train_clients.iloc[0]))
    assert result["decision"] == 1


def test_prediction_is_deterministic(model, complete_client):
    assert model.predict(complete_client) == model.predict(complete_client)


def test_feature_order_has_no_effect(model, complete_client):
    reversed_client = dict(reversed(list(complete_client.items())))
    assert model.predict(reversed_client)["probabilite_defaut"] == pytest.approx(
        model.predict(complete_client)["probabilite_defaut"]
    )


def test_absent_features_are_treated_as_missing(model, complete_client):
    partial = {"EXT_SOURCE_2": complete_client["EXT_SOURCE_2"], "EXT_SOURCE_3": 0.2}
    result = model.predict(partial)
    assert result["features_renseignees"] == 2
    assert 0 <= result["probabilite_defaut"] <= 1


def test_null_values_are_accepted(model, complete_client):
    with_nulls = {**complete_client, "EXT_SOURCE_1": None, "EXT_SOURCE_3": None}
    result = model.predict(with_nulls)
    assert not math.isnan(result["probabilite_defaut"])


def test_unknown_feature_is_rejected(model, complete_client):
    with pytest.raises(UnknownFeatures) as error:
        model.predict({**complete_client, "REVENU_IMAGINAIRE": 1.0})
    assert error.value.names == ["REVENU_IMAGINAIRE"]


def test_score_reacts_to_external_sources(model, complete_client):
    # les scores externes sont les variables les plus prédictives : de bons scores
    # doivent réduire le risque par rapport à de mauvais scores
    good = {**complete_client, "EXT_SOURCE_1": 0.9, "EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.9}
    bad = {**complete_client, "EXT_SOURCE_1": 0.05, "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.05}
    assert model.predict(good)["probabilite_defaut"] < model.predict(bad)["probabilite_defaut"]
