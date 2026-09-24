"""Tests unitaires : chargement du modèle champion et logique de prédiction."""

import math

import pytest

from api.scoring import FEATURE_BOUNDS, OutOfBoundsValues, UnknownFeatures, client_to_dict


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


@pytest.mark.parametrize(
    ("feature", "value"),
    [
        ("DAYS_BIRTH", 1825.0),  # date de naissance dans le futur : âge de -5 ans
        ("DAYS_BIRTH", -1825.0),  # âge de 5 ans : en dessous de la majorité
        ("AMT_INCOME_TOTAL", 0.0),  # revenu nul : impossible dans les 307 511 dossiers
        ("AMT_INCOME_TOTAL", -30000.0),
        ("AMT_CREDIT", 0.0),
        ("EXT_SOURCE_2", 1.5),  # score externe normalisé, donc borné à 1
        ("EXT_SOURCE_3", -0.2),
        ("CNT_CHILDREN", -1.0),
        ("DAYS_EMPLOYED", 365243.0),  # anomalie Home Credit, remplacée par NaN au préprocessing
    ],
)
def test_out_of_bounds_value_is_rejected(model, complete_client, feature, value):
    with pytest.raises(OutOfBoundsValues) as error:
        model.predict({**complete_client, feature: value})
    assert feature in error.value.details[0]


def test_bounds_report_every_faulty_feature(model, complete_client):
    aberrant = {**complete_client, "AMT_INCOME_TOTAL": 0.0, "EXT_SOURCE_1": 42.0}
    with pytest.raises(OutOfBoundsValues) as error:
        model.predict(aberrant)
    assert [nom for nom, *_ in error.value.faults] == ["AMT_INCOME_TOTAL", "EXT_SOURCE_1"]


@pytest.mark.parametrize("feature", sorted(FEATURE_BOUNDS))
def test_bounds_accept_their_own_limits(model, complete_client, feature):
    low, high = FEATURE_BOUNDS[feature]
    for limite in (low, high):
        assert model.predict({**complete_client, feature: limite})["decision"] in (0, 1)


def test_engineered_features_are_not_bounded(model, complete_client):
    # seules les variables du dossier client ont des bornes métier ; les agrégats
    # calculés (ici une somme de montants) ne doivent pas être contraints
    assert "BURO_AMT_CREDIT_SUM_MAX" not in FEATURE_BOUNDS
    result = model.predict({**complete_client, "BURO_AMT_CREDIT_SUM_MAX": 5e8})
    assert 0 <= result["probabilite_defaut"] <= 1


def test_missing_bounded_feature_is_allowed(model):
    # une valeur manquante n'est pas une valeur aberrante : elle reste acceptée
    assert model.predict({"AMT_INCOME_TOTAL": None, "EXT_SOURCE_2": 0.5})["features_renseignees"] == 1


def test_score_reacts_to_external_sources(model, complete_client):
    # les scores externes sont les variables les plus prédictives : de bons scores
    # doivent réduire le risque par rapport à de mauvais scores
    good = {**complete_client, "EXT_SOURCE_1": 0.9, "EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.9}
    bad = {**complete_client, "EXT_SOURCE_1": 0.05, "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.05}
    assert model.predict(good)["probabilite_defaut"] < model.predict(bad)["probabilite_defaut"]
