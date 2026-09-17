"""Tests unitaires : chargement du modèle champion et logique de prédiction."""

import math

import pytest

from api.scoring import FeaturesInconnues, client_en_dict


def test_modele_exporte_est_le_champion(modele):
    assert modele.metadonnees["nom"] == "credit_scoring_lgbm"
    assert modele.metadonnees["alias"] == "champion"
    assert len(modele.features) == 795


def test_seuil_embarque_est_le_seuil_metier(modele):
    # seuil optimisé sur le coût métier, cohérent avec le tag du registry (et non 0.5)
    assert modele.seuil == pytest.approx(float(modele.metadonnees["tags"]["seuil_metier"]))
    assert modele.seuil < 0.5


def test_clients_exemple_ont_toutes_les_features(modele, clients_exemple):
    assert list(clients_exemple.columns) == modele.features
    assert len(clients_exemple) > 0


def test_prediction_client_complet(modele, client_complet):
    resultat = modele.predire(client_complet)
    assert 0 <= resultat["probabilite_defaut"] <= 1
    assert resultat["decision"] in (0, 1)
    assert resultat["seuil"] == modele.seuil


@pytest.mark.parametrize("index", range(5))
def test_decision_coherente_avec_le_seuil(modele, clients_train, index):
    resultat = modele.predire(client_en_dict(clients_train.iloc[index]))
    attendu = int(resultat["probabilite_defaut"] >= modele.seuil)
    assert resultat["decision"] == attendu
    assert resultat["libelle_decision"] == ("crédit refusé" if attendu else "crédit accordé")


def test_defaut_connu_est_refuse(modele, clients_train):
    # le 1er client d'exemple du train est un défaut réel (TARGET = 1)
    resultat = modele.predire(client_en_dict(clients_train.iloc[0]))
    assert resultat["decision"] == 1


def test_prediction_deterministe(modele, client_complet):
    assert modele.predire(client_complet) == modele.predire(client_complet)


def test_ordre_des_features_sans_effet(modele, client_complet):
    inverse = dict(reversed(list(client_complet.items())))
    assert modele.predire(inverse)["probabilite_defaut"] == pytest.approx(
        modele.predire(client_complet)["probabilite_defaut"]
    )


def test_features_absentes_traitees_comme_manquantes(modele, client_complet):
    partiel = {"EXT_SOURCE_2": client_complet["EXT_SOURCE_2"], "EXT_SOURCE_3": 0.2}
    resultat = modele.predire(partiel)
    assert resultat["features_renseignees"] == 2
    assert 0 <= resultat["probabilite_defaut"] <= 1


def test_valeurs_nulles_acceptees(modele, client_complet):
    avec_nulls = {**client_complet, "EXT_SOURCE_1": None, "EXT_SOURCE_3": None}
    resultat = modele.predire(avec_nulls)
    assert not math.isnan(resultat["probabilite_defaut"])


def test_feature_inconnue_rejetee(modele, client_complet):
    with pytest.raises(FeaturesInconnues) as erreur:
        modele.predire({**client_complet, "REVENU_IMAGINAIRE": 1.0})
    assert erreur.value.noms == ["REVENU_IMAGINAIRE"]


def test_score_sensible_aux_sources_externes(modele, client_complet):
    # les scores externes sont les variables les plus prédictives : de bons scores
    # doivent réduire le risque par rapport à de mauvais scores
    bons = {**client_complet, "EXT_SOURCE_1": 0.9, "EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.9}
    mauvais = {**client_complet, "EXT_SOURCE_1": 0.05, "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.05}
    assert modele.predire(bons)["probabilite_defaut"] < modele.predire(mauvais)["probabilite_defaut"]
