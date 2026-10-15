"""Tests du suivi de production : journalisation, stockage, dérive, exploitation."""

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from api.monitoring import PREFIXE, SUFFIXE, ProductionLog, evenement_prediction
from monitoring import drift, operations, report, storage


# --------------------------------------------------------------------------- journal


@pytest.fixture
def journal(tmp_path):
    return ProductionLog(tmp_path / "logs")


def test_journal_ecrit_une_ligne_json(journal):
    journal.record(evenement_prediction(
        entrees={"EXT_SOURCE_2": 0.5}, sortie={"probabilite_defaut": 0.1, "decision": 1},
        latence_ms=42.0, modele={"nom": "m", "version": "1", "seuil": 0.08}))
    lignes = journal.fichier_du_jour().read_text(encoding="utf-8").strip().splitlines()
    assert len(lignes) == 1
    evenement = json.loads(lignes[0])
    assert evenement["evenement"] == "prediction"
    assert evenement["entrees"] == {"EXT_SOURCE_2": 0.5}
    assert evenement["sortie"]["decision"] == 1
    assert evenement["latence_ms"] == 42.0
    assert evenement["horodatage"] and evenement["requete_id"]


def test_journal_ajoute_sans_ecraser(journal):
    for _ in range(3):
        journal.record(evenement_prediction(entrees={"EXT_SOURCE_2": 0.5}, latence_ms=1.0))
    assert len(journal.fichier_du_jour().read_text(encoding="utf-8").strip().splitlines()) == 3


def test_journal_un_fichier_par_jour(journal):
    assert date.today().isoformat() in journal.fichier_du_jour().name


def test_journal_desactive_n_ecrit_rien(tmp_path):
    muet = ProductionLog(tmp_path / "logs", actif=False)
    muet.record(evenement_prediction(entrees={"EXT_SOURCE_2": 0.5}))
    assert muet.fichiers() == []


def test_journal_restreint_aux_features_suivies(tmp_path):
    restreint = ProductionLog(tmp_path / "logs", features=["EXT_SOURCE_2"])
    restreint.record(evenement_prediction(entrees={"EXT_SOURCE_2": 0.5, "AMT_CREDIT": 1e5}))
    evenement = json.loads(restreint.fichier_du_jour().read_text(encoding="utf-8"))
    assert evenement["entrees"] == {"EXT_SOURCE_2": 0.5}
    assert evenement["entrees_tronquees"] is True


def test_purge_respecte_la_retention(journal):
    ancien = journal.dossier / f"{PREFIXE}{(date.today() - timedelta(days=120)).isoformat()}{SUFFIXE}"
    journal.dossier.mkdir(parents=True, exist_ok=True)
    ancien.write_text("{}\n", encoding="utf-8")
    journal.record(evenement_prediction(entrees={"EXT_SOURCE_2": 0.5}))

    supprimes = journal.purge_anciens_journaux(retention_jours=90)
    assert ancien.name in supprimes
    assert not ancien.exists() and journal.fichier_du_jour().exists()


# ------------------------------------------------------------------- journal via l'API


@pytest.fixture
def journal_api(tmp_path, monkeypatch):
    """Redirige le journal de l'API vers un dossier temporaire."""
    import api.main as module

    temporaire = ProductionLog(tmp_path / "logs")
    monkeypatch.setattr(module, "production_log", temporaire)
    return temporaire


def _evenements(journal, type_evenement=None):
    evenements = [json.loads(ligne)
                  for fichier in journal.fichiers()
                  for ligne in fichier.read_text(encoding="utf-8").splitlines() if ligne]
    return [e for e in evenements if type_evenement in (None, e["evenement"])]


def test_api_journalise_entrees_sortie_et_latence(api, journal_api, complete_client):
    reponse = api.post("/predict", json={"features": complete_client})
    assert reponse.status_code == 200

    predictions = _evenements(journal_api, "prediction")
    assert len(predictions) == 1
    evenement = predictions[0]
    # les entrées journalisées sont exactement celles reçues : c'est ce qui rend
    # l'analyse de dérive possible après coup
    assert evenement["entrees"] == complete_client
    assert "entrees_tronquees" not in evenement
    assert evenement["sortie"]["probabilite_defaut"] == reponse.json()["probabilite_defaut"]
    assert evenement["latence_ms"] > 0
    assert evenement["modele"]["seuil"] == reponse.json()["seuil"]


def test_api_journalise_les_entrees_refusees(api, journal_api):
    api.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 0}})
    evenement = _evenements(journal_api, "prediction")[0]
    assert evenement["statut"] == 422
    assert evenement["erreur"]["type"] == "OutOfBoundsValues"
    assert "AMT_INCOME_TOTAL" in evenement["erreur"]["message"]
    assert evenement["sortie"] is None


def test_api_journalise_les_requetes_et_leur_statut(api, api_anonyme, journal_api):
    api.get("/health")
    api_anonyme.get("/model")           # 401
    requetes = _evenements(journal_api, "requete")
    assert {("/health", 200), ("/model", 401)} <= {(e["route"], e["statut"]) for e in requetes}
    assert all(e["latence_ms"] >= 0 for e in requetes)


def test_api_ne_journalise_pas_l_interface(api, journal_api):
    api.get("/ui/")
    assert [e for e in _evenements(journal_api) if e["route"].startswith("/ui")] == []


def test_api_ne_journalise_aucun_secret(api, journal_api, complete_client):
    api.post("/predict", json={"features": complete_client})
    contenu = journal_api.fichier_du_jour().read_text(encoding="utf-8")
    import conftest
    assert conftest.API_KEY not in contenu
    assert "x-api-key" not in contenu.lower()


# --------------------------------------------------------------------------- stockage


@pytest.fixture
def base_peuplee(tmp_path):
    """Une base SQLite alimentée par un journal fabriqué pour le test."""
    dossier = tmp_path / "logs"
    journal = ProductionLog(dossier)
    for i in range(5):
        # horodatages fixés : l'horloge Windows (granularité ~1 ms) produirait des
        # ex æquo et rendrait les tests de fenêtre temporelle indéterministes
        horodatage = f"2026-10-08T10:00:0{i}+00:00"
        journal.record({"requete_id": f"r{i}", "horodatage": horodatage, **evenement_prediction(
            entrees={"EXT_SOURCE_2": 0.3 + i / 100}, sortie={"probabilite_defaut": 0.05 + i / 100,
                                                             "decision": int(i > 2),
                                                             "features_renseignees": 1},
            latence_ms=40 + i, latence_modele_ms=38 + i,
            modele={"nom": "credit_scoring_lgbm", "version": "1", "seuil": 0.08})})
        journal.record({"requete_id": f"r{i}", "horodatage": horodatage,
                        **operations_evenement_requete(statut=200, latence=40 + i)})
    with open(journal.fichier_du_jour(), "a", encoding="utf-8") as f:
        f.write('{"requete_id": "tronqu\n')          # ligne coupée par un arrêt brutal
    base = tmp_path / "production.db"
    return dossier, base


def operations_evenement_requete(statut, latence):
    from api.monitoring import evenement_requete
    return evenement_requete("/predict", "POST", statut, latence)


def test_ingestion_charge_les_evenements(base_peuplee):
    dossier, base = base_peuplee
    recap = storage.ingerer(dossier, base)
    assert recap["evenements"] == 10
    assert recap["illisibles"] == 1          # la ligne tronquée est comptée, pas ingérée
    assert recap["total_en_base"] == 10


def test_ingestion_est_idempotente(base_peuplee):
    dossier, base = base_peuplee
    storage.ingerer(dossier, base)
    recap = storage.ingerer(dossier, base)
    assert recap["total_en_base"] == 10      # réingérer ne duplique pas


def test_chargement_deplie_les_features(base_peuplee):
    dossier, base = base_peuplee
    storage.ingerer(dossier, base)
    predictions = storage.charger_predictions(base, features=["EXT_SOURCE_2"])
    assert len(predictions) == 5
    assert predictions["EXT_SOURCE_2"].between(0.3, 0.35).all()
    assert predictions["probabilite_defaut"].notna().all()
    assert storage.charger_requetes(base).shape[0] == 5


def test_fenetre_temporelle(base_peuplee):
    dossier, base = base_peuplee
    storage.ingerer(dossier, base)
    toutes = storage.charger_predictions(base, features=[])
    milieu = toutes["horodatage"].iloc[2]
    assert len(storage.charger_predictions(base, features=[], depuis=milieu)) == 3
    assert len(storage.charger_predictions(base, features=[], jusqu_a=milieu)) == 3


def test_volumetrie(base_peuplee):
    dossier, base = base_peuplee
    storage.ingerer(dossier, base)
    mesures = storage.volumetrie(base)
    assert mesures["predictions"] == 5 and mesures["evenements"] == 10
    assert mesures["taille_base_mo"] > 0 and mesures["octets_par_prediction"] > 0


# ------------------------------------------------------------------------------ dérive


@pytest.fixture
def aleatoire():
    return np.random.default_rng(1001)


def test_psi_nul_sur_la_meme_distribution(aleatoire):
    echantillon = aleatoire.normal(size=5000)
    assert drift.psi(echantillon, echantillon) == pytest.approx(0, abs=1e-9)
    assert drift.psi(echantillon, aleatoire.normal(size=5000)) < drift.PSI_MODERE


def test_psi_detecte_un_decalage(aleatoire):
    reference = aleatoire.normal(size=5000)
    assert drift.psi(reference, aleatoire.normal(loc=1.0, size=2000)) > drift.PSI_DERIVE


def test_psi_croit_avec_le_decalage(aleatoire):
    reference = aleatoire.normal(size=5000)
    valeurs = [drift.psi(reference, aleatoire.normal(loc=d, size=3000)) for d in (0.1, 0.5, 1.5)]
    assert valeurs == sorted(valeurs)


def test_psi_ignore_les_valeurs_manquantes(aleatoire):
    reference = aleatoire.normal(size=2000)
    avec_nan = np.concatenate([reference[:1000], np.full(500, np.nan)])
    assert drift.psi(reference, avec_nan) < drift.PSI_MODERE


def test_psi_non_evaluable_sur_une_feature_constante():
    assert np.isnan(drift.psi(np.ones(500), np.ones(100)))


def test_kolmogorov_smirnov(aleatoire):
    reference = aleatoire.normal(size=2000)
    _, p_identique = drift.kolmogorov_smirnov(reference, aleatoire.normal(size=2000))
    _, p_decale = drift.kolmogorov_smirnov(reference, aleatoire.normal(loc=0.8, size=2000))
    assert p_identique > 0.05 and p_decale < 1e-6


def test_analyse_features_verdicts(aleatoire):
    reference = pd.DataFrame({
        "stable": aleatoire.normal(size=3000),
        "decalee": aleatoire.normal(size=3000),
        "constante": np.ones(3000),
    })
    production = pd.DataFrame({
        "stable": aleatoire.normal(size=600),
        "decalee": aleatoire.normal(loc=1.5, size=600),
        "constante": np.ones(600),
    })
    resultats = drift.analyser_features(reference, production).set_index("feature")
    assert resultats.loc["decalee", "verdict"] == "dérive"
    assert resultats.loc["stable", "verdict"] == "stable"
    assert resultats.loc["constante", "verdict"] == "non évaluable"
    assert resultats.index[0] == "decalee"          # trié par PSI décroissant


def test_analyse_features_detecte_les_valeurs_manquantes(aleatoire):
    reference = pd.DataFrame({"f": aleatoire.normal(size=2000)})
    production = pd.DataFrame({"f": np.concatenate([aleatoire.normal(size=200),
                                                    np.full(400, np.nan)])})
    resultats = drift.analyser_features(reference, production)
    assert resultats.loc[0, "verdict"] == "dérive des valeurs manquantes"


def test_echantillon_insuffisant(aleatoire):
    reference = pd.DataFrame({"f": aleatoire.normal(size=2000)})
    production = pd.DataFrame({"f": aleatoire.normal(loc=3, size=10)})
    assert drift.analyser_features(reference, production).loc[0, "verdict"] \
        == "échantillon insuffisant"


def test_derive_des_predictions(aleatoire):
    reference = aleatoire.beta(2, 20, size=3000)
    identique = aleatoire.beta(2, 20, size=800)
    plus_risquee = aleatoire.beta(4, 20, size=800)
    stable = drift.analyser_predictions(reference, identique, seuil=0.08)
    derive = drift.analyser_predictions(reference, plus_risquee, seuil=0.08)
    assert stable["verdict"] == "stable"
    assert derive["verdict"].startswith("dérive")
    assert derive["ecart_taux_refus"] > stable["ecart_taux_refus"]
    assert derive["taux_refus_production"] > derive["taux_refus_reference"]


def test_synthese_niveaux(aleatoire):
    reference = pd.DataFrame({f"f{i}": aleatoire.normal(size=2000) for i in range(5)})
    stable = pd.DataFrame({f"f{i}": aleatoire.normal(size=500) for i in range(5)})
    derivee = pd.DataFrame({f"f{i}": aleatoire.normal(loc=1.5, size=500) for i in range(5)})

    assert drift.synthese(drift.analyser_features(reference, stable))["niveau"] == "STABLE"
    synthese_derive = drift.synthese(drift.analyser_features(reference, derivee))
    assert synthese_derive["niveau"] == "ALERTE"
    assert synthese_derive["features_en_derive"] == 5 and synthese_derive["motifs"]


def test_synthese_alerte_si_le_taux_de_refus_bouge(aleatoire):
    reference = pd.DataFrame({f"f{i}": aleatoire.normal(size=2000) for i in range(5)})
    stable = pd.DataFrame({f"f{i}": aleatoire.normal(size=500) for i in range(5)})
    predictions = {"verdict": "stable", "ecart_taux_refus": 0.12, "psi": 0.05}
    assert drift.synthese(drift.analyser_features(reference, stable),
                          predictions)["niveau"] == "ALERTE"


# ------------------------------------------------------------------------ exploitation


def _requetes(statuts, latences):
    return pd.DataFrame({
        "horodatage": [f"2026-10-08T10:{i // 60:02d}:{i % 60:02d}" for i in range(len(statuts))],
        "route": ["/predict"] * len(statuts),
        "statut": statuts,
        "latence_ms": latences,
    })


def test_percentiles():
    mesures = operations.percentiles(list(range(1, 101)))
    assert mesures["p50"] == pytest.approx(50.5)
    assert mesures["p95"] == pytest.approx(95.05, abs=0.1)
    assert mesures["max"] == 100


def test_exploitation_stable():
    mesures = operations.analyser(_requetes([200] * 100, [40.0] * 100))
    assert mesures["niveau"] == "STABLE" and mesures["anomalies"] == []
    assert mesures["taux_succes"] == 1.0
    assert mesures["par_statut"] == {200: 100}


def test_exploitation_detecte_les_erreurs_serveur():
    mesures = operations.analyser(_requetes([200] * 95 + [500] * 5, [40.0] * 100))
    assert mesures["niveau"] == "ALERTE"
    assert mesures["taux_erreur_serveur"] == pytest.approx(0.05)
    assert any("erreurs serveur" in a for a in mesures["anomalies"])


def test_exploitation_detecte_les_erreurs_client():
    mesures = operations.analyser(_requetes([200] * 60 + [422] * 40, [40.0] * 100))
    assert any("erreurs client" in a for a in mesures["anomalies"])


def test_exploitation_detecte_une_latence_hors_slo():
    mesures = operations.analyser(_requetes([200] * 100, [40.0] * 90 + [900.0] * 10))
    assert any("p95" in a and "ms" in a for a in mesures["anomalies"])


def test_exploitation_detecte_une_degradation_progressive():
    latences = [5.0] * 80 + [40.0] * 20            # sous le SLO, mais ×8
    mesures = operations.analyser(_requetes([200] * 100, latences))
    assert mesures["degradation_latence"]["facteur"] > 3
    assert any("multipliée" in a for a in mesures["anomalies"])


def test_exploitation_sans_donnees():
    vide = pd.DataFrame(columns=["horodatage", "route", "statut", "latence_ms"])
    mesures = operations.analyser(vide)
    assert mesures["appels"] == 0 and mesures["niveau"] == "STABLE"


def test_erreurs_frequentes():
    predictions = pd.DataFrame({
        "erreur_type": ["OutOfBoundsValues", "OutOfBoundsValues", "UnknownFeatures", None],
        "erreur_message": ["revenu nul", "revenu nul", "FOO", None],
    })
    frequentes = operations.erreurs_frequentes(predictions)
    assert frequentes[0] == {"type": "OutOfBoundsValues", "message": "revenu nul", "occurrences": 2}


# ---------------------------------------------------------------------------- rapport


def test_rapport_html_complet(aleatoire, tmp_path):
    features = [f"f{i}" for i in range(4)]
    reference = pd.DataFrame({f: aleatoire.normal(size=1500) for f in features})
    reference["probabilite_defaut"] = aleatoire.beta(2, 20, size=1500)
    production = pd.DataFrame({f: aleatoire.normal(loc=1.2, size=400) for f in features})
    production["probabilite_defaut"] = aleatoire.beta(4, 20, size=400)
    production["latence_ms"] = aleatoire.normal(45, 5, size=400)

    resultats = drift.analyser_features(reference, production, features)
    predictions = drift.analyser_predictions(reference["probabilite_defaut"],
                                             production["probabilite_defaut"], 0.08)
    synthese = drift.synthese(resultats, predictions)
    exploitation = operations.analyser(_requetes([200] * 400, production["latence_ms"].tolist()))
    volumetrie = {"evenements": 800, "taille_base_mo": 1.2, "octets_par_prediction": 1500}

    html = report.construire_rapport(synthese, resultats, predictions, exploitation,
                                     volumetrie, reference, production)
    assert "<svg" in html and html.count("<svg") >= 4
    assert "ALERTE" in html
    assert "nan" not in html.lower().replace("nannyml", "")
    assert all(f in html for f in features)          # le tableau liste toutes les features
    assert "RGPD" in html

    resume = report.ecrire_json(tmp_path / "resume.json", synthese, resultats, predictions,
                               exploitation, volumetrie)
    recharge = json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))
    assert recharge["drift"]["niveau"] == "ALERTE"
    assert len(recharge["features"]) == len(features)
    assert resume["volumetrie"]["evenements"] == 800


def test_journal_indisponible_ne_casse_pas_l_api(tmp_path, monkeypatch):
    """Un dossier de journaux inutilisable désactive le suivi, sans faire tomber l'API."""
    import api.monitoring as module

    def mkdir_refuse(*args, **kwargs):
        raise PermissionError("lecture seule")

    monkeypatch.setattr(module.Path, "mkdir", mkdir_refuse)
    journal = ProductionLog(tmp_path / "interdit")
    assert journal.actif is False
    assert journal.record(evenement_prediction(entrees={"A": 1.0})) is not None
    assert journal.fichiers() == []
