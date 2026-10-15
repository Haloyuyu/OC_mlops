"""Tests de l'optimisation de l'inférence : non-régression et suivi des ressources.

Le chemin optimisé (Booster LightGBM direct) doit rendre exactement les mêmes
probabilités et décisions que le modèle MLflow tel qu'enregistré au registry. Ces
tests tournent dans la CI, avant la construction de l'image : une optimisation qui
changerait une seule décision bloquerait le déploiement.
"""

import json
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from api.monitoring import ProductionLog, evenement_prediction, ressources_processus
from api.scoring import MODEL_DIR, ScoringModel, client_to_dict
from monitoring import operations, storage


@pytest.fixture(scope="session")
def reference():
    """Le modèle tel que servi avant optimisation (pyfunc MLflow)."""
    return ScoringModel(MODEL_DIR, backend="pyfunc")


# --------------------------------------------------------------------------- non-régression

def test_backend_optimise_par_defaut(model):
    assert model.backend == "booster"
    assert model.num_threads == 1


def test_backend_inconnu_refuse():
    with pytest.raises(ValueError, match="MODEL_BACKEND"):
        ScoringModel(MODEL_DIR, backend="tensorrt")


def test_meme_sortie_que_le_pyfunc_sur_les_clients_exemple(model, reference, example_clients):
    for _, ligne in example_clients.iterrows():
        client = client_to_dict(ligne)
        assert model.predict(client) == reference.predict(client)


def test_meme_sortie_que_le_pyfunc_sur_les_clients_du_train(model, reference, train_clients):
    for _, ligne in train_clients.iterrows():
        client = client_to_dict(ligne)
        assert model.predict(client) == reference.predict(client)


def test_meme_sortie_avec_valeurs_manquantes(model, reference, complete_client):
    # un client partiel : 3 features renseignées, les 792 autres manquantes
    partiel = {nom: complete_client[nom] for nom in ("EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH")}
    assert model.predict(partiel) == reference.predict(partiel)
    assert model.predict(partiel)["features_renseignees"] == 3


def test_probabilites_en_lot_identiques_au_bit_pres(model, reference, example_clients):
    matrice = example_clients.reindex(columns=model.features).astype("float64")
    attendu = reference.pyfunc.predict(matrice)["probabilite_defaut"].to_numpy()
    np.testing.assert_array_equal(model.predict_proba(matrice.to_numpy()), attendu)


def test_nombre_de_threads_sans_effet_sur_le_resultat(complete_client):
    un = ScoringModel(MODEL_DIR, num_threads=1).predict(complete_client)
    quatre = ScoringModel(MODEL_DIR, num_threads=4).predict(complete_client)
    assert un == quatre


def test_predictions_concurrentes_coherentes(model, example_clients):
    """Le Booster est partagé entre les threads de FastAPI : pas d'interférence."""
    clients = [client_to_dict(ligne) for _, ligne in example_clients.iterrows()]
    attendu = [model.predict(c) for c in clients]
    with ThreadPoolExecutor(max_workers=8) as pool:
        obtenu = list(pool.map(model.predict, clients * 5))
    assert obtenu == attendu * 5


# --------------------------------------------------------------------------- monitoring

def test_ecritures_concurrentes_du_journal_restent_lisibles(tmp_path):
    """Régression : des lignes de ~16 Ko écrites par 16 threads s'entrelaçaient."""
    journal = ProductionLog(tmp_path / "logs")
    entrees = {f"FEATURE_{i}": float(i) for i in range(800)}       # ligne de ~20 Ko

    def ecrire(i):
        journal.record(evenement_prediction(entrees=entrees, latence_ms=float(i)))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(ecrire, range(400)))
    lignes = journal.fichier_du_jour().read_bytes().splitlines()
    assert len(lignes) == 400
    assert all(len(json.loads(ligne)["entrees"]) == 800 for ligne in lignes)


def _ecrire_depuis_un_processus(dossier, n):
    journal = ProductionLog(dossier)
    entrees = {f"FEATURE_{i}": float(i) for i in range(800)}
    for i in range(n):
        journal.record(evenement_prediction(entrees=entrees, latence_ms=float(i)))


@pytest.mark.skipif(sys.platform == "win32",
                    reason="O_APPEND n'est atomique qu'en POSIX ; l'image de production est Linux")
def test_ecritures_de_plusieurs_workers_restent_lisibles(tmp_path):
    """Plusieurs workers uvicorn (processus) écrivent le même journal du jour.

    Sous Linux, une écriture O_APPEND unique n'est jamais entrelacée avec une autre :
    ce test, exécuté par la CI (ubuntu), valide la configuration multi-workers de
    l'image. Sous Windows, le CRT émule O_APPEND (déplacement puis écriture) et des
    lignes se mélangent : on y garde un seul worker.
    """
    dossier = tmp_path / "logs"
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_ecrire_depuis_un_processus, [dossier] * 4, [100] * 4))
    lignes = ProductionLog(dossier).fichier_du_jour().read_bytes().splitlines()
    assert len(lignes) == 400
    assert all(len(json.loads(ligne)["entrees"]) == 800 for ligne in lignes)


def test_evenement_porte_cpu_et_memoire(monkeypatch):
    import api.monitoring as module

    monkeypatch.setattr(module, "FENETRE_CPU_S", 0.0)    # fenêtre close à chaque appel
    sum(i * i for i in range(200_000))                   # un peu de travail CPU
    ressources = ressources_processus()
    assert ressources["cpu_pct"] >= 0
    assert ressources["memoire_mo"] > 0
    evenement = evenement_prediction(entrees={}, ressources=ressources)
    assert evenement["cpu_pct"] == ressources["cpu_pct"]
    assert evenement["memoire_mo"] == ressources["memoire_mo"]


def test_predict_journalise_les_ressources(api, complete_client):
    import api.main as module

    reponse = api.post("/predict", json={"features": complete_client})
    assert reponse.status_code == 200
    requete_id = reponse.headers["X-Request-Id"]
    lignes = module.production_log.fichier_du_jour().read_text(encoding="utf-8").splitlines()
    evenement = next(json.loads(ligne) for ligne in reversed(lignes)
                     if requete_id in ligne and '"prediction"' in ligne)
    assert "cpu_pct" in evenement and evenement["memoire_mo"] > 0


def test_stockage_des_ressources_et_analyse(tmp_path):
    journal = ProductionLog(tmp_path / "logs")
    for cpu in (10.0, 20.0, 30.0):
        journal.record(evenement_prediction(
            entrees={"A": 1.0}, sortie={"probabilite_defaut": 0.1, "decision": 1},
            latence_ms=1.0, ressources={"cpu_pct": cpu, "memoire_mo": 300.0 + cpu}))
    base = tmp_path / "production.db"
    storage.ingerer(tmp_path / "logs", base)
    predictions = storage.charger_predictions(base, features=[])
    resultat = operations.ressources(predictions)
    assert resultat["mesures"] == 3
    assert resultat["cpu_pct"]["p50"] == 20.0
    assert resultat["memoire_mo_max"] == 330.0


def test_ressources_absentes_des_anciens_journaux():
    anciens = pd.DataFrame({"statut": [200], "cpu_pct": [None], "memoire_mo": [None]})
    assert operations.ressources(anciens)["mesures"] == 0
    assert operations.ressources(pd.DataFrame())["mesures"] == 0


def test_base_existante_completee_sans_perte(tmp_path):
    """Une base créée avant l'ajout des colonnes CPU/mémoire est migrée à l'ouverture."""
    base = tmp_path / "ancienne.db"
    ancien_schema = storage.SCHEMA.replace("    cpu_pct              REAL,\n", "") \
                                   .replace("    memoire_mo           REAL,\n", "")
    with sqlite3.connect(base) as cnx:
        cnx.executescript(ancien_schema)
        cnx.execute("INSERT INTO appels (requete_id, evenement, horodatage) "
                    "VALUES ('ancien', 'prediction', '2026-10-01T00:00:00')")
    with storage.connexion(base) as cnx:
        colonnes = {ligne[1] for ligne in cnx.execute("PRAGMA table_info(appels)")}
        assert {"cpu_pct", "memoire_mo"} <= colonnes
        assert cnx.execute("SELECT COUNT(*) FROM appels").fetchone()[0] == 1
