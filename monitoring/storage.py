"""Stockage des données de production : journaux JSON Lines -> base SQLite.

Deux étages, comme en production réelle :

1. l'API écrit un événement JSON par appel, en local, fichier par jour
   (`logs/production-AAAA-MM-JJ.jsonl`) — format transférable tel quel depuis un
   hôte distant ;
2. ces journaux sont ingérés dans une base SQLite (`monitoring/production.db`),
   interrogeable, sur laquelle tourne l'analyse.

SQLite tient le rôle que PostgreSQL tiendrait en production : même modèle de données,
même SQL, zéro serveur à administrer pour un PoC local. L'ingestion est idempotente
(clé primaire `requete_id` + `evenement`) : réingérer les mêmes journaux ne duplique
rien, ce qui permet de la rejouer sans précaution.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
BASE_DEFAUT = RACINE / "monitoring" / "production.db"
JOURNAUX_DEFAUT = RACINE / "logs"

SCHEMA = """
CREATE TABLE IF NOT EXISTS appels (
    requete_id           TEXT NOT NULL,
    evenement            TEXT NOT NULL,
    horodatage           TEXT NOT NULL,
    route                TEXT,
    methode              TEXT,
    statut               INTEGER,
    latence_ms           REAL,
    latence_modele_ms    REAL,
    modele_nom           TEXT,
    modele_version       TEXT,
    seuil                REAL,
    probabilite_defaut   REAL,
    decision             INTEGER,
    features_renseignees INTEGER,
    erreur_type          TEXT,
    erreur_message       TEXT,
    entrees              TEXT,
    PRIMARY KEY (requete_id, evenement)
);
CREATE INDEX IF NOT EXISTS idx_appels_horodatage ON appels (horodatage);
CREATE INDEX IF NOT EXISTS idx_appels_evenement ON appels (evenement, statut);
"""

COLONNES = [
    "requete_id", "evenement", "horodatage", "route", "methode", "statut",
    "latence_ms", "latence_modele_ms", "modele_nom", "modele_version", "seuil",
    "probabilite_defaut", "decision", "features_renseignees",
    "erreur_type", "erreur_message", "entrees",
]


def connexion(base=BASE_DEFAUT):
    """Ouvre la base (la crée si besoin) avec son schéma."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    cnx = sqlite3.connect(base)
    cnx.executescript(SCHEMA)
    return cnx


def _aplatir(evenement):
    """Événement JSON -> tuple de colonnes SQL."""
    modele = evenement.get("modele") or {}
    sortie = evenement.get("sortie") or {}
    erreur = evenement.get("erreur") or {}
    entrees = evenement.get("entrees")
    return (
        evenement.get("requete_id"),
        evenement.get("evenement"),
        evenement.get("horodatage"),
        evenement.get("route"),
        evenement.get("methode"),
        evenement.get("statut"),
        evenement.get("latence_ms"),
        evenement.get("latence_modele_ms"),
        modele.get("nom"),
        modele.get("version"),
        modele.get("seuil"),
        sortie.get("probabilite_defaut"),
        sortie.get("decision"),
        sortie.get("features_renseignees"),
        erreur.get("type"),
        erreur.get("message"),
        None if entrees is None else json.dumps(entrees, ensure_ascii=False),
    )


def ingerer_fichier(cnx, fichier):
    """Ingère un journal JSON Lines. Renvoie (lignes lues, lignes illisibles)."""
    lignes, illisibles = [], 0
    with open(fichier, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                evenement = json.loads(ligne)
            except json.JSONDecodeError:
                illisibles += 1          # ligne tronquée (arrêt brutal de l'API) : ignorée
                continue
            if evenement.get("requete_id") and evenement.get("evenement"):
                lignes.append(_aplatir(evenement))
            else:
                illisibles += 1
    marques = ", ".join("?" * len(COLONNES))
    cnx.executemany(f"INSERT OR REPLACE INTO appels VALUES ({marques})", lignes)
    cnx.commit()
    return len(lignes), illisibles


def ingerer(dossier=JOURNAUX_DEFAUT, base=BASE_DEFAUT):
    """Ingère tous les journaux d'un dossier. Renvoie un récapitulatif."""
    dossier = Path(dossier)
    fichiers = sorted(dossier.glob("production-*.jsonl"))
    recap = {"fichiers": len(fichiers), "evenements": 0, "illisibles": 0, "base": str(base)}
    with connexion(base) as cnx:
        for fichier in fichiers:
            lus, illisibles = ingerer_fichier(cnx, fichier)
            recap["evenements"] += lus
            recap["illisibles"] += illisibles
        recap["total_en_base"] = cnx.execute("SELECT COUNT(*) FROM appels").fetchone()[0]
    return recap


def _lire(base, condition, depuis=None, jusqu_a=None):
    requete = f"SELECT * FROM appels WHERE {condition}"
    parametres = []
    if depuis:
        requete += " AND horodatage >= ?"
        parametres.append(depuis)
    if jusqu_a:
        requete += " AND horodatage <= ?"
        parametres.append(jusqu_a)
    with connexion(base) as cnx:
        return pd.read_sql_query(requete + " ORDER BY horodatage", cnx, params=parametres)


def charger_requetes(base=BASE_DEFAUT, depuis=None, jusqu_a=None):
    """Événements d'exploitation (toutes routes) : statut et latence."""
    return _lire(base, "evenement = 'requete'", depuis, jusqu_a)


def charger_predictions(base=BASE_DEFAUT, depuis=None, features=None, jusqu_a=None):
    """Appels au modèle, avec les features demandées dépliées en colonnes.

    `features` restreint les colonnes extraites du JSON : on ne déplie que les
    features suivies, pas les 795, sans quoi la table serait inexploitable.
    """
    predictions = _lire(base, "evenement = 'prediction'", depuis, jusqu_a)
    if predictions.empty:
        return predictions
    entrees = predictions["entrees"].apply(lambda t: json.loads(t) if t else {})
    if features is None:
        features = sorted({nom for dico in entrees for nom in dico})
    valeurs = pd.DataFrame(
        [{nom: dico.get(nom) for nom in features} for dico in entrees],
        index=predictions.index, columns=features,
    ).astype("float64")
    return pd.concat([predictions.drop(columns=["entrees"]), valeurs], axis=1)


def volumetrie(base=BASE_DEFAUT):
    """Taille de la base et nombre d'événements — pour le suivi du coût de stockage."""
    chemin = Path(base)
    with connexion(base) as cnx:
        total = cnx.execute("SELECT COUNT(*) FROM appels").fetchone()[0]
        predictions = cnx.execute(
            "SELECT COUNT(*) FROM appels WHERE evenement = 'prediction'").fetchone()[0]
    octets = chemin.stat().st_size if chemin.exists() else 0
    return {
        "evenements": total,
        "predictions": predictions,
        "taille_base_mo": round(octets / 1e6, 2),
        "octets_par_prediction": round(octets / predictions) if predictions else 0,
    }
