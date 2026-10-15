"""Journal de production de l'API : un événement JSON par appel.

Chaque ligne du fichier est un objet JSON autonome (format JSON Lines), ce qui permet
de l'analyser plus tard sans parser de logs textuels, et de le transférer tel quel
depuis un hôte distant vers un poste d'analyse.

Ce qui est enregistré : horodatage, identifiant de requête, route, statut HTTP,
latences (requête complète et inférence seule), ressources du processus (CPU, mémoire),
entrées du modèle, sortie du modèle, et l'erreur éventuelle. C'est le minimum pour analyser ensuite la dérive des données
(entrées), la dérive des prédictions (sorties) et les incidents d'exploitation
(taux d'erreur, latence).

RGPD : aucune donnée directement identifiante n'est stockée — ni identifiant client,
ni adresse IP, ni clé d'API. Les features restent des données personnelles (situation
financière d'un demandeur), d'où une rétention limitée (`MONITORING_RETENTION_DAYS`,
90 jours par défaut) appliquée par `purge_anciens_journaux`.

Réglages par variables d'environnement :

    MONITORING_DIR               dossier des journaux (défaut : logs/)
    MONITORING_ENABLED           0 pour désactiver complètement
    MONITORING_RETENTION_DAYS    nombre de jours conservés (défaut : 90)
    MONITORING_FEATURES          `all` (défaut) ou liste de features à journaliser

Coût de stockage, mesuré : journaliser les 795 features coûte ~16 Ko par appel, soit
~1,6 Go pour 100 000 appels. Restreindre `MONITORING_FEATURES` aux 33 features suivies
(colonnes de `model/reference_drift.csv`) ramène à ~1 Ko par appel, au prix de l'
impossibilité d'analyser plus tard la dérive d'une feature non retenue.
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psutil

RACINE = Path(__file__).resolve().parent.parent
DEFAULT_DIR = RACINE / "logs"
RETENTION_DAYS = int(os.getenv("MONITORING_RETENTION_DAYS", "90"))
PREFIXE = "production-"
SUFFIXE = ".jsonl"


def features_journalisees():
    """Features à journaliser : toutes par défaut, sinon la liste configurée."""
    reglage = (os.getenv("MONITORING_FEATURES") or "all").strip()
    if reglage.lower() in ("all", "toutes", ""):
        return None
    return [nom.strip() for nom in reglage.split(",") if nom.strip()]


def nouvel_identifiant():
    return uuid.uuid4().hex


def _maintenant():
    return datetime.now(timezone.utc)


class ProductionLog:
    """Écrit les événements de production en JSON Lines, un fichier par jour.

    La date dans le nom du fichier assure la rotation quotidienne sans dépendance
    ni handler de logging à configurer ; le fichier est ouvert en ajout à chaque
    événement (~0,3 ms pour une ligne de 16 Ko, à comparer aux ~5 ms d'une requête).
    """

    def __init__(self, dossier=None, actif=True, features=None):
        self.dossier = Path(dossier or os.getenv("MONITORING_DIR") or DEFAULT_DIR)
        self.actif = actif and os.getenv("MONITORING_ENABLED", "1") != "0"
        self.features = features if features is not None else features_journalisees()
        if self.actif:
            try:
                self.dossier.mkdir(parents=True, exist_ok=True)
            except OSError as erreur:
                # un journal qui ne peut pas écrire (dossier en lecture seule, volume
                # non monté) ne doit pas empêcher l'API de servir des prédictions
                self.actif = False
                logging.getLogger("api.monitoring").error(
                    "Journal de production désactivé : %s inutilisable (%s)", self.dossier, erreur)

    _verrou = threading.Lock()

    def _ajouter(self, octets):
        """Ajoute une ligne au journal du jour en une seule écriture atomique.

        Une ligne fait ~16 Ko : écrite par `open(..., "a").write`, elle part en
        plusieurs appels système, et des requêtes simultanées (pool de threads de
        FastAPI) entrelaçaient leurs morceaux — le test de charge à 16 clients
        produisait 2 % de lignes illisibles, ignorées à l'ingestion. Le verrou
        sérialise les threads ; `O_APPEND` + un unique `os.write` couvre plusieurs
        processus (workers uvicorn) écrivant le même fichier — sous Linux, où POSIX
        rend l'écriture en ajout atomique. Le CRT Windows l'émule sans cette garantie :
        en local sous Windows, garder un seul worker.
        """
        with self._verrou:
            fd = os.open(self.fichier_du_jour(),
                         os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
            try:
                ecrits = os.write(fd, octets)
                while ecrits < len(octets):   # écriture partielle (rare) : on complète
                    ecrits += os.write(fd, octets[ecrits:])
            finally:
                os.close(fd)

    def fichier_du_jour(self, jour=None):
        return self.dossier / f"{PREFIXE}{jour or date.today().isoformat()}{SUFFIXE}"

    def record(self, evenement):
        """Ajoute un événement (dict) au journal du jour et le renvoie complété."""
        if not self.actif:
            return evenement
        evenement.setdefault("horodatage", _maintenant().isoformat())
        evenement.setdefault("requete_id", nouvel_identifiant())
        if self.features is not None and isinstance(evenement.get("entrees"), dict):
            entrees = evenement["entrees"]
            evenement = {**evenement,
                         "entrees": {nom: entrees[nom] for nom in self.features if nom in entrees},
                         "entrees_tronquees": True}
        ligne = json.dumps(evenement, ensure_ascii=False, default=str)
        try:
            self._ajouter((ligne + "\n").encode("utf-8"))
        except OSError as erreur:
            # idem : perdre le suivi est préférable à perdre le service
            self.actif = False
            logging.getLogger("api.monitoring").error(
                "Écriture du journal impossible, suivi désactivé (%s)", erreur)
        return evenement

    def fichiers(self):
        """Journaux présents, du plus ancien au plus récent."""
        return sorted(self.dossier.glob(f"{PREFIXE}*{SUFFIXE}")) if self.dossier.exists() else []

    def purge_anciens_journaux(self, retention_jours=RETENTION_DAYS):
        """Supprime les journaux dépassant la durée de rétention (RGPD)."""
        limite = date.today() - timedelta(days=retention_jours)
        supprimes = []
        for fichier in self.fichiers():
            try:
                jour = date.fromisoformat(fichier.stem[len(PREFIXE):])
            except ValueError:
                continue
            if jour < limite:
                fichier.unlink()
                supprimes.append(fichier.name)
        return supprimes


_PROCESSUS = psutil.Process()
FENETRE_CPU_S = 1.0
_cpu = {"debut": time.monotonic(), "temps_cpu": sum(_PROCESSUS.cpu_times()[:2]), "pct": None}
_cpu_verrou = threading.Lock()


def _cpu_pct():
    """Occupation CPU du processus, moyennée sur des fenêtres d'au moins une seconde.

    Mesurer entre deux appels successifs (quelques ms) ne veut rien dire : l'horloge
    CPU du système avance par pas de 10 à 16 ms, et la plupart des intervalles
    vaudraient 0 %. On renvoie donc la moyenne de la dernière fenêtre close.
    """
    with _cpu_verrou:
        maintenant = time.monotonic()
        ecoule = maintenant - _cpu["debut"]
        if ecoule >= FENETRE_CPU_S:
            temps_cpu = sum(_PROCESSUS.cpu_times()[:2])          # user + system
            _cpu["pct"] = round(100 * (temps_cpu - _cpu["temps_cpu"]) / ecoule, 1)
            _cpu["debut"], _cpu["temps_cpu"] = maintenant, temps_cpu
        return _cpu["pct"]


def ressources_processus():
    """CPU et mémoire du processus de l'API, au moment de l'appel.

    `cpu_pct` : occupation CPU moyenne du processus sur la dernière fenêtre d'une
    seconde (100 % = un cœur plein, 400 % = quatre cœurs), `None` tant que la première
    fenêtre n'est pas close. C'est ce qui relie une latence dégradée à une saturation
    CPU. Pas de mesure GPU : le modèle (arbres LightGBM) s'exécute uniquement sur CPU.
    """
    return {
        "cpu_pct": _cpu_pct(),
        "memoire_mo": round(_PROCESSUS.memory_info().rss / 2**20, 1),
    }


def evenement_prediction(entrees, sortie=None, erreur=None, statut=200,
                         latence_ms=None, latence_modele_ms=None, modele=None,
                         ressources=None):
    """Construit l'événement d'un appel à /predict (succès comme échec)."""
    ressources = ressources or {}
    return {
        "evenement": "prediction",
        "route": "/predict",
        "methode": "POST",
        "statut": statut,
        "latence_ms": None if latence_ms is None else round(latence_ms, 3),
        "latence_modele_ms": None if latence_modele_ms is None else round(latence_modele_ms, 3),
        "cpu_pct": ressources.get("cpu_pct"),
        "memoire_mo": ressources.get("memoire_mo"),
        "modele": modele,
        "entrees": entrees,
        "sortie": sortie,
        "erreur": erreur,
    }


def evenement_requete(route, methode, statut, latence_ms):
    """Construit l'événement d'exploitation d'une requête HTTP quelconque."""
    return {
        "evenement": "requete",
        "route": route,
        "methode": methode,
        "statut": statut,
        "latence_ms": round(latence_ms, 3),
    }
