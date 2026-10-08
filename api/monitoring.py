"""Journal de production de l'API : un événement JSON par appel.

Chaque ligne du fichier est un objet JSON autonome (format JSON Lines), ce qui permet
de l'analyser plus tard sans parser de logs textuels, et de le transférer tel quel
depuis un hôte distant vers un poste d'analyse.

Ce qui est enregistré : horodatage, identifiant de requête, route, statut HTTP,
latences (requête complète et inférence seule), entrées du modèle, sortie du modèle,
et l'erreur éventuelle. C'est le minimum pour analyser ensuite la dérive des données
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
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
    événement (quelques dizaines de microsecondes, à comparer aux ~40 ms d'inférence).
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
            with open(self.fichier_du_jour(), "a", encoding="utf-8") as f:
                f.write(ligne + "\n")
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


def evenement_prediction(entrees, sortie=None, erreur=None, statut=200,
                         latence_ms=None, latence_modele_ms=None, modele=None):
    """Construit l'événement d'un appel à /predict (succès comme échec)."""
    return {
        "evenement": "prediction",
        "route": "/predict",
        "methode": "POST",
        "statut": statut,
        "latence_ms": None if latence_ms is None else round(latence_ms, 3),
        "latence_modele_ms": None if latence_modele_ms is None else round(latence_modele_ms, 3),
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
