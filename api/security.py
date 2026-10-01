"""Authentification de l'API par clé.

La clé vient de la variable d'environnement `API_KEY` : secret GitHub dans le
pipeline, variable d'environnement en local. **Si aucune clé n'est fournie, une clé
aléatoire est générée au démarrage et affichée dans les logs** : l'API n'est donc
jamais exposée sans clé, même par oubli de configuration.

Deux portes mènent au modèle, et toutes les deux sont fermées par ce secret :
l'en-tête `X-API-Key` pour les routes REST, un formulaire de connexion pour
l'interface Gradio (qui appelle le modèle en interne, sans passer par REST).
"""

import logging
import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

HEADER_NAME = "X-API-Key"
UI_USERNAME = os.getenv("API_USER", "analyste")

logger = logging.getLogger("api.security")


def _load_key():
    """Renvoie (clé, générée ?) — jamais de clé vide."""
    key = os.getenv("API_KEY", "").strip()
    return (key, False) if key else (secrets.token_urlsafe(32), True)


API_KEY, KEY_WAS_GENERATED = _load_key()

_api_key_header = APIKeyHeader(name=HEADER_NAME, auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)):
    """Dépendance FastAPI : refuse la requête si la clé est absente ou fausse.

    `compare_digest` évite de fuiter la clé par le temps de comparaison.
    """
    if key is None or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail=f"Clé d'API absente ou invalide : renseignez l'en-tête {HEADER_NAME}.",
        )
    return key


def log_key_status():
    """Trace au démarrage l'origine de la clé (jamais la clé fournie par secret)."""
    if KEY_WAS_GENERATED:
        logger.warning(
            "Aucune variable API_KEY fournie : clé générée pour cette session.\n"
            "    Clé : %s\n"
            "    Interface /ui : utilisateur « %s », mot de passe = cette clé.\n"
            "    Définissez API_KEY pour une clé stable.",
            API_KEY, UI_USERNAME,
        )
    else:
        logger.info("Clé d'API chargée depuis la variable d'environnement API_KEY.")
