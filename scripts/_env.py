"""Lecture du fichier .env pour les scripts en ligne de commande.

L'API reçoit sa configuration par variables d'environnement (`uvicorn --env-file`,
`docker run --env-file`, secrets GitHub). Les scripts, eux, sont lancés à la main :
sans ce petit chargeur, il faudrait penser à exporter `API_KEY` à chaque fois.

Bibliothèque standard uniquement, et les variables déjà définies dans
l'environnement gagnent toujours : un `API_KEY=...` explicite ou un secret de CI ne
doit jamais être écrasé par un fichier local.
"""

import os
from pathlib import Path

ENV_DEFAUT = Path(__file__).resolve().parent.parent / ".env"


def charger_env(chemin=ENV_DEFAUT):
    """Charge les variables manquantes depuis .env. Renvoie les noms ajoutés."""
    chemin = Path(chemin)
    if not chemin.exists():
        return []
    ajoutees = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        nom, _, valeur = ligne.partition("=")
        nom, valeur = nom.strip(), valeur.strip().strip("\"'")
        if nom and not os.getenv(nom):
            os.environ[nom] = valeur
            ajoutees.append(nom)
    return ajoutees
