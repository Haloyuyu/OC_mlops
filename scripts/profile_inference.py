"""Profilage cProfile d'une prédiction, pour localiser les goulots d'étranglement.

Rejoue N appels à `ScoringModel.predict` (le chemin réellement servi par l'API, avec
validation des entrées) sur les clients d'exemple, sous cProfile, puis :

- regroupe le temps par étape (validation de schéma MLflow, pandas, LightGBM…) ;
- écrit le profil brut (`.prof`, lisible avec `snakeviz` ou `pstats`) et le top des
  fonctions par temps cumulé dans `docs/optimisation/`.

    python scripts/profile_inference.py --backend pyfunc
    python scripts/profile_inference.py --backend booster
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from api.scoring import BACKENDS, ScoringModel, client_to_dict, load_example_clients  # noqa: E402

SORTIE = RACINE / "docs" / "optimisation"

# Étapes repérées par le fichier et la fonction qui les portent. Le temps cumulé
# d'une étape englobe tout ce qu'elle appelle (pandas compris).
ETAPES = {
    "validation de schéma MLflow": ("mlflow/models/utils.py", "_enforce_schema"),
    "inférence LightGBM (Booster.predict)": ("lightgbm/basic.py", "predict"),
    "wrapper pyfunc + sklearn (predict_proba)": ("lightgbm/sklearn.py", "predict_proba"),
    "construction du DataFrame client": ("pandas/core/frame.py", "reindex"),
    "vectorisation numpy": ("api/scoring.py", "vectorize"),
}


def temps_par_etape(stats, total):
    """Temps cumulé (ms par appel) de chaque étape repérée dans le profil."""
    etapes = {}
    for (fichier, _, fonction), (_, _, _, cumul, _) in stats.stats.items():
        fichier = fichier.replace("\\", "/")
        for nom, (motif_fichier, motif_fonction) in ETAPES.items():
            if fichier.endswith(motif_fichier) and fonction == motif_fonction:
                # une fonction récursive apparaît une fois : on garde le maximum
                etapes[nom] = max(etapes.get(nom, 0.0), cumul)
    return {nom: valeur / total * 1000 for nom, valeur in etapes.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=BACKENDS, default="pyfunc")
    parser.add_argument("--n", type=int, default=500, help="nombre d'appels profilés")
    parser.add_argument("--top", type=int, default=25)
    arguments = parser.parse_args()

    modele = ScoringModel(backend=arguments.backend)
    clients = [client_to_dict(ligne) for _, ligne in load_example_clients().iterrows()]
    for client in clients:  # échauffement : caches, première allocation
        modele.predict(client)

    # 1. temps réel, sans profileur (cProfile ralentit le code Python, pas le C)
    debut = time.perf_counter()
    for i in range(arguments.n):
        modele.predict(clients[i % len(clients)])
    ms_reel = (time.perf_counter() - debut) / arguments.n * 1000

    # 2. même boucle sous cProfile
    profil = cProfile.Profile()
    profil.enable()
    for i in range(arguments.n):
        modele.predict(clients[i % len(clients)])
    profil.disable()

    SORTIE.mkdir(parents=True, exist_ok=True)
    chemin_prof = SORTIE / f"profil_{arguments.backend}.prof"
    profil.dump_stats(chemin_prof)

    tampon = io.StringIO()
    stats = pstats.Stats(profil, stream=tampon).strip_dirs().sort_stats("cumulative")
    stats.print_stats(arguments.top)
    stats_completes = pstats.Stats(profil)
    total_profile = stats_completes.total_tt
    etapes = temps_par_etape(stats_completes, arguments.n)

    lignes = [
        f"Profil cProfile — backend « {arguments.backend} », {arguments.n} appels",
        f"Temps réel sans profileur : {ms_reel:.2f} ms par appel",
        f"Temps sous profileur      : {total_profile / arguments.n * 1000:.2f} ms par appel",
        "",
        "Temps cumulé par étape (ms par appel, sous profileur) :",
    ]
    total_ms = total_profile / arguments.n * 1000
    for nom, ms in sorted(etapes.items(), key=lambda e: -e[1]):
        lignes.append(f"  {nom:<45} {ms:8.2f} ms  ({ms / total_ms:5.1%})")
    lignes += ["", tampon.getvalue()]
    texte = "\n".join(lignes)
    chemin_txt = SORTIE / f"profil_{arguments.backend}.txt"
    chemin_txt.write_text(texte, encoding="utf-8")

    print("\n".join(lignes[:5 + len(etapes)]))
    print(f"\nProfil complet : {chemin_txt.relative_to(RACINE)} ; "
          f"brut : {chemin_prof.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
