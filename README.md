# PROJET 6 - OpenClassrooms - Initiez-vous au MLOps (1/2)

Modèle de scoring crédit « Prêt à dépenser » (données Home Credit), optimisé sur un
**coût métier** asymétrique (`10 × FN + 1 × FP`) plutôt que sur l'AUC seule, tracké et
servi avec MLFlow.

## Livrables

| Fichier | Contenu |
|---|---|
| `notebooks/exploration.ipynb` | EDA, contrôle qualité, fusion et feature engineering → `data/processed_data.csv` |
| `preprocessing/preprocess_datasets.py` | Pipeline de préparation (kernel Aguiar adapté) |
| `notebooks/modelisation.ipynb` | Modèles, validation croisée, Optuna, seuil métier, robustesse, registry, serving |
| `test_serving.py` | Test REST du modèle servi |
| `presentation.pptx` | Présentation de synthèse du projet |
| `api/` | API FastAPI (`/predict`, `/model`, `/health`) + interface Gradio (`/ui`) |
| `model/` | Modèle `@champion` exporté du registry + 20 clients d'exemple du jeu test |
| `scripts/export_model.py` | Export du modèle champion du registry vers `model/` |
| `tests/` | Tests unitaires (modèle, logique de scoring) et d'intégration (routes HTTP) |
| `Dockerfile` | Image Docker de l'API |
| `.github/workflows/ci-cd.yml` | Pipeline CI/CD : tests → image Docker → déploiement |

## Installation

```bash
poetry install
```

## Utilisation

```bash
# 1. Serveur de tracking + model registry (terminal 1)
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# 2. Exécuter exploration.ipynb puis modelisation.ipynb
#    (la dernière partie enregistre le modèle champion dans le registry)

# 3. Serving du modèle champion (terminal 2)
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run mlflow models serve -m "models:/credit_scoring_lgbm@champion" `
    --port 5001 --env-manager local

# 4. Test de l'API (terminal 3)
poetry run python test_serving.py
```

## Décision métier

Le modèle servi ne renvoie pas seulement une probabilité : il applique le **seuil métier
de 0,080** (et non 0,5) et renvoie `{probabilite_defaut, decision, seuil_applique}`.
`decision = 1` signifie **crédit refusé**. Le seuil est également stocké en tag de la
version du modèle dans le registry.

## API de scoring (FastAPI + Gradio)

L'API charge le modèle `@champion` figé dans `model/` : ni l'image Docker ni la CI n'ont
accès au serveur MLFlow local. Après un nouvel enregistrement au registry :

```bash
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000   # terminal 1
poetry run python scripts/export_model.py                                   # terminal 2
```

### Lancer l'API

```bash
# en local
pip install -r requirements-dev.txt
uvicorn api.main:app --port 8000

# ou avec Docker
docker build -t api-scoring .
docker run -p 8000:8000 api-scoring
```

- Interface Gradio : http://localhost:8000/ui
- Documentation OpenAPI : http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"      -d '{"features": {"EXT_SOURCE_2": 0.79, "EXT_SOURCE_3": 0.16, "PAYMENT_RATE": 0.036}}'
# {"probabilite_defaut": 0.066, "decision": 0, "libelle_decision": "crédit accordé",
#  "seuil": 0.08, "features_renseignees": 3}
```

### Validation des entrées et gestion des erreurs

| Cas | Réponse |
|---|---|
| Feature absente ou `null` | Valeur manquante, gérée nativement par LightGBM comme à l'entraînement |
| Feature inconnue du modèle | `422` + `features_inconnues` |
| Valeur hors bornes métier (âge négatif, revenu nul, `EXT_SOURCE` > 1…) | `422` + `features_hors_bornes` |
| Type incorrect (texte au lieu d'un nombre), `features` vide ou absent | `422` (validation Pydantic) |
| Panne imprévue du modèle | `500` + message JSON, trace côté serveur |

Les bornes métier (`FEATURE_BOUNDS` dans [api/scoring.py](api/scoring.py)) ne portent que sur
les variables issues directement du dossier client, avec une marge sur les valeurs observées
dans les 307 511 dossiers d'entraînement : les ~780 autres features sont des agrégats
calculés, dont la plage n'est pas un contrat métier. Les cas d'erreur sont documentés dans
Swagger (`GET /docs`) ; la liste des 795 features est donnée par `GET /model`.

### Ressources et performances

Le modèle est chargé **une seule fois au démarrage** (module `api.main`), jamais par requête :
un chargement coûte environ 2,5 s. Mesures sur l'API en fonctionnement, avec le modèle
champion et les 795 features :

| | Mesure |
|---|---|
| Mémoire (RSS), stable après 150 requêtes | ~260 Mo |
| Latence moyenne par prédiction | ~45 ms |
| Image Docker | ~5 Go de disque (LightGBM, scikit-learn, pandas, Gradio) |

Le conteneur est lancé avec `--memory=1g --cpus=1`, et l'étape de déploiement refuse un hôte
offrant moins de 1 Go de RAM disponible ou 8 Go de disque.

### Tests

```bash
pytest
```

60 tests : chargement du modèle champion, cohérence de la décision avec le seuil métier,
valeurs manquantes, valeurs aberrantes, types incorrects, routes HTTP, documentation OpenAPI,
non-rechargement du modèle entre deux requêtes.

### Pipeline CI/CD (GitHub Actions)

À chaque push sur `master` :

1. **Tests** : tests unitaires et d'intégration (`pytest`), rapport JUnit publié en artefact.
2. **Build** (si les tests passent) : construction de l'image Docker, lancement du
   conteneur et test de fumée, puis publication sur GitHub Container Registry
   (`ghcr.io/<owner>/<repo>-api`, tags `<sha>` et `latest`).
3. **Déploiement** (si l'image est publiée) : environnement `production` **simulé** sur
   le runner, qui vérifie d'abord les ressources de l'hôte, tire l'image depuis le
   registry, lance le conteneur et contrôle `/health`, `/predict` et `/ui`
   (`scripts/smoke_test.py`).

Sur une pull request, les tests et la construction de l'image s'exécutent, sans
publication ni déploiement.

Aucun identifiant n'est écrit dans le dépôt : l'authentification au registry utilise le
secret `GITHUB_TOKEN` fourni par GitHub Actions, et chaque job ne demande que les
permissions dont il a besoin (`packages: write` au build, `packages: read` au déploiement).
