# PROJET 6 - OpenClassrooms - Initiez-vous au MLOps (1/2)

Modèle de scoring crédit « Prêt à dépenser » (données Home Credit), optimisé sur un
**coût métier** asymétrique (`10 × FN + 1 × FP`) plutôt que sur l'AUC seule, tracké et
servi avec MLFlow.

## Livrables

| Fichier | Contenu |
|---|---|
| `exploration.ipynb` | EDA, contrôle qualité, fusion et feature engineering → `data/processed_data.csv` |
| `preprocess_datasets.py` | Pipeline de préparation (kernel Aguiar adapté) |
| `modelisation.ipynb` | Modèles, validation croisée, Optuna, seuil métier, robustesse, registry, serving |
| `test_serving.py` | Test REST du modèle servi |
| `presentation.html` | Présentation de synthèse du projet |

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
