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
| `model/` | Modèle `@champion` exporté du registry, 20 clients d'exemple, référence de dérive |
| `scripts/export_model.py` | Export du modèle champion du registry vers `model/` |
| `monitoring/` | Stockage des données de production et analyse (dérive, exploitation) |
| `scripts/simulate_traffic.py` | Génération de trafic de production, nominal ou dérivé |
| `scripts/analyze_production.py` | Analyse automatique : dérive, anomalies, rapport HTML |
| `docs/rapport_optimisation.md` | Rapport d'optimisation de l'inférence : goulots, tests, résultats, configuration finale |
| `scripts/profile_inference.py`, `benchmark_inference.py`, `load_test.py` | Profilage cProfile, banc d'essai des variantes (vitesse, précision, biais), test de charge HTTP |
| `docs/optimisation/` | Résultats bruts des mesures (profils, banc d'essai, tests de charge) |
| `.env.example` | Modèle de configuration locale (clé d'API), copier en `.env` |
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
pip install -r requirements-dev.txt

# 1. générer une clé d'API et la placer dans .env (ignoré par git, cf. .env.example)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. en local
uvicorn api.main:app --port 8000 --env-file .env

# ou avec Docker
docker build -t api-scoring .
docker run --env-file .env -p 8000:8000 api-scoring
```

- Interface Gradio : http://localhost:8000/ui
- Documentation OpenAPI : http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -H "X-API-Key: une-cle-de-votre-choix" \
     -d '{"features": {"EXT_SOURCE_2": 0.79, "EXT_SOURCE_3": 0.16, "PAYMENT_RATE": 0.036}}'
# {"probabilite_defaut": 0.066, "decision": 0, "libelle_decision": "crédit accordé",
#  "seuil": 0.08, "features_renseignees": 3}
```

### Authentification

Les routes de scoring exigent la clé d'API dans l'en-tête **`X-API-Key`**. Seul `/health`
reste public, car Docker et le pipeline s'en servent comme sonde de vie.

L'interface Gradio est protégée par la **même clé** : utilisateur `analyste` (configurable
par `API_USER`), mot de passe = la clé. Sans cela, `/ui` appellerait le modèle en interne et
contournerait l'authentification des routes REST.

La clé vient de la variable d'environnement `API_KEY`. Si aucune clé n'est fournie, l'API en
**génère une au démarrage et l'affiche dans ses logs** — elle n'est donc jamais ouverte par
oubli de configuration. Dans Swagger, le bouton **Authorize** permet de saisir la clé.

| Où | Comment |
|---|---|
| En local | Fichier `.env` (ignoré par git), lu par `uvicorn --env-file .env` ou `docker run --env-file .env` |
| Dans le pipeline | Secret de dépôt GitHub `API_KEY` (Settings → Secrets and variables → Actions) |

Écrire `API_KEY=valeur` **sans espace** autour du `=` : c'est la seule forme acceptée aussi
bien par `python-dotenv` que par le `--env-file` de Docker. Une clé ne doit jamais être
commitée : `.env.example` sert de modèle versionné, `.env` reste local.

### Validation des entrées et gestion des erreurs

| Cas | Réponse |
|---|---|
| Clé d'API absente ou invalide | `401` (sauf sur `/health`) |
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
| Mémoire (RSS) | ~325 Mo |
| Temps de réponse de `/predict` (1 client) | p50 4,7 ms, p95 6,0 ms |
| Inférence seule | ~0,5 ms |
| Débit (1 worker, 16 clients simultanés) | ~270 appels/s |
| Image Docker publiée (compressée sur le registry) | 240 Mo |

Le conteneur est lancé avec `--memory=1g --cpus=1`, et l'étape de déploiement refuse un hôte
offrant moins de 1 Go de RAM disponible ou 2 Go de disque.

### Optimisation de l'inférence

Le temps de réponse est passé de 37 ms (p95 121 ms) à 4,7 ms (p95 6 ms) **sans changer le
modèle ni une seule décision** : le profilage a montré que 99 % du temps partait dans la
validation de schéma MLflow et pandas, et non dans les arbres LightGBM. L'API appelle
désormais directement le Booster LightGBM sur un vecteur numpy. ONNX Runtime et la
quantification ont été évalués puis écartés. Détails, mesures et justification de la
configuration : [`docs/rapport_optimisation.md`](docs/rapport_optimisation.md).

| Variable d'environnement | Défaut | Rôle |
|---|---|---|
| `MODEL_BACKEND` | `booster` | `booster` (optimisé) ou `pyfunc` (modèle MLflow tel qu'au registry, référence et retour arrière) |
| `LGBM_NUM_THREADS` | `1` | Threads OpenMP par prédiction |
| `WEB_CONCURRENCY` | `1` | Workers uvicorn ; un par vCPU alloué (Linux uniquement au-delà de 1) |

```bash
pip install -r requirements-optimisation.txt     # + onnx, onnxruntime, onnxmltools
python scripts/profile_inference.py --backend pyfunc
python scripts/benchmark_inference.py --clients 20000
python scripts/load_test.py --label booster --journaux logs
```

### Tests

```bash
pytest
```

122 tests : chargement du modèle champion, cohérence de la décision avec le seuil métier,
valeurs manquantes, valeurs aberrantes, types incorrects, routes HTTP, authentification
(clé absente, fausse, tronquée ; `/health` public ; connexion Gradio), documentation
OpenAPI, non-rechargement du modèle entre deux requêtes, journalisation de production
(entrées, sorties, latences, erreurs, secrets jamais journalisés), ingestion SQLite
idempotente, PSI et Kolmogorov-Smirnov, seuils d'exploitation et rapport HTML,
non-régression de l'inférence optimisée (sorties identiques au modèle du registry),
intégrité du journal sous écritures concurrentes, ressources CPU et mémoire.

## Suivi des données de production (dérive et anomalies)

### Ce qui est stocké, et où

```
API ──► logs/production-AAAA-MM-JJ.jsonl ──► monitoring/production.db ──► rapport HTML
        (1 ligne JSON par appel)              (SQLite, interrogeable)      + résumé JSON
```

Deux étages, comme en production réelle : l'API écrit des journaux **JSON Lines**
transférables tels quels depuis l'hôte, qui sont ensuite ingérés dans une base
**SQLite** sur laquelle tourne l'analyse. SQLite tient ici le rôle de PostgreSQL —
même modèle de données, même SQL, aucun serveur à administrer. L'ingestion est
idempotente (clé `requete_id` + `evenement`) : on peut la rejouer sans précaution.

Chaque appel produit deux événements : un événement **`prediction`** (entrées du
modèle, sortie, latence d'inférence, CPU et mémoire du processus, erreur éventuelle) et un événement
**`requete`** (route, statut HTTP, latence totale), y compris pour les `401` et les
`422`. C'est ce qui permet d'analyser séparément la dérive du modèle et la santé du
service.

| Variable d'environnement | Rôle |
|---|---|
| `MONITORING_DIR` | Dossier des journaux (défaut `logs/`) |
| `MONITORING_ENABLED` | `0` désactive le suivi |
| `MONITORING_RETENTION_DAYS` | Rétention des journaux (défaut 90 jours) |
| `MONITORING_FEATURES` | `all` (défaut) ou liste de features à journaliser |

Avec Docker, monter le dossier pour que les journaux survivent au conteneur :
`docker run --env-file .env -v ./logs:/app/logs -p 8000:8000 api-scoring`.

**Coût de stockage, mesuré** : journaliser les 795 features coûte ~16 Ko par appel
(33 Ko en base, index compris), soit ~1,6 Go pour 100 000 appels. Restreindre
`MONITORING_FEATURES` aux 33 features suivies ramène à ~1 Ko par appel, au prix de
l'impossibilité d'analyser plus tard une feature non retenue. Le suivi ne peut jamais
faire tomber l'API : si le dossier est inaccessible, le journal se désactive et
l'erreur est tracée.

**RGPD** : aucune donnée directement identifiante n'est journalisée — ni identifiant
client, ni adresse IP, ni clé d'API. Les features restent des données personnelles
(situation financière d'un demandeur) : d'où une rétention de 90 jours, appliquée par
`python scripts/analyze_production.py --purger`, et des journaux exclus de git.

### Lancer l'analyse

```bash
# 1. générer du trafic (API démarrée ; la clé est lue dans .env)
python scripts/simulate_traffic.py --n 250                      # population nominale
python scripts/simulate_traffic.py --n 180 --profil derive --erreurs 24

# 2. ingestion + analyse + rapport
python scripts/analyze_production.py
python scripts/analyze_production.py --evidently                # rapport Evidently en plus
```

Le script enchaîne ingestion, détection de dérive, analyse d'exploitation, rapport
HTML (`rapports/monitoring.html`) et résumé JSON. **Il renvoie le code 1 en cas
d'ALERTE** : il peut donc servir de sonde dans une tâche planifiée, sans avoir à lire
le rapport pour savoir s'il faut regarder.

| Ce qui est détecté | Méthode |
|---|---|
| Dérive d'une feature | PSI (seuils 0,1 / 0,25) **et** Kolmogorov-Smirnov, seuil corrigé par Bonferroni |
| Feature qui cesse d'être renseignée | Écart du taux de valeurs manquantes > 10 points |
| Dérive des prédictions | PSI sur les scores + écart du taux de refus |
| Taux d'erreur | 5xx > 1 %, 4xx > 20 % |
| Latence anormale | p95 > 100 ms (objectif de service) **ou** p95 récent > 3 × p95 précédent |

Deux indicateurs plutôt qu'un seul : KS devient significatif pour des écarts
minuscules dès que le volume monte, et le PSI seul ne dit rien de la fiabilité de
l'écart. Une feature n'est déclarée en dérive que si l'effet est net (PSI ≥ 0,25), ou
modéré **et** statistiquement significatif.

Evidently est **optionnel** (`pip install -r requirements-monitoring.txt`) : la
bibliothèque et ses dépendances pèsent ~800 Mo, hors de proportion avec une image
d'API de 240 Mo. L'analyse n'en dépend pas ; `--evidently` ajoute un rapport
complémentaire.

### Étude de la dérive : ce que les données montrent

Référence : 2000 dossiers du **jeu d'entraînement**, 33 features suivies (les 30 plus
influentes du modèle et les variables du dossier client). Production simulée : des
dossiers du **jeu test**, c'est-à-dire des clients que le modèle n'a jamais vus.

| Période | Verdict | Features en dérive | Taux de refus |
|---|---|---|---|
| Trafic nominal (250 appels) | VIGILANCE | 4 / 33 | 31,9 % contre 30,3 % en référence |
| Population dérivée (180 appels) | ALERTE | 9 / 33 | 55,6 % contre 30,3 % |

**Le jeu test de Home Credit dérive réellement du jeu d'entraînement.** Sur du trafic
nominal, sans aucune perturbation, `PAYMENT_RATE` affiche un PSI de 0,50 et
`AMT_CREDIT` / `AMT_GOODS_PRICE` environ 0,17. Vérification sur les données brutes :
la médiane de `PAYMENT_RATE` passe de 0,050 (train) à 0,061 (test) et celle de
`AMT_CREDIT` de 513 k à 450 k. Le jeu test est une fenêtre temporelle postérieure, pas
un tirage aléatoire du train — des crédits plus petits, remboursés plus vite. Ce n'est
pas un artefact du simulateur : un échantillon aléatoire de 1000 dossiers test donne
le même résultat.

**Points de vigilance qui en découlent**

1. **Une dérive des features ne se traduit pas mécaniquement en dérive des décisions.**
   Malgré un PSI de 0,50 sur la feature la plus influente du modèle, le taux de refus
   ne bouge que de 1,6 point sur le trafic nominal. C'est précisément pour cela qu'il
   faut surveiller les deux : la dérive des entrées alerte tôt, celle des sorties
   mesure l'impact métier.
2. **Le seuil de 0,080 a été optimisé sur la distribution d'entraînement.** Sur la
   population dérivée, le taux de refus grimpe à 55,6 % : le compromis coût
   faux-négatifs / faux-positifs validé en modélisation n'est plus celui qui
   s'applique. Un seuil doit être réévalué quand la population bouge, pas seulement
   quand le modèle est réentraîné.
3. **Les valeurs manquantes sont un angle mort.** LightGBM les accepte sans erreur :
   une feature qui cesse d'être alimentée dégrade le score en silence, sans aucune
   trace dans les erreurs HTTP. D'où leur suivi séparé du PSI.
4. **La référence vieillit.** Elle est liée au modèle `@champion` exporté ; tout
   réentraînement impose de régénérer `model/reference_drift.csv`, sinon la
   comparaison perd son sens.
5. **La dérive observée est mesurée sur quelques centaines d'appels.** En deçà de
   30 appels, l'analyse refuse de conclure (« échantillon insuffisant ») plutôt que de
   produire un verdict dont la variance serait plus grande que l'effet.

### Pipeline CI/CD (GitHub Actions)

À chaque push sur `master` :

1. **Tests** : tests unitaires et d'intégration (`pytest`), rapport JUnit publié en artefact.
2. **Build** (si les tests passent) : construction de l'image Docker, lancement du
   conteneur et test de fumée, puis publication sur GitHub Container Registry
   (`ghcr.io/<owner>/<repo>-api`, tags `<sha>` et `latest`).
3. **Déploiement** (si l'image est publiée) : environnement `production` **simulé** sur
   le runner, qui vérifie d'abord les ressources de l'hôte, tire l'image depuis le
   registry, lance le conteneur et contrôle `/health`, `/predict` et `/ui`
   (`scripts/smoke_test.py`), ainsi que le **temps de réponse** : un p95 de `/predict`
   supérieur à 100 ms fait échouer le déploiement.

Sur une pull request, les tests et la construction de l'image s'exécutent, sans
publication ni déploiement.

#### Gestion des secrets

Aucun identifiant n'est écrit dans le dépôt :

- **Registry** : authentification via le secret `GITHUB_TOKEN` fourni par GitHub Actions ;
  chaque job ne demande que les permissions dont il a besoin (`packages: write` au build,
  `packages: read` au déploiement).
- **Clé d'API** : le pipeline lit le secret de dépôt **`API_KEY`** (Settings → Secrets and
  variables → Actions). S'il n'existe pas, le job tire une clé éphémère valable le temps du
  job, de façon à ne jamais démarrer l'API sans clé. Dans les deux cas la clé est masquée
  dans les logs (`::add-mask::`) et transmise au conteneur par `docker run -e API_KEY`.
