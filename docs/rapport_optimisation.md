# Rapport d'optimisation de l'inférence — API de scoring crédit

*Octobre 2026 — modèle `credit_scoring_lgbm@champion` (LightGBM, 2 550 arbres, 795 features, seuil métier 0,080)*

## Synthèse

| Indicateur | Avant | Après | Gain |
|---|---|---|---|
| Inférence seule, p50 (appel isolé) | 36,4 ms | 0,47 ms | **×77** |
| Inférence seule, p95 (appel isolé) | 113 ms | 0,84 ms | **×135** |
| Temps de réponse HTTP, p50 (1 client) | 37,1 ms | 4,7 ms | **×7,9** |
| Temps de réponse HTTP, p95 (1 client) | 121 ms | 6,0 ms | **×20** |
| Débit maximal, 1 processus (16 clients) | 22 appels/s | 268 appels/s | **×12** |
| Temps de réponse p50 / p95, conteneur déployé par la CI (1 vCPU) | — | 2,8 / 3,0 ms | seuil : 100 ms |
| CPU consommé par prédiction | ~300 ms CPU | ~4 ms CPU | **×70** |
| Décisions modifiées (20 000 clients) | — | **0** | sorties identiques au bit près |

Le modèle n'a pas changé : ce sont les **mêmes arbres**, appelés autrement. Le temps
d'un appel partait à **99 % dans les couches au-dessus de LightGBM** (validation de
schéma MLflow, pandas, scikit-learn). Les arbres eux-mêmes coûtent 0,3 ms. La version
optimisée appelle directement le moteur LightGBM sur un vecteur numpy. Ses
probabilités sont strictement égales à celles du modèle du registry, ce qui exclut
toute régression de précision ou de biais.

ONNX Runtime et la quantification ont été testés. ONNX gagne encore 0,24 ms par
appel, mais il change une décision sur 20 000 et ajoute une dépendance. Ce gain pèse
moins de 5 % du temps de réponse. ONNX n'a donc pas été retenu (§ 5).

---

## 1. Point de départ : ce que disent les données de monitoring

Le journal de production (`logs/production-2026-10-08.jsonl`, 943 prédictions, trafic
simulé par `scripts/simulate_traffic.py`) donnait :

| Mesure journalisée | p50 | p95 | p99 | max |
|---|---|---|---|---|
| `latence_modele_ms` (appel de `ScoringModel.predict`) | 28,6 ms | 110,8 ms | 124,0 ms | 142,8 ms |
| `latence_ms` (requête HTTP complète) | 29,1 ms | 110,9 ms | 126,0 ms | 145,3 ms |

Trois constats ont orienté les hypothèses :

1. **Le temps de réponse est presque entièrement dans `ScoringModel.predict`.** La
   requête complète et l'appel au modèle ne diffèrent que de 0,5 ms : FastAPI, la
   sérialisation et le journal ne sont pas le problème.
2. **La traîne est anormalement longue.** Un p95 à 4 fois le p50 signale un phénomène
   intermittent, et non un coût constant.
3. **Le journal ne contenait ni CPU ni mémoire.** Impossible de dire si la latence
   venait d'une saturation CPU.

Le monitoring a donc d'abord été complété : chaque prédiction enregistre désormais
`cpu_pct` (CPU du processus moyenné sur une fenêtre d'une seconde) et `memoire_mo`
(RSS). Ces champs sont ingérés en base SQLite (migration automatique des bases
existantes), puis restitués par `scripts/analyze_production.py`. Il n'y a pas de
mesure GPU : un modèle à arbres de décision s'exécute sur CPU (§ 5.5).

Hypothèses de départ :

- **H1.** Le surcoût vient des couches au-dessus de LightGBM : validation de schéma
  MLflow sur 795 colonnes, construction de DataFrames pandas.
- **H2.** La traîne vient de la concurrence entre threads. Le modèle avait
  `n_jobs=4`, donc 4 threads OpenMP par prédiction, en plus du pool de threads de
  FastAPI.
- **H3.** L'écriture du journal (~16 Ko de JSON par appel) pèse sur la latence.

## 2. Méthode et outils

Tous les résultats se reproduisent avec les scripts du dépôt :

| Outil | Rôle | Sorties |
|---|---|---|
| `scripts/profile_inference.py` | cProfile sur 500 appels réels de `ScoringModel.predict` | `docs/optimisation/profil_*.{txt,prof}` |
| `scripts/benchmark_inference.py` | Décomposition sans profileur, latence unitaire, charge concurrente, précision et biais sur 20 000 clients, pour chaque variante | `docs/optimisation/benchmark_inference.{md,json}` |
| `scripts/load_test.py` | Test de charge HTTP de l'API réelle (1, 4, 16 clients simultanés), complété par la vue serveur issue du journal de production | `docs/optimisation/load_test_*.json` |
| `scripts/analyze_production.py` | Analyse du journal (latences, erreurs, CPU, mémoire, dérive) | rapport HTML |

Machine de mesure : Intel x86-64, 12 cœurs logiques, Windows 11, Python 3.12.7.
Les mesures absolues dépendent de la machine ; les **rapports** avant/après sont
stables d'une exécution à l'autre (trois exécutions du banc d'essai : latences à moins de 10 % près).

## 3. Goulots d'étranglement identifiés

### 3.1 Profil cProfile : la validation de schéma MLflow domine

`python scripts/profile_inference.py --backend pyfunc` (500 appels) :

```
Temps réel sans profileur : 39.37 ms par appel
Temps sous profileur      : 100.72 ms par appel

Temps cumulé par étape (ms par appel, sous profileur) :
  validation de schéma MLflow                      70.40 ms  (69.9%)
  wrapper pyfunc + sklearn (predict_proba)          5.18 ms  ( 5.1%)
  inférence LightGBM (Booster.predict)              4.45 ms  ( 4.4%)
```

La fonction la plus coûteuse est `mlflow.models.utils._enforce_named_col_schema`. Elle
vérifie et convertit chacune des 795 colonnes une par une, ce qui représente 159 000
accès `DataFrame.__getitem__` pour 200 appels. cProfile gonfle cependant le code Python
d'un facteur 2,5, car il instrumente chaque appel de fonction. Les proportions ont donc
été confirmées sans profileur.

### 3.2 Décomposition sans profileur

Chaque étape est chronométrée dans sa propre boucle, sur un DataFrame neuf, en médiane
sur 200 appels :

| Étape d'un appel pyfunc | ms | Part |
|---|---|---|
| Construction du DataFrame pandas (1 × 795) | 6,40 | 18 % |
| **Validation de schéma MLflow** | **21,90** | **62 %** |
| Surcoût scikit-learn + conversion pandas → LightGBM | 6,21 | 18 % |
| **Arbres LightGBM (le calcul utile)** | **0,30** | **0,9 %** |
| Contrôles d'entrée et mise en forme de la sortie | 0,46 | 1 % |
| **Total (hors ramasse-miettes)** | **35,3** | |

**H1 confirmée.** Moins de 1 % du temps sert à évaluer le modèle.

### 3.3 La traîne de latence vient du ramasse-miettes Python, pas de la concurrence

La traîne existe **même en appels séquentiels** : p50 à 36 ms, p95 à 113 ms, sans
aucune concurrence. H2 ne l'explique donc pas. Chaque appel pyfunc crée des dizaines
de milliers d'objets pandas temporaires, ce qui déclenche régulièrement une **collecte
complète (génération 2)** du ramasse-miettes :

| Mesure (300 appels pyfunc séquentiels) | p50 | p95 | p99 |
|---|---|---|---|
| Ramasse-miettes actif | 35,4 ms | 109,8 ms | 116,6 ms |
| Ramasse-miettes désactivé (diagnostic uniquement) | 35,0 ms | **37,5 ms** | 38,7 ms |

On observe **65 collectes de génération 2 pour 1 000 appels pyfunc, contre 0** pour
la version optimisée. Désactiver le ramasse-miettes n'est pas une solution de
production (la mémoire croîtrait sans limite). La vraie solution consiste à ne plus
créer ces objets.

### 3.4 Le parallélisme OpenMP gaspille le CPU sous charge

Avec 8 requêtes simultanées et le Booster direct :

| Threads OpenMP par prédiction | p95 | Débit | CPU du processus |
|---|---|---|---|
| 4 (`n_jobs` du modèle entraîné) | 8,7 ms | 2 088 /s | 856 % |
| tous (12) | 8,0 ms | 2 241 /s | 922 % |
| **1** | **7,1 ms** | **2 278 /s** | **175 %** |

**H2 confirmée en partie.** Les threads supplémentaires n'apportent rien en latence
ni en débit sur une ligne de 795 valeurs, mais ils **quintuplent le CPU consommé**.
Sur le conteneur de production, limité à 1 vCPU (`--cpus=1`), quatre threads OpenMP
se disputeraient un seul cœur.

### 3.5 Le journal n'est pas un goulot, mais il était défaillant sous concurrence

H3 est infirmée sur la latence : (dé)sérialiser et écrire environ 16 Ko coûte environ
0,5 ms. En revanche, le test de charge a révélé **un défaut d'intégrité**. À 16
requêtes simultanées, 42 lignes sur 1 850 (2,3 %) du journal étaient entrelacées, et
donc illisibles. Elles étaient ignorées sans bruit à l'ingestion. Un test unitaire
reproduisant 16 threads perdait 80 événements sur 400 sous Windows. Cause : chaque
ligne partait en plusieurs écritures système, sans verrou.

Correctif dans `api/monitoring.py` : un verrou entre threads, et l'écriture de chaque
ligne en **un seul `os.write` en mode `O_APPEND`**. Résultat : 0 ligne perdue sur
6 086 événements. Le cas multi-processus (plusieurs workers uvicorn) est garanti par
POSIX sous Linux, l'environnement de l'image Docker. Il est vérifié par un test exécuté
dans la CI Ubuntu (`test_ecritures_de_plusieurs_workers_restent_lisibles`). Sous
Windows, où le CRT émule `O_APPEND` sans cette garantie, on garde un seul worker.

## 4. Stratégies testées

Toutes les variantes ont été mesurées sur le même modèle et les mêmes clients
(`docs/optimisation/benchmark_inference.md`).

| Variante | Stratégie | p50 isolé | p95 isolé | Débit (8 threads) | CPU (8 threads) |
|---|---|---|---|---|---|
| `pyfunc` | Référence : modèle MLflow du registry | 36,4 ms | 113 ms | 24 /s | 749 % |
| `booster-4t` | Optimisation de code, 4 threads | 0,58 ms | 0,83 ms | 2 088 /s | 856 % |
| **`booster-1t`** | **Optimisation de code, 1 thread** | **0,47 ms** | **0,84 ms** | **2 278 /s** | **175 %** |
| `booster-tous` | Optimisation de code, 12 threads | 0,60 ms | 0,78 ms | 2 241 /s | 922 % |
| `onnx-fp32` | Conversion ONNX + ONNX Runtime | 0,24 ms | 0,38 ms | 3 900 /s | 175 % |
| `onnx-int8` | ONNX + quantification dynamique int8 | 0,23 ms | 0,36 ms | 4 182 /s | 174 % |

### 4.1 Optimisation de code : appel direct du Booster LightGBM

`ScoringModel` charge toujours le même artefact MLflow (`model/credit_scoring_lgbm`,
exporté du registry), d'où il tire les features, le seuil et les arbres. À
l'inférence, en revanche :

- le client est placé dans un vecteur numpy pré-alloué de 795 `NaN`, grâce à un index
  `{feature: position}` calculé une fois au chargement ;
- `Booster.predict(vecteur, num_threads=1)` calcule la probabilité ;
- la décision applique le seuil exact du modèle enregistré.

Les contrôles d'entrée (features inconnues, bornes métier) sont inchangés. Le
contrat de l'API ne change pas non plus. L'ancien chemin reste disponible par
`MODEL_BACKEND=pyfunc` : il sert de référence aux tests de non-régression, et de retour
arrière immédiat sans reconstruire l'image.

### 4.2 ONNX Runtime

Le modèle est converti par `onnxmltools.convert_lightgbm` (opset 15, 1,8 s, fichier de
2,9 Mo), puis exécuté par ONNX Runtime 1.31 sur 1 thread. La latence est **deux fois
plus faible que celle du Booster** : 0,24 ms contre 0,47 ms. Mais :

- l'opérateur `TreeEnsembleClassifier` d'ONNX travaille en **float32**, alors que
  LightGBM compare les features en float64. Un client proche d'un seuil de coupure
  change de branche : l'écart de probabilité maximal atteint 0,0039, et **une décision
  sur 20 000 bascule** (§ 6) ;
- le modèle servi ne serait plus l'artefact du registry, mais une conversion de cet
  artefact, avec une étape et deux dépendances de plus (`onnxruntime`, `onnxmltools`).

### 4.3 Quantification

`onnxruntime.quantization.quantize_dynamic` (int8) produit un graphe **strictement
identique** : même opérateurs (`TreeEnsembleClassifier`, `Cast`, `Mul`), même taille
(2,9 Mo), mêmes sorties. La quantification dynamique d'ONNX Runtime cible les
opérateurs matriciels des réseaux de neurones (`MatMul`, `Conv`, `Gather`). Un
ensemble d'arbres n'en contient aucun : ses paramètres sont des seuils de comparaison,
non des poids à multiplier. Le seul effet de « quantification » mesurable est le
passage float64 → float32 imposé par ONNX (§ 4.2), qui dégrade la précision sans gain
propre. Conclusion : **la quantification n'est pas un levier pertinent pour ce modèle.**

### 4.4 Configuration logicielle et matérielle : workers et cœurs

Une fois l'inférence ramenée à 0,5 ms, un processus uvicorn plafonne vers 270 appels/s :
le verrou global de l'interpréteur Python (GIL) sérialise la couche HTTP. Le levier
suivant est le nombre de processus (`WEB_CONCURRENCY`) :

| Workers uvicorn (backend booster) | p50, 1 client | p95, 16 clients | Débit, 16 clients | RSS par worker |
|---|---|---|---|---|
| 1 | 4,7 ms | 69,6 ms | 268 /s | 325 Mo |
| 2 | 4,8 ms | 51,0 ms | 479 /s | 319 Mo |
| 4 | 4,9 ms | 29,1 ms | 816 /s | 317 Mo |

Le débit croît presque linéairement avec le nombre de processus, tant qu'il reste des
cœurs libres. La latence d'un appel isolé ne change pas. Chaque worker coûte environ
320 Mo de mémoire. Les mesures CPU par worker de ces deux essais ne sont pas citées :
elles ont été prises avant la correction de la fenêtre de mesure CPU (§ 7).

### 4.5 GPU : écarté

Un GPU n'apporterait rien ici :

- **Taille du calcul.** Une prédiction parcourt 2 550 arbres pour **un seul client**,
  soit environ 0,3 ms sur un cœur CPU. Le transfert vers un GPU et le lancement de
  noyaux coûtent à eux seuls de l'ordre de la dizaine à la centaine de
  microsecondes, sans parallélisme à exploiter sur une ligne unique.
- **Support logiciel.** LightGBM ne propose l'accélération GPU qu'à l'entraînement,
  pas à l'inférence.
- **Coût.** Une instance GPU coûte un ordre de grandeur plus cher qu'un vCPU, pour un
  service qui n'utilise que 4 ms de CPU par prédiction.

## 5. Choix de la variante déployée

La consigne de départ était : retenir l'optimisation de code si elle est la meilleure
option, et documenter ONNX. Après mesure, **l'appel direct du Booster (1 thread) est
retenu**. Les raisons :

1. **Précision.** Le Booster rend des sorties **identiques au bit près** à celles du
   modèle du registry. ONNX change une décision sur 20 000. Pour un refus de crédit,
   c'est un client traité différemment sans raison métier.
2. **Gain marginal d'ONNX.** Il gagne 0,24 ms par appel, sur un temps de réponse de
   4,7 ms : **5 %**, en dessous de la variabilité réseau d'un vrai client.
3. **Simplicité.** Aucune dépendance ajoutée, aucune étape de conversion à maintenir
   ou à re-valider à chaque réentraînement : le modèle servi reste l'artefact du
   registry.
4. **Ressources.** Avec 1 thread OpenMP, la consommation CPU est la même qu'avec ONNX
   (175 %), et le débit est plus de 80 fois supérieur au besoin observé en production.

ONNX deviendrait pertinent si le temps de réponse visé descendait sous la
milliseconde, ou si le service tournait sur un runtime sans Python. Les scripts et
`requirements-optimisation.txt` permettent de rejouer cette évaluation.

## 6. Non-régression : précision et biais

L'échantillon compte 20 000 clients d'entraînement tirés au hasard, avec leur TARGET.
Il s'agit de données d'entraînement : l'AUC est donc optimiste en valeur absolue
(0,857 contre 0,79 en validation croisée), mais la comparaison entre variantes, elle,
est exacte.

| Variante | Écart de probabilité max | Décisions changées | AUC | Coût métier / client | Taux de refus |
|---|---|---|---|---|---|
| pyfunc (référence) | 0 | 0 / 20 000 | 0,856840 | 0,392200 | 30,420 % |
| **booster-1t (déployé)** | **0** | **0 / 20 000** | **0,856840** | **0,392200** | **30,420 %** |
| onnx-fp32 | 3,9·10⁻³ | 1 / 20 000 | 0,856841 | 0,392250 | 30,425 % |
| onnx-int8 | 3,9·10⁻³ | 1 / 20 000 | 0,856841 | 0,392250 | 30,425 % |

Taux de refus par sous-groupe, pour vérifier l'absence de biais introduit :

| Groupe | Clients | pyfunc | booster-1t | onnx-fp32 |
|---|---|---|---|---|
| CODE_GENDER = 0 | 6 836 | 40,535 % | 40,535 % (0 chg.) | 40,550 % (1 chg.) |
| CODE_GENDER = 1 | 13 164 | 25,167 % | 25,167 % (0 chg.) | 25,167 % (0 chg.) |
| âge < 30 ans | 2 911 | 48,368 % | 48,368 % (0 chg.) | 48,403 % (1 chg.) |
| âge 30-44 | 8 036 | 33,449 % | 33,449 % (0 chg.) | 33,449 % (0 chg.) |
| âge 45-59 | 6 768 | 24,409 % | 24,409 % (0 chg.) | 24,409 % (0 chg.) |
| âge 60+ | 2 285 | 14,705 % | 14,705 % (0 chg.) | 14,705 % (0 chg.) |

La version déployée ne modifie aucune décision, dans aucun sous-groupe. Les écarts de
taux de refus entre groupes, eux, existent déjà dans le modèle d'origine : ils
relèvent de l'analyse d'équité du modèle et non de cette optimisation.

Ces garanties sont automatisées dans `tests/test_optimisation.py`, exécuté par la CI
**avant** la construction de l'image :

- égalité exacte avec le pyfunc sur les clients d'exemple, sur les clients du train
  et sur un client à 792 valeurs manquantes ;
- égalité au bit près des probabilités en lot ;
- indépendance du résultat vis-à-vis du nombre de threads ;
- cohérence sous 8 threads concurrents.

## 7. Gain de bout en bout et validation en conditions réelles

### 7.1 Temps de réponse HTTP

Test de charge de l'API réelle (uvicorn, 1 worker, `scripts/load_test.py`) :

| Clients simultanés | | p50 | p95 | p99 | Débit |
|---|---|---|---|---|---|
| 1 | avant | 37,1 ms | 121,3 ms | 132,9 ms | 23 /s |
| | **après** | **4,7 ms** | **6,0 ms** | **6,9 ms** | **197 /s** |
| 4 | avant | 154,9 ms | 251,7 ms | 282,7 ms | 23 /s |
| | **après** | **13,8 ms** | **17,6 ms** | **42,0 ms** | **266 /s** |
| 16 | avant | 727,6 ms | 829,3 ms | 888,1 ms | 22 /s |
| | **après** | **55,4 ms** | **69,6 ms** | **186,7 ms** | **268 /s** |

Vue serveur, relevée par le monitoring de production pendant ces tests :

| | Latence du modèle p50 / p95 | CPU du processus p50 | Mémoire max |
|---|---|---|---|
| Avant (pyfunc) | 54,3 / 328,1 ms | 668 % pour 22 appels/s | 342 Mo |
| **Après (booster)** | **1,2 / 3,6 ms** | **114 % pour 268 appels/s** | **325 Mo** |

Le CPU consommé par prédiction passe d'environ 300 ms (6,7 cœurs pour 22 appels/s) à
environ 4 ms (1,1 cœur pour 268 appels/s).

### 7.2 Où passe le temps restant

Dans les 4,7 ms restantes, LightGBM ne pèse plus qu'environ 0,3 ms. Le reste se
répartit, sans goulot dominant, entre :

- la boucle asyncio et le pool de threads de Starlette ;
- le décodage JSON et la validation Pydantic de 795 champs ;
- deux écritures de journal par requête (événement HTTP et événement de prédiction).

Gagner davantage demanderait de changer le contrat de l'API (un vecteur au lieu d'un
dictionnaire de 795 clés) ou de journaliser moins de features
(`MONITORING_FEATURES`). Ces changements toucheraient les appelants et l'analyse de
dérive, pour un gain de l'ordre de la milliseconde.

### 7.3 Corrections du monitoring apportées en chemin

- **Mesure CPU.** Un premier essai mesurait le CPU entre deux appels successifs. À
  plusieurs centaines d'appels par seconde, ces intervalles (quelques ms) sont plus
  courts que la résolution de l'horloge CPU du système (10 à 16 ms), et la plupart
  des valeurs valaient 0 %. Le CPU est désormais moyenné sur des fenêtres d'au moins
  une seconde.
- **Objectif de service (SLO).** L'alerte « latence p95 » était fixée à 500 ms,
  dimensionnée sur l'ancienne latence. Elle passe à **100 ms**, environ 15 fois le
  nouveau p95 et en dessous de l'ancien (120 ms) : un retour accidentel à l'ancien
  chemin d'inférence déclencherait l'alerte.
- **Intégrité du journal** sous concurrence (§ 3.5).

### 7.4 Compatibilité avec l'environnement de production

- L'image Docker (Linux, Python 3.12) utilise exactement les bibliothèques de
  l'entraînement : LightGBM 4.7.0, MLflow 3.14.0. La seule dépendance ajoutée est
  `psutil`, pour le monitoring des ressources.
- La CI rejoue les tests de non-régression sous Ubuntu, y compris le test du journal
  multi-processus, qui ne peut pas s'exécuter sous Windows.
- Le **test de fumée** du pipeline mesure désormais le p95 de `/predict` sur 50 appels,
  sur le conteneur construit puis sur le conteneur déployé (1 vCPU, 1 Go). Au-delà de
  100 ms (`SMOKE_P95_MAX_MS`), le déploiement échoue. Contrôle en local : le chemin
  optimisé passe (p95 de 32 ms, avec une nouvelle connexion TCP par appel) ; l'ancien
  chemin pyfunc est rejeté (p95 de 111 ms). La latence mesurée sur le conteneur
  déployé est publiée dans le résumé du job `deploy`.

### 7.5 Mesure sur le conteneur déployé par le pipeline

La version optimisée a été déployée par le pipeline CI/CD le 15 octobre 2026 : tests
(y compris le test du journal multi-processus sous Linux), construction de l'image,
puis déploiement. Le test de fumée du conteneur déployé (Linux, 1 vCPU, 1 Go de RAM)
a mesuré :

```
Latence /predict (50 appels) -> p50 2.8 ms, p95 3.0 ms (max toléré 100 ms)
```

Ce temps de réponse est mesuré de bout en bout, vu d'un client qui ouvre une nouvelle
connexion à chaque appel. Il est inférieur à la mesure locale sous Windows (p50 de
4,7 ms avec connexion persistante), car l'ouverture d'une connexion TCP locale coûte
beaucoup moins cher sous Linux. Il est aussi **40 fois sous le seuil** de 100 ms. Avec
une seule vCPU, la limite de ressources de la production n'a donc pas dégradé le gain
mesuré en développement.

## 8. Configuration finale et justification

| Élément | Choix | Justification |
|---|---|---|
| Moteur d'inférence | `lightgbm.Booster.predict` sur numpy (`MODEL_BACKEND=booster`) | ×77 sur l'inférence, sorties identiques au bit près (§ 4.1, § 6) |
| Artefact | Modèle MLflow `@champion` exporté du registry (inchangé) | Une seule source de vérité : arbres, features et seuil viennent du registry |
| Threads LightGBM | 1 (`LGBM_NUM_THREADS=1`, `OMP_NUM_THREADS=1`) | Même latence que 4 ou 12 threads, 5 fois moins de CPU sous charge (§ 3.4) |
| ONNX Runtime | Non retenu (évalué) | +0,24 ms seulement, 1 décision changée sur 20 000, dépendance de plus (§ 5) |
| Quantification | Non applicable | Aucun opérateur quantifiable dans un ensemble d'arbres (§ 4.3) |
| Serveur | uvicorn, `WEB_CONCURRENCY=1` | 1 worker par vCPU ; la cible de déploiement est `--cpus=1` |
| Matériel | 1 vCPU, 1 Go de RAM, sans GPU | 268 appels/s pour 325 Mo de RSS. On passe à l'échelle en ajoutant des vCPU et des workers (~320 Mo chacun, § 4.4). GPU sans intérêt (§ 4.5) |
| Python | 3.12 | Imposé par la sérialisation cloudpickle du modèle |
| Monitoring | Journal JSON Lines + CPU/mémoire par fenêtre d'1 s, SLO p95 de 100 ms | Les décisions d'optimisation reposent sur ces mesures (§ 1) |

Pour une charge plus forte, augmenter `--cpus` et `WEB_CONCURRENCY` ensemble. Avec
plusieurs workers, l'interface Gradio (`/ui`) devra être servie derrière une
répartition de charge à affinité de session : ses sessions sont propres à chaque
processus.

## 9. Reproduire

```bash
pip install -r requirements-optimisation.txt      # dev + onnx, onnxruntime, onnxmltools

python scripts/profile_inference.py --backend pyfunc
python scripts/profile_inference.py --backend booster
python scripts/benchmark_inference.py --clients 20000

# test de charge HTTP, une configuration à la fois (port et dossier de journaux dédiés)
MODEL_BACKEND=pyfunc  MONITORING_DIR=logs/charge-pyfunc  uvicorn api.main:app --port 8001 --env-file .env
python scripts/load_test.py --url http://127.0.0.1:8001 --label pyfunc --n 300 --journaux logs/charge-pyfunc
MODEL_BACKEND=booster MONITORING_DIR=logs/charge-booster uvicorn api.main:app --port 8001 --env-file .env
python scripts/load_test.py --url http://127.0.0.1:8001 --label booster --n 1000 --journaux logs/charge-booster
```
