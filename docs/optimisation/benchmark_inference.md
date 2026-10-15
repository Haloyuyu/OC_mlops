# Banc d'essai de l'inférence — 2026-10-15

Machine : Intel64 Family 6 Model 167 Stepping 1, GenuineIntel, 12 cœurs logiques, Windows 11, Python 3.12.7. Échantillon : 20000 clients d'entraînement.

## Décomposition d'un appel pyfunc (sans profileur)

| Étape | ms par appel |
|---|---|
| appel_complet_hors_ramasse_miettes_ms | 35.28 |
| construction_dataframe_pandas_ms | 6.40 |
| validation_schema_mlflow_ms | 21.90 |
| surcout_sklearn_et_conversion_pandas_ms | 6.21 |
| arbres_lightgbm_4_threads_ms | 0.30 |
| reste_controles_et_mise_en_forme_ms | 0.46 |
| arbres_lightgbm_1_thread_ms | 0.26 |

## Latence d'un appel isolé (ms)

| Variante | Description | p50 | p95 | p99 | Gain p50 | Collectes gen2 / 1000 appels |
|---|---|---|---|---|---|---|
| pyfunc | MLflow pyfunc (avant optimisation) | 36.413 | 113.142 | 118.304 | ×1 | 65 |
| booster-4t | Booster LightGBM direct, 4 thread(s) | 0.581 | 0.831 | 0.975 | ×63 | 0 |
| booster-1t | Booster LightGBM direct, 1 thread(s) | 0.473 | 0.835 | 1.067 | ×77 | 0 |
| booster-tous | Booster LightGBM direct, tous les cœurs | 0.604 | 0.776 | 0.960 | ×60 | 0 |
| onnx-fp32 | ONNX Runtime, float32, 1 thread | 0.237 | 0.383 | 0.520 | ×154 | 0 |
| onnx-int8 | ONNX Runtime, quantification dynamique int8 | 0.230 | 0.360 | 0.504 | ×158 | 0 |

## Sous charge (8 threads concurrents)

| Variante | p50 (ms) | p95 (ms) | p99 (ms) | Débit (appels/s) | CPU processus |
|---|---|---|---|---|---|
| pyfunc | 323.57 | 602.43 | 760.24 | 24 | 749 % |
| booster-4t | 3.09 | 8.68 | 12.15 | 2088 | 856 % |
| booster-1t | 2.90 | 7.06 | 9.41 | 2278 | 175 % |
| booster-tous | 2.80 | 8.03 | 11.27 | 2241 | 922 % |
| onnx-fp32 | 1.11 | 5.56 | 15.97 | 3900 | 175 % |
| onnx-int8 | 0.98 | 6.88 | 16.12 | 4182 | 174 % |

## Non-régression (référence : pyfunc)

| Variante | Écart proba max | Décisions changées | AUC | Coût métier / client | Taux de refus |
|---|---|---|---|---|---|
| pyfunc | 0.00e+00 | 0 / 20000 | 0.856840 | 0.392200 | 30.4200% |
| booster-4t | 0.00e+00 | 0 / 20000 | 0.856840 | 0.392200 | 30.4200% |
| booster-1t | 0.00e+00 | 0 / 20000 | 0.856840 | 0.392200 | 30.4200% |
| booster-tous | 0.00e+00 | 0 / 20000 | 0.856840 | 0.392200 | 30.4200% |
| onnx-fp32 | 3.94e-03 | 1 / 20000 | 0.856841 | 0.392250 | 30.4250% |
| onnx-int8 | 3.94e-03 | 1 / 20000 | 0.856841 | 0.392250 | 30.4250% |

## Taux de refus par sous-groupe

| Groupe | Clients | pyfunc | booster-4t | booster-1t | booster-tous | onnx-fp32 | onnx-int8 |
|---|---|---|---|---|---|---|---|
| CODE_GENDER=0 | 6836 | 40.535% (0 chg.) | 40.535% (0 chg.) | 40.535% (0 chg.) | 40.535% (0 chg.) | 40.550% (1 chg.) | 40.550% (1 chg.) |
| CODE_GENDER=1 | 13164 | 25.167% (0 chg.) | 25.167% (0 chg.) | 25.167% (0 chg.) | 25.167% (0 chg.) | 25.167% (0 chg.) | 25.167% (0 chg.) |
| age=30-44 | 8036 | 33.449% (0 chg.) | 33.449% (0 chg.) | 33.449% (0 chg.) | 33.449% (0 chg.) | 33.449% (0 chg.) | 33.449% (0 chg.) |
| age=45-59 | 6768 | 24.409% (0 chg.) | 24.409% (0 chg.) | 24.409% (0 chg.) | 24.409% (0 chg.) | 24.409% (0 chg.) | 24.409% (0 chg.) |
| age=60+ | 2285 | 14.705% (0 chg.) | 14.705% (0 chg.) | 14.705% (0 chg.) | 14.705% (0 chg.) | 14.705% (0 chg.) | 14.705% (0 chg.) |
| age=<30 | 2911 | 48.368% (0 chg.) | 48.368% (0 chg.) | 48.368% (0 chg.) | 48.368% (0 chg.) | 48.403% (1 chg.) | 48.403% (1 chg.) |
