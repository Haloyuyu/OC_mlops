"""Banc d'essai des stratégies d'optimisation de l'inférence : vitesse ET précision.

Compare, sur le même modèle champion :

    pyfunc          modèle MLflow tel qu'au registry (version déployée avant optimisation)
    booster-Nt      appel direct du Booster LightGBM sur numpy, N threads OpenMP
    onnx-fp32       conversion ONNX (onnxmltools), exécution ONNX Runtime
    onnx-int8       la même, après quantification dynamique (onnxruntime.quantization)

Pour chaque variante :

1. latence d'un appel isolé (p50/p95/p99), préparation de l'entrée comprise ;
2. latence et débit sous charge concurrente (pool de threads, comme FastAPI), avec
   l'occupation CPU du processus ;
3. non-régression sur un échantillon de clients d'entraînement : écart de
   probabilité, décisions changées au seuil métier, AUC, coût métier, et taux de
   refus par sous-groupe (genre, âge) pour contrôler l'absence de biais introduit.

    pip install -r requirements-optimisation.txt
    python scripts/benchmark_inference.py --clients 20000

Résultats : docs/optimisation/benchmark_inference.{json,md}.
"""

import argparse
import gc
import json
import platform
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import roc_auc_score

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from api.scoring import ScoringModel  # noqa: E402

DONNEES = RACINE / "data" / "processed_data.csv"
CACHE = RACINE / "data" / "clients_benchmark.csv"
ONNX_DIR = RACINE / "data" / "optimisation"
SORTIE = RACINE / "docs" / "optimisation"

COUT_FN, COUT_FP = 10, 1


# --------------------------------------------------------------------------- données

def charger_clients(n, graine=42):
    """Échantillon aléatoire de clients d'entraînement (TARGET connue), mis en cache."""
    if CACHE.exists():
        clients = pd.read_csv(CACHE)
        if len(clients) >= n:
            return clients.head(n)
    morceaux = []
    for morceau in pd.read_csv(DONNEES, chunksize=50_000):
        morceau.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in morceau.columns]
        train = morceau.loc[morceau["TARGET"].notna()]
        if not train.empty:
            morceaux.append(train.sample(frac=min(1.0, 1.5 * n / 307_511), random_state=graine))
    clients = pd.concat(morceaux).sample(frac=1.0, random_state=graine).head(n)
    clients.to_csv(CACHE, index=False)
    print(f"{len(clients)} clients d'entraînement mis en cache dans {CACHE.relative_to(RACINE)}")
    return clients


def en_dict(ligne):
    return {nom: (None if pd.isna(valeur) else float(valeur)) for nom, valeur in ligne.items()}


# --------------------------------------------------------------------------- variantes

class Variante:
    """Une stratégie d'inférence : un appel unitaire (dict) et un appel en lot (matrice)."""

    nom = ""
    description = ""

    def proba_client(self, features):
        raise NotImplementedError

    def proba_lot(self, matrice):
        raise NotImplementedError


class Pyfunc(Variante):
    nom = "pyfunc"
    description = "MLflow pyfunc (avant optimisation)"

    def __init__(self, modele):
        self.modele = modele          # ScoringModel(backend="pyfunc")

    def proba_client(self, features):
        return self.modele.predict(features)["probabilite_defaut"]

    def proba_lot(self, matrice):
        cadre = pd.DataFrame(matrice, columns=self.modele.features)
        return self.modele.pyfunc.predict(cadre)["probabilite_defaut"].to_numpy()


class Booster(Variante):
    def __init__(self, modele, threads):
        self.modele = ScoringModel(backend="booster", num_threads=threads)
        self.nom = f"booster-{threads}t" if threads else "booster-tous"
        self.description = (f"Booster LightGBM direct, {threads} thread(s)" if threads
                            else "Booster LightGBM direct, tous les cœurs")

    def proba_client(self, features):
        return self.modele.predict(features)["probabilite_defaut"]

    def proba_lot(self, matrice):
        return self.modele.predict_proba(matrice)


class Onnx(Variante):
    def __init__(self, modele, chemin, nom, description):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(chemin), options,
                                            providers=["CPUExecutionProvider"])
        self.entree = self.session.get_inputs()[0].name
        self.modele = modele          # pour la vectorisation et les contrôles d'entrée
        self.nom, self.description = nom, description
        self.taille_mo = Path(chemin).stat().st_size / 2**20

    def _proba(self, matrice):
        sorties = self.session.run(None, {self.entree: matrice.astype(np.float32)})
        probabilites = sorties[1]     # [label, probabilities] avec zipmap=False
        return np.asarray(probabilites)[:, 1]

    def proba_client(self, features):
        return float(self._proba(self.modele.vectorize(features))[0])

    def proba_lot(self, matrice):
        return np.concatenate([self._proba(matrice[i:i + 5000])
                               for i in range(0, len(matrice), 5000)])


def construire_onnx(modele):
    """Convertit le LGBMClassifier en ONNX, puis en version quantifiée int8."""
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32, int8 = ONNX_DIR / "credit_scoring_lgbm.onnx", ONNX_DIR / "credit_scoring_lgbm_int8.onnx"
    classifieur = modele.pyfunc.unwrap_python_model().model
    debut = time.perf_counter()
    onx = onnxmltools.convert_lightgbm(
        classifieur, initial_types=[("input", FloatTensorType([None, len(modele.features)]))],
        zipmap=False, target_opset=15)
    fp32.write_bytes(onx.SerializeToString())
    duree = time.perf_counter() - debut
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"Conversion ONNX : {duree:.1f} s → {fp32.relative_to(RACINE)} "
          f"({fp32.stat().st_size / 2**20:.1f} Mo), int8 {int8.stat().st_size / 2**20:.1f} Mo")
    return fp32, int8


# --------------------------------------------------------------------------- mesures

def percentiles(valeurs):
    v = np.asarray(valeurs)
    return {"moyenne": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))}


def latence_unitaire(variante, clients, n):
    for client in clients[:20]:
        variante.proba_client(client)
    mesures = []
    for i in range(n):
        debut = time.perf_counter()
        variante.proba_client(clients[i % len(clients)])
        mesures.append((time.perf_counter() - debut) * 1000)
    return percentiles(mesures)


def sous_charge(variante, clients, n, workers):
    """n appels répartis sur `workers` threads : latence vue par appel, débit, CPU."""
    def appel(i):
        debut = time.perf_counter()
        variante.proba_client(clients[i % len(clients)])
        return (time.perf_counter() - debut) * 1000

    processus = psutil.Process()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(appel, range(workers * 2)))           # échauffement
        processus.cpu_percent()
        debut = time.perf_counter()
        latences = list(pool.map(appel, range(n)))
        duree = time.perf_counter() - debut
        cpu = processus.cpu_percent()
    return {**percentiles(latences), "debit_par_s": n / duree, "cpu_pct": cpu,
            "workers": workers}


def decomposition_pyfunc(modele, clients, n=200):
    """Où passe le temps d'un appel pyfunc, mesuré sans profileur (médianes, ms par appel).

    Chaque étape est chronométrée dans sa propre boucle, sur un DataFrame neuf à
    chaque appel comme en production (réutiliser le même objet profiterait du cache
    de colonnes de pandas et diviserait par deux le coût de la validation MLflow).
    Le coût d'une étape s'obtient par différence avec l'étape qu'elle englobe. Le
    ramasse-miettes est suspendu pour isoler le coût déterministe ; son effet (la
    traîne de latence) est mesuré à part.
    """
    classifieur = modele.pyfunc.unwrap_python_model().model

    def cadre(i):
        return pd.DataFrame([clients[i]]).reindex(columns=modele.features).astype("float64")

    def mediane(fonction):
        for i in range(20):
            fonction(i)
        mesures = []
        for i in range(n):
            debut = time.perf_counter()
            fonction(i % 20)
            mesures.append((time.perf_counter() - debut) * 1000)
        return float(np.median(mesures))

    vecteurs = [modele.vectorize(c) for c in clients[:20]]
    gc.collect()
    gc.disable()
    try:
        total = mediane(lambda i: modele.predict(clients[i]))
        t_cadre = mediane(cadre)
        t_pyfunc = mediane(lambda i: modele.pyfunc.predict(cadre(i)))
        t_sklearn = mediane(lambda i: classifieur.predict_proba(cadre(i)))
        t_4t = mediane(lambda i: modele.booster.predict(vecteurs[i], num_threads=4))
        t_1t = mediane(lambda i: modele.booster.predict(vecteurs[i], num_threads=1))
    finally:
        gc.enable()
    return {
        "appel_complet_hors_ramasse_miettes_ms": total,
        "construction_dataframe_pandas_ms": t_cadre,
        "validation_schema_mlflow_ms": t_pyfunc - t_sklearn,
        "surcout_sklearn_et_conversion_pandas_ms": t_sklearn - t_cadre - t_4t,
        "arbres_lightgbm_4_threads_ms": t_4t,
        "reste_controles_et_mise_en_forme_ms": total - t_pyfunc,
        "arbres_lightgbm_1_thread_ms": t_1t,
    }


def collectes_gen2(variante, clients, n):
    """Nombre de collectes complètes (génération 2) du ramasse-miettes pendant n appels."""
    compteur = [0]

    def rappel(phase, info):
        if phase == "start" and info["generation"] == 2:
            compteur[0] += 1

    gc.callbacks.append(rappel)
    try:
        for i in range(n):
            variante.proba_client(clients[i % len(clients)])
    finally:
        gc.callbacks.remove(rappel)
    return compteur[0]


def cout_metier(y, decision):
    fn = int(((y == 1) & (decision == 0)).sum())
    fp = int(((y == 0) & (decision == 1)).sum())
    return (COUT_FN * fn + COUT_FP * fp) / len(y)


def precision(reference, proba, y, seuil, groupes):
    """Non-régression d'une variante face aux probabilités de référence (pyfunc)."""
    decision_ref, decision = reference >= seuil, proba >= seuil
    ecart = np.abs(proba - reference)
    par_groupe = {}
    for nom_groupe, valeurs in groupes.items():
        for modalite in sorted(valeurs.dropna().unique()):
            masque = (valeurs == modalite).to_numpy()
            par_groupe[f"{nom_groupe}={modalite}"] = {
                "clients": int(masque.sum()),
                "taux_refus_reference": float(decision_ref[masque].mean()),
                "taux_refus": float(decision[masque].mean()),
                "decisions_changees": int((decision_ref[masque] != decision[masque]).sum()),
            }
    return {
        "ecart_proba_max": float(ecart.max()),
        "ecart_proba_moyen": float(ecart.mean()),
        "decisions_changees": int((decision_ref != decision).sum()),
        "accord_decisions": float((decision_ref == decision).mean()),
        "auc": float(roc_auc_score(y, proba)),
        "cout_metier_par_client": cout_metier(y, decision.astype(int)),
        "taux_refus": float(decision.mean()),
        "par_groupe": par_groupe,
    }


# --------------------------------------------------------------------------- rapport

def tableau(entetes, lignes):
    sortie = ["| " + " | ".join(entetes) + " |", "|" + "---|" * len(entetes)]
    sortie += ["| " + " | ".join(str(c) for c in ligne) + " |" for ligne in lignes]
    return "\n".join(sortie)


def rapport_markdown(resultats):
    env = resultats["environnement"]
    md = [f"# Banc d'essai de l'inférence — {resultats['date']}", "",
          f"Machine : {env['processeur']}, {env['coeurs_logiques']} cœurs logiques, "
          f"{env['systeme']}, Python {env['python']}. "
          f"Échantillon : {resultats['clients']} clients d'entraînement.", "",
          "## Décomposition d'un appel pyfunc (sans profileur)", "",
          tableau(["Étape", "ms par appel"],
                  [[k, f"{v:.2f}"] for k, v in resultats["decomposition_pyfunc"].items()]),
          "", "## Latence d'un appel isolé (ms)", ""]
    reference = resultats["variantes"]["pyfunc"]["unitaire"]["p50"]
    md.append(tableau(
        ["Variante", "Description", "p50", "p95", "p99", "Gain p50",
         "Collectes gen2 / 1000 appels"],
        [[nom, v["description"], f"{v['unitaire']['p50']:.3f}", f"{v['unitaire']['p95']:.3f}",
          f"{v['unitaire']['p99']:.3f}", f"×{reference / v['unitaire']['p50']:.0f}",
          v["collectes_gen2_pour_1000_appels"]]
         for nom, v in resultats["variantes"].items()]))
    md += ["", f"## Sous charge ({resultats['workers']} threads concurrents)", ""]
    md.append(tableau(
        ["Variante", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Débit (appels/s)", "CPU processus"],
        [[nom, f"{v['charge']['p50']:.2f}", f"{v['charge']['p95']:.2f}",
          f"{v['charge']['p99']:.2f}", f"{v['charge']['debit_par_s']:.0f}",
          f"{v['charge']['cpu_pct']:.0f} %"]
         for nom, v in resultats["variantes"].items()]))
    md += ["", "## Non-régression (référence : pyfunc)", ""]
    md.append(tableau(
        ["Variante", "Écart proba max", "Décisions changées", "AUC", "Coût métier / client",
         "Taux de refus"],
        [[nom, f"{v['precision']['ecart_proba_max']:.2e}",
          f"{v['precision']['decisions_changees']} / {resultats['clients']}",
          f"{v['precision']['auc']:.6f}", f"{v['precision']['cout_metier_par_client']:.6f}",
          f"{v['precision']['taux_refus']:.4%}"]
         for nom, v in resultats["variantes"].items()]))
    md += ["", "## Taux de refus par sous-groupe", ""]
    groupes = list(resultats["variantes"]["pyfunc"]["precision"]["par_groupe"])
    md.append(tableau(
        ["Groupe", "Clients"] + list(resultats["variantes"]),
        [[g, resultats["variantes"]["pyfunc"]["precision"]["par_groupe"][g]["clients"]]
         + [f"{v['precision']['par_groupe'][g]['taux_refus']:.3%}"
            f" ({v['precision']['par_groupe'][g]['decisions_changees']} chg.)"
            for v in resultats["variantes"].values()]
         for g in groupes]))
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", type=int, default=20_000,
                        help="taille de l'échantillon de non-régression")
    parser.add_argument("--appels", type=int, default=1000, help="appels unitaires chronométrés")
    parser.add_argument("--workers", type=int, default=8, help="threads de la mesure sous charge")
    parser.add_argument("--sans-onnx", action="store_true",
                        help="ne pas évaluer ONNX (onnxruntime non installé)")
    arguments = parser.parse_args()

    clients_df = charger_clients(arguments.clients)
    reference_modele = ScoringModel(backend="pyfunc")
    features = reference_modele.features
    y = clients_df["TARGET"].astype(int).to_numpy()
    matrice = clients_df.reindex(columns=features).astype("float64").to_numpy()
    clients = [en_dict(ligne) for _, ligne in clients_df[features].head(500).iterrows()]
    age = (-clients_df["DAYS_BIRTH"] / 365.25)
    groupes = {
        "CODE_GENDER": clients_df["CODE_GENDER"],
        "age": pd.cut(age, [0, 30, 45, 60, 120], labels=["<30", "30-44", "45-59", "60+"])
               .astype(str).where(age.notna()),
    }

    variantes = [Pyfunc(reference_modele)] + [Booster(reference_modele, t)
                                              for t in (4, 1, 0)]
    if not arguments.sans_onnx:
        fp32, int8 = construire_onnx(reference_modele)
        variantes += [
            Onnx(reference_modele, fp32, "onnx-fp32", "ONNX Runtime, float32, 1 thread"),
            Onnx(reference_modele, int8, "onnx-int8", "ONNX Runtime, quantification dynamique int8"),
        ]

    seuil = reference_modele._seuil
    proba_reference = Pyfunc(reference_modele).proba_lot(matrice)
    resultats = {
        "date": time.strftime("%Y-%m-%d"),
        "clients": len(clients_df),
        "workers": arguments.workers,
        "seuil": seuil,
        "environnement": {
            "processeur": platform.processor() or platform.machine(),
            "coeurs_logiques": psutil.cpu_count(),
            "systeme": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "decomposition_pyfunc": decomposition_pyfunc(reference_modele, clients),
        "variantes": {},
    }
    for variante in variantes:
        lent = isinstance(variante, Pyfunc)
        print(f"· {variante.nom} …", flush=True)
        resultats["variantes"][variante.nom] = {
            "description": variante.description,
            "unitaire": latence_unitaire(variante, clients,
                                         arguments.appels // 5 if lent else arguments.appels),
            "collectes_gen2_pour_1000_appels": round(
                collectes_gen2(variante, clients, 200) * 5) if lent
            else collectes_gen2(variante, clients, 1000),
            "charge": sous_charge(variante, clients, 400 if lent else 4000, arguments.workers),
            "precision": precision(proba_reference, variante.proba_lot(matrice), y, seuil, groupes),
        }

    SORTIE.mkdir(parents=True, exist_ok=True)
    (SORTIE / "benchmark_inference.json").write_text(
        json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = rapport_markdown(resultats)
    (SORTIE / "benchmark_inference.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
