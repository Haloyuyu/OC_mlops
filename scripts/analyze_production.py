"""Analyse automatique des données de production : dérive et anomalies d'exploitation.

Enchaîne les quatre étapes du suivi, sans intervention :

    journaux JSON Lines  ->  base SQLite  ->  analyse  ->  rapport HTML + résumé JSON

    python scripts/analyze_production.py
    python scripts/analyze_production.py --depuis 2026-10-01 --evidently

Le code de sortie vaut 1 en cas d'ALERTE : le script peut donc servir de sonde dans
une tâche planifiée ou un pipeline, sans qu'on ait à lire le rapport pour savoir
s'il faut regarder.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from monitoring import drift, operations, report, storage  # noqa: E402

REFERENCE = RACINE / "model" / "reference_drift.csv"
RAPPORTS = RACINE / "rapports"
HORS_FEATURES = {"probabilite_defaut", "decision", "TARGET"}


def charger_reference(chemin=REFERENCE):
    if not Path(chemin).exists():
        raise SystemExit(f"Référence absente : {chemin}\n"
                         "Lancez `python scripts/export_model.py` (serveur MLFlow démarré).")
    return pd.read_csv(chemin)


def rapport_evidently(reference, production, features, destination):
    """Rapport Evidently en complément, si la bibliothèque est installée.

    Optionnel à dessein : Evidently et ses dépendances pèsent ~800 Mo, hors de
    proportion avec l'image de l'API. L'analyse de ce script n'en dépend pas.
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    definition = DataDefinition(numerical_columns=list(features))
    jeux = [Dataset.from_pandas(df[list(features)], data_definition=definition)
            for df in (production, reference)]
    execution = Report([DataDriftPreset()]).run(*jeux)
    execution.save_html(str(destination))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", default=storage.JOURNAUX_DEFAUT, help="dossier des journaux")
    parser.add_argument("--base", default=storage.BASE_DEFAUT, help="base SQLite")
    parser.add_argument("--reference", default=REFERENCE, help="référence de dérive (CSV)")
    parser.add_argument("--depuis", default=None, help="n'analyser qu'à partir de cet horodatage ISO")
    parser.add_argument("--jusqu-a", default=None, dest="jusqu_a",
                        help="borne haute de la fenêtre (ISO) : permet de comparer deux périodes")
    parser.add_argument("--purger", action="store_true",
                        help="supprimer les journaux au-delà de la durée de rétention (RGPD)")
    parser.add_argument("--rapport", default=RAPPORTS / "monitoring.html")
    parser.add_argument("--json", default=RAPPORTS / "monitoring.json")
    parser.add_argument("--evidently", action="store_true",
                        help="produire en plus le rapport Evidently (si installé)")
    parser.add_argument("--sans-ingestion", action="store_true",
                        help="analyser la base telle quelle, sans relire les journaux")
    arguments = parser.parse_args()

    if not arguments.sans_ingestion:
        recap = storage.ingerer(arguments.logs, arguments.base)
        print(f"Ingestion : {recap['evenements']} événements depuis {recap['fichiers']} journal(ux)"
              + (f", {recap['illisibles']} ligne(s) illisible(s)" if recap["illisibles"] else "")
              + f" — {recap['total_en_base']} en base")

    reference = charger_reference(arguments.reference)
    features = [c for c in reference.columns if c not in HORS_FEATURES]

    if arguments.purger:
        from api.monitoring import ProductionLog
        supprimes = ProductionLog(arguments.logs).purge_anciens_journaux()
        print(f"Purge RGPD : {len(supprimes)} journal(ux) au-delà de la rétention supprimé(s)")

    requetes = storage.charger_requetes(arguments.base, arguments.depuis, arguments.jusqu_a)
    predictions = storage.charger_predictions(arguments.base, arguments.depuis, features=features,
                                              jusqu_a=arguments.jusqu_a)
    if predictions.empty:
        raise SystemExit("Aucune prédiction journalisée : lancez du trafic "
                         "(scripts/simulate_traffic.py) avant d'analyser.")

    reussies = predictions[predictions["statut"] == 200]
    resultats = drift.analyser_features(reference, reussies, features)
    seuil = float(reussies["seuil"].dropna().iloc[0]) if reussies["seuil"].notna().any() else 0.08
    derive_predictions = drift.analyser_predictions(
        reference["probabilite_defaut"], reussies["probabilite_defaut"], seuil)
    synthese = drift.synthese(resultats, derive_predictions)
    exploitation = operations.analyser(requetes)
    exploitation["erreurs_frequentes"] = operations.erreurs_frequentes(predictions)
    exploitation["ressources"] = operations.ressources(predictions)
    volumetrie = storage.volumetrie(arguments.base)

    Path(arguments.rapport).parent.mkdir(parents=True, exist_ok=True)
    html = report.construire_rapport(synthese, resultats, derive_predictions, exploitation,
                                     volumetrie, reference, reussies)
    Path(arguments.rapport).write_text(html, encoding="utf-8")
    report.ecrire_json(arguments.json, synthese, resultats, derive_predictions,
                       exploitation, volumetrie)

    print(f"\nDérive       : {synthese['niveau']} — {synthese['features_en_derive']} feature(s) "
          f"en dérive sur {synthese['features_evaluables']} évaluables")
    for motif in synthese["motifs"]:
        print(f"  · {motif}")
    print(f"Prédictions  : taux de refus {derive_predictions['taux_refus_production']:.1%} "
          f"contre {derive_predictions['taux_refus_reference']:.1%} en référence "
          f"({derive_predictions['ecart_taux_refus']:+.1%}), PSI {derive_predictions['psi']:.3f}")
    print(f"Exploitation : {exploitation['niveau']} — {exploitation['appels']} appels, "
          f"4xx {exploitation['taux_erreur_client']:.1%}, 5xx "
          f"{exploitation['taux_erreur_serveur']:.1%}, p95 "
          f"{exploitation['latences_ms']['p95']:.0f} ms")
    for anomalie in exploitation["anomalies"]:
        print(f"  · {anomalie}")
    ressources = exploitation["ressources"]
    if ressources["mesures"]:
        print(f"Ressources   : CPU p50 {ressources['cpu_pct']['p50']:.0f} %, "
              f"p95 {ressources['cpu_pct']['p95']:.0f} % (100 % = un cœur), "
              f"mémoire max {ressources['memoire_mo_max']:.0f} Mo")
    print(f"Stockage     : {volumetrie['evenements']} événements, "
          f"{volumetrie['taille_base_mo']} Mo "
          f"({volumetrie['octets_par_prediction']} octets par prédiction)")
    print(f"Rapport      : {arguments.rapport}")

    if arguments.evidently:
        chemin = rapport_evidently(reference, reussies, features,
                                   Path(arguments.rapport).with_name("evidently.html"))
        print(f"Evidently    : {chemin}" if chemin
              else "Evidently    : non installé (pip install -r requirements-monitoring.txt)")

    return 1 if "ALERTE" in (synthese["niveau"], exploitation["niveau"]) else 0


if __name__ == "__main__":
    sys.exit(main())
