"""Rapport HTML autonome de suivi de production (dérive + exploitation).

Un seul fichier, sans dépendance réseau ni bibliothèque de graphiques : les figures
sont du SVG généré ici, avec infobulles natives au survol. Le rapport s'ouvre dans un
navigateur, se transmet par mail et s'archive tel quel.

Palette : les deux premiers créneaux de la palette de référence du projet (bleu =
référence d'entraînement, orange = production), utilisés sans modification — leur
séparation est validée pour les daltonismes sur paires adjacentes. Les couleurs de
statut (vert/jaune/rouge) ne portent jamais seules le sens : elles vont toujours avec
une icône et un libellé.
"""

import html
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

COULEURS = {
    "reference": "#2a78d6",
    "production": "#eb6834",
    "stable": "#0ca30c",
    "vigilance": "#fab219",
    "alerte": "#d03b3b",
}
ICONES = {"STABLE": "●", "VIGILANCE": "▲", "ALERTE": "■"}
N_BINS_FIGURE = 12
N_FEATURES_FIGURES = 3

STYLE = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axe: #c3c2b7; --bord: rgba(11,11,11,0.10);
  --reference: #2a78d6; --production: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axe: #383835; --bord: rgba(255,255,255,0.10);
    --reference: #3987e5; --production: #d95926;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--plane); color: var(--ink);
       font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 32px 0 12px; }
h3 { font-size: .95rem; margin: 20px 0 8px; color: var(--ink2); font-weight: 600; }
p, li { color: var(--ink2); }
.sous-titre { color: var(--muted); margin: 0 0 20px; }
.verdict { display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px;
           border-radius: 8px; font-weight: 650; color: #fff; margin: 4px 0 16px; }
.tuiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tuile { background: var(--surface); border: 1px solid var(--bord); border-radius: 10px;
         padding: 12px 14px; }
.tuile .valeur { font-size: 1.5rem; font-weight: 650; display: block; }
.tuile .libelle { color: var(--muted); font-size: .8rem; }
.tuile .detail { color: var(--ink2); font-size: .8rem; }
figure { background: var(--surface); border: 1px solid var(--bord); border-radius: 10px;
         padding: 14px; margin: 12px 0; }
figcaption { color: var(--muted); font-size: .8rem; margin-top: 6px; }
svg { width: 100%; height: auto; display: block; }
.legende { display: flex; gap: 16px; flex-wrap: wrap; font-size: .82rem; color: var(--ink2);
           margin-bottom: 8px; }
.legende span { display: inline-flex; align-items: center; gap: 6px; }
.pastille { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.tableau { overflow-x: auto; background: var(--surface); border: 1px solid var(--bord);
           border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--grid);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--muted); font-weight: 600; }
tr:last-child td { border-bottom: none; }
.etiquette { display: inline-flex; align-items: center; gap: 5px; }
ul { padding-left: 20px; }
footer { margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--grid);
         color: var(--muted); font-size: .8rem; }
code { background: var(--surface); border: 1px solid var(--bord); border-radius: 4px;
       padding: 1px 5px; font-size: .85em; }
"""


def _couleur_niveau(niveau):
    return COULEURS.get(niveau.lower(), COULEURS["vigilance"])


def _e(valeur):
    return html.escape(str(valeur))


def _format(valeur, gabarit="{:.3f}", defaut="—"):
    if valeur is None or (isinstance(valeur, float) and not np.isfinite(valeur)):
        return defaut
    return gabarit.format(valeur)


def _histogramme(reference, production, titre, unite=""):
    """Histogramme comparé : part des effectifs par classe, deux séries côte à côte."""
    ref = pd.Series(reference).dropna().to_numpy(dtype="float64")
    prod = pd.Series(production).dropna().to_numpy(dtype="float64")
    if len(ref) == 0 or len(prod) == 0:
        return ""
    bornes = np.histogram_bin_edges(np.concatenate([ref, prod]), bins=N_BINS_FIGURE)
    part_ref = np.histogram(ref, bins=bornes)[0] / len(ref) * 100
    part_prod = np.histogram(prod, bins=bornes)[0] / len(prod) * 100

    largeur, hauteur = 640, 210
    marge = {"haut": 12, "bas": 34, "gauche": 42, "droite": 8}
    aire_l = largeur - marge["gauche"] - marge["droite"]
    aire_h = hauteur - marge["haut"] - marge["bas"]
    maximum = max(part_ref.max(), part_prod.max(), 1e-9)
    pas = aire_l / len(part_ref)
    barre = max((pas - 6) / 2, 1.5)      # 2 px de respiration entre les deux séries

    elements = []
    for i, graduation in enumerate(np.linspace(0, maximum, 4)):
        y = marge["haut"] + aire_h - (graduation / maximum) * aire_h
        elements.append(f'<line x1="{marge["gauche"]}" y1="{y:.1f}" x2="{largeur - marge["droite"]}" '
                        f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        elements.append(f'<text x="{marge["gauche"] - 6}" y="{y + 3.5:.1f}" text-anchor="end" '
                        f'font-size="10" fill="var(--muted)">{graduation:.0f}%</text>')

    for i, (valeur_ref, valeur_prod) in enumerate(zip(part_ref, part_prod)):
        x = marge["gauche"] + i * pas + 2
        for valeur, couleur, nom, decalage in (
            (valeur_ref, "var(--reference)", "référence", 0),
            (valeur_prod, "var(--production)", "production", barre + 2),
        ):
            h = (valeur / maximum) * aire_h
            y = marge["haut"] + aire_h - h
            info = (f"{nom} : {valeur:.1f} % des appels entre "
                    f"{bornes[i]:.4g} et {bornes[i + 1]:.4g}{unite}")
            elements.append(
                f'<rect x="{x + decalage:.1f}" y="{y:.1f}" width="{barre:.1f}" '
                f'height="{max(h, 0.6):.1f}" rx="2" fill="{couleur}">'
                f'<title>{_e(info)}</title></rect>')

    for i in (0, len(part_ref) // 2, len(part_ref)):
        x = marge["gauche"] + i * pas
        elements.append(f'<text x="{x:.1f}" y="{hauteur - 14}" text-anchor="middle" '
                        f'font-size="10" fill="var(--muted)">{bornes[i]:.4g}</text>')
    elements.append(f'<line x1="{marge["gauche"]}" y1="{marge["haut"] + aire_h}" '
                    f'x2="{largeur - marge["droite"]}" y2="{marge["haut"] + aire_h}" '
                    f'stroke="var(--axe)" stroke-width="1"/>')
    elements.append(f'<text x="{largeur / 2}" y="{hauteur - 1}" text-anchor="middle" '
                    f'font-size="10" fill="var(--muted)">{_e(titre)}</text>')
    return (f'<svg viewBox="0 0 {largeur} {hauteur}" role="img" '
            f'aria-label="Distribution comparée : {_e(titre)}">' + "".join(elements) + "</svg>")


def _courbe_latence(latences, seuil_ms):
    """Latence appel par appel, avec l'objectif de service en repère."""
    valeurs = pd.Series(latences).dropna().to_numpy(dtype="float64")
    if len(valeurs) < 2:
        return ""
    largeur, hauteur = 640, 200
    marge = {"haut": 12, "bas": 28, "gauche": 46, "droite": 8}
    aire_l = largeur - marge["gauche"] - marge["droite"]
    aire_h = hauteur - marge["haut"] - marge["bas"]
    maximum = max(valeurs.max(), seuil_ms * 0.4, 1e-9)

    points = " ".join(
        f"{marge['gauche'] + i / (len(valeurs) - 1) * aire_l:.1f},"
        f"{marge['haut'] + aire_h - valeurs[i] / maximum * aire_h:.1f}"
        for i in range(len(valeurs)))

    elements = []
    for graduation in np.linspace(0, maximum, 4):
        y = marge["haut"] + aire_h - graduation / maximum * aire_h
        elements.append(f'<line x1="{marge["gauche"]}" y1="{y:.1f}" x2="{largeur - marge["droite"]}" '
                        f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        elements.append(f'<text x="{marge["gauche"] - 6}" y="{y + 3.5:.1f}" text-anchor="end" '
                        f'font-size="10" fill="var(--muted)">{graduation:.0f}</text>')
    if seuil_ms <= maximum:
        y = marge["haut"] + aire_h - seuil_ms / maximum * aire_h
        elements.append(f'<line x1="{marge["gauche"]}" y1="{y:.1f}" x2="{largeur - marge["droite"]}" '
                        f'y2="{y:.1f}" stroke="{COULEURS["alerte"]}" stroke-width="2" '
                        f'stroke-dasharray="5 4"/>')
        elements.append(f'<text x="{largeur - marge["droite"]}" y="{y - 5:.1f}" text-anchor="end" '
                        f'font-size="10" fill="{COULEURS["alerte"]}">objectif '
                        f'{seuil_ms:.0f} ms</text>')
    elements.append(f'<polyline points="{points}" fill="none" stroke="var(--production)" '
                    f'stroke-width="2" stroke-linejoin="round"/>')
    elements.append(f'<text x="{largeur / 2}" y="{hauteur - 2}" text-anchor="middle" '
                    f'font-size="10" fill="var(--muted)">appels dans l\'ordre chronologique '
                    f'(latence en ms)</text>')
    return (f'<svg viewBox="0 0 {largeur} {hauteur}" role="img" '
            f'aria-label="Latence par appel">' + "".join(elements) + "</svg>")


def _barres_statuts(par_statut):
    """Répartition des codes HTTP : une magnitude par catégorie, une seule teinte."""
    if not par_statut:
        return ""
    statuts = [(str(k), int(v)) for k, v in sorted(par_statut.items())]
    largeur, hauteur = 640, 180
    marge = {"haut": 16, "bas": 36, "gauche": 46, "droite": 8}
    aire_l = largeur - marge["gauche"] - marge["droite"]
    aire_h = hauteur - marge["haut"] - marge["bas"]
    maximum = max(valeur for _, valeur in statuts)
    pas = aire_l / len(statuts)
    barre = min(pas - 14, 70)

    elements = [f'<line x1="{marge["gauche"]}" y1="{marge["haut"] + aire_h}" '
                f'x2="{largeur - marge["droite"]}" y2="{marge["haut"] + aire_h}" '
                f'stroke="var(--axe)" stroke-width="1"/>']
    for i, (statut, valeur) in enumerate(statuts):
        h = valeur / maximum * aire_h
        x = marge["gauche"] + i * pas + (pas - barre) / 2
        y = marge["haut"] + aire_h - h
        couleur = "var(--reference)" if statut.startswith("2") else COULEURS[
            "alerte" if statut.startswith("5") else "vigilance"]
        elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{barre:.1f}" height="{h:.1f}" '
                        f'rx="4" fill="{couleur}"><title>HTTP {statut} : {valeur} appels</title></rect>')
        elements.append(f'<text x="{x + barre / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" '
                        f'font-size="11" fill="var(--ink)">{valeur}</text>')
        elements.append(f'<text x="{x + barre / 2:.1f}" y="{hauteur - 18}" text-anchor="middle" '
                        f'font-size="11" fill="var(--ink2)">{statut}</text>')
    elements.append(f'<text x="{largeur / 2}" y="{hauteur - 2}" text-anchor="middle" '
                    f'font-size="10" fill="var(--muted)">code HTTP</text>')
    return (f'<svg viewBox="0 0 {largeur} {hauteur}" role="img" '
            f'aria-label="Répartition des codes HTTP">' + "".join(elements) + "</svg>")


def _legende():
    return (f'<div class="legende">'
            f'<span><i class="pastille" style="background:var(--reference)"></i>'
            f'référence (entraînement)</span>'
            f'<span><i class="pastille" style="background:var(--production)"></i>'
            f'production</span></div>')


def _tuile(valeur, libelle, detail=""):
    return (f'<div class="tuile"><span class="valeur">{_e(valeur)}</span>'
            f'<span class="libelle">{_e(libelle)}</span>'
            + (f'<div class="detail">{_e(detail)}</div>' if detail else "")
            + "</div>")


def _tableau_drift(resultats):
    entetes = ["Feature", "PSI", "KS", "p-value", "Moy. réf.", "Moy. prod.",
               "Manquants réf.", "Manquants prod.", "Verdict"]
    lignes = []
    for ligne in resultats.itertuples():
        couleur = (COULEURS["alerte"] if ligne.verdict.startswith("dérive")
                   and "modérée" not in ligne.verdict else
                   COULEURS["vigilance"] if ligne.verdict != "stable" else COULEURS["stable"])
        icone = ("■" if couleur == COULEURS["alerte"]
                 else "▲" if couleur == COULEURS["vigilance"] else "●")
        lignes.append(
            "<tr>"
            f"<td>{_e(ligne.feature)}</td>"
            f"<td>{_format(ligne.psi)}</td>"
            f"<td>{_format(ligne.ks_statistique)}</td>"
            f"<td>{_format(ligne.ks_p_value, '{:.2e}')}</td>"
            f"<td>{_format(ligne.moyenne_reference, '{:,.4g}')}</td>"
            f"<td>{_format(ligne.moyenne_production, '{:,.4g}')}</td>"
            f"<td>{_format(ligne.manquants_reference, '{:.0%}')}</td>"
            f"<td>{_format(ligne.manquants_production, '{:.0%}')}</td>"
            f'<td><span class="etiquette" style="color:{couleur}">{icone} '
            f"{_e(ligne.verdict)}</span></td></tr>")
    return ('<div class="tableau"><table><thead><tr>'
            + "".join(f"<th>{_e(e)}</th>" for e in entetes)
            + "</tr></thead><tbody>" + "".join(lignes) + "</tbody></table></div>")


def _vigilances(synthese_drift, predictions, exploitation, resultats):
    """Points de vigilance déduits des mesures — le propos de l'étude, pas du décor."""
    points = []
    pires = resultats[resultats["verdict"].str.startswith("dérive")]
    if len(pires):
        noms = ", ".join(f"{l.feature} (PSI {l.psi:.2f})" for l in pires.head(3).itertuples())
        points.append(
            f"<strong>{len(pires)} feature(s) s'écartent de la référence</strong> — les plus "
            f"marquées : {noms}. Une dérive sur les variables les plus influentes du modèle "
            "déplace le score sans qu'aucune erreur ne soit levée : c'est la panne silencieuse "
            "à surveiller en premier.")
    if predictions and np.isfinite(predictions.get("ecart_taux_refus") or float("nan")):
        ecart = predictions["ecart_taux_refus"]
        points.append(
            f"<strong>Taux de refus : {predictions['taux_refus_production']:.1%} en production "
            f"contre {predictions['taux_refus_reference']:.1%} sur la référence "
            f"({ecart:+.1%})</strong>. Le seuil métier a été optimisé sur la distribution "
            "d'entraînement ; s'il s'applique à une population différente, le compromis "
            "coût-faux-négatifs/faux-positifs n'est plus celui qui a été validé.")
    if exploitation["anomalies"]:
        points.append("<strong>Exploitation</strong> : "
                      + " ; ".join(_e(a) for a in exploitation["anomalies"])
                      + ". Un taux d'erreurs client élevé signale en général un appelant mal "
                      "intégré plutôt qu'une panne du modèle.")
    manquants = resultats[(resultats["manquants_production"]
                           - resultats["manquants_reference"]).abs() > 0.1]
    if len(manquants):
        points.append(
            f"<strong>{len(manquants)} feature(s) changent de taux de valeurs manquantes</strong> "
            "de plus de 10 points. LightGBM accepte les valeurs manquantes sans erreur : une "
            "feature qui cesse d'être alimentée dégrade le score en silence.")
    if not points:
        points.append("Aucune anomalie au-delà des seuils. La référence reste le jeu "
                      "d'entraînement : elle devra être réactualisée après tout réentraînement, "
                      "sinon la comparaison perdra son sens.")
    return "".join(f"<li>{p}</li>" for p in points)


def construire_rapport(synthese_drift, resultats_features, predictions, exploitation,
                       volumetrie, reference, production, periode=None):
    """Assemble le rapport HTML complet et le renvoie sous forme de chaîne."""
    niveau = synthese_drift["niveau"] if synthese_drift["niveau"] != "STABLE" \
        else exploitation["niveau"]
    niveau_global = "ALERTE" if "ALERTE" in (synthese_drift["niveau"], exploitation["niveau"]) \
        else niveau
    couleur = _couleur_niveau(niveau_global)
    latences = exploitation["latences_ms"]

    tuiles = [
        _tuile(f"{exploitation['appels']}", "appels journalisés",
               f"dont {exploitation['appels_predict']} sur /predict"),
        _tuile(_format(exploitation["taux_erreur_client"], "{:.1%}"), "erreurs client (4xx)",
               f"serveur (5xx) : {exploitation['taux_erreur_serveur']:.1%}"),
        _tuile(_format(latences["p95"], "{:.0f} ms"), "latence p95",
               f"p50 {_format(latences['p50'], '{:.0f}')} ms · "
               f"p99 {_format(latences['p99'], '{:.0f}')} ms"),
        _tuile(f"{synthese_drift['features_en_derive']}/{synthese_drift['features_evaluables']}",
               "features en dérive", f"dont {synthese_drift['features_derive_moderee']} modérées"),
    ]
    if predictions:
        tuiles.append(_tuile(_format(predictions["taux_refus_production"], "{:.1%}"),
                             "taux de refus",
                             f"référence : {predictions['taux_refus_reference']:.1%}"))

    figures_features = ""
    for ligne in resultats_features.head(N_FEATURES_FIGURES).itertuples():
        if ligne.feature not in production.columns:
            continue
        figure = _histogramme(reference[ligne.feature], production[ligne.feature], ligne.feature)
        if figure:
            figures_features += (
                f"<figure>{figure}<figcaption>{_e(ligne.feature)} — PSI "
                f"{_format(ligne.psi)}, KS {_format(ligne.ks_statistique)} "
                f"(p = {_format(ligne.ks_p_value, '{:.1e}')}) : {_e(ligne.verdict)}"
                "</figcaption></figure>")

    figure_scores = ""
    if predictions and "probabilite_defaut" in reference.columns:
        svg = _histogramme(reference["probabilite_defaut"],
                           production["probabilite_defaut"].dropna(),
                           "probabilité de défaut prédite")
        if svg:
            figure_scores = (
                f"<figure>{svg}<figcaption>Distribution des scores — PSI "
                f"{_format(predictions['psi'])}, score moyen "
                f"{_format(predictions['score_moyen_reference'], '{:.3f}')} en référence contre "
                f"{_format(predictions['score_moyen_production'], '{:.3f}')} en production."
                "</figcaption></figure>")

    periode = periode or f"{exploitation['periode_debut']} → {exploitation['periode_fin']}"
    genere = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Suivi de production — scoring crédit</title>
<style>{STYLE}</style></head>
<body><main>
<h1>Suivi de production — API de scoring crédit</h1>
<p class="sous-titre">Période analysée : {_e(periode)} · rapport généré le {_e(genere)}</p>
<div class="verdict" style="background:{couleur}">{ICONES.get(niveau_global, '▲')}
  Verdict global : {_e(niveau_global)}</div>
<div class="tuiles">{''.join(tuiles)}</div>

<h2>Dérive des données</h2>
<p>Référence : {synthese_drift['features_suivies']} features suivies sur un échantillon du jeu
d'entraînement ({len(reference)} dossiers). Une feature est déclarée en dérive si l'écart est
net (PSI ≥ 0,25) ou modéré et statistiquement significatif (KS, seuil corrigé par Bonferroni).</p>
{'<ul>' + ''.join(f'<li>{_e(m)}</li>' for m in synthese_drift['motifs']) + '</ul>'
    if synthese_drift['motifs'] else '<p>Aucune dérive au-delà des seuils.</p>'}
{_legende()}
{figures_features}
<h3>Toutes les features suivies</h3>
{_tableau_drift(resultats_features)}

<h2>Dérive des prédictions</h2>
{_legende() + figure_scores if figure_scores else '<p>Pas assez de prédictions.</p>'}

<h2>Exploitation</h2>
<figure>{_barres_statuts(exploitation['par_statut'])}
<figcaption>Répartition des codes HTTP sur la période.</figcaption></figure>
<figure>{_courbe_latence(production['latence_ms'] if 'latence_ms' in production else [], 500)}
<figcaption>Latence de chaque appel à /predict, comparée à l'objectif de service.</figcaption></figure>

<h2>Points de vigilance</h2>
<ul>{_vigilances(synthese_drift, predictions, exploitation, resultats_features)}</ul>

<footer>
<p><strong>Stockage</strong> : {volumetrie['evenements']} événements en base
({volumetrie['taille_base_mo']} Mo, soit {volumetrie['octets_par_prediction']} octets par
prédiction). Journaux conservés 90 jours puis purgés.</p>
<p><strong>RGPD</strong> : ni identifiant client, ni adresse IP, ni clé d'API dans les journaux —
seules les features du modèle, la prédiction et les métriques techniques.</p>
<p>Reproduire : <code>python scripts/analyze_production.py --rapport rapports/monitoring.html</code></p>
</footer>
</main></body></html>
"""


def ecrire_json(chemin, synthese_drift, resultats_features, predictions, exploitation, volumetrie):
    """Résumé machine de l'analyse, pour l'archivage ou une alerte automatique."""
    contenu = {
        "genere_le": datetime.now(timezone.utc).isoformat(),
        "drift": synthese_drift,
        "predictions": predictions,
        "exploitation": {k: v for k, v in exploitation.items() if k != "par_statut"},
        "par_statut": {str(k): int(v) for k, v in (exploitation.get("par_statut") or {}).items()},
        "volumetrie": volumetrie,
        "features": resultats_features.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(contenu, f, indent=2, ensure_ascii=False)
    return contenu
