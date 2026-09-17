"""Interface Gradio montée sur l'API FastAPI (route /ui)."""

import json

import gradio as gr

from api.scoring import FeaturesInconnues, client_en_dict

# Les 7 features les plus utilisées par le LightGBM champion (feature_importances_) :
# ce sont celles qu'un chargé de clientèle a intérêt à pouvoir faire varier.
FEATURES_CLES = [
    "PAYMENT_RATE",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "DAYS_BIRTH",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
]


def formater_resultat(resultat):
    icone = "🔴" if resultat["decision"] == 1 else "🟢"
    return (
        f"## {icone} {resultat['libelle_decision'].capitalize()}\n\n"
        f"| Probabilité de défaut | Seuil métier | Features renseignées |\n"
        f"|---|---|---|\n"
        f"| **{resultat['probabilite_defaut']:.1%}** | {resultat['seuil']:.1%} "
        f"| {resultat['features_renseignees']} |\n\n"
        "Le crédit est refusé dès que la probabilité de défaut atteint le seuil métier "
        "(coût d'un défaut non détecté = 10 × coût d'un bon client refusé)."
    )


def creer_interface(modele, clients_exemple):
    def valeurs_client(sk_id):
        ligne = clients_exemple.loc[int(sk_id)]
        return [None if ligne.isna()[nom] else float(ligne[nom]) for nom in FEATURES_CLES]

    def scorer_client(sk_id, *valeurs_cles):
        features = client_en_dict(clients_exemple.loc[int(sk_id)])
        features.update(dict(zip(FEATURES_CLES, valeurs_cles)))
        return formater_resultat(modele.predire(features))

    def scorer_json(texte):
        try:
            features = json.loads(texte)
            if not isinstance(features, dict):
                raise ValueError("le JSON doit être un objet {feature: valeur}")
            return formater_resultat(modele.predire(features))
        except (ValueError, TypeError, FeaturesInconnues) as erreur:
            return f"⚠️ Entrée invalide : {erreur}"

    ids = [str(i) for i in clients_exemple.index]
    version = modele.metadonnees

    with gr.Blocks(title="Scoring crédit — Prêt à dépenser") as interface:
        gr.Markdown(
            "# Scoring crédit — Prêt à dépenser\n"
            f"Modèle `{version['nom']}` v{version['version']} (alias `@{version['alias']}`), "
            f"{len(modele.features)} features, seuil métier {modele.seuil:.3f}."
        )
        with gr.Tab("Client existant"):
            choix = gr.Dropdown(ids, value=ids[0], label="Client (SK_ID_CURR, jeu test)")
            with gr.Row():
                champs = [gr.Number(label=nom) for nom in FEATURES_CLES]
            bouton = gr.Button("Évaluer le client", variant="primary")
            sortie = gr.Markdown()
            choix.change(valeurs_client, choix, champs)
            bouton.click(scorer_client, [choix, *champs], sortie)
            interface.load(valeurs_client, choix, champs)

        with gr.Tab("Saisie JSON"):
            exemple = {nom: client_en_dict(clients_exemple.iloc[0])[nom] for nom in FEATURES_CLES}
            texte = gr.Code(json.dumps(exemple, indent=2), language="json",
                            label="Features du client (les features absentes sont traitées comme manquantes)")
            bouton_json = gr.Button("Évaluer", variant="primary")
            sortie_json = gr.Markdown()
            bouton_json.click(scorer_json, texte, sortie_json)

    return interface
