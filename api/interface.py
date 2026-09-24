"""Interface Gradio montée sur l'API FastAPI (route /ui)."""

import json

import gradio as gr

from api.scoring import UnknownFeatures, client_to_dict

# Les 7 features les plus utilisées par le LightGBM champion (feature_importances_) :
# ce sont celles qu'un chargé de clientèle a intérêt à pouvoir faire varier.
KEY_FEATURES = [
    "PAYMENT_RATE",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "DAYS_BIRTH",
    "AMT_ANNUITY",
    "DAYS_EMPLOYED",
]


def format_result(result):
    icon = "🔴" if result["decision"] == 1 else "🟢"
    return (
        f"## {icon} {result['libelle_decision'].capitalize()}\n\n"
        f"| Probabilité de défaut | Seuil métier | Features renseignées |\n"
        f"|---|---|---|\n"
        f"| **{result['probabilite_defaut']:.1%}** | {result['seuil']:.1%} "
        f"| {result['features_renseignees']} |\n\n"
        "Le crédit est refusé dès que la probabilité de défaut atteint le seuil métier "
        "(coût d'un défaut non détecté = 10 × coût d'un bon client refusé)."
    )


def create_interface(model, example_clients):
    def client_values(sk_id):
        row = example_clients.loc[int(sk_id)]
        return [None if row.isna()[name] else float(row[name]) for name in KEY_FEATURES]

    def score_client(sk_id, *key_values):
        features = client_to_dict(example_clients.loc[int(sk_id)])
        features.update(dict(zip(KEY_FEATURES, key_values)))
        return format_result(model.predict(features))

    def score_json(text):
        try:
            features = json.loads(text)
            if not isinstance(features, dict):
                raise ValueError("le JSON doit être un objet {feature: valeur}")
            return format_result(model.predict(features))
        except (ValueError, TypeError, UnknownFeatures) as error:
            return f"⚠️ Entrée invalide : {error}"

    ids = [str(i) for i in example_clients.index]
    version = model.metadata

    with gr.Blocks(title="Scoring crédit — Prêt à dépenser") as interface:
        gr.Markdown(
            "# Scoring crédit — Prêt à dépenser\n"
            f"Modèle `{version['nom']}` v{version['version']} (alias `@{version['alias']}`), "
            f"{len(model.features)} features, seuil métier {model.threshold:.3f}."
        )
        with gr.Tab("Client existant"):
            choice = gr.Dropdown(ids, value=ids[0], label="Client (SK_ID_CURR, jeu test)")
            with gr.Row():
                fields = [gr.Number(label=name) for name in KEY_FEATURES]
            button = gr.Button("Évaluer le client", variant="primary")
            output = gr.Markdown()
            choice.change(client_values, choice, fields)
            button.click(score_client, [choice, *fields], output)
            interface.load(client_values, choice, fields)

        with gr.Tab("Saisie JSON"):
            example = {name: client_to_dict(example_clients.iloc[0])[name] for name in KEY_FEATURES}
            text = gr.Code(json.dumps(example, indent=2), language="json",
                           label="Features du client (les features absentes sont traitées comme manquantes)")
            json_button = gr.Button("Évaluer", variant="primary")
            json_output = gr.Markdown()
            json_button.click(score_json, text, json_output)

    return interface
