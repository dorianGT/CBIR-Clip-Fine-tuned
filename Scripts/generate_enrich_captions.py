import os
import json
import argparse
import logging

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration logging
logging.basicConfig(
    filename='enrichment_log.txt',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_print(message):
    # On peut également print() ici si besoin
    logging.info(message)

# Chargement du modèle Mistral
print("Chargement du modèle Mistral...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Modèle Mistral chargé.")



def load_descriptions(caption_csv_path):
    """
    Charge les descriptions à partir d’un fichier CSV sans en-tête.
    Utilise uniquement le nom de fichier (sans chemin ni extension) comme clé.
    Filtre les prompts selon un préfixe.

    Args:
        caption_csv_path (str): Chemin vers le fichier CSV contenant les colonnes [name;prompt;description].

    Returns:
        dict: Dictionnaire associant le nom de fichier (sans extension) à sa description.
    """
    # Lecture brute sans en-tête
    df = pd.read_csv(caption_csv_path, sep=";", header=None, names=["name", "prompt", "description"])

    # Filtrage du prompt
    prompt_filter = "a newspaper clipping from the early 1900s showing"
    df = df[df["prompt"].str.startswith(prompt_filter, na=False)]

    # Nettoyage du nom : on enlève chemin + extension
    df["name"] = df["name"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    # Dictionnaire: nom - description
    return dict(zip(df["name"], df["description"]))


def build_fusion_prompt(metadata_row, visual_caption):
    """
    Construit un prompt pour fusionner les métadonnées et la caption visuelle.

    Args:
        metadata_row (pd.Series): Ligne de métadonnées contenant possiblement les colonnes 'date', 'description', 'legende', 'legende_2'.
        visual_caption (str): Caption générée à partir de l'image.

    Returns:
        str: Prompt textuel formaté pour instruct-tuning, destiné à une génération de texte.
    """
    meta_description = []

    if pd.notna(metadata_row.get("date")):
        try:
            date_obj = pd.to_datetime(metadata_row["date"], errors='raise')
            formatted_date = date_obj.strftime("%d %B %Y")
            meta_description.append(f"dated {formatted_date}")
        except Exception:
            meta_description.append(f"dated {metadata_row['date']}")

    # Vérifie la présence des metadata        
    if pd.notna(metadata_row.get("description")):
        meta_description.append(metadata_row["description"].strip())
    if pd.notna(metadata_row.get("legende")):
        meta_description.append(metadata_row["legende"].strip())
    if pd.notna(metadata_row.get("legende_2")):
        meta_description.append(metadata_row["legende_2"].strip())
    # if pd.notna(metadata_row.get("type")):
    #     meta_description.append(f"Type: {metadata_row['type']}")

    alttext = ". ".join(meta_description)

    prompt = (
        "Rephrase and merge the following two sentences into a single, fluent sentence in proper and concise English.\n"
        "Follow this order when combining the information: (1) Description, (2) Date (if available), (3) Location (if available).\n"
        "Don't talk about date or location if not available\n"
        "Make sure to retain all essential information and ensure the sentence sounds natural and coherent.\n\n"
        "Don't add absent informations.\n\n"
        f"Sentence 1 (Alt-text): {alttext}\n"
        f"Sentence 2 (Visual caption): {visual_caption}\n\n"
        "Final sentence:"
    )

    return prompt

def call_mistral(prompt):
    """
    Appelle le modèle Mistral pour générer une caption enrichie.

    Args:
        prompt (str): Prompt textuel à fournir au modèle, incluant les instructions de fusion.

    Returns:
        str: Phrase générée par le modèle, correspondant à la fusion des informations.
    """
    formatted_prompt = (
        "<s>[INST] You are a helpful assistant tasked with generating image captions. "
        "Follow these instructions:\n"
        "- Merge the metadata and the visual caption into a single short sentence.\n"
        "- Place attributes before noun entities.\n"
        "- Do not start the sentence with 'The image'.\n"
        "- Use grammatically correct and concise English.\n"
        f"{prompt} [/INST]"
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split("[/INST]")[-1].strip()


def enrich_captions(description_dict, metadata_csv, model_call_fn):
    """
    Enrichit les captions en fusionnant avec les métadonnées.
    Affiche les logs détaillés pour chaque étape.
    Distingue les images avec et sans métadonnées textuelles.

    Args:
        description_dict (dict): Dictionnaire {image_name: visual_caption}, les captions visuelles.
        metadata_csv (str): Chemin vers le fichier CSV contenant les métadonnées.
        model_call_fn (Callable[[str], str]): Fonction de génération prenant un prompt et retournant une caption enrichie.

    Returns:
        dict: Dictionnaire enrichi contenant les champs :
            - original_caption (str)
            - metadata_used (dict)
            - fusion_prompt (str ou None)
            - enriched_caption (str)
    """
    df_meta = pd.read_csv(metadata_csv)
    enriched_dict = {}

    # Création d’un dictionnaire rapide d’accès
    metadata_names = {
        os.path.splitext(row["name"])[0]: row
        for _, row in df_meta.iterrows()
    }

    # Séparer les images avec et sans métadonnées textuelles
    with_meta = {}
    without_meta = {}

    for image_name, visual_caption in description_dict.items():
        image_base = os.path.splitext(image_name)[0]

        if image_base not in metadata_names:
            log_print(f"Image non trouvée dans les métadonnées : {image_name}")
            continue

        metadata_row = metadata_names[image_base]

        # Si toutes les colonnes sont vides
        if all(pd.isna(metadata_row.get(col)) for col in ["description", "legende", "legende_2"]):
            without_meta[image_name] = (visual_caption, metadata_row)
        else:
            with_meta[image_name] = (visual_caption, metadata_row)

    # Affichage du nombre d'images sans métadonnées textuelles
    print(f"\nImages sans métadonnées textuelles : {len(without_meta)}")
    print(f"Images avec métadonnées textuelles : {len(with_meta)}\n")

    # Étape 1 : Stocker telles quelles celles sans métadonnées
    for image_name, (visual_caption, metadata_row) in without_meta.items():
        metadata_used = metadata_row.to_dict()
        enriched_dict[image_name] = {
            "original_caption": visual_caption,
            "metadata_used": metadata_used,
            "fusion_prompt": None,
            "enriched_caption": visual_caption
        }

    # Étape 2 : Enrichissement pour celles qui ont des métadonnées
    for image_name, (visual_caption, metadata_row) in tqdm(with_meta.items(), desc="Enrichissement des captions"):
        metadata_used = metadata_row.to_dict()
        log_print(f"\nTraitement de l'image : {image_name}")
        log_print(f"Caption initiale : {visual_caption}")
        log_print("Métadonnées utilisées :")
        for k, v in metadata_used.items():
            log_print(f"   - {k}: {v}")

        # Création du prompt et génération
        prompt = build_fusion_prompt(metadata_row, visual_caption)
        try:
            enriched_caption = model_call_fn(prompt)
            enriched_dict[image_name] = {
                "original_caption": visual_caption,
                "metadata_used": metadata_used,
                "fusion_prompt": prompt,
                "enriched_caption": enriched_caption
            }
            log_print(f"Caption enrichie : {enriched_caption}")
        except Exception as e:
            log_print(f"Erreur pendant l'enrichissement : {e}")

    return enriched_dict



def save_enriched_captions(enriched_dict, output_path):
    """
    Sauvegarde les captions enrichies dans un fichier JSON.

    Args:
        enriched_dict (dict): Dictionnaire des captions enrichies à sauvegarder.
        output_path (str): Chemin du fichier JSON de sortie.
    """
    with open(output_path, "w") as f:
        json.dump(enriched_dict, f, indent=4)
    print(f"Captions enrichies sauvegardées dans {output_path}")


def afficher_taux_remplissage(metadata_csv):
    """
    Affiche le taux de remplissage de chaque colonne dans le fichier de métadonnées.

    Args:
        metadata_csv (str): Chemin vers le fichier CSV contenant les métadonnées.
    """
    df_meta = pd.read_csv(metadata_csv)
    taux_remplissage = df_meta.notna().mean() * 100
    print("\nTaux de remplissage des métadonnées :")
    print(taux_remplissage.sort_values(ascending=False).round(2).astype(str) + " %")


def main():
    """
    Point d'entrée du script.
    Charge les captions et les métadonnées, enrichit les captions avec un modèle local,
    et sauvegarde les résultats dans un fichier JSON.

    Args:
        --captions (str): Chemin du fichier CSV contenant les captions existantes.
        --metadata (str): Chemin du fichier CSV contenant les métadonnées.
        --output (str, optionnel): Chemin du fichier JSON de sortie (défaut: "enriched_captions.json").
    """
    parser = argparse.ArgumentParser(description="Enrich image captions using metadata and a local Mistral model.")
    parser.add_argument("--captions", required=True, help="Chemin du fichier CSV contenant les captions existantes.")
    parser.add_argument("--metadata", required=True, help="Chemin du fichier CSV contenant les métadonnées.")
    parser.add_argument("--output", default="enriched_captions.json", help="Fichier JSON de sortie.")

    args = parser.parse_args()

    # Affichage du taux de remplissage
    afficher_taux_remplissage(args.metadata)

    # Chargement des captions visuelles
    description_dict = load_descriptions(args.captions)

    # Enrichissement via modèle local
    enriched_dict = enrich_captions(description_dict, args.metadata, call_mistral)

    # Sauvegarde
    save_enriched_captions(enriched_dict, args.output)

if __name__ == "__main__":
    main()