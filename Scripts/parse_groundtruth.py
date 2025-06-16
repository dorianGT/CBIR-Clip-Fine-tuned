import os
import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split
import unicodedata
from PIL import Image
import argparse

def set_seed(seed=2):
    """
    Initialise la seed pour garantir la reproductibilité.

    Args:
        seed (int): La seed. Par défaut 2.
    """
    random.seed(seed)

def load_groundtruth(excel_file):
    """
    Charge les groupes de groundtruth depuis un fichier Excel.

    Chaque ligne correspond à un groupe, et chaque colonne contient un nom de fichier (sans extension).

    Args:
        excel_file (str): Chemin vers le fichier Excel contenant les groupes d'images.

    Returns:
        list: Liste de groupes, chaque groupe étant une liste de noms de fichiers (sans extension).
    """
    df = pd.read_excel(excel_file, header=None)
    
    groups = []
    for _, row in df.iterrows():
        group = [str(cell) for cell in row if pd.notnull(cell)]
        if group:
            groups.append(group)

    return groups

def remove_accents(input_str):
    """
    Supprime les accents d'une chaîne de caractères.

    Args:
        input_str (str): Chaîne d'entrée.

    Returns:
        str: Chaîne sans accents.
    """
    return unicodedata.normalize('NFKD', input_str).encode('ASCII', 'ignore').decode('ASCII')

def correct_filenames(groups, image_folder, description_dict):
    """
    Corrige les noms de fichiers dans les groupes en fonction des fichiers réellement présents.

    Args:
        groups (list): Groupes d'images à corriger.
        image_folder (str): Dossier contenant les images.
        description_dict (dict): Dictionnaire des descriptions.

    Returns:
        list: Groupes avec noms de fichiers corrigés (avec extension).
    """
    available_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder)}
    available_files_no_accents = {remove_accents(k): v for k, v in available_files.items()}

    corrected_groups = []
    corrected_count = 0
    missing_count = 0

    for group in groups:
        corrected = []
        for name in group:
            if name in available_files:
                if name in description_dict:
                    corrected.append(available_files[name])
                else:
                    print(f"no caption for file: '{name}'")
            elif remove_accents(name) in available_files_no_accents:
                corrected_name = available_files_no_accents[remove_accents(name)]
                if corrected_name in description_dict:
                    print(f"Corrected '{name}' -> '{corrected_name}'")
                    corrected.append(corrected_name)
                    corrected_count += 1
                else:
                    print(f"no caption for file: '{name}'")         
            else:
                print(f"Missing file for '{name}'")
                missing_count += 1
        corrected_groups.append(corrected)

    print(f"\nRésultat final :")
    print(f" - {corrected_count} noms corrigés")
    print(f" - {missing_count} noms manquants")

    return corrected_groups

def get_singletons(groups, image_folder, description_dict, move_unused=True):
    """
    Trouve les images qui ne sont dans aucun groupe.
    Si elles ne sont pas dans le dictionnaire de descriptions, les déplace dans un dossier 'not_used'.

    Args:
        groups (list): Groupes d'images.
        image_folder (str): Dossier contenant les images.
        description_dict (dict): Dictionnaire des descriptions.
        move_unused (bool): Si True, déplace les images inutilisées vers un dossier "not_used".

    Returns:
        list: Liste des images valides (avec caption) non incluses dans les groupes.
    """
    all_images = set(f for f in os.listdir(image_folder)
                 if os.path.isfile(os.path.join(image_folder, f)))

    grouped_images = set(img for group in groups for img in group)
    singletons = all_images - grouped_images
    valid_singletons = []

    not_used_dir = os.path.join(image_folder, "not_used")
    if move_unused and not os.path.exists(not_used_dir):
        os.makedirs(not_used_dir)

    for img in singletons:
        base_name = os.path.splitext(img)[0]
        if base_name in description_dict:
            valid_singletons.append(img)
        elif move_unused:
            src = os.path.join(image_folder, img)
            dst = os.path.join(not_used_dir, img)
            print(f"Déplacement de {img} vers 'not_used'")
            os.rename(src, dst)

    return valid_singletons


def get_caption(image_name, description_dict):
    """
    Récupère la légende d'une image à partir du dictionnaire de descriptions.

    Args:
        image_name (str): Nom de l'image (avec extension).
        description_dict (dict): Dictionnaire de descriptions {nom_sans_extension: description}.

    Returns:
        str or None: Description si disponible, sinon None.
    """
    base_name = os.path.splitext(image_name)[0]
    if description_dict and base_name in description_dict:
        return description_dict[base_name]
    else:
        return None

def print_stats(groups, train, val, output_file):
    """
    Affiche des statistiques sur les groupes d'images et les jeux d'entraînement/validation.

    Args:
        groups (list): Liste des groupes d'images.
        train (list): Liste des images pour l'entraînement.
        val (list): Liste des images pour la validation.
        output_file (str): Nom du fichier de sortie.
    """
    group_sizes = [len(group) for group in groups]
    total_images_in_groups = sum(group_sizes)
    num_groups = len(groups)
    min_size = min(group_sizes) if group_sizes else 0
    max_size = max(group_sizes) if group_sizes else 0
    mean_size = total_images_in_groups / num_groups if num_groups > 0 else 0

    print("\nStatistiques des groupes :")
    print(f" - Nombre de groupes          : {num_groups}")
    print(f" - Nombre total d'images      : {total_images_in_groups}")
    print(f" - Taille min d'un groupe     : {min_size}")
    print(f" - Taille max d'un groupe     : {max_size}")
    print(f" - Taille moyenne d'un groupe : {mean_size:.2f}")

    print("\nStatistiques des jeux de données :")
    print(f" - Images dans train : {len(train)}")
    print(f" - Images dans val   : {len(val)}")

    print(f"\nSauvegardé dans : {output_file}")


def save_groundtruth_json(groups, train, val, output_file, description_dict):
    """
    Sauvegarde les groupes, jeux d'entraînement et de validation avec leurs captions dans un fichier JSON.

    Args:
        groups (list): Groupes d'images.
        train (list): Liste des images pour l'entraînement.
        val (list): Liste des images pour la validation.
        output_file (str): Chemin du fichier JSON à sauvegarder.
        description_dict (dict): Dictionnaire des descriptions.
    """
    train_texts = []
    new_train = []
    val_texts = []
    new_val = []

    train_missing = 0
    val_missing = 0

    for img in train:
        text = get_caption(img, description_dict)
        if text is not None:
            train_texts.append(text)
            new_train.append(img)
        else:
            train_missing += 1

    for img in val:
        text = get_caption(img, description_dict)
        if text is not None:
            val_texts.append(text)
            new_val.append(img)
        else:
            val_missing += 1

    groups_with_captions = []
    for group in groups:
        group_info = []
        for img in group:
            group_info.append({
                "image": img,
                "caption": get_caption(img, description_dict)
            })
        groups_with_captions.append(group_info)

    data = {
        "groups": groups_with_captions,
        "train": new_train,
        "train_texts": train_texts,
        "val": new_val,
        "val_texts": val_texts
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print_stats(groups, new_train, new_val, output_file)


def load_groundtruth_json(json_file):
    """
    Charge les données (groupes, train, val et leurs textes) depuis un fichier JSON.

    Args:
        json_file (str): Chemin du fichier JSON.

    Returns:
        tuple: (groups, train, train_texts, val, val_texts)
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data["groups"], data["train"], data["train_texts"], data["val"], data["val_texts"]

def load_descriptions(csv_file):
    """
    Charge les descriptions depuis un fichier CSV.
    Ne garde que les captions dont le prompt commence par :
    "a newspaper clipping from the early 1900s showing".

    Args:
        csv_file (str): Chemin du fichier CSV contenant les descriptions.

    Returns:
        dict: Dictionnaire {nom_image_sans_extension: description}.
    """
    df = pd.read_csv(csv_file, sep=";", header=None)
    description_dict = {}

    for _, row in df.iterrows():
        image_path = row[0]
        prompt = str(row[1]).strip().lower()
        description = str(row[2]).strip()

        if pd.isna(description) or description.upper() == "EMPTY":
            continue

        if prompt.startswith("a newspaper clipping from the early 1900s showing"):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            if image_name not in description_dict:
                description_dict[image_name] = description

    return description_dict

def load_veclip_captions(json_file):
    """
    Charge les captions enrichies depuis un fichier JSON généré par VeCLIP.

    Args:
        json_file (str): Chemin vers le fichier JSON VeCLIP.

    Returns:
        dict: Dictionnaire {nom_image_sans_extension: enriched_caption}.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    description_dict = {}
    for key, value in data.items():
        enriched_caption = value.get("enriched_caption", None)
        if enriched_caption:
            description_dict[key] = enriched_caption

    return description_dict


def main():
    parser = argparse.ArgumentParser(description="Génère un fichier JSON de ground truth à partir d'un Excel de groupes et d'un CSV de descriptions.")
    parser.add_argument("--image_folder", type=str, required=True, help="Chemin vers le dossier contenant les images.")
    parser.add_argument("--excel_file", type=str, required=True, help="Chemin vers le fichier Excel contenant les groupes d'images similaires.")
    parser.add_argument("--csv_descriptions_file", type=str, required=True, help="Chemin vers le fichier CSV contenant les descriptions des images.")
    parser.add_argument("--use_veclip_caption", action="store_true",
                    help="Si activé, utilise les captions enrichies de VECLIP depuis un fichier JSON.")
    parser.add_argument("--veclip_json_file", type=str, default=None,
                    help="Chemin vers le fichier JSON contenant les captions enrichies (obligatoire si --use_veclip_caption est activé).")
    parser.add_argument("--output_json", type=str, default="ground_truth.json", help="Nom du fichier de sortie JSON.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion des singletons à utiliser pour la validation.")
    parser.add_argument("--seed", type=int, default=2, help="Seed pour la reproductibilité du split.")

    args = parser.parse_args()

    # Définir la seed
    set_seed(args.seed)

    # Charger les descriptions
    if args.use_veclip_caption:
        if not args.veclip_json_file:
            raise ValueError("Le fichier JSON VECLIP est requis si --use_veclip_caption est activé.")
        description_dict = load_veclip_captions(args.veclip_json_file)
    else:
        description_dict = load_descriptions(args.csv_descriptions_file)

    # Charger et corriger les groupes
    original_groups = correct_filenames(load_groundtruth(args.excel_file), args.image_folder, description_dict)

    # Obtenir les singletons (images non présentes dans les groupes)
    singletons = get_singletons(original_groups, args.image_folder, description_dict)

    # Diviser les singletons en train/val
    train_singletons, val_singletons = train_test_split(singletons, test_size=args.test_size, random_state=args.seed)

    # Sauvegarder dans un JSON
    save_groundtruth_json(original_groups, train_singletons, val_singletons, args.output_json, description_dict)


if __name__ == "__main__":
    main()