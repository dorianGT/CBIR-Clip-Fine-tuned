import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from parse_groundtruth import load_groundtruth_json
import clip

def get_image_paths(folder):
    """
    Récupère les chemins des images valides dans un dossier.

    Args:
        folder (str): Dossier contenant les images.

    Returns:
        List[str]: Liste des chemins d'images.
    """
    exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def get_caption_for_image(image_name, train, train_texts, val, val_texts, groups):
    """
    Trouve la légende (caption) associée à une image donnée.

    Args:
        image_name (str): Nom du fichier image.
        train (List[str]): Liste des noms d'images d'entraînement.
        train_texts (List[str]): Captions associées aux images d'entraînement.
        val (List[str]): Liste des noms d'images de validation.
        val_texts (List[str]): Captions associées aux images de validation.
        groups (List[List[Dict]]): Groupes d'images avec leur caption (issues de ground_truth.json).

    Returns:
        str: Caption associée à l'image, ou chaîne vide si non trouvée.
    """
    if image_name in train:
        idx = train.index(image_name)
        return train_texts[idx]
    elif image_name in val:
        idx = val.index(image_name)
        return val_texts[idx]
    else:
        for group in groups:
            for img in group:
                if img["image"] == image_name:
                    return img["caption"]
    return ""


def encode(model, preprocess, device, image_paths, train, train_texts, val, val_texts, groups):
    """
    Encode une liste d'images en vecteurs d'embeddings à l'aide d'un modèle, 
    et retourne également les captions correspondantes.

    Args:
        model (torch.nn.Module): Modèle d'encodage (ex : modèle CLIP).
        preprocess (Callable): Fonction de prétraitement des images.
        device (str): Appareil à utiliser ("cuda" ou "cpu").
        image_paths (List[str]): Liste des chemins complets des images à encoder.
        train (List[str]): Liste des noms d'images d'entraînement.
        train_texts (List[str]): Captions associées aux images d'entraînement.
        val (List[str]): Liste des noms d'images de validation.
        val_texts (List[str]): Captions associées aux images de validation.
        groups (List[List[Dict]]): Groupes d'images avec leur caption (issues de ground_truth.json).

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: 
            - Embeddings des images (shape: [N, D])
            - Embeddings des captions associées (shape: [N, D])
            - Liste des noms de fichiers images correspondants
    """
    embeddings_image = []
    embeddings_text = []
    valid_paths = []

    for path in tqdm(image_paths[::-1], desc="Encoding images", unit="image"):
        try:
            with Image.open(path) as img:
                img = preprocess(img).unsqueeze(0).to(device)
                image_name = os.path.basename(path)
                caption = get_caption_for_image(image_name, train, train_texts, val, val_texts, groups)
                with torch.no_grad():
                    embedding_image = model.encode_image(img)
                    if model.__module__.startswith("clip"):
                        text_tokens = clip.tokenize(caption).to(device)
                    else:
                        # open_clip case
                        text_tokens = model.tokenize(caption).to(device)
                    embedding_text = model.encode_text(text_tokens)

                embeddings_image.append(embedding_image.cpu().numpy())
                embeddings_text.append(embedding_text.cpu().numpy())
                valid_paths.append(image_name)

        except (UnidentifiedImageError, IOError):
            print(f"Skipped: {path}")

    return np.vstack(embeddings_image), np.vstack(embeddings_text), valid_paths


def save_embeddings(model_folder, embeddings_image, embeddings_text,embeddings_combined, paths):
    """
    Sauvegarde les embeddings des images, les embeddings des captions, et les embeddings combinés dans des fichiers .npz.

    Args:
        model_folder (str): Dossier où sauvegarder les fichiers embeddings.
        embeddings_image (np.ndarray): Embeddings des images.
        embeddings_text (np.ndarray): Embeddings des captions associées.
        embeddings_combined (np.ndarray): Embeddings combinés image + texte.
        paths (List[str]): Liste des noms de fichiers images correspondants.
    """
    np.savez(os.path.join(model_folder, "embeddings_image.npz"), embeddings=embeddings_image, paths=np.array(paths, dtype="object"))
    np.savez(os.path.join(model_folder, "embeddings_text.npz"), embeddings=embeddings_text, paths=np.array(paths, dtype="object"))
    np.savez(os.path.join(model_folder, "embeddings_combined.npz"), embeddings=embeddings_combined, paths=np.array(paths, dtype="object"))

def generate_embeddings(model_folder, image_folder, model, preprocess, device):
    """
    Génère et sauvegarde les embeddings d'un ensemble d'images et de leurs captions associées.

    Args:
        model_folder (str): Dossier où sauvegarder les fichiers d'embeddings.
        image_folder (str): Dossier contenant les images à encoder.
        model (torch.nn.Module): Modèle utilisé pour encoder les images et les textes.
        preprocess (Callable): Fonction de prétraitement des images.
        device (str): Appareil à utiliser ("cuda" ou "cpu").

    Effets de bord:
        - Sauvegarde les fichiers embeddings_image.npz, embeddings_text.npz et embeddings_combined.npz dans model_folder.
        - Affiche le nombre total d'embeddings sauvegardés.
    """
    groups, train, train_texts, val, val_texts = load_groundtruth_json("ground_truth.json")
    image_paths = get_image_paths(image_folder)
    embeddings_image,embeddings_text, valid_paths = encode(model, preprocess, device, image_paths, train, train_texts, val, val_texts, groups)
    
    alpha = 0.5  # pondération entre 0 et 1
    combined_embeddings = alpha * embeddings_image + (1 - alpha) * embeddings_text

    save_embeddings(model_folder, embeddings_image,embeddings_text,combined_embeddings, valid_paths)

    print(f"{len(valid_paths)} embeddings sauvegardés dans embeddings_image.npz, embeddings_text.npz et embeddings_combined.npz.")

# from models import load_clip_model, load_fine_tuned_model, load_fine_tuned_model_with_lora

# def main():
#     """
#     Fonction principale : charge le modèle, encode les images et sauvegarde les embeddings.
#     """
#     model, preprocess, device = load_fine_tuned_model_with_lora("fine_tuned_clip_model_with_miner_3")
#     generate_embeddings("fine_tuned_clip_model_with_miner_3","HighVision_Corpus_Groundtruth/historicaldataset", model, preprocess, device)

# if __name__ == "__main__":
#     main()
