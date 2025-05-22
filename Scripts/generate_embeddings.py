import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

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

def encode_images(model, preprocess, device, image_paths):
    """
    Encode une liste d'images en vecteurs d'embedding à l'aide d'un modèle.

    Args:
        model (torch.nn.Module): Modèle d'encodage (ex : CLIP).
        preprocess (Callable): Fonction de prétraitement des images.
        device (str): Appareil à utiliser ("cuda" ou "cpu").
        image_paths (List[str]): Liste des chemins d'images à encoder.

    Returns:
        np.ndarray: Matrice d'embeddings de taille (N, D).
        List[str]: Chemins valides des images encodées (correspondant aux embeddings).
    """
    embeddings, valid_paths = [], []

    for path in tqdm(image_paths[::-1], desc="Encoding images", unit="image"):
        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                try:
                    img.load()
                except:
                    print(f"Erreur: Impossible de charger {path}")
                    continue
                img = preprocess(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = model.encode_image(img)

                embeddings.append(embedding.cpu().numpy())
                valid_paths.append(os.path.basename(path))

        except (UnidentifiedImageError, IOError):
            print(f"Skipped: {path}")

    if not embeddings:
        raise RuntimeError("Aucune image valide trouvée.")

    return np.vstack(embeddings), valid_paths

def save_embeddings(model_folder, embeddings, paths):
    """
    Sauvegarde les embeddings et les chemins d’images associés dans un fichier .npz.

    Args:
        model_folder (str): Dossier où sauvegarder le fichier embeddings.npz.
        embeddings (np.ndarray): Embeddings des images.
        paths (List[str]): Noms de fichiers images correspondants.
    """
    np.savez(os.path.join(model_folder, "embeddings.npz"), embeddings=embeddings, paths=np.array(paths, dtype="object"))

def generate_embeddings(model_folder, image_folder, model, preprocess, device):
    """
    Génère et sauvegarde les embeddings pour toutes les images valides d’un dossier.

    Args:
        model_folder (str): Dossier où sauvegarder les résultats (embeddings.npz).
        image_folder (str): Dossier contenant les images à encoder.
        model (torch.nn.Module): Modèle utilisé pour encoder les images.
        preprocess (Callable): Fonction de prétraitement des images.
        device (str): Appareil à utiliser ("cuda" ou "cpu").
    """
    image_paths = get_image_paths(image_folder)
    embeddings, valid_paths = encode_images(model, preprocess, device, image_paths)
    save_embeddings(model_folder, embeddings, valid_paths)

    print(f"Saved {len(valid_paths)} embeddings.")

# from models import load_clip_model, load_fine_tuned_model, load_fine_tuned_model_with_lora

# def main():
#     """
#     Fonction principale : charge le modèle, encode les images et sauvegarde les embeddings.
#     """
#     model, preprocess, device = load_fine_tuned_model_with_lora("fine_tuned_clip_model_with_miner_3")
#     generate_embeddings("fine_tuned_clip_model_with_miner_3","HighVision_Corpus_Groundtruth/historicaldataset", model, preprocess, device)

# if __name__ == "__main__":
#     main()
