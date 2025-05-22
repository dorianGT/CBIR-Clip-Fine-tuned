import os
import shutil
import argparse
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

def is_image_valid(path):
    """
    Tente d'ouvrir et de charger complètement une image pour vérifier qu'elle n'est pas corrompue.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # Vérification rapide sans chargement complet
        with Image.open(path) as img:
            img.load()    # Chargement complet des données de l'image
        return True
    except Exception:
        return False

def find_corrupted_images(image_dir, log_file="corrupted_images.txt"):
    """
    Parcourt un dossier d'images, détecte celles corrompues (impossibles à charger),
    les déplace dans un sous-dossier 'Corrupted' et enregistre leurs noms dans un fichier.

    Args:
        image_dir (str): Chemin du dossier contenant les images à vérifier.
        log_file (str): Nom du fichier où enregistrer les noms des images corrompues.
    """
    corrupted = []
    corrupted_dir = os.path.join(image_dir, "Corrupted")

    os.makedirs(corrupted_dir, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]

    for filename in tqdm(files[:-1], desc="Vérification images", unit="image"):
        file_path = os.path.join(image_dir, filename)

        if not is_image_valid(file_path):
            print(f"Image corrompue détectée : {filename}")
            corrupted.append(filename)

            dest_path = os.path.join(corrupted_dir, filename)
            shutil.move(file_path, dest_path)

    # Écriture des noms des images corrompues dans un fichier log
    if corrupted:
        log_path = os.path.join(image_dir, log_file)
        with open(log_path, "w") as f:
            for name in corrupted:
                f.write(name + "\n")
        print(f"{len(corrupted)} images corrompues déplacées vers '{corrupted_dir}' et listées dans '{log_file}'.")
    else:
        print("Aucune image corrompue détectée.")

def main():
    parser = argparse.ArgumentParser(description="Détecte et déplace les images corrompues dans un dossier.")
    parser.add_argument("--image_folder", type=str, help="Chemin du dossier contenant les images à vérifier.")
    args = parser.parse_args()

    if not os.path.isdir(args.image_folder):
        print(f"Erreur : '{args.image_folder}' n'est pas un dossier valide.")
        return

    find_corrupted_images(args.image_folder)

if __name__ == "__main__":
    main()
