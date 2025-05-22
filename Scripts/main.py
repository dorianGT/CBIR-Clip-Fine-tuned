import argparse
import random
import numpy as np
import torch
import os

from fine_tune_model import fine_tune
from models import load_clip_model, load_clip_model_with_lora, load_fine_tuned_model, load_fine_tuned_model_with_lora
from generate_embeddings import generate_embeddings
from faiss_index import create_faiss_index
from evaluation import evaluate

def set_seed(seed=2):
    """
    Fixe toutes les seeds pour assurer la reproductibilité des résultats.
    
    Args:
        seed (int): Valeur de la seed à utiliser pour random, numpy et torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(model_folder, image_folder, load_method, do_finetune=True, gen_emb = True, epochs=20, batch_size=64, patience=3, learning_rate=1e-4, seed = 2):
    """
    Fonction principale du pipeline : chargement du modèle, fine-tuning (optionnel),
    génération d'embeddings, création de l'index FAISS et évaluation.
    
    Args:
        model_folder (str): Dossier de sauvegarde/chargement du modèle.
        load_method (str): Méthode de chargement du modèle parmi ["clip", "clip+lora", "finetuned", "finetuned+lora"].
        do_finetune (bool): Indique s'il faut faire le fine-tuning ou non.
        gen_emb (bool): Indique s'l faut générer les embeddings ou non.
        epochs (int): Nombre d'époques pour l'entraînement.
        batch_size (int): Taille des batchs pour le fine-tuning.
        patience (int): Patience pour l'early stopping.
        learning_rate (float): Taux d'apprentissage.
        seed (int): Seed pour reproductibilité.
    """
    set_seed(seed)

    print("[1/6] Chargement du modèle...")
    if load_method == "clip":
        model, preprocess, device = load_clip_model()
    elif load_method == "clip+lora":
        model, preprocess, device = load_clip_model_with_lora()
    elif load_method == "finetuned":
        model, preprocess, device = load_fine_tuned_model(model_folder)
    elif load_method == "finetuned+lora":
        model, preprocess, device = load_fine_tuned_model_with_lora(model_folder)
    else:
        raise ValueError(f"Méthode de chargement inconnue : {load_method}")
    print("[1/6] Modèle chargé avec succès.")

    os.makedirs(model_folder, exist_ok=True)

    if do_finetune:
        print("[2/6] Démarrage du fine-tuning...")
        use_lora = load_method in ["clip+lora", "finetuned+lora"]
        fine_tune(
            model_folder, model, preprocess, device,seed,image_folder,
            epochs=epochs, batch_size=batch_size, patience=patience,
            learning_rate=learning_rate, use_lora=use_lora
        )
        print("[2/6] Fine-tuning terminé.")

        print("[3/6] Rechargement du modèle fine-tuné...")
        if use_lora:
            model, preprocess, device = load_fine_tuned_model_with_lora(model_folder)
        else:
            model, preprocess, device = load_fine_tuned_model(model_folder)
        print("[3/6] Modèle fine-tuné rechargé.")

    if generate_embeddings :
        print("[4/6] Génération des embeddings...")
        generate_embeddings(model_folder, image_folder, model, preprocess, device)
        print("[4/6] Embeddings générés avec succès.")

        print("[5/6] Création de l'index FAISS...")
        create_faiss_index(model_folder)
        print("[5/6] Index FAISS créé.")

    print("[6/6] Évaluation du modèle...")
    evaluate(model_folder)
    print("[6/6] Évaluation terminée.")

    info_path = os.path.join(model_folder, "training_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Load Method: {load_method}\n")
        f.write(f"Fine-tuning: {do_finetune}\n")
        if do_finetune:
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Patience: {patience}\n")
            f.write(f"Learning Rate: {learning_rate}\n")
    print(f"Informations sauvegardées dans '{info_path}'")

if __name__ == "__main__":
    # Parsing des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Fine-tune, embed, index and evaluate a model.")
    parser.add_argument("--model_folder", type=str, required=True, help="Chemin du dossier du modèle")
    parser.add_argument("--image_folder", type=str, required=True, help="Chemin du dossier contenant les images")
    parser.add_argument("--load_method", type=str, choices=["clip", "clip+lora", "finetuned", "finetuned+lora"], required=True, help="Méthode de chargement du modèle")
    parser.add_argument("--do_finetune", type=lambda x: (str(x).lower() == 'true'), default=True, help="Faire le fine-tuning (True/False)")
    parser.add_argument("--generate_embeddings", type=lambda x: (str(x).lower() == 'true'), default=True, help="Générer les embeddings (True/False)")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques pour le fine-tuning")
    parser.add_argument("--batch_size", type=int, default=64, help="Taille des batchs pour l'entraînement")
    parser.add_argument("--patience", type=int, default=3, help="Patience pour l'early stopping")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Taux d'apprentissage pour l'optimiseur")
    parser.add_argument("--seed", type=int, default=2, help="Seed pour reproductibilité")
    
    args = parser.parse_args()
    main(args.model_folder, args.image_folder, args.load_method, args.do_finetune, args.generate_embeddings, args.epochs, args.batch_size, args.patience, args.learning_rate, args.seed)