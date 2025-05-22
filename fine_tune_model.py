from parse_groundtruth import load_groundtruth_json
from models import CLIPWithProjector
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch

from models import load_clip_model_with_lora
from torchvision import transforms

import random
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

class ImageTextDataset(Dataset):
    """
    Dataset personnalisé pour associer des images et des descriptions textuelles.

    Args:
        image_folder (str): Dossier contenant les images.
        image_names (List[str]): Liste des noms de fichiers image.
        descriptions (List[str]): Liste des descriptions textuelles.
        transform (callable): Transformations à appliquer aux images.
    """
    def __init__(self, image_folder, image_names, descriptions, transform):
        self.image_folder = image_folder
        self.image_names = image_names
        self.descriptions = descriptions
        self.transform = transform
        self.data = [(img, desc) for img, desc in zip(image_names, descriptions)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name, description = self.data[idx]
        img_path = os.path.join(self.image_folder, img_name)

        image = Image.open(img_path).convert("RGB")

        return self.transform(image), description

def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Calcule la perte contrastive CLIP (InfoNCE symétrique).

    Args:
        image_embeddings (Tensor): Embeddings des images.
        text_embeddings (Tensor): Embeddings des textes.
        temperature (float): Température pour la normalisation des logits.

    Returns:
        torch.Tensor: Perte contrastive moyenne.
    """
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    logits_per_image = image_embeddings @ text_embeddings.T
    logits_per_text = text_embeddings @ image_embeddings.T

    logits_per_image /= temperature
    logits_per_text /= temperature

    labels = torch.arange(len(image_embeddings)).to(image_embeddings.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2

def fine_tune_clip(model_folder, model: CLIPWithProjector, train_dataset, val_dataset,
                   device, epochs=20, batch_size=16, lr=1e-5, patience=5):
    """
    Fine-tune un modèle CLIP avec projecteur à l'aide d'une perte contrastive.

    Args:
        model_folder (str): Emplacement de sauvegarde du modèle.
        model (CLIPWithProjector): Modèle CLIP avec projecteur.
        train_dataset (Dataset): Dataset d'entraînement.
        val_dataset (Dataset): Dataset de validation.
        device (torch.device): Périphérique d'entraînement.
        epochs (int): Nombre d'époques.
        batch_size (int): Taille des batches.
        lr (float): Taux d'apprentissage.
        patience (int): Nombre d'époques sans amélioration avant early stopping.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (images, texts) in enumerate(train_loader):
            images = images.to(device)
            tokenized = model.tokenize(texts).to(device)

            image_embeds, text_embeds = model(images, tokenized)
            loss = clip_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            print(f"[Train] Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"[Train] Epoch {epoch+1}, Average Loss: {avg_train_loss:.4f}")

        # Phase de validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device)
                tokenized = model.tokenize(texts).to(device)

                image_embeds, text_embeds = model(images, tokenized)
                loss = clip_loss(image_embeds, text_embeds)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        val_losses.append(avg_val_loss)
        print(f"[Val] Epoch {epoch+1}, Average Validation Loss: {avg_val_loss:.4f}")

        # Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"--> Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"--> No improvement. Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Chargement du meilleur modèle
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Sauvegarde des courbes de loss
    with open(f"{model_folder}/losses.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(model_folder, "loss_curve.png"))
    plt.close()

    print("Fine-tuning terminé!")

def fine_tune(model_folder, model_clip, preprocess, device, seed, image_folder, ground_truth_file = "ground_truth.json",
              epochs=20, batch_size=64, patience=3, learning_rate=1e-4, use_lora=False):
    """
    Fonction principale de fine-tuning du modèle CLIP avec ajout d'un projecteur.

    Args:
        model_folder (str): Dossier où sauvegarder le modèle.
        model_clip (nn.Module): Modèle CLIP de base (chargé depuis OpenCLIP ou HuggingFace).
        preprocess (Callable): Transformations d'entrée pour CLIP.
        device (torch.device): Appareil d'entraînement.
        seed (int): La seed.
        image_folder (str): Dossier contenant les images.
        epochs (int): Nombre d'époques.
        batch_size (int): Taille des lots.
        patience (int): Patience pour early stopping.
        learning_rate (float): Taux d'apprentissage.
        use_lora (bool): Indique si LoRA est utilisé (pour sauvegarde HuggingFace).
    """
    
    set_seed(seed)

    # Charger les données texte/image à partir d'un fichier JSON
    groups, train, train_texts, val, val_texts = load_groundtruth_json(ground_truth_file)

    # Bloquer tous les paramètres pour le fine-tuning
    for param in model_clip.parameters():
        param.requires_grad = False

    # Unfreeze partiel des 6 derniers blocs du Transformer visuel
    # for block in model_clip.visual.transformer.resblocks[-3:]:
    #     for param in block.parameters():
    #         param.requires_grad = True

    # # Unfreeze partiel des 6 derniers blocs du Transformer textuel
    # for block in model_clip.transformer.resblocks[-6:]:
    #     for param in block.parameters():
    #         param.requires_grad = True

    # Création du modèle CLIP avec tête de projection
    model = CLIPWithProjector(model_clip)
    model.to(device)

    # Transforms pour les datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    train_dataset = ImageTextDataset(image_folder, train, train_texts, transform)
    val_dataset = ImageTextDataset(image_folder, val, val_texts, transform)

    # Fine-tuning du modèle
    fine_tune_clip(model_folder=model_folder,
                   model=model,
                   train_dataset=train_dataset,
                   val_dataset=val_dataset,
                   device=device,
                   epochs=epochs,
                   batch_size=batch_size,
                   patience=patience,
                   lr=learning_rate)

    # Sauvegarde du modèle fine-tuné
    if use_lora:
        model.clip_model.save_pretrained(model_folder)  # Format HuggingFace
    else:
        torch.save(model.clip_model.state_dict(), os.path.join(model_folder, "clip_model.pth"))

    torch.save(model.image_projector.state_dict(), os.path.join(model_folder, "image_projector.pth"))
    torch.save(model.text_projector.state_dict(), os.path.join(model_folder, "text_projector.pth"))

    print("Modèle fine-tuné et sauvegardé!")


# def main():
#     """
#     Fonction principale qui charge les données, initialise le modèle, entraîne et sauvegarde le modèle fine-tuné.
#     """
#     model_clip, preprocess, device = load_clip_model_with_lora()
#     fine_tune("fine_tuned_clip_model_with_miner_3", model_clip, preprocess, device)
 
# if __name__ == "__main__":
#     main()
