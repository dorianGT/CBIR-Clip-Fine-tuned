import torch
import torch.nn.functional as F
import clip
import os

from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig

class CLIPWithProjector(torch.nn.Module):
    """
    Enveloppe personnalisée autour d'un modèle CLIP avec des projecteurs (MLP) 
    appliqué sur les embeddings d'image et de texte.
    """

    def __init__(self, clip_model, projector_dim=512):
        """
        Initialise l'enveloppe CLIP avec des projecteurs.

        Args:
            clip_model (torch.nn.Module): Modèle CLIP pré-entraîné.
            projector_dim (int): Dimension de sortie du projecteur.
        """
        super(CLIPWithProjector, self).__init__()
        self.clip_model = clip_model.float()

        # Projecteur pour les features image
        self.image_projector = torch.nn.Sequential(
            torch.nn.Linear(clip_model.visual.output_dim, projector_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(projector_dim, projector_dim),
        )

        # Projecteur pour les features texte
        self.text_projector = torch.nn.Sequential(
            torch.nn.Linear(clip_model.text_projection.shape[1], projector_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(projector_dim, projector_dim),
        )

    def tokenize(self, texts):
        """Tokenise une liste de textes à l’aide du tokenizer CLIP."""
        return clip.tokenize(texts)

    def encode_image(self, images):
        """
        Encode les images en passant par CLIP puis applique un projecteur.

        Args:
            images (torch.Tensor): Batch d'images [B, 3, H, W].

        Returns:
            torch.Tensor: Vecteurs projetés normalisés.
        """
        features = self.clip_model.encode_image(images)
        features = F.normalize(features, dim=-1, p=2).to(dtype=torch.float32)
        projected = self.image_projector(features)
        return F.normalize(projected, dim=-1, p=2)

    def encode_text(self, tokenized_texts):
        """
        Encode les textes à l’aide du modèle CLIP puis applique le projecteur.

        Args:
            tokenized_texts (torch.Tensor): Batch de textes tokenisés.

        Returns:
            torch.Tensor: Vecteurs de texte normalisés.
        """
        features = self.clip_model.encode_text(tokenized_texts)
        features = F.normalize(features.to(dtype=torch.float32), dim=-1, p=2)
        projected = self.text_projector(features)
        return F.normalize(projected, dim=-1, p=2)

    def forward(self, images, texts):
        """
        Applique les encodeurs image et texte pour obtenir des représentations projetées.

        Args:
            images (torch.Tensor): Batch d'images.
            texts (torch.Tensor): Batch de textes tokenisés.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features projetés d'image et texte.
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # # Sanity check pour éviter les NaNs
        # assert not torch.isnan(images).any(), "Images contain NaNs"
        # assert not torch.isnan(image_features).any(), "Image features contain NaNs"
        # assert not torch.isnan(text_features).any(), "Text features contain NaNs"

        return image_features, text_features


def load_clip_model_with_lora():
    """
    Charge le modèle CLIP ViT-B/32 avec un adapter LoRA pour fine-tuning léger.

    Returns:
        model (torch.nn.Module): CLIP avec LoRA.
        preprocess (Callable): Fonction de prétraitement d'images.
        device (str): "cuda" si dispo, sinon "cpu".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Définition de la configuration LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["attn.out_proj", "mlp.c_fc", "mlp.c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # Ajout des adaptateurs LoRA au modèle
    model = get_peft_model(model, config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")

    return model, preprocess, device


def load_clip_model():
    """
    Charge le modèle CLIP ViT-B/32 et déverrouille tous les paramètres pour un fine-tuning complet.

    Returns:
        model (torch.nn.Module): CLIP prêt pour fine-tuning.
        preprocess (Callable): Fonction de prétraitement d'images.
        device (str): Appareil utilisé.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Permet de mettre à jour tous les poids pendant le fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    return model, preprocess, device


def load_fine_tuned_model(model_folder):
    """
    Charge un modèle CLIP fine-tuné avec projecteurs, sans LoRA.

    Args:
        model_folder (str): Dossier contenant les fichiers clip_model.pth et projector.pth.

    Returns:
        model (torch.nn.Module): CLIP enrichi des projecteurs.
        preprocess (Callable): Fonction de prétraitement.
        device (str): "cuda" ou "cpu".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger les poids du modèle CLIP
    clip_model_path = os.path.join(model_folder, "clip_model.pth")
    base_model, preprocess = clip.load("ViT-B/32", device=device)
    base_model.load_state_dict(torch.load(clip_model_path, map_location=device))

    # Ajout du projecteur
    model = CLIPWithProjector(base_model)

    # Charger les poids des projecteur
    projector_path = os.path.join(model_folder, "image_projector.pth")
    model.image_projector.load_state_dict(torch.load(projector_path, map_location=device))

    projector_path = os.path.join(model_folder, "text_projector.pth")
    model.text_projector.load_state_dict(torch.load(projector_path, map_location=device))

    model.to(device)
    model.eval()

    return model, preprocess, device


def load_fine_tuned_model_with_lora(model_folder):
    """
    Charge un modèle CLIP fine-tuné avec LoRA et un projecteur.

    Args:
        model_folder (str): Dossier contenant les poids fine-tunés de LoRA + projector.pth.

    Returns:
        model (torch.nn.Module): CLIP avec adaptateurs LoRA et projecteur.
        preprocess (Callable): Fonction de prétraitement.
        device (str): Appareil utilisé.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger le modèle CLIP de base
    base_model, preprocess = clip.load("ViT-B/32", device=device)

    # Reproduire la config LoRA utilisée à l'entraînement
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["attn.out_proj", "mlp.c_fc", "mlp.c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # Appliquer LoRA
    model_lora = get_peft_model(base_model, config)

    # Charger les poids fine-tunés de LoRA
    model_lora = PeftModel.from_pretrained(model_lora, model_folder)

    # Ajouter le projecteur
    model = CLIPWithProjector(model_lora)

    # Charger les poids du projecteur
    projector_path = os.path.join(model_folder, "image_projector.pth")
    model.image_projector.load_state_dict(torch.load(projector_path, map_location=device))

    projector_path = os.path.join(model_folder, "text_projector.pth")
    model.text_projector.load_state_dict(torch.load(projector_path, map_location=device))

    model.to(device)
    model.eval()

    return model, preprocess, device
